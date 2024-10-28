import torch
import torch.nn as nn                   # for builtin modules including Linear, Conv2d, MultiheadAttention, LayerNorm, etc
import torch.nn.functional as F
from torch.nn import ModuleDict         # for layer naming when nn.Sequential is not viable
import numpy as np                      # for some basic dimention computation, might be redundent

from math import ceil, floor
from collections import OrderedDict

# typing
from torch import Tensor, LongTensor
from typing import Dict, List, Tuple
from ctypes import Union

from config.model_config import ModelConfig as Config


##################
## Actual model ##
##################
class Model(nn.Module):
    def __init__(self, ModelConfig):
        super(Model, self).__init__()
        # feature configure parameter
        self.model_name = Config.NETWORK_NAME

        # build onnx
        self.build_onnx = False

        # lstm
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_size = Config.LSTM_UNIT_SIZE

        # data
        self.hero_data_split_shape = Config.HERO_DATA_SPLIT_SHAPE
        self.hero_seri_vec_split_shape = Config.HERO_SERI_VEC_SPLIT_SHAPE
        self.hero_feature_img_channel = Config.HERO_FEATURE_IMG_CHANNEL
        self.hero_label_size_list = Config.HERO_LABEL_SIZE_LIST
        self.hero_is_reinforce_task_list = Config.HERO_IS_REINFORCE_TASK_LIST

        # loss
        self.learning_rate = Config.INIT_LEARNING_RATE_START
        self.var_beta = Config.BETA_START
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.min_policy = Config.MIN_POLICY
        self.embedding_trainable = False
        self.value_head_num = Config.VALUE_HEAD_NUM

        # value
        self.value_head_num = Config.VALUE_HEAD_NUM
        self.hero_policy_weight = Config.HERO_POLICY_WEIGHT

        # target attention
        self.target_embedding_dim = Config.TARGET_EMBEDDING_DIM
        # num
        self.hero_num = 3
        self.hero_data_len = sum(Config.data_shapes[0])

        # build network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 18, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(18, 12, 3, 1, 1)
        )
        self.hero_share_mlp = nn.Sequential(
            nn.Linear(251, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.hero_frd_mlp = nn.Linear(256, 128)
        self.hero_emy_mlp = nn.Linear(256, 128)
        self.public_info_mlp = nn.Sequential(
            nn.Linear(44, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.soldier_share_mlp = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.soldier_frd_mlp = nn.Linear(64, 64)
        self.soldier_emy_mlp = nn.Linear(64, 64)
        self.organ_share_mlp = nn.Sequential(
            nn.Linear(29, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.organ_frd_mlp = nn.Linear(64, 64)
        self.organ_emy_mlp = nn.Linear(64, 64)
        self.monster_mlp = nn.Sequential(
            nn.Linear(28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.concat_mlp = nn.Sequential(
            nn.Linear(768+128*2+64+64+64*2+64*2+68, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.target_embed = nn.Linear(64, 64)
        self.action_heads = nn.ModuleList()
        for action_dim in (13, 25, 42, 42):
            self.action_heads.append(
                nn.Sequential(
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim)
                )
            )
        self.target_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.value_mlp = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data_list, data_list_with_label=None, inference=False):
        if self.build_onnx:
            data_list = self.format_data(data_list)
        
        if data_list_with_label is not None:
            data_list_with_label = self.format_data_with_label(data_list_with_label)

        all_hero_result_list = []
        hero_public_first_result_list = []
        hero_public_second_result_list = []
        all_hero_target_list = []

        for hero_index, hero_data in enumerate(data_list):
            if not inference:
                hero_feature = hero_data[0]
            else:
                hero_feature = hero_data

            feature_img, feature_vec = hero_feature.split([17 * 17 * 6, 2852], dim = 1)

            feature_img = feature_img.reshape((-1, 6, 17, 17))
            feature_vec = feature_vec.reshape((-1, 2852))

            conv_hidden = self.conv_layers(feature_img).flatten(start_dim = 1) # B*12*8*8

            hero_frd, hero_emy, public_info, soldier_frd, soldier_emy, organ_frd, organ_emy, monster_vec, global_info = feature_vec.split(
                [251 * 3, 251 * 3, 44, 25 * 10, 25 * 10, 29 * 3, 29 * 3, 28 * 20, 68], dim = 1
            )
            #改成reshape
            hero_frd_list = hero_frd.reshape((-1, 3, 251))
            hero_emy_list = hero_emy.reshape((-1, 3, 251))
            soldier_frd_list = soldier_frd.reshape((-1, 10, 25))
            soldier_emy_list = soldier_emy.reshape((-1, 10, 25))
            organ_frd_list = organ_frd.reshape((-1, 3, 29))
            organ_emy_list = organ_emy.reshape((-1, 3, 29))
            monster_list = monster_vec.reshape((-1, 20, 28))

            hero_target_list = []

            hero_frd_hidden = self.hero_frd_mlp(self.hero_share_mlp(hero_frd_list))
            hero_target_list.append(hero_frd_hidden[:,:,-64:]) # 3 frd hero
            hero_frd_hidden_pool, _ = hero_frd_hidden.max(dim = 1)
            hero_emy_hidden = self.hero_emy_mlp(self.hero_share_mlp(hero_emy_list))
            hero_target_list.append(hero_emy_hidden[:,:,-64:]) # 3 emy hero
            hero_emy_hidden_pool, _ = hero_emy_hidden.max(dim = 1)
            public_info_hidden = self.public_info_mlp(public_info)
            hero_target_list.append(public_info_hidden.unsqueeze(1)) # 1 public info
            monster_hidden = self.monster_mlp(monster_list)
            monster_hidden_pool, _ = monster_hidden.max(dim = 1)
            hero_target_list.append(monster_hidden) # 20 monster
            soldier_frd_hidden = self.soldier_frd_mlp(self.soldier_share_mlp(soldier_frd_list))
            soldier_frd_hidden_pool, _ = soldier_frd_hidden.max(dim = 1)
            soldier_emy_hidden = self.soldier_emy_mlp(self.soldier_share_mlp(soldier_emy_list))
            soldier_emy_hidden_pool, _ = soldier_emy_hidden.max(dim = 1)
            hero_target_list.append(soldier_emy_hidden) # 10 emy soldier
            organ_frd_hidden = self.organ_frd_mlp(self.organ_share_mlp(organ_frd_list))
            organ_frd_hidden_pool, _ = organ_frd_hidden.max(dim = 1)
            organ_emy_hidden = self.organ_emy_mlp(self.organ_share_mlp(organ_emy_list))
            organ_emy_hidden_pool, _ = organ_emy_hidden.max(dim = 1)
            hero_target_list.append(organ_emy_hidden_pool.reshape((-1, 1, 64))) # 1 emy organ
            hero_target_list.insert(0, torch.ones_like(hero_target_list[2], dtype=torch.float32) * 0.1)
            all_hero_target_list.append(torch.cat(hero_target_list, dim = 1))

            concat_hidden = torch.cat([conv_hidden, hero_frd_hidden_pool, hero_emy_hidden_pool, public_info_hidden, soldier_frd_hidden_pool, soldier_emy_hidden_pool, organ_frd_hidden_pool, organ_emy_hidden_pool, monster_hidden_pool, global_info], dim = 1) # 768+64+64+32+32+32+32+32+32+68=1156
            concat_hidden = self.concat_mlp(concat_hidden)

            concat_hidden_split = concat_hidden.split((64, 192), dim = 1)
            hero_public_first_result_list.append(concat_hidden_split[0])
            hero_public_second_result_list.append(concat_hidden_split[1])

        pool_hero_public, _ = torch.stack(hero_public_first_result_list, dim=1).max(dim=1)

        for hero_index in range(self.hero_num):
            hero_result_list = []
            fc_public_result = torch.cat([pool_hero_public, hero_public_second_result_list[hero_index]], dim = 1)
            # 4 action head
            for action_head in self.action_heads:
                hero_result_list.append(action_head(fc_public_result))
            # target head
            target_embedding = self.target_embed(all_hero_target_list[hero_index]) # B*39*32
            target_key = self.target_head(fc_public_result).unsqueeze(-1) # B*32*1
            target_logits = torch.matmul(target_embedding, target_key).squeeze(-1) # B*39
            hero_result_list.append(target_logits)
            # value head
            hero_result_list.append(self.value_mlp(fc_public_result))

            all_hero_result_list.append(hero_result_list)
        
        # return all_hero_result_list

        ### Output Projection ###
        all_hero_predict_result_list = all_hero_result_list

        # rst_list = all_hero_predict_result_list
        # total_loss, info_list = self.compute_loss(data_list, rst_list)  # for debug. merge loss compute
        # return total_loss, info_list

        self.probs_h0 = torch.flatten(torch.cat(all_hero_predict_result_list[0], 1), start_dim=1)
        self.probs_h1 = torch.flatten(torch.cat(all_hero_predict_result_list[1], 1), start_dim=1)
        self.probs_h2 = torch.flatten(torch.cat(all_hero_predict_result_list[2], 1), start_dim=1)

        # self.lstm_cell_output = self.lstm_cell
        # self.lstm_hidden_output = self.lstm_hidden

        self.lstm_cell_output = torch.tensor([])
        self.lstm_hidden_output = torch.tensor([])

        rst_list = [self.probs_h0, self.probs_h1, self.probs_h2, self.lstm_cell_output, self.lstm_hidden_output]

        if not inference:
            print([x.shape for x in rst_list])
        return rst_list

    def format_data(self, datas, inference=False):
        hero_data_list = []
        for hero_index in range(self.hero_num):
            hero_data = datas[hero_index].float()
            hero_data.view(-1, self.hero_data_split_shape[hero_index][0])
            hero_data = torch.unsqueeze(hero_data, 0)
            hero_data_list.append(hero_data)
        self.lstm_cell = datas[-2].float()
        self.lstm_hidden = datas[-1].float()

        return hero_data_list

    def format_data_with_label(self, datas):
        datas = datas.view(-1, self.hero_num, self.hero_data_len)
        data_list = datas.permute(1, 0, 2)

        hero_data_list = []
        for hero_index in range(self.hero_num):
            # calculate length of each frame
            hero_each_frame_data_length = np.sum(np.array(self.hero_data_split_shape[hero_index]))
            hero_sequence_data_length = hero_each_frame_data_length * self.lstm_time_steps
            hero_sequence_data_split_shape = [hero_sequence_data_length, self.lstm_unit_size, self.lstm_unit_size]

            sequence_data, lstm_cell_data, lstm_hidden_data = data_list[hero_index].float().split(
                hero_sequence_data_split_shape, dim=1)
            reshape_sequence_data = sequence_data.reshape(-1, hero_each_frame_data_length)
            hero_data = reshape_sequence_data.split(self.hero_data_split_shape[hero_index], dim=1)
            hero_data = list(hero_data)  # convert from tuple to list
            hero_data.append(lstm_cell_data)
            hero_data.append(lstm_hidden_data)
            hero_data_list.append(hero_data)
        return hero_data_list

if __name__ == '__main__':
    net_torch = Model()
    print(net_torch)
