import sys
import torch
import torch.nn as nn  # for builtin modules including Linear, Conv2d, MultiheadAttention, LayerNorm, etc
import torch.nn.functional as F
from torch.nn import ModuleDict  # for layer naming when nn.Sequential is not viable
import numpy as np  # for some basic dimention computation, might be redundent

from math import ceil, floor
from collections import OrderedDict

# typing
from torch import Tensor, LongTensor
from typing import Dict, List, Tuple
from ctypes import Union

from config.Config import Config


##################
## Actual model ##
##################
class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()
        # feature configure parameter
        self.model_name = Config.NETWORK_NAME
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

        self.distill_temperature = Config.DISTILL_TEMPERATURE
        self.distill_weight = Config.DISTILL_WEIGHT

        print("model file: ", __name__)
        # loss weights
        coefficients = {
            "hard loss weight": Config.HARD_WEIGHT,
            "soft loss weight": Config.SOFT_WEIGHT,
            "distill loss weight": self.distill_weight,
            "distill temperature": self.distill_temperature
        }
        for k, v in coefficients.items():
            print(f"{k}: {v}")
        sys.stdout.flush()

        self.out_dim = 128

        # build network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 18, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(18, 12, 3, 1, 1),
            nn.MaxPool2d(2)
        )  # 192
        # self.img_mlp = nn.Sequential(
        #     nn.Linear(768, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512)
        # )
        self.hero_share_mlp = nn.Sequential(
            nn.Linear(251, 256),
            nn.ReLU(),
        )
        self.hero_frd_mlp = nn.Linear(256, self.out_dim)
        self.hero_emy_mlp = nn.Linear(256, self.out_dim)
        self.public_info_mlp = nn.Sequential(
            nn.Linear(44, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )
        self.soldier_share_mlp = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
        )
        self.soldier_frd_mlp = nn.Linear(128, self.out_dim)
        self.soldier_emy_mlp = nn.Linear(128, self.out_dim)
        self.organ_share_mlp = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
        )
        self.organ_frd_mlp = nn.Linear(128, self.out_dim)
        self.organ_emy_mlp = nn.Linear(128, self.out_dim)
        self.monster_mlp = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(68, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )
        # 192+128*9=1344
        self.concat_mlp = nn.Sequential(
            nn.Linear(1344, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.action_heads = nn.ModuleList()
        for action_dim in (13, 25, 42, 42):
            self.action_heads.append(
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_dim)
                )
            )
        self.target_embed = nn.Linear(self.out_dim, self.out_dim)
        self.target_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.out_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data_list):
        all_hero_result_list = []
        hero_public_first_result_list = []
        hero_public_second_result_list = []
        all_hero_target_list = []

        for hero_index, hero_data in enumerate(data_list):
            hero_feature = hero_data[0]

            feature_img, hero_frd, hero_emy, public_info, soldier_frd, soldier_emy, organ_frd, organ_emy, monster_vec, global_info = hero_feature.split(
                [17 * 17 * 6, 251 * 3, 251 * 3, 44, 25 * 10, 25 * 10, 29 * 3, 29 * 3, 28 * 20, 68], dim=1)
            feature_img = feature_img.reshape((-1, 6, 17, 17))

            conv_hidden = self.conv_layers(feature_img).flatten(start_dim=1)  # B*12*4*4
            # img_hidden = self.img_mlp(conv_hidden)
            # 改成reshape
            hero_frd_list = hero_frd.reshape((-1, 3, 251))
            hero_emy_list = hero_emy.reshape((-1, 3, 251))
            soldier_frd_list = soldier_frd.reshape((-1, 10, 25))
            soldier_emy_list = soldier_emy.reshape((-1, 10, 25))
            organ_frd_list = organ_frd.reshape((-1, 3, 29))
            organ_emy_list = organ_emy.reshape((-1, 3, 29))
            monster_list = monster_vec.reshape((-1, 20, 28))

            hero_target_list = []

            hero_frd_hidden = self.hero_frd_mlp(self.hero_share_mlp(hero_frd_list))
            hero_target_list.append(hero_frd_hidden)  # 3 frd hero
            hero_frd_hidden_pool, _ = hero_frd_hidden.max(dim=1)
            hero_emy_hidden = self.hero_emy_mlp(self.hero_share_mlp(hero_emy_list))
            hero_target_list.append(hero_emy_hidden)  # 3 emy hero
            hero_emy_hidden_pool, _ = hero_emy_hidden.max(dim=1)
            public_info_hidden = self.public_info_mlp(public_info)
            hero_target_list.append(public_info_hidden.reshape((-1, 1, self.out_dim)))  # 1 public info
            monster_hidden = self.monster_mlp(monster_list)
            monster_hidden_pool, _ = monster_hidden.max(dim=1)
            hero_target_list.append(monster_hidden)  # 20 monster
            soldier_frd_hidden = self.soldier_frd_mlp(self.soldier_share_mlp(soldier_frd_list))
            soldier_frd_hidden_pool, _ = soldier_frd_hidden.max(dim=1)
            soldier_emy_hidden = self.soldier_emy_mlp(self.soldier_share_mlp(soldier_emy_list))
            soldier_emy_hidden_pool, _ = soldier_emy_hidden.max(dim=1)
            hero_target_list.append(soldier_emy_hidden)  # 10 emy soldier
            organ_frd_hidden = self.organ_frd_mlp(self.organ_share_mlp(organ_frd_list))
            organ_frd_hidden_pool, _ = organ_frd_hidden.max(dim=1)
            organ_emy_hidden = self.organ_emy_mlp(self.organ_share_mlp(organ_emy_list))
            organ_emy_hidden_pool, _ = organ_emy_hidden.max(dim=1)
            global_hidden = self.global_mlp(global_info)
            hero_target_list.append(organ_emy_hidden_pool.reshape((-1, 1, self.out_dim)))  # 1 emy organ
            hero_target_list.insert(0, torch.ones_like(hero_target_list[2], dtype=torch.float32) * 0.1)
            all_hero_target_list.append(torch.cat(hero_target_list, dim=1))

            concat_hidden = torch.cat(
                [conv_hidden, hero_frd_hidden_pool, hero_emy_hidden_pool,
                 public_info_hidden, soldier_frd_hidden_pool,
                 soldier_emy_hidden_pool, organ_frd_hidden_pool,
                 organ_emy_hidden_pool, monster_hidden_pool,
                 global_hidden], dim=1)
            concat_hidden = self.concat_mlp(concat_hidden)

            concat_hidden_split = concat_hidden.split((128, 384), dim=1)
            hero_public_first_result_list.append(concat_hidden_split[0])
            hero_public_second_result_list.append(concat_hidden_split[1])

        pool_hero_public, _ = torch.stack(hero_public_first_result_list, dim=1).max(dim=1)

        for hero_index in range(self.hero_num):
            hero_result_list = []
            fc_public_result = torch.cat([pool_hero_public, hero_public_second_result_list[hero_index]], dim=1)
            # 4 action head
            for action_head in self.action_heads:
                hero_result_list.append(action_head(fc_public_result))
            # target head
            target_embedding = self.target_embed(all_hero_target_list[hero_index])  # B*39*128
            target_key = self.target_head(fc_public_result).reshape((-1, self.out_dim, 1))  # B*128*1
            target_logits = torch.matmul(target_embedding, target_key).reshape((-1, 39))  # B*39
            hero_result_list.append(target_logits)
            # value head
            hero_result_list.append(target_logits[:, 0:1])

            all_hero_result_list.append(hero_result_list)

        return all_hero_result_list

    def _calculate_single_hero_hard_loss(self, unsqueeze_label_list, fc2_label_list, unsqueeze_weight_list):
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(torch.squeeze(ele, dim=1).long())
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(torch.squeeze(weight, dim=1))

        cost_p_label_list = []
        for i in range(len(label_list)):
            weight = (weight_list[i] != torch.tensor(0, dtype=torch.float32)).float()
            label_loss = F.cross_entropy(fc2_label_list[i], label_list[i], reduction='none')
            label_final_loss = torch.mean(weight * label_loss)
            cost_p_label_list.append(label_final_loss)
        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(cost_p_label_list)):
            loss = loss + cost_p_label_list[i]
        return loss, cost_p_label_list

    def _calculate_single_hero_soft_loss(self, student_logits_list, teacher_logits_list, unsqueeze_weight_list):
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(torch.squeeze(weight, dim=1))

        cost_p_label_list = []
        for i in range(len(student_logits_list)):
            weight = (weight_list[i] != torch.tensor(0, dtype=torch.float32)).float()
            # Calculate soft label loss
            teacher_probs = F.softmax(teacher_logits_list[i], dim=1)
            soft_label_loss = F.cross_entropy(student_logits_list[i], teacher_probs, reduction='none')
            label_final_loss = torch.mean(weight * soft_label_loss)
            cost_p_label_list.append(label_final_loss)
        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(cost_p_label_list)):
            loss = loss + cost_p_label_list[i]
        return loss, cost_p_label_list

    def _calculate_single_hero_distill_loss(self, unsqueeze_label_list, student_logits_list, teacher_logits_list,
                                            unsqueeze_weight_list, temperature=4.0, lambda_weight=0.5):
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(torch.squeeze(ele, dim=1).long())
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(torch.squeeze(weight, dim=1))

        cost_p_label_list = []
        for i in range(len(label_list)):
            weight = (weight_list[i] != torch.tensor(0, dtype=torch.float32)).float()

            # Calculate hard label loss
            hard_label_loss = F.cross_entropy(student_logits_list[i], label_list[i], reduction='none')
            hard_label_final_loss = torch.mean(weight * hard_label_loss)

            # Calculate soft label loss
            student_logits_temperature = student_logits_list[i] / temperature
            teacher_logits_temperature = teacher_logits_list[i] / temperature
            teacher_probs = F.softmax(teacher_logits_temperature, dim=1)

            soft_label_loss = F.cross_entropy(student_logits_temperature, teacher_probs, reduction='none')
            soft_label_final_loss = torch.mean(weight * soft_label_loss)

            # Combine the hard and soft label losses with the specified weight
            final_loss = (1 - lambda_weight) * hard_label_final_loss + (
                    temperature ** 2) * lambda_weight * soft_label_final_loss

            cost_p_label_list.append(final_loss)

        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(cost_p_label_list)):
            loss = loss + cost_p_label_list[i]
        return loss, cost_p_label_list

    def _calculate_single_temperatured_soft_loss(self, unsqueeze_label_list, student_logits_list, teacher_logits_list,
                                                 unsqueeze_weight_list, temperature=4.0):
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(torch.squeeze(ele, dim=1).long())
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(torch.squeeze(weight, dim=1))

        cost_p_label_list = []
        for i in range(len(label_list)):
            weight = (weight_list[i] != torch.tensor(0, dtype=torch.float32)).float()

            # Calculate soft label loss
            student_logits_temperature = student_logits_list[i] / temperature
            teacher_logits_temperature = teacher_logits_list[i] / temperature
            teacher_probs = F.softmax(teacher_logits_temperature, dim=1)

            soft_label_loss = F.cross_entropy(student_logits_temperature, teacher_probs, reduction='none')
            soft_label_final_loss = torch.mean(weight * soft_label_loss)

            cost_p_label_list.append(soft_label_final_loss)

        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(cost_p_label_list)):
            loss = loss + cost_p_label_list[i]
        return loss, cost_p_label_list

    def _calculate_single_hero_kl_div_loss(self, unsqueeze_label_list, student_logits_list, teacher_logits_list,
                                           unsqueeze_weight_list, temperature=4.0):
        label_list = [torch.squeeze(ele, dim=1).long() for ele in unsqueeze_label_list]
        weight_list = [torch.squeeze(weight, dim=1) for weight in unsqueeze_weight_list]

        loss_list = []
        for i in range(len(label_list)):
            weight = (weight_list[i] != 0).float()

            student_logits_temperature = student_logits_list[i] / temperature
            teacher_logits_temperature = teacher_logits_list[i] / temperature

            # 使用 log_softmax, 避免指数运行带来的数值不稳定
            student_probs_log = F.log_softmax(student_logits_temperature, dim=1)
            teacher_probs_log = F.log_softmax(teacher_logits_temperature, dim=1)

            # 计算 KL 散度，log_target=True 表示教师模型的目标已是 log 概率
            kl_div_loss = F.kl_div(student_probs_log, teacher_probs_log, reduction='none', log_target=True)
            # 通过对类别维度进行求和或均值来得到每个样本的损失
            kl_div_loss = kl_div_loss.sum(dim=1)  # 或使用 .mean(dim=1)

            soft_label_final_loss = torch.mean(weight * kl_div_loss)

            loss_list.append(soft_label_final_loss)

        loss = torch.sum(torch.stack(loss_list))
        return loss, loss_list

    def compute_loss(self, data_list, rst_list):
        cost_all = torch.tensor(0.0, dtype=torch.float32)
        all_hero_loss_list = []
        all_acc_list = []

        top_k = 5

        # sub_action_mask，从官方数据集中提取得到
        hero_legal_sub_action_list = [
            [
                [0, 0, 0, 0, 0, 0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [5.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [6.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [7, 0, 0, 0, 0, 0],
                [8.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [9, 0, 0, 0, 0, 0],
                [10, 0, 0, 0, 0, 0],
                [11.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [12, 0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [5.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [6.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [7, 0, 0, 0, 0, 0],
                [8.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [9, 0, 0, 0, 0, 0],
                [10, 0, 0, 0, 0, 0],
                [11.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [12, 0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [5.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [6.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [7, 0, 0, 0, 0, 0],
                [8.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [9, 0, 0, 0, 0, 0],
                [10, 0, 0, 0, 0, 0],
                [11.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [12, 0, 0, 0, 0, 0]
            ]
        ]
        hero_legal_sub_action_list = np.array(hero_legal_sub_action_list)
        hero_legal_sub_action_list = torch.tensor(hero_legal_sub_action_list).to(rst_list[0][0].device)

        for hero_index in range(len(data_list)):
            this_hero_label_task_count = len(self.hero_label_size_list[hero_index])
            data_index = 1

            # legal action
            official_legal_action_list = data_list[hero_index][1: 1 + 5]
            data_index += this_hero_label_task_count

            # reward
            data_index += 1

            # advantage
            data_index += 1

            # action (label)
            this_hero_action_list = data_list[hero_index][data_index:(data_index + this_hero_label_task_count)]
            data_index += this_hero_label_task_count

            # action (prob lists, each corresponds to a sub-task)
            this_hero_probability_list = data_list[hero_index][data_index:(data_index + this_hero_label_task_count)]
            this_hero_logits_list = this_hero_probability_list
            data_index += this_hero_label_task_count

            # is_train
            data_index += 1

            # sub_action
            this_hero_weight_list = data_list[hero_index][data_index:(data_index + this_hero_label_task_count)]
            this_hero_weight_list_new = [torch.ones_like(t) for t in this_hero_weight_list]
            data_index += this_hero_label_task_count  # originally (task_num + 1)

            # policy network output
            this_hero_fc_label_list = rst_list[hero_index][:-1]

            # value network output
            this_hero_value = rst_list[hero_index][-1]

            # teacher的logits
            teacher_logits_list = this_hero_logits_list
            # 使用legal_action作为掩码对logits_list进行处理
            teacher_masked_logits_list = []
            for logits, legal_action in zip(teacher_logits_list, official_legal_action_list):
                new_logits = logits.clone()
                # new_logits[legal_action == 0] = float('-inf')
                new_logits[legal_action == 0] = -1e8
                teacher_masked_logits_list.append(new_logits)
            teacher_topk_action_list = [torch.topk(x, top_k, dim=-1).indices for x in teacher_masked_logits_list]
            # 替换sub_action
            new_button = teacher_topk_action_list[0].reshape(-1).int()

            new_sub_action_list1 = hero_legal_sub_action_list[hero_index][new_button]
            new_sub_action_list = new_sub_action_list1.reshape(-1, top_k, 6)

            # student 的logits
            student_logits_list = this_hero_fc_label_list
            # 使用legal_action作为掩码对logits_list进行处理
            student_masked_logits_list = []
            for logits, legal_action in zip(student_logits_list, official_legal_action_list):
                new_logits = logits.clone()
                # new_logits[legal_action == 0] = float('-inf')
                new_logits[legal_action == 0] = -1e8
                student_masked_logits_list.append(new_logits)
            student_topk_action_list = [torch.topk(x, top_k, dim=-1).indices for x in student_masked_logits_list]

            temperature = self.distill_temperature
            lambda_weight = self.distill_weight

            # hard label loss
            this_hero_hard_loss_list = self._calculate_single_hero_hard_loss(this_hero_action_list,
                                                                             this_hero_fc_label_list,
                                                                             this_hero_weight_list)

            # soft label loss
            this_hero_soft_loss_list = self._calculate_single_hero_soft_loss(this_hero_fc_label_list,
                                                                             this_hero_logits_list,
                                                                             this_hero_weight_list_new)
            # distill loss
            this_hero_distill_loss_list = self._calculate_single_hero_distill_loss(this_hero_action_list,
                                                                                   this_hero_fc_label_list,
                                                                                   this_hero_logits_list,
                                                                                   this_hero_weight_list_new,
                                                                                   temperature=temperature,
                                                                                   lambda_weight=lambda_weight)
            # temperatured soft loss
            this_hero_temperatured_soft_loss_list = self._calculate_single_temperatured_soft_loss(this_hero_action_list,
                                                                                                  this_hero_fc_label_list,
                                                                                                  this_hero_logits_list,
                                                                                                  this_hero_weight_list_new,
                                                                                                  temperature=temperature)
            # kl_div loss
            this_hero_kl_div_loss_list = self._calculate_single_hero_kl_div_loss(this_hero_action_list,
                                                                                 this_hero_fc_label_list,
                                                                                 this_hero_logits_list,
                                                                                 this_hero_weight_list_new,
                                                                                 temperature=temperature)

            # cost_all = cost_all + \
            #            Config.HARD_WEIGHT * this_hero_hard_loss_list[0] + \
            #            Config.SOFT_WEIGHT * this_hero_soft_loss_list[0] + \
            #            Config.DISTILL_WEIGHT * this_hero_distill_loss_list[0] + \
            #            0.0 * torch.sum(this_hero_value)  # loss item (scalar)

            cost_all = cost_all + this_hero_kl_div_loss_list[0] * (temperature ** 2)

            all_hero_loss_list.append(
                [
                    this_hero_hard_loss_list[0],
                    this_hero_soft_loss_list[0],
                    this_hero_distill_loss_list[0],
                    this_hero_temperatured_soft_loss_list[0],
                    this_hero_kl_div_loss_list[0]
                ]
            )

            topk_acc_list = torch.zeros((5, top_k))
            # 计算 top k action 准确率，student 相对 teacher
            for action_idx in range(5):
                teacher_actions = teacher_topk_action_list[action_idx]
                student_actions = student_topk_action_list[action_idx]
                teacher_sub_action_mask = new_sub_action_list[:, :, action_idx + 1].int()
                is_equal = (student_actions == teacher_actions) & teacher_sub_action_mask
                for kk in range(1, top_k + 1):
                    count1 = torch.sum(is_equal[:, kk - 1])
                    count2 = torch.sum(teacher_sub_action_mask[:, kk - 1])
                    if count2 != 0:
                        topk_acc_list[action_idx][kk - 1] = (count1 / count2).cpu()

            all_acc_list.append(topk_acc_list)

        # all_hero_loss_list (3, 5)   all_acc_list (5, top_k)
        out_all_acc_list = torch.mean(torch.stack(all_acc_list), dim=0).cpu().numpy()
        return cost_all, all_hero_loss_list, out_all_acc_list

    def format_data(self, datas):
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
    net_torch = NetworkModel()
    print(net_torch)
