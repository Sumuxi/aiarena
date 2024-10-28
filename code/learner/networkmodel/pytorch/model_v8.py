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

        print("model file: ", __name__)
        # loss weights
        coefficients = {"hard loss weight": Config.HARD_WEIGHT,
                        "soft loss weight": Config.SOFT_WEIGHT,
                        "distill loss weight": Config.DISTILL_WEIGHT}
        for k, v in coefficients.items():
            print(f"{k}: {v}")
        print("DISTILL_TEMPERATURE: ", Config.DISTILL_TEMPERATURE)
        sys.stdout.flush()

        # build network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 18, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(18, 12, 3, 1, 1),
            # nn.MaxPool2d(2)
        )
        self.img_mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.hero_share_mlp = nn.Sequential(
            nn.Linear(251, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.hero_frd_mlp = nn.Linear(512, 128)
        self.hero_emy_mlp = nn.Linear(512, 128)
        self.public_info_mlp = nn.Sequential(
            nn.Linear(44, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.soldier_share_mlp = nn.Sequential(
            nn.Linear(25, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.soldier_frd_mlp = nn.Linear(256, 128)
        self.soldier_emy_mlp = nn.Linear(256, 128)
        self.organ_share_mlp = nn.Sequential(
            nn.Linear(29, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.organ_frd_mlp = nn.Linear(256, 128)
        self.organ_emy_mlp = nn.Linear(256, 128)
        self.monster_mlp = nn.Sequential(
            nn.Linear(28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(68, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.concat_mlp = nn.Sequential(
            nn.Linear(1664, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.action_heads = nn.ModuleList()
        for action_dim in (13, 25, 42, 42):
            self.action_heads.append(
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim)
                )
            )
        self.target_embed = nn.Linear(128, 128)
        self.target_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
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
            img_hidden = self.img_mlp(conv_hidden)
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
            hero_target_list.append(public_info_hidden.reshape((-1, 1, 128)))  # 1 public info
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
            hero_target_list.append(organ_emy_hidden_pool.reshape((-1, 1, 128)))  # 1 emy organ
            hero_target_list.insert(0, torch.ones_like(hero_target_list[2], dtype=torch.float32) * 0.1)
            all_hero_target_list.append(torch.cat(hero_target_list, dim=1))

            concat_hidden = torch.cat(
                [img_hidden, hero_frd_hidden_pool, hero_emy_hidden_pool,
                 public_info_hidden, soldier_frd_hidden_pool,
                 soldier_emy_hidden_pool, organ_frd_hidden_pool,
                 organ_emy_hidden_pool, monster_hidden_pool,
                 global_hidden], dim=1)  # 192+64+64+32+32+32+32+32+32+68=580
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
            target_key = self.target_head(fc_public_result).reshape((-1, 128, 1))  # B*128*1
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

    def compute_loss(self, data_list, rst_list):
        cost_all = torch.tensor(0.0, dtype=torch.float32)
        all_hero_loss_list = []
        for hero_index in range(len(data_list)):
            this_hero_label_task_count = len(self.hero_label_size_list[hero_index])
            data_index = 1

            # legal action
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

            # hard label loss
            this_hero_hard_loss_list = self._calculate_single_hero_hard_loss(this_hero_action_list,
                                                                             this_hero_fc_label_list,
                                                                             this_hero_weight_list_new)

            # soft label loss
            this_hero_soft_loss_list = self._calculate_single_hero_soft_loss(this_hero_fc_label_list,
                                                                             this_hero_probability_list,
                                                                             this_hero_weight_list_new)

            # distill loss
            this_hero_distill_loss_list = self._calculate_single_hero_distill_loss(this_hero_action_list,
                                                                                   this_hero_fc_label_list,
                                                                                   this_hero_probability_list,
                                                                                   this_hero_weight_list_new,
                                                                                   temperature=Config.DISTILL_TEMPERATURE,
                                                                                   lambda_weight=Config.DISTILL_LAMBDA_WEIGHT)

            cost_all = cost_all + \
                       Config.HARD_WEIGHT * this_hero_hard_loss_list[0] + \
                       Config.SOFT_WEIGHT * this_hero_soft_loss_list[0] + \
                       Config.DISTILL_WEIGHT * this_hero_distill_loss_list[0] + \
                       0.0 * torch.sum(this_hero_value)  # loss item (scalar)
            all_hero_loss_list.append(
                [this_hero_hard_loss_list[0], this_hero_soft_loss_list[0], this_hero_distill_loss_list[0]]
            )

            # 计算acc
            pred_logits_list = this_hero_fc_label_list
            # 直接对logits_list进行处理
            # softmax 求出 probs
            pred_probs_list = [torch.softmax(x, dim=-1) for x in pred_logits_list]
            # argmax 求出 labels
            pred_action_list = [torch.argmax(x, dim=-1, keepdim=True) for x in pred_probs_list]
            sub_action_mask_list = this_hero_weight_list

            is_equal_list = [x == y for x, y in zip(pred_action_list, this_hero_action_list)]
            for x, y in zip(is_equal_list, sub_action_mask_list):
                x[y == 0] = 0
            # is_equal_list_copy = []
            # for x, y in zip(is_equal_list, sub_action_mask_list):
            #     x_copy = x.clone()
            #     x_copy[y == 0] = 0
            #     is_equal_list_copy.append(x_copy)

            acc_list = [(torch.sum(x) / torch.sum(y)).item() for x, y in zip(is_equal_list, sub_action_mask_list)]

        return cost_all, [[all_hero_loss_list[0][0], all_hero_loss_list[0][1], all_hero_loss_list[0][2]],
                          [all_hero_loss_list[1][0], all_hero_loss_list[1][1], all_hero_loss_list[1][2]],
                          [all_hero_loss_list[2][0], all_hero_loss_list[2][1], all_hero_loss_list[2][2]]], acc_list

    def compute_rl_loss(self, data_list, rst_list):
        all_hero_loss_list = []
        total_loss = torch.tensor(0.0)
        for hero_index, hero_data in enumerate(data_list):
            # _calculate_single_hero_loss
            label_task_count = len(self.hero_label_size_list[hero_index])
            data_index = 1
            hero_legal_action_flag_list = hero_data[data_index: (data_index + label_task_count)]
            data_index += label_task_count

            unsqueeze_reward = hero_data[data_index]
            data_index += 1

            hero_advantage = hero_data[data_index]
            data_index += 1

            hero_action_list = hero_data[data_index:(data_index + label_task_count)]
            data_index += label_task_count

            hero_probability_list = hero_data[data_index:(data_index + label_task_count)]
            data_index += label_task_count

            hero_frame_is_train = hero_data[data_index]
            data_index += 1

            hero_weight_list = hero_data[data_index:(data_index + label_task_count + 1)]
            data_index += label_task_count + 1

            hero_fc_label_result = rst_list[hero_index][:-1]
            hero_value_result = rst_list[hero_index][-1]
            hero_label_size_list = self.hero_label_size_list[hero_index]

            _hero_legal_action_flag_list = hero_legal_action_flag_list

            unsqueeze_reward_list = unsqueeze_reward.split(unsqueeze_reward.shape[1] // self.value_head_num,
                                                           dim=1)
            _hero_reward_list = [item.squeeze(1) for item in unsqueeze_reward_list]
            _hero_advantage = hero_advantage.squeeze(1)
            _hero_action_list = [item.squeeze(1) for item in hero_action_list]
            _hero_probability_list = hero_probability_list
            _hero_frame_is_train = hero_frame_is_train.squeeze(1)
            _hero_weight_list = [item.squeeze(1) for item in hero_weight_list]
            _hero_fc_label_result = hero_fc_label_result
            _hero_value_result = hero_value_result
            # _hero_value_result = hero_value_result.squeeze(1)
            _hero_label_size_list = hero_label_size_list

            train_frame_count = _hero_frame_is_train.sum()
            train_frame_count = torch.maximum(train_frame_count, torch.tensor(1.0))  # prevent division by 0

            # loss of value net
            # value_cost = torch.square((_hero_reward_list[0] - _hero_value_result)) * _hero_frame_is_train
            # value_cost = 0.5 * value_cost.sum(0) / train_frame_count

            unsqueeze_fc2_value_result_list = _hero_value_result.split(
                _hero_value_result.shape[1] // self.value_head_num, dim=1)

            # loss of value net
            value_cost = []
            for value_index in range(self.value_head_num):
                fc2_value_result_squeezed = unsqueeze_fc2_value_result_list[value_index].squeeze(1)

                value_cost_tmp = torch.tensor(0.)
                value_cost_tmp = torch.square((_hero_reward_list[value_index] - fc2_value_result_squeezed))
                value_cost_tmp = 0.5 * torch.mean(value_cost_tmp, dim=0)

                value_cost.append(value_cost_tmp)

            # for entropy loss calculate
            label_logits_subtract_max_list = []
            label_sum_exp_logits_list = []
            label_probability_list = []

            policy_cost = torch.tensor(0.0)
            for task_index, is_reinforce_task in enumerate(self.hero_is_reinforce_task_list[hero_index]):
                if is_reinforce_task:
                    final_log_p = torch.tensor(0.0)
                    boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                    one_hot_actions = nn.functional.one_hot(_hero_action_list[task_index].long(),
                                                            _hero_label_size_list[task_index])
                    legal_action_flag_list_max_mask = (1 - _hero_legal_action_flag_list[task_index]) * boundary

                    # print("task index: {}".format(task_index))
                    # print(_hero_fc_label_result[task_index].shape)
                    # print(legal_action_flag_list_max_mask.shape)
                    label_logits_subtract_max = torch.clamp(
                        _hero_fc_label_result[task_index] - torch.max(
                            _hero_fc_label_result[task_index] - legal_action_flag_list_max_mask, dim=1, keepdim=True
                        ).old_values, -boundary, 1)

                    label_logits_subtract_max_list.append(label_logits_subtract_max)

                    label_exp_logits = _hero_legal_action_flag_list[task_index] * torch.exp(
                        label_logits_subtract_max) + self.min_policy
                    label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                    label_sum_exp_logits_list.append(label_sum_exp_logits)

                    label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                    label_probability_list.append(label_probability)

                    policy_p = (one_hot_actions * label_probability).sum(1)
                    policy_log_p = torch.log(policy_p)
                    old_policy_p = (one_hot_actions * _hero_probability_list[task_index]).sum(1)
                    old_policy_log_p = torch.log(old_policy_p)
                    final_log_p = final_log_p + policy_log_p - old_policy_log_p
                    ratio = torch.exp(final_log_p)
                    clip_ratio = ratio.clamp(0.0, 3.0)

                    surr1 = clip_ratio * _hero_advantage
                    surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * _hero_advantage
                    policy_cost = policy_cost - torch.sum(torch.minimum(surr1, surr2) * (
                        _hero_weight_list[task_index].float()) * _hero_frame_is_train) / torch.maximum(
                        torch.sum((_hero_weight_list[task_index].float()) * _hero_frame_is_train), torch.tensor(1.0))

            # cross entropy loss
            current_entropy_loss_index = 0
            entropy_loss_list = []
            for task_index, is_reinforce_task in enumerate(self.hero_is_reinforce_task_list[hero_index]):
                if is_reinforce_task:
                    temp_entropy_loss = -torch.sum(
                        label_probability_list[current_entropy_loss_index] * _hero_legal_action_flag_list[
                            task_index] * torch.log(label_probability_list[current_entropy_loss_index]), dim=1)
                    temp_entropy_loss = -torch.sum((temp_entropy_loss * _hero_weight_list[
                        task_index].float() * _hero_frame_is_train)) / torch.maximum(
                        torch.sum(_hero_weight_list[task_index].float() * _hero_frame_is_train),
                        torch.tensor(1.0))  # add - because need to minize
                    entropy_loss_list.append(temp_entropy_loss)
                    current_entropy_loss_index = current_entropy_loss_index + 1
                else:
                    temp_entropy_loss = torch.tensor(0.0)
                    entropy_loss_list.append(temp_entropy_loss)
            entropy_cost = torch.tensor(0.0)
            for entropy_element in entropy_loss_list:
                entropy_cost = entropy_cost + entropy_element

            value_cost_all = torch.tensor(0.0)
            for value_ele in value_cost:
                value_cost_all = value_cost_all + value_ele
            # cost_all
            cost_all = value_cost_all + self.hero_policy_weight * policy_cost + self.var_beta * entropy_cost

            _hero_all_loss_list = [cost_all, value_cost_all, policy_cost, entropy_cost]

            total_loss = total_loss + _hero_all_loss_list[0]
            all_hero_loss_list.append(_hero_all_loss_list)
        return total_loss, [total_loss, [all_hero_loss_list[0][1], all_hero_loss_list[0][2], all_hero_loss_list[0][3]],
                            [all_hero_loss_list[1][1], all_hero_loss_list[1][2], all_hero_loss_list[1][3]],
                            [all_hero_loss_list[2][1], all_hero_loss_list[2][2], all_hero_loss_list[2][3]]]

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
