import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.Config import Config


##################
## Actual model ##
##################
class NetworkModel(nn.Module):
    def __init__(self, kwargs):
        super(NetworkModel, self).__init__()
        # feature configure parameter
        self.model_name = Config.NETWORK_NAME
        # lstm
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_size = Config.LSTM_UNIT_SIZE

        # data
        self.hero_data_split_shape = Config.HERO_DATA_SPLIT_SHAPE

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

        # image like feature，我方英雄 敌方英雄 主英雄 我方小兵 敌方小兵 我方防御塔 敌方防御塔 野怪 全局信息
        # feature_img, hero_frd, hero_emy, public_info, soldier_frd, soldier_emy, organ_frd, organ_emy, monster_vec, global_info
        # [17 * 17 * 6, 251 * 3, 251 * 3, 44, 25 * 10, 25 * 10, 29 * 3, 29 * 3, 28 * 20, 68]
        # [1734, 753, 753, 44, 250, 250, 87, 87, 560, 68]

        # 有哪些结构参数？
        # conv_layers 可替换5x5 3x3
        # 是否需要 img_mlp
        # 9个特征处理 mlp 的尺寸
        # concat_mlp 的尺寸
        # 分类头的尺寸

        # build network
        # image like feature
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
        # vector feature
        # image like feature，我方英雄 敌方英雄 主英雄 我方小兵 敌方小兵 我方防御塔 敌方防御塔 野怪 全局信息
        # feature_img, hero_frd, hero_emy, public_info, soldier_frd, soldier_emy, organ_frd, organ_emy, monster_vec, global_info
        # [17 * 17 * 6, 251 * 3, 251 * 3, 44, 25 * 10, 25 * 10, 29 * 3, 29 * 3, 28 * 20, 68]
        # [1734, 753, 753, 44, 250, 250, 87, 87, 560, 68]

        for k, v in kwargs.items():
            setattr(self, k, v)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data_list):
        all_hero_result_list = []
        hero_public_first_result_list = []
        hero_public_second_result_list = []

        for hero_index, hero_data in enumerate(data_list):
            hero_feature = hero_data[0]

            feature_img, hero_frd, hero_emy, public_info, soldier_frd, soldier_emy, organ_frd, organ_emy, monster_vec, global_info = hero_feature.split(
                [17 * 17 * 6, 251 * 3, 251 * 3, 44, 25 * 10, 25 * 10, 29 * 3, 29 * 3, 28 * 20, 68], dim=1)
            feature_img = feature_img.reshape((-1, 6, 17, 17))

            conv_hidden = self.conv_layers(feature_img).flatten(start_dim=1)  # B*12*4*4
            # 改成reshape
            hero_frd_list = hero_frd.reshape((-1, 3, 251))
            hero_emy_list = hero_emy.reshape((-1, 3, 251))
            soldier_frd_list = soldier_frd.reshape((-1, 10, 25))
            soldier_emy_list = soldier_emy.reshape((-1, 10, 25))
            organ_frd_list = organ_frd.reshape((-1, 3, 29))
            organ_emy_list = organ_emy.reshape((-1, 3, 29))
            monster_list = monster_vec.reshape((-1, 20, 28))

            hero_frd_hidden = self.hero_frd_mlp(self.hero_share_mlp(hero_frd_list))
            hero_frd_hidden_pool, _ = hero_frd_hidden.max(dim=1)
            hero_emy_hidden = self.hero_emy_mlp(self.hero_share_mlp(hero_emy_list))
            hero_emy_hidden_pool, _ = hero_emy_hidden.max(dim=1)
            public_info_hidden = self.public_info_mlp(public_info)
            monster_hidden = self.monster_mlp(monster_list)
            monster_hidden_pool, _ = monster_hidden.max(dim=1)
            soldier_frd_hidden = self.soldier_frd_mlp(self.soldier_share_mlp(soldier_frd_list))
            soldier_frd_hidden_pool, _ = soldier_frd_hidden.max(dim=1)
            soldier_emy_hidden = self.soldier_emy_mlp(self.soldier_share_mlp(soldier_emy_list))
            soldier_emy_hidden_pool, _ = soldier_emy_hidden.max(dim=1)
            organ_frd_hidden = self.organ_frd_mlp(self.organ_share_mlp(organ_frd_list))
            organ_frd_hidden_pool, _ = organ_frd_hidden.max(dim=1)
            organ_emy_hidden = self.organ_emy_mlp(self.organ_share_mlp(organ_emy_list))
            organ_emy_hidden_pool, _ = organ_emy_hidden.max(dim=1)
            global_hidden = self.global_mlp(global_info)

            concat_hidden = torch.cat(
                [conv_hidden, hero_frd_hidden_pool, hero_emy_hidden_pool,
                 public_info_hidden, soldier_frd_hidden_pool,
                 soldier_emy_hidden_pool, organ_frd_hidden_pool,
                 organ_emy_hidden_pool, monster_hidden_pool,
                 global_hidden], dim=1)  # 192+128*9=1344
            concat_hidden = self.concat_mlp(concat_hidden)

            concat_hidden_split = concat_hidden.split((128, 384), dim=1)
            hero_public_first_result_list.append(concat_hidden_split[0])
            hero_public_second_result_list.append(concat_hidden_split[1])

        pool_hero_public, _ = torch.stack(hero_public_first_result_list, dim=1).max(dim=1)

        for hero_index in range(self.hero_num):
            hero_result_list = []
            fc_public_result = torch.cat([pool_hero_public, hero_public_second_result_list[hero_index]], dim=1)
            # 5 action head
            for action_head in self.action_heads:
                hero_result_list.append(action_head(fc_public_result))
            # value head
            hero_result_list.append(torch.zeros((hero_result_list[-1].shape[0], 1)).to(hero_result_list[-1].device))

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
            # 通过对类别维度进行求和来得到每个样本的损失
            kl_div_loss = kl_div_loss.sum(dim=1)  # 或使用 .mean(dim=1)

            soft_label_final_loss = torch.mean(weight * kl_div_loss)

            loss_list.append(soft_label_final_loss)

        loss = torch.sum(torch.stack(loss_list))
        return loss, loss_list

    def _calculate_single_hero_topk_kl_div_loss(self,
                                                unsqueeze_label_list,
                                                student_logits_list,
                                                teacher_logits_list,
                                                topk_weight_list,
                                                top_k=3):
        label_list = [torch.squeeze(ele, dim=1).long() for ele in unsqueeze_label_list]
        # weight_list = [torch.squeeze(weight[], dim=1) for weight in topk_weight_list]

        teacher_masked_logits_list = teacher_logits_list
        teacher_topk_action_list = [torch.topk(x, top_k, dim=-1).indices for x in teacher_masked_logits_list]

        loss_list = []
        for i in range(len(label_list)):
            weight_i = topk_weight_list[:, :, i + 1]
            weight = (weight_i != 0).float()

            # 使用 log_softmax, 避免指数运行带来的数值不稳定
            student_probs_log = F.log_softmax(student_logits_list[i], dim=1)
            teacher_probs_log = F.log_softmax(teacher_masked_logits_list[i], dim=1)

            # 计算全类别上的 KL 散度
            kl_div_loss = F.kl_div(student_probs_log, teacher_probs_log, reduction='none', log_target=True)
            # kl_div_loss_full = kl_div_loss.sum(dim=1)  # 全类别 KL 散度

            # 只计算前 k 个类别的 KL 散度
            # top_k_indices = torch.topk(teacher_probs_log, top_k, dim=1).indices
            top_k_indices = teacher_topk_action_list[i]
            # topk_kl_div_loss = kl_div_loss.gather(dim=1, index=top_k_indices).sum(dim=1)  # 前 k 个类别的KL散度
            topk_kl_loss = kl_div_loss.gather(dim=1, index=top_k_indices)  # 前 k 个类别的KL散度
            topk_kl_loss = torch.sum(weight * topk_kl_loss, dim=-1)
            kl_div_final_loss = torch.mean(topk_kl_loss)

            loss_list.append(kl_div_final_loss)

        loss = torch.sum(torch.stack(loss_list))
        return loss, loss_list

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
    from helper.arch_config import v1

    net_torch = NetworkModel(v1)
    print(net_torch)
