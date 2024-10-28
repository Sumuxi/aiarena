import argparse
import os
import re

import numpy as np
import torch

from networkmodel.pytorch.NetworkModel import NetworkModel as TeacherNet

"""
计算baseline模型的logit输出相对于官方数据集的动作标签的 top k action 准确率
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--id', type=str, default='3', help='')
args = parser.parse_args()
id = int(args.id)

npz_directory = "/mnt/storage/yxh/competition24/lightweight/dataset/valid"

npz_list = [f for f in os.listdir(npz_directory) if f.endswith('.npz')]
npz_list = sorted(npz_list)

batch_size = 500
device = "cuda:" + str(id)

teacher_net = TeacherNet()
state_dict = torch.load("/mnt/storage/yxh/competition24/lightweight/assets/baseline/model.pth", map_location="cpu")
missing_keys, unexpected_keys = teacher_net.load_state_dict(state_dict["network_state_dict"], strict=True)
print(f"missing_keys: {missing_keys}")
print(f"unexpected_keys: {unexpected_keys}")
teacher_net.to(device)

# sub_action_mask，从官方数据集中提取得到
hero_legal_sub_action_list = [
    [
        [0, 1, 0, 0, 0, 0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [4.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [5.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [6.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [7, 1, 0, 0, 0, 0],
        [8.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [9, 1, 0, 0, 0, 0],
        [10, 1, 0, 0, 0, 0],
        [11.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [12, 1, 0, 0, 0, 0]
    ],
    [
        [0, 1, 0, 0, 0, 0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [4.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [5.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [6.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [7, 1, 0, 0, 0, 0],
        [8.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [9, 1, 0, 0, 0, 0],
        [10, 1, 0, 0, 0, 0],
        [11.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [12, 1, 0, 0, 0, 0]
    ],
    [
        [0, 1, 0, 0, 0, 0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [4.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [5.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [6.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [7, 1, 0, 0, 0, 0],
        [8.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [9, 1, 0, 0, 0, 0],
        [10, 1, 0, 0, 0, 0],
        [11.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [12, 1, 0, 0, 0, 0]
    ]
]
hero_legal_sub_action_list = np.array(hero_legal_sub_action_list)
hero_legal_sub_action_list = torch.tensor(hero_legal_sub_action_list).to(device)

torch.set_printoptions(precision=8, sci_mode=False)


# 提取文件名中的数字并按照数字大小排序
def extract_number(file_name):
    # 使用正则表达式找到文件名中的数字
    match = re.search(r'(\d+)', file_name)
    # 如果找到数字，返回整数形式的数字；否则返回0（或根据需要返回其他默认值）
    return int(match.group(1)) if match else -1

with torch.no_grad():
    for npz in npz_list:
        with open(os.path.join(npz_directory, npz), 'rb') as file:
            numpy_data = np.load(file)
            samples = numpy_data["samples"]
            # samples 是一个 numpy 数组
            num_samples = samples.shape[0]

            count_K = 10
            k = count_K
            official_topk_acc_list = []
            official_topk_prob_list = []
            teacher_topk_acc_list = []
            teacher_topk_prob_list = []

            # 分批次进行推理
            for i in range(0, num_samples, batch_size):
                batch_samples = samples[i:min(i + batch_size, num_samples)]
                input_data = torch.Tensor(batch_samples).to(device)
                data_list = teacher_net.format_data(input_data)
                teacher_rst_list = teacher_net(data_list)

                official_topk_acc = torch.zeros((count_K, 3, 5))
                official_topk_prob = torch.zeros((count_K, 3, 5))
                teacher_topk_acc = torch.zeros((count_K, 3, 5))
                teacher_topk_prob = torch.zeros((count_K, 3, 5))
                for hero_idx in range(0, 3):
                    # 来自官方数据集
                    official_legal_action_list = data_list[hero_idx][1: 1 + 5]
                    official_action_list = data_list[hero_idx][8: 8 + 5]
                    official_probs_list = data_list[hero_idx][13: 13 + 5]
                    official_sub_action_list = data_list[hero_idx][19: 19 + 5]

                    # official_masked_probs_list = []
                    # for probs, legal_action in zip(official_probs_list, official_legal_action_list):
                    #     new_probs = probs.clone()
                    #     new_probs[legal_action == 0] = 1e-9
                    #     official_masked_probs_list.append(new_probs)

                    official_topk_probs_list = [torch.topk(x, k, dim=-1).values for x in official_probs_list]
                    for action_idx in range(5):
                        for kk in range(1, count_K + 1):
                            official_topk_prob[kk - 1][hero_idx][action_idx] = torch.mean(torch.sum(official_topk_probs_list[action_idx][:, :kk], dim=-1))

                    # 计算 top k action 准确率，官方数据集
                    official_topk_action_list = [torch.topk(x, k, dim=-1).indices for x in official_probs_list]

                    for action_idx in range(5):
                        official_actions = official_action_list[action_idx]
                        official_topk_actions = official_topk_action_list[action_idx]
                        sub_action_mask = official_sub_action_list[action_idx].int()
                        is_equal = (official_actions == official_topk_actions) & sub_action_mask
                        for kk in range(1, count_K+1):
                            official_topk_acc[kk-1][hero_idx][action_idx] = \
                                torch.sum(torch.any(is_equal[:, :kk], dim=-1)) / torch.sum(sub_action_mask)

                    # for action_idx in range(5):
                    #     official_actions = official_action_list[action_idx]
                    #     official_topk_actions = official_topk_action_list[action_idx]
                    #     sub_action_mask = official_sub_action_list[action_idx]
                    #     is_equal = official_actions == official_topk_actions
                    #     for kk in range(1, count_K+1):
                    #         official_topk_acc[kk-1][hero_idx][action_idx] = torch.sum(torch.any(is_equal[:, :kk], dim=-1)) / official_actions.shape[0]

                    # teacher的logits
                    teacher_logits_list = teacher_rst_list[hero_idx][0:-1]
                    # 使用legal_action作为掩码对logits_list进行处理
                    teacher_masked_logits_list = []
                    for logits, legal_action in zip(teacher_logits_list, official_legal_action_list):
                        new_logits = logits.clone()
                        new_logits[legal_action == 0] = float('-inf')
                        # new_logits[legal_action == 0] = -1e9
                        teacher_masked_logits_list.append(new_logits)

                    # softmax 求出 probs
                    teacher_masked_probs_list = [torch.softmax(x, dim=-1) for x in teacher_masked_logits_list]
                    teacher_topk_probs_list = [torch.topk(x, k, dim=-1).values for x in teacher_masked_probs_list]
                    for action_idx in range(5):
                        for kk in range(1, count_K + 1):
                            teacher_topk_prob[kk - 1][hero_idx][action_idx] = torch.mean(
                                torch.sum(teacher_topk_probs_list[action_idx][:, :kk], dim=-1))
                    # 计算 top k action 准确率，teacher 相对于官方数据集
                    teacher_topk_action_list = [torch.topk(x, k, dim=-1).indices for x in teacher_masked_logits_list]
                    for action_idx in range(5):
                        official_actions = official_action_list[action_idx]
                        teacher_topk_actions = teacher_topk_action_list[action_idx]
                        sub_action_mask = official_sub_action_list[action_idx].int()
                        is_equal = (official_actions == teacher_topk_actions) & sub_action_mask
                        for kk in range(1, count_K + 1):
                            teacher_topk_acc[kk-1][hero_idx][action_idx] = \
                                torch.sum(torch.any(is_equal[:, :kk], dim=-1)) / torch.sum(sub_action_mask)

                official_topk_acc_list.append(official_topk_acc)
                official_topk_prob_list.append(official_topk_prob)
                teacher_topk_acc_list.append(teacher_topk_acc)
                teacher_topk_prob_list.append(teacher_topk_prob)

            numpy_data.close()

            print(f"acc from npz file: {npz}")
            official_topk_acc_list = torch.mean(torch.stack(official_topk_acc_list), dim=(0, 2)).cpu().numpy()
            print("official_topk_acc:")
            for k in range(count_K):
                print(f"top_{k + 1} action acc: {official_topk_acc_list[k]}")

            official_topk_prob_list = torch.mean(torch.stack(official_topk_prob_list), dim=(0, 2)).cpu().numpy()
            print("official_topk_prob:")
            for k in range(count_K):
                print(f"top_{k + 1} prob: {official_topk_prob_list[k]}")

            teacher_topk_acc_list = torch.mean(torch.stack(teacher_topk_acc_list), dim=(0, 2)).cpu().numpy()
            print("baseline_topk_acc:")
            for k in range(count_K):
                print(f"top_{k + 1} action acc: {teacher_topk_acc_list[k]}")

            teacher_topk_prob_list = torch.mean(torch.stack(teacher_topk_prob_list), dim=(0, 2)).cpu().numpy()
            print("baseline_topk_prob:")
            for k in range(count_K):
                print(f"top_{k + 1} prob: {teacher_topk_prob_list[k]}")

            print()


