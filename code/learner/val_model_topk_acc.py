import os
import re
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from networkmodel.pytorch.NetworkModel import NetworkModel as TeacherNet
from networkmodel.pytorch.final_v1 import NetworkModel as StudentNet

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--id', type=str, default='5', help='')
args = parser.parse_args()
id = int(args.id)

count_K = 3

exp_name = "f1e1"

log_dir = f"/aiarena/output/metrics/{exp_name}"
writer = SummaryWriter(os.path.join(log_dir, "val"))
stu_to_official_writers = [
    SummaryWriter(os.path.join(log_dir, "stu_to_official", s)) for s in [f"top{k}" for k in range(1, 1+count_K)]
]
stu_to_tea_writers = [
    SummaryWriter(os.path.join(log_dir, "stu_to_teacher", s)) for s in [f"top{k}" for k in range(1, 1+count_K)]
]


npz_directory = "/aiarena/dataset/valid"
ckpt_directory = f"/aiarena/output/{exp_name}/ckpt"

# npz_list = [f for f in os.listdir(npz_directory) if f.endswith('.npz')]
# npz_list = sorted(npz_list)

batch_size = 500
device = "cuda:" + str(id)

teacher_net = TeacherNet()
state_dict = torch.load("/aiarena/code/assets/baseline/model.pth", map_location="cpu")
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

torch.set_printoptions(precision=6, sci_mode=False)


# for npz in npz_list:

npz = 'all_samples_5_1723147132.npz'

# 提取文件名中的数字并按照数字大小排序
def extract_number(file_name):
    # 使用正则表达式找到文件名中的数字
    match = re.search(r'(\d+)', file_name)
    # 如果找到数字，返回整数形式的数字；否则返回0（或根据需要返回其他默认值）
    return int(match.group(1)) if match else -1

with torch.no_grad():
    ckpt_list = [f for f in os.listdir(ckpt_directory) if f.endswith('.pth')]
    ckpt_list = sorted(ckpt_list, key=extract_number)

    for ckpt in ckpt_list:
        step = extract_number(ckpt)
        if step < 1 or step % 10000 != 0:
            continue

        student_net = StudentNet()
        state_dict = torch.load(
            os.path.join(ckpt_directory, ckpt),
            map_location="cpu")
        missing_keys, unexpected_keys = student_net.load_state_dict(state_dict["network_state_dict"], strict=True)
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")
        student_net.to(device)


        with open(os.path.join(npz_directory, npz), 'rb') as file:
            numpy_data = np.load(file)
            samples = numpy_data["samples"]
            # samples 是一个 numpy 数组
            num_samples = samples.shape[0]


            k = count_K

            student_to_official_topk_acc_list = []
            student_to_teacher_topk_acc_list = []
            student_topk_prob_list = []

            cost_all_list = []
            all_hero_loss_list = []
            all_acc_list = []

            # 分批次进行推理
            for i in range(0, num_samples, batch_size):
                batch_samples = samples[i:min(i + batch_size, num_samples)]
                input_data = torch.Tensor(batch_samples).to(device)
                data_list = teacher_net.format_data(input_data)
                teacher_rst_list = teacher_net(data_list)
                student_rst_list = student_net(data_list)

                student_to_official_topk_acc = torch.zeros((count_K, 3, 5))
                student_to_teacher_topk_acc = torch.zeros((count_K, 3, 5))
                student_topk_prob = torch.zeros((count_K, 3, 5))
                for hero_idx in range(0, 3):
                    # 来自官方数据集
                    official_legal_action_list = data_list[hero_idx][1: 1 + 5]
                    official_action_list = data_list[hero_idx][8: 8 + 5]
                    official_probs_list = data_list[hero_idx][13: 13 + 5]
                    official_sub_action_list = data_list[hero_idx][19: 19 + 5]

                    official_topk_action_list = [torch.topk(x, k, dim=-1).indices for x in official_probs_list]

                    # teacher的logits
                    teacher_logits_list = teacher_rst_list[hero_idx][0:-1]
                    # 使用legal_action作为掩码对logits_list进行处理
                    teacher_masked_logits_list = []
                    for logits, legal_action in zip(teacher_logits_list, official_legal_action_list):
                        new_logits = logits.clone()
                        new_logits[legal_action == 0] = float('-inf')
                        # new_logits[legal_action == 0] = -1e9
                        teacher_masked_logits_list.append(new_logits)
                    teacher_topk_action_list = [torch.topk(x, k, dim=-1).indices for x in teacher_masked_logits_list]

                    # student 的logits
                    student_logits_list = student_rst_list[hero_idx][0:-1]
                    # 使用legal_action作为掩码对logits_list进行处理
                    student_masked_logits_list = []
                    for logits, legal_action in zip(student_logits_list, official_legal_action_list):
                        new_logits = logits.clone()
                        new_logits[legal_action == 0] = float('-inf')
                        # new_logits[legal_action == 0] = -1e9
                        student_masked_logits_list.append(new_logits)

                    # softmax 求出 probs
                    student_masked_probs_list = [torch.softmax(x, dim=-1) for x in student_masked_logits_list]
                    student_topk_probs_list = [torch.topk(x, k, dim=-1).values for x in student_masked_probs_list]
                    for action_idx in range(5):
                        for kk in range(1, count_K + 1):
                            student_topk_prob[kk - 1][hero_idx][action_idx] = torch.mean(
                                torch.sum(student_topk_probs_list[action_idx][:, :kk], dim=-1))

                    student_topk_action_list = [torch.topk(x, k, dim=-1).indices for x in student_masked_logits_list]

                    # 计算 top k action 准确率，student 相对于官方数据集
                    for action_idx in range(5):
                        official_topk_actions = official_topk_action_list[action_idx]
                        student_topk_actions = student_topk_action_list[action_idx]
                        is_equal = (student_topk_actions == official_topk_actions)
                        for kk in range(1, count_K + 1):
                            student_to_official_topk_acc[kk - 1, hero_idx, action_idx] = torch.sum(is_equal[:, kk - 1]).item() / is_equal.shape[0]

                    # 计算 top k action 准确率，student 相对 teacher
                    for action_idx in range(5):
                        teacher_actions = teacher_topk_action_list[action_idx]
                        student_actions = student_topk_action_list[action_idx]
                        is_equal = (student_actions == teacher_actions)
                        for kk in range(1, count_K + 1):
                            student_to_teacher_topk_acc[kk - 1, hero_idx, action_idx] = \
                                torch.sum(is_equal[:, kk - 1]).item() / is_equal.shape[0]

                    # softmax 求出 probs
                    teacher_probs_list = [torch.softmax(x, dim=-1) for x in teacher_logits_list]
                    # argmax 求出 labels
                    teacher_action_list = [torch.argmax(x, dim=-1, keepdim=True) for x in teacher_probs_list]
                    # 替换datalist
                    # 替换 action
                    data_list[hero_idx][8: 8 + 5] = teacher_action_list
                    # 替换 logits
                    data_list[hero_idx][13: 13 + 5] = teacher_logits_list
                    # 替换sub_action
                    new_button = teacher_action_list[0].reshape(-1).int()

                    res_arr = hero_legal_sub_action_list[hero_idx][new_button]
                    for s in range(5):
                        # 替换sub_action
                        data_list[hero_idx][19 + s] = res_arr[:, 1 + s].reshape(
                            data_list[hero_idx][19 + s].shape).int()



                total_loss, info_list, acc_list = student_net.compute_loss(data_list, student_rst_list)

                cost_all_list.append(total_loss)
                all_hero_loss_list.append(info_list)
                all_acc_list.append(acc_list)

                student_to_official_topk_acc_list.append(student_to_official_topk_acc)
                student_to_teacher_topk_acc_list.append(student_to_teacher_topk_acc)
                student_topk_prob_list.append(student_topk_prob)

            numpy_data.close()

            all_hero_loss = torch.mean(torch.Tensor(all_hero_loss_list), dim=0)
            all_acc = torch.mean(torch.Tensor(all_acc_list), dim=0)
            for m in range(3):
                for n in range(5):
                    writer.add_scalar(f'loss/loss_{m}_{n}', all_hero_loss[m][n].item(), step)
                for n in range(3):
                    writer.add_scalar(f'action_accuracy/acc_{m}_{n}', all_acc[m][n].item(), step)


            print(f"write topk acc of step = {step}")
            topk_acc_list1 = torch.mean(torch.stack(student_to_official_topk_acc_list), dim=(0, 2)).cpu().numpy()
            topk_acc_list3 = torch.mean(torch.stack(student_topk_prob_list), dim=(0, 2)).cpu().numpy()
            for n in range(5):
                for k in range(count_K):
                    stu_to_official_writers[k].add_scalar(f'action_accuracy/acc_{n}', topk_acc_list1[k, n], step)
                    stu_to_official_writers[k].add_scalar(f'probs/{n}', topk_acc_list3[k, n], step)

            topk_acc_list2 = torch.mean(torch.stack(student_to_teacher_topk_acc_list), dim=(0, 2)).cpu().numpy()
            for n in range(5):
                for k in range(count_K):
                    stu_to_tea_writers[k].add_scalar(f'action_accuracy/acc_{n}', topk_acc_list2[k, n], step)
