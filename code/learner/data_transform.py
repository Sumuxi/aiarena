import argparse
import os

import numpy as np
import torch

from networkmodel.pytorch.NetworkModel import NetworkModel

"""
替换数据集，
将被掩码处理过的probability替换为用baseline模型推理生成的logits
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--id', type=str, default='0', help='')
args = parser.parse_args()

id = int(args.id)

npz_directory = "/mnt/storage/yxh/competition24/lightweight/dataset/train"
new_npz_directory = "/mnt/storage/yxh/competition24/lightweight/dataset/train_with_logits"

npz_list = [f for f in os.listdir(npz_directory) if f.endswith('.npz')]
npz_list = sorted(npz_list)

# range_list = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
# npz_list = npz_list[range_list[id]: range_list[id + 1]]

# range_list = [0, 400, 800, 1200, 1600, 2000]
# npz_list = npz_list[range_list[id-3]: range_list[id-3 + 1]]

batch_size = 200
device = "cuda:" + str(id)

net = NetworkModel()
state_dict = torch.load("/mnt/storage/yxh/competition24/lightweight/aiarena/code/assets/baseline/model.pth", map_location="cpu")
missing_keys, unexpected_keys = net.load_state_dict(state_dict["network_state_dict"], strict=True)
print(f"missing_keys: {missing_keys}")
print(f"unexpected_keys: {unexpected_keys}")
net.to(device)

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
hero_legal_sub_action_list = torch.tensor(hero_legal_sub_action_list).to(device)

torch.set_printoptions(precision=6, sci_mode=False)

with torch.no_grad():
    for npz in npz_list:
        with open(os.path.join(npz_directory, npz), 'rb') as file:
            numpy_data = np.load(file)
            samples = numpy_data["samples"]
            # samples 是一个 numpy 数组
            num_samples = samples.shape[0]

            results = []
            # 分批次进行推理
            for i in range(0, num_samples, batch_size):
                batch_samples = samples[i:min(i + batch_size, num_samples)]
                input_data = torch.Tensor(batch_samples).to(device)
                data_list = net.format_data(input_data)
                rst_list = net(data_list)

                for hero_idx in range(0, 3):
                    # 来自官方数据集
                    old_legal_action_list = data_list[hero_idx][1: 1 + 5]
                    old_action_list = data_list[hero_idx][8: 8 + 5]
                    old_probs_list = data_list[hero_idx][13: 13 + 5]
                    old_sub_action_list = data_list[hero_idx][19: 19 + 5]

                    old_action_list_from_probs = [torch.argmax(x, dim=-1, keepdim=True) for x in old_probs_list]
                    # 计算old_action_list_from_probs和old_action_list相同的比例
                    print("action acc1: ", [torch.sum(x == y).item() / y.shape[0] for x, y in zip(old_action_list_from_probs, old_action_list)])

                    # 来自baseline的推理结果
                    logits_list = rst_list[hero_idx][0:-1]
                    hero_value = rst_list[hero_idx][-1]

                    # 使用legal_action作为掩码对logits_list进行处理
                    masked_logits_list = []
                    for logits, legal_action in zip(logits_list, old_legal_action_list):
                        new_logits = logits.clone()
                        # new_logits[legal_action == 0] = float('-inf')
                        new_logits[legal_action == 0] = -1e9
                        masked_logits_list.append(new_logits)
                    # softmax 求出 probs
                    masked_probs_list = [torch.softmax(x, dim=-1) for x in masked_logits_list]
                    # argmax 求出 labels
                    masked_action_list = [torch.argmax(x, dim=-1, keepdim=True) for x in masked_probs_list]
                    # 计算action_list和old_action_list相同的比例
                    print("action acc2: ", [(torch.sum(x == y)/y.shape[0]).item() for x, y in zip(masked_action_list, old_action_list)])
                    print("logits diff1:", [torch.mean(torch.abs(x - y)).item() for x, y in zip(masked_probs_list, old_probs_list)])

                    # 直接对logits_list进行处理
                    # softmax 求出 probs
                    probs_list = [torch.softmax(x, dim=-1) for x in logits_list]
                    # argmax 求出 labels
                    action_list = [torch.argmax(x, dim=-1, keepdim=True) for x in probs_list]
                    print("action acc3: ", [(torch.sum(x == y) / y.shape[0]).item() for x, y in zip(action_list, old_action_list)])
                    print("logits diff2:", [torch.mean(torch.abs(x - y)).item() for x, y in zip(probs_list, old_probs_list)])
                    print("action acc4: ",
                          [(torch.sum(x == y) / y.shape[0]).item() for x, y in zip(masked_action_list, old_action_list_from_probs)])

                    # 替换datalist
                    # 替换 action
                    data_list[hero_idx][8: 8 + 5] = action_list
                    # 替换 logits
                    data_list[hero_idx][13: 13 + 5] = logits_list
                    # 替换sub_action
                    new_button = action_list[0].reshape(-1).int()
                    # assert torch.isin(new_button, torch.tensor([1, 2, 3, 4, 5, 6, 8, 11]).to(device)).all().item()
                    count1 = torch.sum(
                        torch.isin(new_button, torch.tensor([1, 2, 3, 4, 5, 6, 8, 11]).to(device))).item()
                    count2 = new_button.shape[0]
                    print(f"合法子动作比例 hero{hero_idx}：{count1}/{count2}")

                    res_arr = hero_legal_sub_action_list[hero_idx][new_button]
                    for k in range(5):
                        # 替换sub_action
                        data_list[hero_idx][19 + k] = res_arr[:, 1 + k].reshape(data_list[hero_idx][19 + k].shape).int()

                # 将data_list转换为numpy
                out_arr = net.convert_to_datas(data_list)
                results.append(out_arr.cpu().numpy())

            # result_ndarray = np.concatenate(results, axis=0)
            # file_name = os.path.join(new_npz_directory, npz)
            # np.savez_compressed(file_name, samples=result_ndarray)
            # print(f"saved {npz}, data shape is {result_ndarray.shape}")
            numpy_data.close()
