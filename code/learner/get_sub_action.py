import argparse
import os

import numpy as np
import torch

from networkmodel.pytorch.NetworkModel import NetworkModel

"""
从数据集中提取完整的 sub_action_mask
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--id', type=str, help='', required=True)
args = parser.parse_args()

id = int(args.id)

npz_directory = "/mnt/storage/yxh/competition24/lightweight/dataset/train"

npz_list = [f for f in os.listdir(npz_directory) if f.endswith('.npz')]
npz_list = sorted(npz_list)

# range_list = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
# npz_list = npz_list[range_list[id]: range_list[id + 1]]

batch_size = 500
device = "cuda:" + str(id)

net = NetworkModel()
state_dict = torch.load("/aiarena/code/assets/baseline/model.pth", map_location="cpu")
missing_keys, unexpected_keys = net.load_state_dict(state_dict["network_state_dict"], strict=True)
print(f"missing_keys: {missing_keys}")
print(f"unexpected_keys: {unexpected_keys}")
net.to(device)

hero_sub_action_list = [
    [[i, 0, 0, 0, 0, 0] for i in range(13)],
    [[i, 0, 0, 0, 0, 0] for i in range(13)],
    [[i, 0, 0, 0, 0, 0] for i in range(13)],
]

with torch.no_grad():
    for npz in npz_list:
        print(f"handle file {npz}")
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

                for hero_idx in range(0, 3):
                    old_legal_action = data_list[hero_idx][1: 1 + 5]
                    old_action_list = data_list[hero_idx][8: 8 + 5]
                    old_logits_list = data_list[hero_idx][13: 13 + 5]
                    old_sub_action_list = data_list[hero_idx][19: 19 + 5]

                    action_sub_action = torch.cat([old_action_list[0], *old_sub_action_list], dim=-1)

                    for button in range(13):
                        indexes = torch.where(action_sub_action[:, 0] == button)[0]
                        if indexes.shape[0] > 0:
                            # 找到至少一个
                            hero_sub_action_list[hero_idx][button] = action_sub_action[indexes[0]].cpu().numpy().tolist()

            numpy_data.close()

        print("result:")
        print(hero_sub_action_list)

    print("last result:")
    print(hero_sub_action_list)
