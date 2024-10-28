import sys
import os
import torch
from torch import Tensor

# model_path = sys.argv[1]
model_path = '/media/storage/yxh/competition24/lightweight/exp_dir/v15e36/ckpt/model_step100.pth'
abs_model_path = os.path.abspath(model_path)

model = torch.load(abs_model_path, map_location=torch.device('cpu'))

# print('model step: ', model['step'])
# print(model.keys())

def print_dict(dictionary, indent=0):
    for key, value in dictionary.items():
        print(' ' * indent + str(key) + ': ' + (str(value.shape) +", " + str(value.dtype) if isinstance(value, Tensor) else ""))
        if isinstance(value, dict):
            print_dict(value, indent + 4)  # 增加4个空格的缩进级别

# print_dict(model)

for k,v in model['network_state_dict'].items():
    min_val, max_val, scale, zero_point = '','','',''
    if 'min_val' in k:
        min_val = v.item()
    if 'max_val' in k:
        max_val = v.item()
    if 'scale' in k:
        scale = v.item()
    if 'zero_point' in k:
        zero_point = v.item()
    print(k, min_val, max_val, scale, zero_point)