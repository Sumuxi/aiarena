import os
import numpy as np

folder_path = '/dataset/train'

file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
print(f"npz size: {len(file_list)}")

total_samples_shape_0 = 0

# 遍历排序后的文件名列表
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    
    # 加载 .npz 文件
    with np.load(file_path) as data:
        samples = data['samples']
        shape = samples.shape
        
        # 输出文件名和 shape
        print(f'File: {filename}, Shape: {shape}')
        
        # 累加第 0 维度的大小
        total_samples_shape_0 += shape[0]

print(f'total samples: {total_samples_shape_0}')
