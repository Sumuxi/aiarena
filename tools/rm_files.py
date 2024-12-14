import os
import re


def extract_number(file_name):
    # 使用正则表达式找到文件名中的数字
    match = re.search(r'step(\d+)', file_name)
    # 如果找到数字，返回整数形式的数字；否则返回0（或根据需要返回其他默认值）
    return int(match.group(1)) if match else -1


directory = '/aiarena/output'

for root, dirs, files in os.walk(directory):
    for file in files:
        full_path = os.path.join(root, file)
        step = extract_number(file)
        if step >= 0 and file.endswith('.zip'):
            print(f"remove {full_path}")
            os.remove(full_path)
        if file.endswith('.pth') and step % 10000 != 0:
            print(f"remove {full_path}")
            os.remove(full_path)
