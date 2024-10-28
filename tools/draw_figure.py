import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

save_path = "/aiarena/output/figures"
os.makedirs(save_path, exist_ok=True)

# exps_to_draw = ['f1e1','f1e2','f2e2','f3e1', 'f3e2', 'f3e3','f4e2']
# exps_to_draw = ['f3e1', 'f3e2', 'f3e3']
# exps_to_draw = ['f1e1', 'f2e1', 'f3e1', 'f4e1', 'v15e30']
exps_to_draw = ['f1e1', 'f3e1', 'f3e3', 'v15e30']

win_rate_dict = [
    {
        'exp_name': 'f1e1',
        'exp_label': 'f1: kl',
        'win_rate': [
            (200, (17 + 14 + 18 + 14 + 10 + 147) / (40 * 5 + 400)),
            (180, 12 / 40),
            (160, 11 / 40),
            (140, 12 / 40),
            (120, 13 / 40),
            (100, 11 / 40),
            (80, (16 + 16 + 20 + 15 + 15 + 17 + 71 + 150) / (6 * 40 + 200 + 400)),
            (60, 10 / 40),
            (55, 55 / 200),
            (51, 57 / 200),
            (50, (19 + 16 + 10 + 70) / (3 * 40 + 200)),
            (49, (13 + 14 + 72) / (2 * 40 + 200)),
            (48, 43 / 200),
            (45, 68 / 200),
            (40, (14 + 49) / (40 + 200)),
            (20, 10 / 40),
        ],
    },
    {
        'exp_name': 'f1e2',
        'exp_label': 'f1: top3 kl + kl',
        'win_rate': [
            (100, 9 / 40),
            (50, 7 / 40),
            (20, 5 / 40),
        ],
    },
    {
        'exp_name': 'f2e1',
        'exp_label': 'f2: kl',
        'win_rate': [
            (100, 9 / 40),
            (50, 12 / 40),
            (40, 7 / 40),
        ],
    },
    {
        'exp_name': 'f2e2',
        'exp_label': 'f2: top3 kl + kl',
        'win_rate': [
            (200, 12 / 40),
            (160, 11 / 40),
            (140, 11 / 40),
            (120, 10 / 40),
            (100, 14 / 40),
            (90, 13 / 40),
            (50, 8 / 40),
            (20, 5 / 40),
        ],
    },
    {
        'exp_name': 'f3e1',
        'exp_label': 'f3: kl',
        'win_rate': [
            (200, 13 / 40),
            (180, 15 / 40),
            (170, 11 / 40),
            (160, (18 + 14 + 14 + 14 + 84 + 160) / (4 * 40 + 200 + 400)),
            (155, 14 / 40),
            (150, (18 + 9 + 81) / (40 + 40 + 200)),
            (140, 14 / 40),
            (120, 12 / 40),
            (100, 12 / 40),
            (80, 11 / 40),
            (60, 12 / 40),
            (50, (18 + 60) / (40 + 200)),
            (40, (17 + 71) / (40 + 200)),
            (30, 10 / 40),
            (20, 14 / 40),
        ],
    },
    {
        'exp_name': 'f3e2',
        'exp_label': 'f3: top3 kl + kl',
        'win_rate': [
            (200, 9 / 40),
            (180, 11 / 40),
            (160, 12 / 40),
            (140, 11 / 40),
            (120, 12 / 40),
            (110, 14 / 40),
            (100, (18 + 10 + 17 + 12 + 14) / 5 / 40),
            (90, 10 / 40),
            (80, 15 / 40),
            (70, 10 / 40),
            (60, 11 / 40),

        ],
    },
    {
        'exp_name': 'f3e3',
        'exp_label': 'f3: cross entropy',
        'win_rate': [
            (180, 11 / 40),
            (160, 11 / 40),
            (150, (17 + 71) / (40 + 200)),
            (145, 84 / 200),
            (140, (17 + 17 + 80) / (40 + 40 + 200)),
            (120, 15 / 40),
            (100, 11 / 40),
            (80, 15 / 40),
            (60, 12 / 40),
            (40, 13 / 40),
        ],
    },
    {
        'exp_name': 'f4e1',
        'exp_label': 'f4: kl',
        'win_rate': [
            (180, 11 / 40),
            (160, 8 / 40),
            (150, 14 / 40),
            (140, 15 / 40),
            (120, 9 / 40),
            (100, 11 / 40),
            (80, 11 / 40),
            (50, 9 / 40),
            (40, 14 / 40),
        ],
    },
    {
        'exp_name': 'f4e2',
        'exp_label': 'f4: top3 kl + kl',
        'win_rate': [
            (40, 11 / 40),
            (60, 10 / 40),
            (80, 8 / 40),
            (100, 12 / 40),
            (120, 14 / 40),
            (200, 10 / 40),
        ],
    },
    {
        'exp_name': 'v15e30',
        'exp_label': 'v15: kl',
        'win_rate': [
            (180, 7 / 40),
            (160, 12 / 40),
            (140, (15 + 13) / (2 * 40)),
            (120, 13 / 40),
            (100, 4 / 40),
            (80, 7 / 40),
            (70, 10 / 40),
            (50, 10 / 40),
            (40, (16 + 9 + 11) / 3 / 40),
            (30, 12 / 40),
            (20, 6 / 40),
        ],
    },

]

# 创建一幅图
plt.figure(figsize=(10, 6))

# 颜色或线型的列表，用于区分不同实验的曲线
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '*', 'x']

# 遍历实验
idx = 0
for exp in win_rate_dict:
    exp_name = exp['exp_name']
    if exp_name not in exps_to_draw:
        continue

    exp_label = exp['exp_label']
    rates = [(x, y) for x, y in exp['win_rate']]

    # 使用 zip 函数将 step 和 win_rate 打包成键值对
    # combined = list(zip(steps, rates))
    combined = list(rates)
    # 根据 step 对键值对排序
    combined_sorted = sorted(combined, key=lambda x: x[0])
    # 解压排序后的键值对
    sorted_step, sorted_win_rate = zip(*combined_sorted)

    print("exp_name:", exp_name)
    print("exp_label:", exp_label)
    print("win rate by step:")
    for x, y in combined_sorted:
        print(f"({x}: {y})")
    print()

    # 使用不同的颜色、线型和标记绘制曲线
    color = colors[idx % len(colors)]
    linestyle = linestyles[idx % len(linestyles)]
    marker = markers[idx % len(markers)]
    idx += 1

    plt.plot(sorted_step, sorted_win_rate,
             marker=marker, linestyle=linestyle,
             color=color, label=f"{exp_label}")

# 设置标题和标签
plt.title('Win Rate vs Step')
plt.xlabel('Step')
plt.ylabel('Win Rate')

plt.grid(visible=True, which='both')

# 显示图例
plt.legend()

# 显示图表
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Win_Rate_of_Batch.png'), dpi=300, format='png')
plt.close()
