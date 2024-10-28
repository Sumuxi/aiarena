import math


def calculate_single_item_score(reference_strong, reference_weak, actual):
    """
    计算单项评分
    reference_strong: 参考指标strong
    reference_weak: 参考指标weak
    actual: 实测指标
    """
    if actual <= 0:
        return 0

    score = math.log(reference_weak) - math.log(actual)
    denom = math.log(reference_weak) - math.log(reference_strong)
    normalized_score = (score / denom) * (80 - 20) + 20
    # single_item_score = min(normalized_score, 100)
    single_item_score = normalized_score
    return single_item_score


def calculate_original_total_score(avg_time_score, avg_storage_score, model_size_score, avg_energy_score):
    """
    计算原始综合得分
    """
    return (max(avg_time_score, 0) * 0.4 +
            max(avg_storage_score, 0) * 0.2 +
            max(model_size_score, 0) * 0.2 +
            max(avg_energy_score, 0) * 0.2)


def calculate_final_score(target_accuracy, actual_accuracy, target_consistency, avg_consistency, original_total_score):
    """
    计算综合评分
    """
    accuracy = min(target_accuracy, actual_accuracy) / target_accuracy
    consistency = min(target_consistency, avg_consistency) / target_consistency
    final_score = accuracy * consistency * original_total_score
    return final_score


def cal_score(actual_win_rate, avg_consistency, act_time, act_memory, act_size, act_energy):
    target_win_rate = 0.4
    target_consistency = 0.9
    # norms = [5 * 1000, 300 * 1000, 160, 200, 1, 50, 5 * 1024 * 1024, 100 * 1024 * 1024]
    norms = [7800, 315 * 1000, 160, 200, 2, 60, 9 * 1024 * 1024, 140 * 1024 * 1024]

    # 单项评分计算
    time_score = calculate_single_item_score(norms[0], norms[1], act_time)
    memory_score = calculate_single_item_score(norms[2], norms[3], act_memory)
    size_score = calculate_single_item_score(norms[6], norms[7], act_size)
    energy_score = calculate_single_item_score(norms[4], norms[5], act_energy)

    original_total_score = calculate_original_total_score(time_score, memory_score, size_score, energy_score)
    final_score = calculate_final_score(target_win_rate, actual_win_rate, target_consistency, avg_consistency,
                                        original_total_score)
    return final_score, original_total_score, time_score, memory_score, size_score, energy_score


# v15
# avg_time,avg_memory,avg_energy,avg_size
# 1995.67,161.44,0.39,11237507

# v4
# avg_time:  1682.1
# avg_memory:  147.817578125
# avg_energy:  0.005105552822351455
# avg_size:  909962.0

# v0
# avg_time:  10247.5
# avg_memory:  201.6125
# avg_energy:  0.44242222011089327
# avg_size:  37084438.0

raw_data = [
    [0, 0.9448, 0.9686, 0.921, 1556.67, 149.49, 909962, 0.16],
    [0.26, 0.8599, 0.932, 0.7878, 1995.67, 161.44, 7893933, 0.39],
    [0.325, 0.5272, 0.5898, 0.4646, 1976.33, 162.26, 7893932, 0.30],
    [0.325, 0.5021, 0.574, 0.4302, 2027.00, 160.78, 7893933, 0.29],
    [0.2733, 0.444, 0.5098, 0.3782, 1954.67, 163.15, 7893933, 0.39],
    [0.35, 0.9464, 0.955, 0.9378, 2457.33, 155.17, 7814613, 0.30],
    [0.388333333, 0.959, 0.9652, 0.9528, 2208.67, 149.53, 12152530, 0.34],
    [0.366, 0.9486, 0.958, 0.9392, 2474, 149.9921875, 12153867, 0.352796316],
    [0.3857, 0.9483, 0.9582, 0.9384, 3535.40, 150.18, 20849019, 0.52],
    [0.3875, 0.9446, 0.9606, 0.9286, 3414.00, 150.84, 20847891, 0.53],
    [0.39, 0.9446, 0.9606, 0.9286, 3606.67, 160.19, 20849019, 0.79],
    [0.295, 0.9548, 0.9606, 0.949, 3520.00, 158.07, 20849019, 0.77],
    [0.407, 0.9508, 0.9612, 0.9404, 3527.00, 151.60, 20849019, 0.53],
    [0.415384615, 0.9505, 0.9644, 0.9366, 3437.67, 152.40, 20847891, 0.54],
    [0.366, 0.9484, 0.9594, 0.9374, 4071.00, 122.47, 20849019, 0.48],
]

exps = [
    'v4e1s30w',
    'v15e1s90w',
    'v15e20s90w',
    'v15e20s100w',
    'v15e23s110w',
    'v15e30s140w',
    'f1e1s80w',
    'f1e1s200w',
    'f3e1s150w',
    'f3e1s160w',
    'f3e1s160w',
    'f3e2s100w',
    'f3e3s140w',
    'f3e3s145w',
    'f3e3s150w',
]

data_list = [{'exp': x, 'raw_data': y} for x, y in zip(exps, raw_data)]

if __name__ == "__main__":
    for data in data_list:
        exp_name, actual_win_rate, avg_consistency, top1_consistency, top2_consistency, act_time, act_memory, act_size, act_energy = \
            data['exp'], data['raw_data'][0], data['raw_data'][1], \
            data['raw_data'][2], data['raw_data'][3], \
            data['raw_data'][-4], data['raw_data'][-3], data['raw_data'][-2], data['raw_data'][-1]
        final_score, original_total_score, time_score, memory_score, size_score, energy_score = \
            cal_score(actual_win_rate, avg_consistency, act_time, act_memory, act_size, act_energy)
        print(
            f"{exp_name}"
            f"\t{final_score}"
            f"\t{original_total_score}"
            f"\t{actual_win_rate}"
            # f"\t{avg_consistency}"
            # f"{top1_consistency}\t"
            # f"{top2_consistency}\t"

            # f"{act_time}\t"
            # f"{act_memory}\t"
            # f"{act_size}\t"
            # f"{act_energy}\t"

            f"\t{time_score}"
            f"\t{memory_score}"
            f"\t{size_score}"
            f"\t{energy_score}"
            f"\t{exp_name}"
        )
