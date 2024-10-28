import math
import os
import numpy as np


def get_env(key):
    env = {
        "hard_weight": 0.0,
        "soft_weight": 0.0,
        "distill_weight": 1.0,
        "file_cnt": 10000000
    }
    for k, v in env.items():
        env[k] = str(v)
    env.update(os.environ)
    return env.get(key).strip() if key in env.keys() else None


class Config:
    NETWORK_NAME = "network"

    HARD_WEIGHT = float(get_env("hard_weight"))
    SOFT_WEIGHT = float(get_env("soft_weight"))
    DISTILL_WEIGHT = float(get_env("distill_weight"))
    FILE_CNT = int(get_env("file_cnt"))

    DISTILL_TEMPERATURE = 4
    DISTILL_LAMBDA_WEIGHT = 0.5

    CAMP_NUM = 2
    game_type = "3v3"

    HERO_NUM = 3
    SOLDIER_NUM = 10
    ORGAN_NUM = 3
    MONSTER_NUM = 20

    HERO_DIM = 251
    MAIN_HERO_DIM = 44
    SOLDIER_DIM = 25
    ORGAN_DIM = 29
    MONSTER_DIM = 28

    GLOBAL_DIM = 68

    VEC_DIM = HERO_NUM * HERO_DIM * CAMP_NUM + MAIN_HERO_DIM + \
              (SOLDIER_NUM * SOLDIER_DIM + ORGAN_NUM * ORGAN_DIM) * CAMP_NUM + \
              MONSTER_NUM * MONSTER_DIM + GLOBAL_DIM

    HERO_SERI_VEC_SPLIT_SHAPE = [[(6, 17, 17), (VEC_DIM,)]] * HERO_NUM  # (1734 + 2852) * 3
    TOTAL_DIM = int(VEC_DIM + np.prod(HERO_SERI_VEC_SPLIT_SHAPE[0][0]))

    TARGET_DIM = 1 + HERO_NUM * CAMP_NUM + 1 + MONSTER_NUM + SOLDIER_NUM + 1  # 39

    BUTTON_DIM = 13  # [Empty, Empty, Move, Attack, Skill1, ...]
    MOVE_DIM = 25
    OFFSET_DIM = 42  # Offset_X / Offset_Z
    REWARD_DIM = 1
    ADVANTAGE_DIM = 1

    MODEL_PRED_DIMS = [BUTTON_DIM, MOVE_DIM, OFFSET_DIM, OFFSET_DIM, TARGET_DIM]

    HERO_DATA_SPLIT_SHAPE = [
                                [TOTAL_DIM]  # feature
                                + MODEL_PRED_DIMS  # legal action
                                + [REWARD_DIM, ADVANTAGE_DIM]  # reward & advantage
                                + [1] * len(MODEL_PRED_DIMS)  # labels (action)
                                + MODEL_PRED_DIMS  # probs
                                + [1]  # is_train
                                + [1] * len(MODEL_PRED_DIMS)  # sub_action
                            ] * HERO_NUM
    HERO_LABEL_SIZE_LIST = [MODEL_PRED_DIMS] * HERO_NUM

    print(HERO_SERI_VEC_SPLIT_SHAPE)
    print(HERO_DATA_SPLIT_SHAPE)
    print(HERO_LABEL_SIZE_LIST)

    ## transformer
    TOKEN_DIM = 224
    HEAD_DIM = 224
    ATT_HEAD_NUM = 4
    ATT_LAYER_NUM = 3
    DK_SCALE = 1.0 / (HEAD_DIM ** 0.5)

    ###### DATA CONFIG ######
    HERO_FEATURE_IMG_CHANNEL = [[6]] * HERO_NUM  # feature image channel for each hero
    HERO_IS_REINFORCE_TASK_LIST = [[True] * len(MODEL_PRED_DIMS)] * HERO_NUM

    ###### NETWORK CONFIG #####
    # lstm
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 1024
    # target attention
    TARGET_EMBEDDING_DIM = 64
    # value
    VALUE_HEAD_NUM = REWARD_DIM  # single-head value for single-branch reward
    HERO_POLICY_WEIGHT = 1.0

    ###### LOSS CONFIG #####
    INIT_LEARNING_RATE_START = 0.0002
    BETA_START = 0.008
    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001

    ###### TASK CONFIG ######
    TASK_ID = 123456
    TASK_UUID = "1234abcd"
    data_keys = "hero1_data,hero2_data,hero3_data,hero4_data,hero5_data"
    data_shapes = [[sum(HERO_DATA_SPLIT_SHAPE[0]) * LSTM_TIME_STEPS + LSTM_UNIT_SIZE * 2]] * HERO_NUM
