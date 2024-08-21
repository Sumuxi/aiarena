# -*- coding:utf-8 -*-
import os


class Config:

    HERO_NUM = 3
    SEND_SAMPLE_FRAME = 963
    # kinghonour: 1e-5 atari: 2.5e-4
    INIT_LEARNING_RATE = 1e-4
    END_LEARNING_RATE = 1e-5
    ALPHA = 0.5
    BETA = 0.01
    EPSILON = 1e-5
    INIT_CLIP_PARAM = 0.1
    # kinghonour:4096 atari:256
    BATCH_SIZE = 4096
    EPISODE = 20000000
    GAMMA = 0.995
    LAMDA = 0.95
    STEPS = 128
    EPOCHES = 4
    MINI_BATCH_NUM = 4
    ENV_NAME = "BowlingNoFrameskip-v4"
    MIN_POLICY = 0.00005
    T = 1
    TASK_NAME = "test"
    MEM_PROCESS_NUM = 8
    DATA_KEYS = "input_data"
    KEY_TYPES = "tf.float32"
    SERVER_PORT = 30166
    ACTOR_NUM = 0
    LEARNER_NUM = 0
    EACH_LEARNER_NUM = 0
    PARAMS_PATH = "/data1/reinforcement_platform/rl_learner_platform/model/update"
    GPU_SERVER_LIST = ""
    UPDATE_PATH = "../model/update"
    INIT_PATH = "../model/init"
    MEM_POOL_PATH = "./config/mem_pool.host_list"
    TASK_UUID = "123"

    ENV_RULE = "none"
    EVAL_FREQ = 10

    # kinghonour_dqn
    DATA_SHAPES = [[5888]]
    DATA_SPLIT_SHAPE = [2823, 79, 2823, 79, 1, 1, 1, 1, 79, 1]
