import torch.nn as nn

from .util import make_mlp_layer, make_linear_layer, make_action_heads

v1 = {
    # v1: 48.737352MFLOPs, 5.958627MParams
    # 特征处理
    'conv_layers': nn.Sequential(
        nn.Conv2d(6, 18, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Conv2d(18, 12, 3, 1, 1),
        nn.MaxPool2d(2)
    ),  # image like feature
    'hero_share_mlp': make_mlp_layer([251, 256]),
    'hero_frd_mlp': make_linear_layer(256, 128),  # 我方英雄
    'hero_emy_mlp': make_linear_layer(256, 128),  # 敌方英雄
    'public_info_mlp': make_mlp_layer([44, 128, 128], activation_last=False),  # 主英雄
    'soldier_share_mlp': make_mlp_layer([25, 128]),
    'soldier_frd_mlp': make_linear_layer(128, 128),  # 我方小兵
    'soldier_emy_mlp': make_linear_layer(128, 128),  # 敌方小兵
    'organ_share_mlp': make_mlp_layer([29, 128]),
    'organ_frd_mlp': make_linear_layer(128, 128),  # 我方防御塔
    'organ_emy_mlp': make_linear_layer(128, 128),  # 敌方防御塔
    'monster_mlp': make_mlp_layer([28, 128, 128], activation_last=False),  # 野怪
    'global_mlp': make_mlp_layer([68, 128, 128], activation_last=False),  # 计分板信息
    # 拼接特征，联合处理
    'concat_mlp': make_mlp_layer([1344, 2048, 768, 512, 512]),
    # 分类头
    'action_heads': make_action_heads((13, 25, 42, 42, 39), [512, 256])
}

# 特征处理MLP的隐藏层都是128
v2 = {
    # 特征处理
    'conv_layers': nn.Sequential(
        nn.Conv2d(6, 18, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Conv2d(18, 12, 3, 1, 1),
        nn.MaxPool2d(2)
    ),  # image like feature
    'hero_share_mlp': make_mlp_layer([251, 128]),
    'hero_frd_mlp': make_linear_layer(128, 128),  # 我方英雄
    'hero_emy_mlp': make_linear_layer(128, 128),  # 敌方英雄
    'public_info_mlp': make_mlp_layer([44, 128, 128], activation_last=False),  # 主英雄
    'soldier_share_mlp': make_mlp_layer([25, 128]),
    'soldier_frd_mlp': make_linear_layer(128, 128),  # 我方小兵
    'soldier_emy_mlp': make_linear_layer(128, 128),  # 敌方小兵
    'organ_share_mlp': make_mlp_layer([29, 128]),
    'organ_frd_mlp': make_linear_layer(128, 128),  # 我方防御塔
    'organ_emy_mlp': make_linear_layer(128, 128),  # 敌方防御塔
    'monster_mlp': make_mlp_layer([28, 128, 128], activation_last=False),  # 野怪
    'global_mlp': make_mlp_layer([68, 128, 128], activation_last=False),  # 计分板信息
    # 拼接特征，联合处理
    'concat_mlp': make_mlp_layer([1344, 2048, 768, 512, 512]),
    # 分类头
    'action_heads': make_action_heads((13, 25, 42, 42, 39), [512, 256])
}

# 特征处理MLP的隐藏层都是128
v3 = {
    # 特征处理
    # image like feature
    'conv_layers': nn.Sequential(
        nn.Conv2d(6, 18, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Conv2d(18, 12, 3, 1, 1),
        nn.MaxPool2d(2)
    ),  # 192
    'hero_share_mlp': make_mlp_layer([251, 128]),
    'hero_frd_mlp': make_linear_layer(128, 64),  # 我方英雄
    'hero_emy_mlp': make_linear_layer(128, 64),  # 敌方英雄
    'public_info_mlp': make_mlp_layer([44, 128, 64], activation_last=False),  # 主英雄
    'soldier_share_mlp': make_mlp_layer([25, 128]),
    'soldier_frd_mlp': make_linear_layer(128, 64),  # 我方小兵
    'soldier_emy_mlp': make_linear_layer(128, 64),  # 敌方小兵
    'organ_share_mlp': make_mlp_layer([29, 128]),
    'organ_frd_mlp': make_linear_layer(128, 64),  # 我方防御塔
    'organ_emy_mlp': make_linear_layer(128, 64),  # 敌方防御塔
    'monster_mlp': make_mlp_layer([28, 128, 64], activation_last=False),  # 野怪
    'global_mlp': make_mlp_layer([68, 128, 64], activation_last=False),  # 计分板信息
    # 拼接特征，联合处理
    # 192 + 64*9
    'concat_mlp': make_mlp_layer([768, 512, 512]),
    # 分类头
    'action_heads': make_action_heads((13, 25, 42, 42, 39), [512, 256])
}
