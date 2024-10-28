import sys
import os

import json
import torch
from torch import Tensor

# model_path = sys.argv[1]
model_path = '/mnt/storage/yxh/competition24/lightweight/exp_dir/v15e35/ckpt/model_step1010000.pth'
abs_model_path = os.path.abspath(model_path)

model = torch.load(abs_model_path, map_location=torch.device('cpu'))

# print('model step: ', model['step'])
# print(model.keys())


act_quant_info = {

}

act_keys = []

for k, v in model['network_state_dict'].items():
    if 'activation_post_process' in k and 'weight_fake_quant' not in k:
        if 'quant' not in k:
            act_keys.append(k.split('.activation_post_process')[0])

# print('\n'.join(act_keys))
# print('\n')
# print('\n')

filter_act_keys = []
for i in range(0, len(act_keys)):
    if i == 0:
        filter_act_keys.append(act_keys[i])
    elif act_keys[i] != filter_act_keys[-1]:
        filter_act_keys.append(act_keys[i])

for k in filter_act_keys:
    print(f'"{k}",')

print('\n')
print('\n')


fake_quant_nodes = [
    "conv_layers.0",
    "conv_layers.3",
    "hero_share_mlp.0",
    "hero_frd_mlp",
    "hero_emy_mlp",
    "public_info_mlp.0",
    "public_info_mlp.2",
    "soldier_share_mlp.0",
    "soldier_frd_mlp",
    "soldier_emy_mlp",
    "organ_share_mlp.0",
    "organ_frd_mlp",
    "organ_emy_mlp",
    "monster_mlp.0",
    "monster_mlp.2",
    "global_mlp.0",
    "global_mlp.2",
    "concat_mlp.0",
    "concat_mlp.2",
    "concat_mlp.4",
    "concat_mlp.6",
    "action_heads.0.0",
    "action_heads.0.2",
    "action_heads.1.0",
    "action_heads.1.2",
    "action_heads.2.0",
    "action_heads.2.2",
    "action_heads.3.0",
    "action_heads.3.2",
    "target_embed",
    "target_head.0",
    "target_head.2",
    "q_concat"
]

edges = {'/conv_layers/conv_layers.0/Relu_output_0': '_conv_layers_conv_layers_0_Relu_output_0',
         '/conv_layers/conv_layers.3/Conv_output_0': '_conv_layers_conv_layers_3_Conv_output_0',
         '/hero_share_mlp/hero_share_mlp.0/Add_output_0': '_hero_share_mlp_hero_share_mlp_0_Add_output_0',
         '/hero_frd_mlp/Add_output_0': '_hero_frd_mlp_Add_output_0',
         '/hero_share_mlp/hero_share_mlp.0_1/Add_output_0': '_hero_share_mlp_hero_share_mlp_0_1_Add_output_0',
         '/hero_emy_mlp/Add_output_0': '_hero_emy_mlp_Add_output_0',
         '/public_info_mlp/public_info_mlp.0/Gemm_output_0': '_public_info_mlp_public_info_mlp_0_Gemm_output_0',
         '/public_info_mlp/public_info_mlp.2/Gemm_output_0': '_public_info_mlp_public_info_mlp_2_Gemm_output_0',
         '/monster_mlp/monster_mlp.0/Add_output_0': '_monster_mlp_monster_mlp_0_Add_output_0',
         '/monster_mlp/monster_mlp.2/Add_output_0': '_monster_mlp_monster_mlp_2_Add_output_0',
         '/soldier_share_mlp/soldier_share_mlp.0/Add_output_0': '_soldier_share_mlp_soldier_share_mlp_0_Add_output_0',
         '/soldier_frd_mlp/Add_output_0': '_soldier_frd_mlp_Add_output_0',
         '/soldier_share_mlp/soldier_share_mlp.0_1/Add_output_0': '_soldier_share_mlp_soldier_share_mlp_0_1_Add_output_0',
         '/soldier_emy_mlp/Add_output_0': '_soldier_emy_mlp_Add_output_0',
         '/organ_share_mlp/organ_share_mlp.0/Add_output_0': '_organ_share_mlp_organ_share_mlp_0_Add_output_0',
         '/organ_frd_mlp/Add_output_0': '_organ_frd_mlp_Add_output_0',
         '/organ_share_mlp/organ_share_mlp.0_1/Add_output_0': '_organ_share_mlp_organ_share_mlp_0_1_Add_output_0',
         '/organ_emy_mlp/Add_output_0': '_organ_emy_mlp_Add_output_0',
         '/global_mlp/global_mlp.0/Gemm_output_0': '_global_mlp_global_mlp_0_Gemm_output_0',
         '/global_mlp/global_mlp.2/Gemm_output_0': '_global_mlp_global_mlp_2_Gemm_output_0',
         '/Concat_output_0': '_Concat_output_0', '/Concat_1_output_0': '_Concat_1_output_0',
         '/concat_mlp/concat_mlp.0/Gemm_output_0': '_concat_mlp_concat_mlp_0_Gemm_output_0',
         '/concat_mlp/concat_mlp.2/Gemm_output_0': '_concat_mlp_concat_mlp_2_Gemm_output_0',
         '/concat_mlp/concat_mlp.4/Gemm_output_0': '_concat_mlp_concat_mlp_4_Gemm_output_0',
         '/concat_mlp/concat_mlp.6/Gemm_output_0': '_concat_mlp_concat_mlp_6_Gemm_output_0',
         '/conv_layers/conv_layers.0_1/Relu_output_0': '_conv_layers_conv_layers_0_1_Relu_output_0',
         '/conv_layers/conv_layers.3_1/Conv_output_0': '_conv_layers_conv_layers_3_1_Conv_output_0',
         '/hero_share_mlp/hero_share_mlp.0_2/Add_output_0': '_hero_share_mlp_hero_share_mlp_0_2_Add_output_0',
         '/hero_frd_mlp_1/Add_output_0': '_hero_frd_mlp_1_Add_output_0',
         '/hero_share_mlp/hero_share_mlp.0_3/Add_output_0': '_hero_share_mlp_hero_share_mlp_0_3_Add_output_0',
         '/hero_emy_mlp_1/Add_output_0': '_hero_emy_mlp_1_Add_output_0',
         '/public_info_mlp/public_info_mlp.0_1/Gemm_output_0': '_public_info_mlp_public_info_mlp_0_1_Gemm_output_0',
         '/public_info_mlp/public_info_mlp.2_1/Gemm_output_0': '_public_info_mlp_public_info_mlp_2_1_Gemm_output_0',
         '/monster_mlp/monster_mlp.0_1/Add_output_0': '_monster_mlp_monster_mlp_0_1_Add_output_0',
         '/monster_mlp/monster_mlp.2_1/Add_output_0': '_monster_mlp_monster_mlp_2_1_Add_output_0',
         '/soldier_share_mlp/soldier_share_mlp.0_2/Add_output_0': '_soldier_share_mlp_soldier_share_mlp_0_2_Add_output_0',
         '/soldier_frd_mlp_1/Add_output_0': '_soldier_frd_mlp_1_Add_output_0',
         '/soldier_share_mlp/soldier_share_mlp.0_3/Add_output_0': '_soldier_share_mlp_soldier_share_mlp_0_3_Add_output_0',
         '/soldier_emy_mlp_1/Add_output_0': '_soldier_emy_mlp_1_Add_output_0',
         '/organ_share_mlp/organ_share_mlp.0_2/Add_output_0': '_organ_share_mlp_organ_share_mlp_0_2_Add_output_0',
         '/organ_frd_mlp_1/Add_output_0': '_organ_frd_mlp_1_Add_output_0',
         '/organ_share_mlp/organ_share_mlp.0_3/Add_output_0': '_organ_share_mlp_organ_share_mlp_0_3_Add_output_0',
         '/organ_emy_mlp_1/Add_output_0': '_organ_emy_mlp_1_Add_output_0',
         '/global_mlp/global_mlp.0_1/Gemm_output_0': '_global_mlp_global_mlp_0_1_Gemm_output_0',
         '/global_mlp/global_mlp.2_1/Gemm_output_0': '_global_mlp_global_mlp_2_1_Gemm_output_0',
         '/Concat_2_output_0': '_Concat_2_output_0', '/Concat_3_output_0': '_Concat_3_output_0',
         '/concat_mlp/concat_mlp.0_1/Gemm_output_0': '_concat_mlp_concat_mlp_0_1_Gemm_output_0',
         '/concat_mlp/concat_mlp.2_1/Gemm_output_0': '_concat_mlp_concat_mlp_2_1_Gemm_output_0',
         '/concat_mlp/concat_mlp.4_1/Gemm_output_0': '_concat_mlp_concat_mlp_4_1_Gemm_output_0',
         '/concat_mlp/concat_mlp.6_1/Gemm_output_0': '_concat_mlp_concat_mlp_6_1_Gemm_output_0',
         '/conv_layers/conv_layers.0_2/Relu_output_0': '_conv_layers_conv_layers_0_2_Relu_output_0',
         '/conv_layers/conv_layers.3_2/Conv_output_0': '_conv_layers_conv_layers_3_2_Conv_output_0',
         '/hero_share_mlp/hero_share_mlp.0_4/Add_output_0': '_hero_share_mlp_hero_share_mlp_0_4_Add_output_0',
         '/hero_frd_mlp_2/Add_output_0': '_hero_frd_mlp_2_Add_output_0',
         '/hero_share_mlp/hero_share_mlp.0_5/Add_output_0': '_hero_share_mlp_hero_share_mlp_0_5_Add_output_0',
         '/hero_emy_mlp_2/Add_output_0': '_hero_emy_mlp_2_Add_output_0',
         '/public_info_mlp/public_info_mlp.0_2/Gemm_output_0': '_public_info_mlp_public_info_mlp_0_2_Gemm_output_0',
         '/public_info_mlp/public_info_mlp.2_2/Gemm_output_0': '_public_info_mlp_public_info_mlp_2_2_Gemm_output_0',
         '/monster_mlp/monster_mlp.0_2/Add_output_0': '_monster_mlp_monster_mlp_0_2_Add_output_0',
         '/monster_mlp/monster_mlp.2_2/Add_output_0': '_monster_mlp_monster_mlp_2_2_Add_output_0',
         '/soldier_share_mlp/soldier_share_mlp.0_4/Add_output_0': '_soldier_share_mlp_soldier_share_mlp_0_4_Add_output_0',
         '/soldier_frd_mlp_2/Add_output_0': '_soldier_frd_mlp_2_Add_output_0',
         '/soldier_share_mlp/soldier_share_mlp.0_5/Add_output_0': '_soldier_share_mlp_soldier_share_mlp_0_5_Add_output_0',
         '/soldier_emy_mlp_2/Add_output_0': '_soldier_emy_mlp_2_Add_output_0',
         '/organ_share_mlp/organ_share_mlp.0_4/Add_output_0': '_organ_share_mlp_organ_share_mlp_0_4_Add_output_0',
         '/organ_frd_mlp_2/Add_output_0': '_organ_frd_mlp_2_Add_output_0',
         '/organ_share_mlp/organ_share_mlp.0_5/Add_output_0': '_organ_share_mlp_organ_share_mlp_0_5_Add_output_0',
         '/organ_emy_mlp_2/Add_output_0': '_organ_emy_mlp_2_Add_output_0',
         '/global_mlp/global_mlp.0_2/Gemm_output_0': '_global_mlp_global_mlp_0_2_Gemm_output_0',
         '/global_mlp/global_mlp.2_2/Gemm_output_0': '_global_mlp_global_mlp_2_2_Gemm_output_0',
         '/Concat_4_output_0': '_Concat_4_output_0', '/Concat_5_output_0': '_Concat_5_output_0',
         '/concat_mlp/concat_mlp.0_2/Gemm_output_0': '_concat_mlp_concat_mlp_0_2_Gemm_output_0',
         '/concat_mlp/concat_mlp.2_2/Gemm_output_0': '_concat_mlp_concat_mlp_2_2_Gemm_output_0',
         '/concat_mlp/concat_mlp.4_2/Gemm_output_0': '_concat_mlp_concat_mlp_4_2_Gemm_output_0',
         '/concat_mlp/concat_mlp.6_2/Gemm_output_0': '_concat_mlp_concat_mlp_6_2_Gemm_output_0',
         '/Concat_7_output_0': '_Concat_7_output_0',
         '/action_heads.0/action_heads.0.0/Gemm_output_0': '_action_heads_0_action_heads_0_0_Gemm_output_0',
         '/action_heads.0/action_heads.0.2/Gemm_output_0': '_action_heads_0_action_heads_0_2_Gemm_output_0',
         '/action_heads.1/action_heads.1.0/Gemm_output_0': '_action_heads_1_action_heads_1_0_Gemm_output_0',
         '/action_heads.1/action_heads.1.2/Gemm_output_0': '_action_heads_1_action_heads_1_2_Gemm_output_0',
         '/action_heads.2/action_heads.2.0/Gemm_output_0': '_action_heads_2_action_heads_2_0_Gemm_output_0',
         '/action_heads.2/action_heads.2.2/Gemm_output_0': '_action_heads_2_action_heads_2_2_Gemm_output_0',
         '/action_heads.3/action_heads.3.0/Gemm_output_0': '_action_heads_3_action_heads_3_0_Gemm_output_0',
         '/action_heads.3/action_heads.3.2/Gemm_output_0': '_action_heads_3_action_heads_3_2_Gemm_output_0',
         '/target_embed/Add_output_0': '_target_embed_Add_output_0',
         '/target_head/target_head.0/Gemm_output_0': '_target_head_target_head_0_Gemm_output_0',
         '/target_head/target_head.2/Gemm_output_0': '_target_head_target_head_2_Gemm_output_0',
         '/Concat_8_output_0': '_Concat_8_output_0',
         '/action_heads.0/action_heads.0.0_1/Gemm_output_0': '_action_heads_0_action_heads_0_0_1_Gemm_output_0',
         '/action_heads.0/action_heads.0.2_1/Gemm_output_0': '_action_heads_0_action_heads_0_2_1_Gemm_output_0',
         '/action_heads.1/action_heads.1.0_1/Gemm_output_0': '_action_heads_1_action_heads_1_0_1_Gemm_output_0',
         '/action_heads.1/action_heads.1.2_1/Gemm_output_0': '_action_heads_1_action_heads_1_2_1_Gemm_output_0',
         '/action_heads.2/action_heads.2.0_1/Gemm_output_0': '_action_heads_2_action_heads_2_0_1_Gemm_output_0',
         '/action_heads.2/action_heads.2.2_1/Gemm_output_0': '_action_heads_2_action_heads_2_2_1_Gemm_output_0',
         '/action_heads.3/action_heads.3.0_1/Gemm_output_0': '_action_heads_3_action_heads_3_0_1_Gemm_output_0',
         '/action_heads.3/action_heads.3.2_1/Gemm_output_0': '_action_heads_3_action_heads_3_2_1_Gemm_output_0',
         '/target_embed_1/Add_output_0': '_target_embed_1_Add_output_0',
         '/target_head/target_head.0_1/Gemm_output_0': '_target_head_target_head_0_1_Gemm_output_0',
         '/target_head/target_head.2_1/Gemm_output_0': '_target_head_target_head_2_1_Gemm_output_0',
         '/Concat_9_output_0': '_Concat_9_output_0',
         '/action_heads.0/action_heads.0.0_2/Gemm_output_0': '_action_heads_0_action_heads_0_0_2_Gemm_output_0',
         '/action_heads.0/action_heads.0.2_2/Gemm_output_0': '_action_heads_0_action_heads_0_2_2_Gemm_output_0',
         '/action_heads.1/action_heads.1.0_2/Gemm_output_0': '_action_heads_1_action_heads_1_0_2_Gemm_output_0',
         '/action_heads.1/action_heads.1.2_2/Gemm_output_0': '_action_heads_1_action_heads_1_2_2_Gemm_output_0',
         '/action_heads.2/action_heads.2.0_2/Gemm_output_0': '_action_heads_2_action_heads_2_0_2_Gemm_output_0',
         '/action_heads.2/action_heads.2.2_2/Gemm_output_0': '_action_heads_2_action_heads_2_2_2_Gemm_output_0',
         '/action_heads.3/action_heads.3.0_2/Gemm_output_0': '_action_heads_3_action_heads_3_0_2_Gemm_output_0',
         '/action_heads.3/action_heads.3.2_2/Gemm_output_0': '_action_heads_3_action_heads_3_2_2_Gemm_output_0',
         '/target_embed_2/Add_output_0': '_target_embed_2_Add_output_0',
         '/target_head/target_head.0_2/Gemm_output_0': '_target_head_target_head_0_2_Gemm_output_0',
         '/target_head/target_head.2_2/Add_output_0': '_target_head_target_head_2_2_Add_output_0',
         '/Concat_10_output_0': '_Concat_10_output_0', '/Concat_11_output_0': '_Concat_11_output_0',
         '/Concat_12_output_0': '_Concat_12_output_0'}


json_data = {
    "activation_encodings": {},
    "param_encodings": {}
}


for k in fake_quant_nodes:
    scale = model['network_state_dict'][f"{k}.activation_post_process.scale"].item()
    zero_point = model['network_state_dict'][f"{k}.activation_post_process.zero_point"].item()
    zero_point = -zero_point
    min_val = model['network_state_dict'][f"{k}.activation_post_process.activation_post_process.min_val"].item()
    max_val = model['network_state_dict'][f"{k}.activation_post_process.activation_post_process.max_val"].item()

    if 'q_concat' in k:
        k = '/Concat'

    for fname in edges.keys():
        if k in fname:
            # print(
            #     f'        "{edges[fname]}": [{{"bitwidth": 8, "min": {min_val}, "max": {max_val}, "scale": {scale},"scale": {scale}, "offset": {zero_point}}}],')
            json_data["activation_encodings"][f"{edges[fname]}"] = []
            tmp_dict = {}
            tmp_dict["bitwidth"] = 8
            tmp_dict["min"] = min_val
            tmp_dict["max"] = max_val
            tmp_dict["scale"] = scale
            tmp_dict["offset"] = zero_point
            json_data["activation_encodings"][f"{edges[fname]}"].append(tmp_dict)


print(len(json_data["activation_encodings"]))


# 写入文件
file_name = "/mnt/storage/yxh/competition24/lightweight/exp_dir/onnx/v15e35/v15e35s110w_1_qat_rename.json"

with open(file_name, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)
