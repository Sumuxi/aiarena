import os
import json
import torch
import re
import onnx
from onnx import helper
import vcaponnx.pytorch_tools as pytorch_tools

def fuse_model(module_in):
    module_in.eval()

    torch.quantization.fuse_modules(module_in.conv_layers, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.hero_share_mlp, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.public_info_mlp, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.soldier_share_mlp, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.organ_share_mlp, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.monster_mlp, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.global_mlp, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.concat_mlp, ['6', '7'], inplace=True)
    torch.quantization.fuse_modules(module_in.concat_mlp, ['4', '5'], inplace=True)
    torch.quantization.fuse_modules(module_in.concat_mlp, ['2', '3'], inplace=True)
    torch.quantization.fuse_modules(module_in.concat_mlp, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.communicate_mlp, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(module_in.communicate_mlp, ['2', '3'], inplace=True)
    torch.quantization.fuse_modules(module_in.communicate_mlp, ['4', '5'], inplace=True)
    # torch.quantization.fuse_modules(module_in.target_head, ['0', '1'], inplace=True)
    # for m in module_in.action_heads:
    #     torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

    module_in.train()
    return module_in


def export_onnx(ckpt_path, onnx_without_fake_quant, onnx_with_fake_quant, is_qat=True):
    import copy

    input_shapes = [
        (1, 4586),
        (1, 4586),
        (1, 4586),
    ]
    # assign names for I/O nodes
    input_names = [
        "feature_hero0",
        "feature_hero1",
        "feature_hero2",
        # "lstm_cell_in",
        # "lstm_hidden_in",
    ]
    output_names = [
        "logits_hero0",
        "logits_hero1",
        "logits_hero2",
        # "lstm_cell_out",
        # "lstm_hidden_out",
    ]
    # graph(注释插入量化和反量化节点)
    net = Model(None)
    net2 = copy.deepcopy(net)

    # 导出带ACT量化信息浮点ONNX模型
    if is_qat:
        net = fuse_model(net)
    # 传入参数2需要与训练时对应
    pytorch_tools.qat_qconfig_and_prepare(net, 'qnnpack', scene_mode=1)
    pytorch_tools.export_onnx_floatmodel_with_actfakequant(net, ckpt_path, onnx_with_fake_quant,
                                                           input_shapes, input_names, output_names)

    # 导出不带量化信息浮点ONNX模型
    if is_qat:
        net2 = fuse_model(net2)
    # 传入参数2需要与训练时对应
    pytorch_tools.qat_qconfig_and_prepare(net2, 'qnnpack', scene_mode=2)
    pytorch_tools.export_onnx_floatmodel_without_fakequant(net2, ckpt_path, onnx_without_fake_quant,
                                                           input_shapes, input_names, output_names)


def handle_qat(ckpt_path, onnx_no_qat, onnx_with_fake_quant, onnx_without_fake_quant, quant_json, is_qat=True):
    # 加载 ONNX 模型
    onnx_fake = onnx.load(onnx_with_fake_quant)
    onnx_no_fake = onnx.load(onnx_without_fake_quant)
    onnx_f32 = onnx.load(onnx_no_qat)

    abs_model_path = os.path.abspath(ckpt_path)
    ckpt = torch.load(abs_model_path, map_location=torch.device('cpu'))

    from onnxsim import simplify
    # convert model
    onnx_fake, check = simplify(onnx_fake)
    assert check, "Simplified ONNX model could not be validated"
    onnx_no_fake, check = simplify(onnx_no_fake)
    assert check, "Simplified ONNX model could not be validated"
    onnx_f32, check = simplify(onnx_f32)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(onnx_fake, onnx_with_fake_quant.split(".onnx")[0] + "_sim.onnx")
    onnx.save(onnx_no_fake, onnx_without_fake_quant.split(".onnx")[0] + "_sim.onnx")
    onnx.save(onnx_f32, onnx_no_qat.split(".onnx")[0] + "_sim.onnx")

    # 获取模型的图
    graph_fake = onnx_fake.graph

    quant_node_dict = {}  # onnx中需要量化的结点
    # 遍历模型中的所有节点
    for node in graph_fake.node:
        if 'fake_quant' in node.op_type.lower():
            input_name = node.input[0]
            output_name = node.output[0]
            prev_node = get_prev_node(graph_fake, input_name)
            next_node = get_next_node(graph_fake, output_name)
            ckpt_key_name = node.input[3].split('.activation_post_process')[0]
            if 'Relu' in next_node.op_type:
                edit_node = next_node
            elif 'Relu' in prev_node.op_type:
                edit_node = prev_node
            elif 'Concat' in prev_node.op_type:
                edit_node = prev_node
            else:
                assert 'Conv' in prev_node.op_type or 'Add' in prev_node.op_type \
                       or 'Gemm' in prev_node.op_type or 'Clip' in prev_node.op_type, prev_node.op_type
                edit_node = prev_node

            quant_node_dict[edit_node.name] = ckpt_key_name

    print(f'fake_quant_nodes: ')
    for k, v in quant_node_dict.items():
        print(f'{k}: {v}')
    print()

    if is_qat:
        graph_edit = onnx_no_fake.graph
        edit_edge_dict = {}  # onnx中需要量化的结点：ckpt对应的结点
        # 遍历模型中的所有节点
        for node_name, ckpt_key in quant_node_dict.items():
            edit_node = get_node(graph_edit, node_name)
            if edit_node is None and 'Add' in node_name:
                edit_node = get_node(graph_edit, re.sub(r'Add', 'Gemm', node_name))
                if edit_node is None:
                    print("debug:", f"{node_name} is None")

            old_edge_name = edit_node.output[0]
            new_edge_name = re.sub(r'[/.]', '_', old_edge_name)

            new_edge = rename_edge(graph_edit, edit_node.name, old_edge_name, new_edge_name)
            edit_edge_dict[new_edge] = ckpt_key

        onnx.save(onnx_no_fake, onnx_without_fake_quant.split(".onnx")[0] + "_sim_rename.onnx")
    else:
        graph_edit = onnx_f32.graph
        edit_edge_dict = {}  # onnx中需要量化的结点：ckpt对应的结点
        # 遍历模型中的所有节点
        for node_name, ckpt_key in quant_node_dict.items():
            edit_node = get_node(graph_edit, node_name)
            if edit_node is None and 'Add' in node_name:
                edit_node = get_node(graph_edit, re.sub(r'Add', 'Gemm', node_name))
            if edit_node is None:
                print("debug:", f"{node_name} is None")

            old_edge_name = edit_node.output[0]
            new_edge_name = re.sub(r'[/.]', '_', old_edge_name)

            new_edge = rename_edge(graph_edit, edit_node.name, old_edge_name, new_edge_name)
            edit_edge_dict[new_edge] = ckpt_key

        onnx.save(onnx_f32, onnx_no_qat.split(".onnx")[0] + "_f32.onnx")

    print(f'graph_edit edit_edges: ')
    for k, v in edit_edge_dict.items():
        print(f'{k}: {v}')
    print()
    print(f'len(fake_quant_nodes): {len(quant_node_dict)}')
    print(f'len(edit_edge_dict): {len(edit_edge_dict)}')

    json_data_full = {
        "activation_encodings": {},
        "param_encodings": {}
    }
    for k,v in edit_edge_dict.items():
        scale = ckpt['network_state_dict'][f"{v}.activation_post_process.scale"].item()
        zero_point = ckpt['network_state_dict'][f"{v}.activation_post_process.zero_point"].item()
        zero_point = -zero_point
        min_val = ckpt['network_state_dict'][f"{v}.activation_post_process.activation_post_process.min_val"].item()
        max_val = ckpt['network_state_dict'][f"{v}.activation_post_process.activation_post_process.max_val"].item()

        json_data_full["activation_encodings"][f"{k}"] = []
        tmp_dict = {"bitwidth": 8, "min": min_val, "max": max_val, "scale": scale, "offset": zero_point}
        json_data_full["activation_encodings"][f"{k}"].append(tmp_dict)

    quant_json = quant_json.split('.json')[0] + str("_qat.json" if is_qat else "_f32.json")
    with open(quant_json, 'w') as json_file:
        json.dump(json_data_full, json_file, indent=4)


def get_node(graph, node_name):
    for node in graph.node:
        if node.name == node_name:
            return node

def get_prev_node(graph, output_name):
    for src_node in graph.node:
        for i, output in enumerate(src_node.output):
            if output == output_name:
                return src_node


def get_next_node(graph, input_name):
    for dest_node in graph.node:
        for i, input in enumerate(dest_node.input):
            if input == input_name:
                return dest_node


def rename_edge(graph, node_name, edge_name, new_edge_name):
    for node in graph.node:
        if node.name == node_name:
            if 'logits' in node.output[0]:
                print("debug: ", node_name, node.output[0])

            # 1. 修改目标节点的输入名称
            old_edge_name = node.output[0]

            finded = False

            # 2. 查找源节点，并修改它的输出名称
            for dest_node in graph.node:
                for i, input in enumerate(dest_node.input):
                    if input == old_edge_name:
                        assert len(dest_node.input) > 0, f"{dest_node.name}, {len(dest_node.input)}"
                        dest_node.input[i] = new_edge_name
                        finded = True

            if finded:
                node.output[0] = new_edge_name

            return node.output[0]


def write_quant_json(ckpt_path, onnx_with_fake_quant, quant_json):
    abs_model_path = os.path.abspath(ckpt_path)

    ckpt = torch.load(abs_model_path, map_location=torch.device('cpu'))
    onnx_qat = onnx.load(onnx_with_fake_quant)

    # act_keys = []
    #
    # for k, v in ckpt['network_state_dict'].items():
    #     if 'activation_post_process' in k and 'weight_fake_quant' not in k:
    #         if 'quant' not in k:
    #             act_keys.append(k.split('.activation_post_process')[0])
    #
    # filter_act_keys = []
    # for i in range(0, len(act_keys)):
    #     if i == 0:
    #         filter_act_keys.append(act_keys[i])
    #     elif act_keys[i] != filter_act_keys[-1]:
    #         filter_act_keys.append(act_keys[i])
    #
    # fake_quant_nodes = filter_act_keys
    # print(f'fake_quant_nodes: ')
    # for k in filter_act_keys:
    #     print(f'"{k}",')

    json_data = {
        "activation_encodings": {},
        "param_encodings": {}
    }

    json_data_full = {
        "activation_encodings": {},
        "param_encodings": {}
    }

    # 遍历计算图中的节点，寻找 `FakeQuantize` 节点
    for node in onnx_qat.graph.node:
        if 'fake_quant' in node.op_type.lower():
            input_name = node.input[0]
            output_name = node.output[0]

            prev_node = get_prev_node(onnx_qat.graph, input_name)
            next_node = get_next_node(onnx_qat.graph, output_name)




            node_name = node.input[3].split('.activation_post_process')[0]

            scale = ckpt['network_state_dict'][f"{node_name}.activation_post_process.scale"].item()
            zero_point = ckpt['network_state_dict'][f"{node_name}.activation_post_process.zero_point"].item()
            zero_point = -zero_point
            min_val = ckpt['network_state_dict'][f"{node_name}.activation_post_process.activation_post_process.min_val"].item()
            max_val = ckpt['network_state_dict'][f"{node_name}.activation_post_process.activation_post_process.max_val"].item()

            json_data["activation_encodings"][f"{input_name}"] = []
            tmp_dict = {"bitwidth": 8, "min": min_val, "max": max_val}
            json_data["activation_encodings"][f"{input_name}"].append(tmp_dict)

            json_data_full["activation_encodings"][f"{input_name}"] = []
            tmp_dict = {"bitwidth": 8, "min": min_val, "max": max_val, "scale": scale, "offset": zero_point}
            json_data_full["activation_encodings"][f"{input_name}"].append(tmp_dict)


    # for k in fake_quant_nodes:
    #     scale = ckpt['network_state_dict'][f"{k}.activation_post_process.scale"].item()
    #     zero_point = ckpt['network_state_dict'][f"{k}.activation_post_process.zero_point"].item()
    #     zero_point = -zero_point
    #     min_val = ckpt['network_state_dict'][f"{k}.activation_post_process.activation_post_process.min_val"].item()
    #     max_val = ckpt['network_state_dict'][f"{k}.activation_post_process.activation_post_process.max_val"].item()
    #
    #     if 'q_concat' in k:
    #         k = '/Concat'
    #
    #     for fname in edges.keys():
    #         if k in fname:
    #             json_data["activation_encodings"][f"{edges[fname]}"] = []
    #             # tmp_dict = {"bitwidth": 8, "min": min_val, "max": max_val, "scale": scale, "offset": zero_point}
    #             tmp_dict = {"bitwidth": 8, "min": min_val, "max": max_val}
    #             json_data["activation_encodings"][f"{edges[fname]}"].append(tmp_dict)

    print('len(json_data["activation_encodings"])', len(json_data["activation_encodings"]))

    with open(quant_json, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    with open(quant_json.split('.json')[0]+'_full.json', 'w') as json_file:
        json.dump(json_data_full, json_file, indent=4)


if __name__ == "__main__":
    from model.pytorch.final_v3_qat import Model

    base_dir = '/aiarena/output/onnx'
    exp_name = 'f3e10qat'
    ckpt_name = 'f3e10'
    ckpt_path = '/media/storage/yxh/competition24/lightweight/exp_dir/f3e10qat/ckpt/model_step1005000.pth'

    os.makedirs(os.path.join(base_dir, exp_name), exist_ok=True)

    onnx_no_qat = '/media/storage/yxh/competition24/lightweight/exp_dir/onnx/f3e2/f3e2s100w.onnx'

    # 输出带伪量化浮点onnx模型路径
    onnx_with_fake_quant = os.path.join(base_dir, exp_name, ckpt_name + "_fake.onnx")
    # 输出不带伪量化浮点onnx模型路径
    onnx_without_fake_quant = os.path.join(base_dir, exp_name, ckpt_name + "_no_fake.onnx")

    quant_json = os.path.join(base_dir, exp_name, ckpt_name + ".json")

    use_qat_model = True

    export_onnx(ckpt_path, onnx_without_fake_quant, onnx_with_fake_quant, is_qat=use_qat_model)
    handle_qat(ckpt_path, onnx_no_qat, onnx_with_fake_quant, onnx_without_fake_quant, quant_json, is_qat=use_qat_model)
