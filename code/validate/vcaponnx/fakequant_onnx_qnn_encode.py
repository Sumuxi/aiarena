'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-03-24 11:17:39
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-11-21 16:12:27
FilePath: \vcap_pytorch_quant\tools\vcaponnx\fakequant_onnx_qnn_encode.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: duanzhijie
Date: 2022-03-24 11:17:39
LastEditTime: 2022-07-13 17:37:12
FilePath: \pytorchquant\vcap_pytorch_quant\QNN\fakequant_onnx_qnn_ecode.py
'''
import numpy as np
import onnx
import argparse
import json
from collections import OrderedDict
from onnx import helper
from onnx import numpy_helper

upper_minmax_list = ["Reshape", "MaxPool", "AvgPool", "Transpose", "Pad", \
                     "AveragePool", "ReduceMax", "ReduceMin", "ReduceMean", \
                     "Resize", "Upsample"]


def computeQnnActQuantizeInfo(min, max, qmin=0, qmax=255):
    qmin_float = float(qmin)
    qmax_float = float(qmax)
    if min > 0:
        min = 0
        scale = (max - min) / (qmax_float - qmin_float)
        return scale, qmin, max, min
    if max < 0:
        max = 0
        scale = (max - min) / (qmax_float - qmin_float)
        return scale, 255, max, min
    if min == max:
        scale = 0
        zero_point = 0
        return scale, int(zero_point), max, min

    if min <= 0 and max > 0:
        scale = (max - min) / (qmax_float - qmin_float)
        zero_point = round(qmin_float - min / scale)
        nudged_min = (qmin_float - zero_point) * scale
        nudged_max = (qmax_float - zero_point) * scale
        return scale, int(zero_point), nudged_max, nudged_min


def save_encodings_with_AIMET_format(encoding_list):
    aimet_format_file = OrderedDict()
    for item in encoding_list:
        if len(item) == 6:
            aimet_format_json = OrderedDict()
            aimet_format_json["bitwidth"] = item[-1]
            aimet_format_json["min"] = item[1]
            aimet_format_json["max"] = item[2]
            aimet_format_json["scale"] = item[3]
            aimet_format_json["offset"] = item[-2]
            aimet_format_file[item[0]] = [aimet_format_json]
        if len(item) == 4:
            aimet_format_json = OrderedDict()
            aimet_format_json["bitwidth"] = item[-1]
            aimet_format_json["min"] = item[1]
            aimet_format_json["max"] = item[2]
            aimet_format_file[item[0]] = [aimet_format_json]
    return aimet_format_file


def compute_concat_min_max(model, nodes_dict, initial_dict):
    nodes = model.graph.node
    concat_dict = {}
    for node in nodes:
        if node.op_type == "Concat":
            input_num = len(node.input)
            act_min = 0
            act_max = 0
            for i in range(input_num):
                input_name = node.input[i]
                fake_node = None
                if input_name in nodes_dict.keys():
                    fake_node = nodes_dict[input_name]
                    if fake_node.op_type == "fused_moving_avg_obs_fake_quant":
                        min_name = fake_node.input[3]
                        max_name = fake_node.input[4]
                        tmp_min = numpy_helper.to_array(initial_dict[min_name])
                        tmp_max = numpy_helper.to_array(initial_dict[max_name])
                        if tmp_min < act_min:
                            act_min = tmp_min
                        if tmp_max > act_max:
                            act_max = tmp_max
                    if fake_node.op_type in ["Reshape", "MaxPool", "AvgPool", "Transpose"]:
                        # print(" concat input is not fakequant node!")
                        loop_node = fake_node
                        while loop_node.input:
                            tmp_name = loop_node.input[0]
                            if tmp_name in nodes_dict.keys():
                                tmp_node = nodes_dict[tmp_name]
                                if tmp_node.op_type == "fused_moving_avg_obs_fake_quant":
                                    min_name = tmp_node.input[3]
                                    max_name = tmp_node.input[4]
                                    tmp_min = numpy_helper.to_array(initial_dict[min_name])
                                    tmp_max = numpy_helper.to_array(initial_dict[max_name])
                                    if tmp_min < act_min:
                                        act_min = tmp_min
                                    if tmp_max > act_max:
                                        act_max = tmp_max
                                    break
                            loop_node = tmp_node

            print("concat :", node.output[0], " :", act_min, act_max)
            concat_dict[node.output[0]] = [act_min, act_max]
    return concat_dict


def compute_actual_min_max(node, needfake_dict):
    if node.output[0] in needfake_dict.keys():
        min_max = needfake_dict[node.output[0]]
        min_value = min_max[0]
        max_value = min_max[1]
        if node.op_type == "Relu":
            if min_value < 0:
                min_value = 0
        if node.op_type == "Relu6":
            if min_value < 0:
                min_value = 0
            if max_value > 6:
                max_value = 6
        return min_value, max_value


def get_all_node_minmax(model):
    # node_name:(min, max)
    node_quantdict = {}
    nodes = model.graph.node
    nodeout_dict = {}
    for node in nodes:
        if node.output:
            # nodeout_dict[node.output[0]] = node
            for x in node.output:
                nodeout_dict[x] = node
    initializers = model.graph.initializer
    initial_dict = {}
    for initial in initializers:
        initial_dict[initial.name] = initial
    concat_dict = compute_concat_min_max(model, nodeout_dict, initial_dict)
    # get all node quantinfo with fakequant min/max
    for i in range(len(nodes)):
        node = nodes[i]
        if node.op_type == "fused_moving_avg_obs_fake_quant":
            # 找到上一个节点
            input_name = node.input[0]
            min_name = node.input[3]
            max_name = node.input[4]
            act_min = numpy_helper.to_array(initial_dict[min_name])
            act_max = numpy_helper.to_array(initial_dict[max_name])
            # graph inupt not encode
            if input_name not in nodeout_dict.keys():
                continue
            encode_node = nodeout_dict[input_name]
            output_name = encode_node.output[0]
            # concat 单独计算min/max
            if encode_node.op_type == "Concat":
                # print(concat_dict)
                if encode_node.output[0] in concat_dict.keys():
                    min_max = concat_dict[encode_node.output[0]]
                    act_min = min_max[0]
                    act_max = min_max[1]
                    act_scale, act_zp, _, _, = computeQnnActQuantizeInfo(act_min, act_max, 0, 255)
            # 因此float model已经把BN fuse了
            if encode_node.op_type == "BatchNormalization":
                bn_input_name = encode_node.input[0]
                tmp_conv_node = nodeout_dict[bn_input_name]
                tmp_conv_name = tmp_conv_node.output[0]
                output_name = tmp_conv_name
            # 同时保留conv:min/max ; fake_quant:min/max
            node_quantdict[output_name] = (act_min, act_max)
            node_quantdict[node.output[0]] = (act_min, act_max)

    for i in range(len(nodes)):
        node = nodes[i]
        if node.op_type in ["Softmax", "Sigmoid"]:
            if node.output:
                node_quantdict[node.output[0]] = (0.0, 1.0)

    # fakequant->transpose->reshape
    # add+fakequant+upsample
    for i in range(len(nodes)):
        node = nodes[i]
        if node.op_type in upper_minmax_list:
            # if node.input and node.input[0] in nodeout_dict:
            if node.input:
                upper_node = nodeout_dict[node.input[0]]
                if upper_node.output[0] in node_quantdict.keys():
                    min_max = node_quantdict[upper_node.output[0]]
                    node_quantdict[node.output[0]] = min_max
    return node_quantdict


def ecoding_fakequant_onnx(onnx_path, json_path):
    model = onnx.load(onnx_path)
    minmax_dict = get_all_node_minmax(model)
    print(minmax_dict)

    inputs = model.graph.input
    nodes = model.graph.node
    nodeout_dict = {}
    fakequant_list = []
    fakequant_input = []
    for node in nodes:
        if node.output:
            nodeout_dict[node.output[0]] = node
    for node in nodes:
        if node.op_type == "fused_moving_avg_obs_fake_quant":
            fakequant_list.append(node)
            if node.input:
                if node.input[0] in nodeout_dict.keys():
                    tmp_node = nodeout_dict[node.input[0]]
                    fakequant_input.append(tmp_node)

    initializers = model.graph.initializer
    initial_dict = {}
    for initial in initializers:
        initial_dict[initial.name] = initial

    # 规避一些需要加fakequant但是没有加fakequant的情况，
    # 比如fakequant+relu+conv，可以通过原理得出范围，但是不包括相同的范围情况
    needfake_node = {}
    for node in nodes:
        if node.input:
            if node.input[0] in nodeout_dict.keys():
                tmp_node = nodeout_dict[node.input[0]]
                if (node not in fakequant_input) and (node.op_type in ["Relu", "Relu6"]):
                    if tmp_node.op_type == "fused_moving_avg_obs_fake_quant":
                        print("warring, this node has not fakequant:", node.output[0])
                        tmp_min_name = tmp_node.input[3]
                        tmp_max_name = tmp_node.input[4]
                        tmp_act_min = numpy_helper.to_array(initial_dict[tmp_min_name])
                        tmp_act_max = numpy_helper.to_array(initial_dict[tmp_max_name])
                        needfake_node[node.output[0]] = (tmp_act_min, tmp_act_max)

    act_symmetric = False
    weights_symmetric = False
    act_bitwidth = 8  # 16
    act_encoding_list = []
    param_encoding_list = []
    aimet_json_file = {}
    for i in range(len(nodes)):
        node = nodes[i]
        if not node.output:
            continue
        if node.output[0] in minmax_dict.keys():
            if node.op_type == "fused_moving_avg_obs_fake_quant":
                continue
            # 找到上一个节点
            (act_min, act_max) = minmax_dict[node.output[0]]
            act_scale, act_zp, _, _, = computeQnnActQuantizeInfo(act_min, act_max, 0, 255)
            output_name = node.output[0]
            print(output_name, " :", act_min, act_max)
            act_encoding_list.append([output_name, float(act_min), \
                                      float(act_max), float(act_scale), -act_zp, act_bitwidth])

        # 没有fakequant的节点，但是可以通过上一层理论值确定范围
        if node.op_type in ["Relu", "Relu6"]:
            if node.output:
                if node.output[0] in needfake_node.keys():
                    node_name = node.output[0]
                    act_min, act_max = compute_actual_min_max(node, needfake_node)
                    print(output_name, " :", act_min, act_max)
                    act_scale, act_zp, _, _, = computeQnnActQuantizeInfo(act_min, act_max, 0, 255)
                    act_encoding_list.append([node_name, float(act_min), \
                                              float(act_max), float(act_scale), -act_zp, act_bitwidth])

    # print(act_encoding_list)
    aimet_json_file['activation_encodings'] = save_encodings_with_AIMET_format(act_encoding_list)
    aimet_json_file['param_encodings'] = save_encodings_with_AIMET_format(param_encoding_list)
    # json_path = onnx_path.split('.onnx')[0] + "_aimet.json"
    with open(json_path, 'w') as json_write:
        json.dump(aimet_json_file, json_write, indent=4)


def main():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-model', help='Input ONNX model')
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-json', help='Output ONNX model')
    args = parser.parse_args()
    input_onnx_path = args.model
    json_path = None
    if args.json:
        json_path = args.json
    else:
        json_path = input_onnx_path.split('.onnx')[0] + "_aimet.json"
    ecoding_fakequant_onnx(input_onnx_path, json_path)


if __name__ == "__main__":
    main()
