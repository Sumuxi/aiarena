'''
Author: duanzhijie
Date: 2022-03-21 14:23:54
LastEditTime: 2022-11-21 21:07:13
FilePath: pytorch_tools.py
'''

import torch
import torch.nn as nn
import os
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import QConfig, FakeQuantize, MovingAverageMinMaxObserver, FusedMovingAvgObsFakeQuantize
from torch.onnx import ExportTypes, OperatorExportTypes


def fakequantnode():
    return FusedMovingAvgObsFakeQuantize(observer=MovingAverageMinMaxObserver,
                                         quant_min=0,
                                         quant_max=255,
                                         dtype=torch.quint8,
                                         reduce_range=False,
                                         qscheme=torch.per_tensor_affine)


def fakequantnodeenpu():
    return FusedMovingAvgObsFakeQuantize(observer=MovingAverageMinMaxObserver,
                                         quant_min=-127,
                                         quant_max=127,
                                         dtype=torch.qint8,
                                         reduce_range=False,
                                         qscheme=torch.per_tensor_symmetric)


'''
@function: 量化相关配置函数
@param model: model graph module
@param platform_mode: pytorch 量化平台模式, 'fbgemm'(用于服务器端推理) ; 'qnnpack' (用于移动端推理); 'tfqat'(tensorflow qat)
@param scene_mode: 场景模式，0(训练模式), 1(导出带ACT量化信息浮点模型), 2(导出无量化信息浮点模型)
'''


def qat_qconfig_and_prepare(model, platform_mode, scene_mode=0):
    # add quant config
    if scene_mode == 0:
        # pytorch quant qat training with qnnpack or fbgemm
        if platform_mode == "tfqat":
            print("training model with tfqat mode!")
            fused_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                          quant_min=0,
                                                                          quant_max=255,
                                                                          dtype=torch.quint8,
                                                                          reduce_range=False,
                                                                          qscheme=torch.per_tensor_affine)
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=False),
                              weight=fused_wt_fake_quant)
            model.qconfig = qconfig
        elif platform_mode == "enpu":
            print("training model with enpu mode!")
            fused_wt_fake_quant_range_neg_127_to_127 = FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=-127,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                eps=2 ** -12)
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=-127,
                                                                                 quant_max=127,
                                                                                 dtype=torch.qint8,
                                                                                 qscheme=torch.per_tensor_symmetric,
                                                                                 eps=2 ** -12),
                              weight=fused_wt_fake_quant_range_neg_127_to_127)
        else:
            model.qconfig = torch.quantization.get_default_qat_qconfig(platform_mode)

    if scene_mode == 1:
        if platform_mode == "qnnpack" or platform_mode == "tfqat":
            print("export model with mode:", platform_mode)
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=False),
                              weight=torch.nn.Identity)
            model.qconfig = qconfig
        if platform_mode == "fbgemm":
            print("export model with mode:", platform_mode)
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=True),
                              weight=torch.nn.Identity)
            model.qconfig = qconfig
        if platform_mode == "enpu":
            print("export model with mode:", platform_mode)
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=-127,
                                                                                 quant_max=127,
                                                                                 dtype=torch.qint8,
                                                                                 qscheme=torch.per_tensor_symmetric,
                                                                                 eps=2 ** -12),
                              weight=torch.nn.Identity)
            model.qconfig = qconfig
    if scene_mode == 2:
        qconfig = QConfig(activation=torch.nn.Identity, weight=torch.nn.Identity)
        model.qconfig = qconfig

    # prepare,插入量化节点
    torch.quantization.prepare_qat(model, inplace=True)


'''
@function:生成不带量化信息的浮点ONNX模型
@param graph: nn.Module, 模型graph 结构
@param dict_path: 训练完成时保存的模型dict
@param onnx_path: 生成onnx模型路径
@param input_shapes: model input shape list, for example [[1, 1, 28, 28], [1, 3, 28, 28]]
@param input_names:  model input name list, for example [ "input0",  "input1"]
@param output_names: model output name list, for example [ "output0",  "output1"]
'''


def export_onnx_floatmodel_without_fakequant(graph, dict_path, onnx_path,
                                             input_shapes, input_names, output_names):
    print("input shapes:", input_shapes)
    print("input names:", input_names)
    print("output names:", output_names)
    from collections import OrderedDict
    net_dict = torch.load(dict_path, map_location=torch.device('cpu'))['network_state_dict']
    out_dict = OrderedDict()
    for key, value in net_dict.items():
        if "weight_fake_quant" in key:
            continue
        if "activation_post_process" in key:
            continue
        if "activation_fakequant" in key:
            continue

        out_dict[key] = value

    missing_keys, unexpected_keys = graph.load_state_dict(out_dict, strict=False)
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)
    graph.build_onnx = True
    graph.eval()

    dummy_input = [
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        # torch.rand(Config.LSTM_UNIT_SIZE),
        # torch.rand(Config.LSTM_UNIT_SIZE),
    ]

    torch.onnx.export(graph,
                      dummy_input,
                      f=onnx_path,
                      verbose=False,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=input_names,
                      output_names=output_names,
                      operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
                      export_params=True,
                      opset_version=11
                      )

    # if len(input_names) == 1:
    #   #conveter to onnx model
    #   dummy_input = None
    #   input_shape = input_shapes[0]
    #   if len(input_shape) == 4:
    #     dummy_input = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    #   if len(input_shape) == 3:
    #     dummy_input = torch.randn(input_shape[0], input_shape[1], input_shape[2])
    #   if len(input_shape) == 2:
    #     dummy_input = torch.randn(input_shape[0], input_shape[1])
    #
    #   input_names = input_names
    #   output_names = output_names
    #   torch.onnx.export(graph,
    #                     dummy_input,
    #                     onnx_path,
    #                     verbose=False,
    #                     do_constant_folding=True,
    #                     input_names=input_names,
    #                     output_names=output_names,
    #                     operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    #                     export_params=True,
    #                     )
    # elif len(input_names) == 2:
    #   #conveter to onnx model
    #   dummy_input0 = None
    #   input_shape0 = input_shapes[0]
    #   if len(input_shape0) == 4:
    #     dummy_input0 = torch.randn(input_shape0[0], input_shape0[1], input_shape0[2], input_shape0[3])
    #   if len(input_shape0) == 3:
    #     dummy_input0 = torch.randn(input_shape0[0], input_shape0[1], input_shape0[2])
    #   if len(input_shape0) == 2:
    #     dummy_input0 = torch.randn(input_shape0[0], input_shape0[1])
    #
    #   dummy_input1 = None
    #   input_shape1 = input_shapes[1]
    #   if len(input_shape1) == 4:
    #     dummy_input1 = torch.randn(input_shape1[0], input_shape1[1], input_shape1[2], input_shape1[3])
    #   if len(input_shape1) == 3:
    #     dummy_input1 = torch.randn(input_shape1[0], input_shape1[1], input_shape1[2])
    #   if len(input_shape1) == 2:
    #     dummy_input1 = torch.randn(input_shape1[0], input_shape1[1])
    #
    #   input_names = input_names
    #   output_names = output_names
    #   torch.onnx.export(graph,
    #                     args=(dummy_input0, dummy_input1),
    #                     f=onnx_path,
    #                     verbose=False,
    #                     do_constant_folding=True,	# 是否执行常量折叠优化
    #                     input_names=input_names,
    #                     output_names=output_names,
    #                     operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    #                     export_params=True,
    #                     )
    # else:
    #   print("do not support this conveter with multi input: ", len(input_names))


'''
@function:生成带activation量化信息的浮点ONNX模型
@param graph: nn.Module, 模型graph 结构
@param dict_path: 训练完成时保存的模型dict
@param onnx_path: 生成onnx模型路径
@param input_shapes: model input shape list, for example [[1, 1, 28, 28], [1, 3, 28, 28]]
@param input_names:  model input name list, for example [ "input0",  "input1"]
@param output_names: model output name list, for example [ "output0",  "output1"]
'''


def export_onnx_floatmodel_with_actfakequant(graph, dict_path, onnx_path,
                                             input_shapes, input_names, output_names):
    print("input shapes:", input_shapes)
    print("input names:", input_names)
    print("output names:", output_names)
    from collections import OrderedDict
    net_dict = torch.load(dict_path, map_location=torch.device('cpu'))['network_state_dict']
    out_dict = OrderedDict()
    for key, value in net_dict.items():
        if "weight_fake_quant" in key:
            continue
        out_dict[key] = value

    missing_keys, unexpected_keys = graph.load_state_dict(out_dict, strict=False)
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)
    graph.build_onnx = True
    graph.eval()

    dummy_input = [
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        # torch.rand(Config.LSTM_UNIT_SIZE),
        # torch.rand(Config.LSTM_UNIT_SIZE),
    ]

    torch.onnx.export(graph,
                    dummy_input,
                    onnx_path,
                    verbose=False,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
                    export_params=True,
                    opset_version=11
                    )

    # if len(input_names) == 1:
    #   #conveter to onnx model
    #   dummy_input = None
    #   input_shape = input_shapes[0]
    #   if len(input_shape) == 4:
    #     dummy_input = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    #   if len(input_shape) == 3:
    #     dummy_input = torch.randn(input_shape[0], input_shape[1], input_shape[2])
    #   if len(input_shape) == 2:
    #     dummy_input = torch.randn(input_shape[0], input_shape[1])
    #
    #   input_names = input_names
    #   output_names = output_names
    #   torch.onnx.export(graph,
    #                     dummy_input,
    #                     onnx_path,
    #                     verbose=False,
    #                     do_constant_folding=True,
    #                     input_names=input_names,
    #                     output_names=output_names,
    #                     operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    #                     export_params=True,
    #                     opset_version=13
    #                     )
    # elif len(input_names) == 2:
    #   #conveter to onnx model
    #   dummy_input0 = None
    #   input_shape0 = input_shapes[0]
    #   if len(input_shape0) == 4:
    #     dummy_input0 = torch.randn(input_shape0[0], input_shape0[1], input_shape0[2], input_shape0[3])
    #   if len(input_shape0) == 3:
    #     dummy_input0 = torch.randn(input_shape0[0], input_shape0[1], input_shape0[2])
    #   if len(input_shape0) == 2:
    #     dummy_input0 = torch.randn(input_shape0[0], input_shape0[1])
    #
    #   dummy_input1 = None
    #   input_shape1 = input_shapes[1]
    #   if len(input_shape1) == 4:
    #     dummy_input1 = torch.randn(input_shape1[0], input_shape1[1], input_shape1[2], input_shape1[3])
    #   if len(input_shape1) == 3:
    #     dummy_input1 = torch.randn(input_shape1[0], input_shape1[1], input_shape1[2])
    #   if len(input_shape1) == 2:
    #     dummy_input1 = torch.randn(input_shape1[0], input_shape1[1])
    #
    #   input_names = input_names
    #   output_names = output_names
    #   torch.onnx.export(graph,
    #                     args=(dummy_input0, dummy_input1),
    #                     f=onnx_path,
    #                     verbose=False,
    #                     do_constant_folding=True,
    #                     input_names=input_names,
    #                     output_names=output_names,
    #                     operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    #                     export_params=True,
    #                     )
    # else:
    #   print("do not support this conveter with multi input: ", len(input_names))
