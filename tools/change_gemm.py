import re

import onnx
import numpy as np
from onnx import helper, numpy_helper

# 加载 ONNX 模型
onnx_model = onnx.load("/media/storage/yxh/competition24/lightweight/exp_dir/onnx/v15e36/v15e36s1002k_float_rename.onnx")

# 初始化新的节点列表
new_nodes = []

# 获取初始化器（包含权重的值）
initializers_map = {init.name: init for init in onnx_model.graph.initializer}

# 遍历模型的节点，找到 Gemm 节点并替换为 MatMul + Add
for node in onnx_model.graph.node:
    if node.op_type == "Gemm":
        print(f"Found Gemm node: {node.name}")

        # 提取 Gemm 的输入：A, B, C
        input_A = node.input[0]  # 输入 A
        input_B = node.input[1]  # 权重 B
        input_C = node.input[2]  # 偏置 C

        # 找到 B 对应的 Initializer (权重值)
        if input_B in initializers_map:
            # 获取权重 B 的值
            weight_B = numpy_helper.to_array(initializers_map[input_B])

            # 对 B 进行转置 (K, N) -> (N, K)
            weight_B_transposed = np.transpose(weight_B, (1, 0))

            # 替换转置后的权重到 Initializer
            transposed_weight_initializer = numpy_helper.from_array(weight_B_transposed, input_B)
            initializers_map[input_B].CopyFrom(transposed_weight_initializer)
        else:
            raise ValueError(f"Initializer for {input_B} not found.")

        # 创建 MatMul 节点，直接使用转置后的 B
        # matmul_output = node.output[0] + "_matmul"  # MatMul 的中间输出名
        matmul_output = re.sub(r'Gemm', 'MatMul', node.output[0])
        matmul_node = helper.make_node(
            "MatMul",
            [input_A, input_B],  # 注意 B 已经在 Initializer 中转置
            [matmul_output],
            name=re.sub(r'Gemm', 'MatMul', node.name)
        )

        # 创建 Add 节点，用 MatMul 的输出加上偏置 C
        add_node = helper.make_node(
            "Add",
            [matmul_output, input_C],
            [node.output[0]],  # 最终输出名
            name=re.sub(r'Gemm', 'Add', node.name)
        )

        # 将 MatMul 和 Add 节点添加到新的节点列表
        new_nodes.append(matmul_node)
        new_nodes.append(add_node)
    else:
        # 如果不是 Gemm 节点，保留原节点
        new_nodes.append(node)

# 清空原有节点，替换为新的节点列表
onnx_model.graph.ClearField("node")
onnx_model.graph.node.extend(new_nodes)

# 替换模型中的 Initializer
onnx_model.graph.ClearField("initializer")
onnx_model.graph.initializer.extend(initializers_map.values())

# 保存修改后的 ONNX 模型
onnx.save(onnx_model, "/media/storage/yxh/competition24/lightweight/exp_dir/onnx/v15e36/model_without_gemm.onnx")

print("Gemm nodes have been successfully replaced with MatMul + Add, and B has been transposed.")
