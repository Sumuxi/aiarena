import onnx
import numpy as np

# 加载 ONNX 模型
onnx_model = onnx.load("/mnt/storage/yxh/competition24/lightweight/exp_dir/onnx/v15e36/v15e36s1002k_qat_float.onnx")

initializers = onnx_model.graph.initializer
initial_dict = {}
for initial in initializers:
    initial_dict[initial.name] = initial

# 遍历计算图中的节点，寻找 `FakeQuantize` 节点
for node in onnx_model.graph.node:
    if 'fake_quant' in node.op_type.lower():
        # print(f"input: {node.name}")

        min_name = node.input[3]
        print(node.input[3].split('.activation_post_process')[0])

        # 假设 `FakeQuantize` 节点的输入包含 scale、zero_point、min_val、max_val 等参数
        # # 输入通常是常量节点，我们可以根据这些输入的名称找到对应的常量值
        # for input_name in node.input:
        #     if 'min_val' in input_name or 'max_val' in input_name:
        #         # 遍历模型的初始化器，找到与 input_name 对应的常量值
        #         for initializer in onnx_model.graph.initializer:
        #             if initializer.name == input_name:
        #                 # 打印初始值
        #                 data = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
        #                 print(f"{initializer.name}: {format(float(data), 'f')}")
