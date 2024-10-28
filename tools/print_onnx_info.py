import re

import onnx

# 加载 ONNX 模型
model = onnx.load("/mnt/storage/yxh/competition24/lightweight/exp_dir/onnx/v15e35/v15e35s101w_1_qat_float.onnx")
model_float = onnx.load("/mnt/storage/yxh/competition24/lightweight/exp_dir/onnx/v15e35/v15e35s101w_1_float.onnx")

# 获取模型的图
graph = model.graph

edges = {}

# 遍历模型中的所有节点
for node in graph.node:
    otype = node.op_type
    if "fake_quant" in otype:
        # print(node.input[0])
        # 打印节点的输入
        # print("Inputs:")
        for i, input_name in enumerate(node.input):
            fname = input_name.lower()
            if "activation_post_process" in fname or 'constant' in fname:
                continue

            old_input_name = input_name
            new_name = re.sub(r'[/.]', '_', input_name)
            edges[old_input_name] = new_name

            # 1. 修改目标节点的输入名称
            node.input[i] = new_name

            # 2. 查找源节点，并修改它的输出名称
            for src_node in graph.node:
                for i, output in enumerate(src_node.output):
                    if output == old_input_name:
                        src_node.output[i] = new_name

onnx.save(model, "/mnt/storage/yxh/competition24/lightweight/exp_dir/onnx/v15e35/v15e35s101w_1_qat_float_rename.onnx")


for k,v in edges.items():
    print(k, ":", v)
print(len(edges))



# 获取模型的图
graph = model_float.graph
# 遍历模型中的所有节点
for node in graph.node:
    for i, input_name in enumerate(node.input):
        if input_name in edges.keys():
            # 1. 修改目标节点的输入名称
            new_name = edges[input_name]
            node.input[i] = new_name

            # 2. 查找源节点，并修改它的输出名称
            for src_node in graph.node:
                for i, output in enumerate(src_node.output):
                    if output == input_name:
                        src_node.output[i] = new_name
onnx.save(model_float, "/mnt/storage/yxh/competition24/lightweight/exp_dir/onnx/v15e35/v15e35s101w_1_float_rename.onnx")

#
# # import onnx
# #
# # # 加载 ONNX 模型
# # model = onnx.load("your_model.onnx")
# #
# # # 获取模型的图
# # graph = model.graph
#
# # 遍历所有节点，找到目标节点
# for node in graph.node:
#     if node.name == "node_to_modify":  # 替换为要修改的目标节点名称
#         old_input_name = node.input[0]  # 假设我们要修改第一个输入
#         new_input_name = "new_input_name_1"  # 你想要的新输入名称
#
#         # 1. 修改目标节点的输入名称
#         node.input[0] = new_input_name
#         print(f"Modified input for node {node.name}: {node.input}")
#
#         # 2. 查找源节点，并修改它的输出名称
#         for src_node in graph.node:
#             for i, output in enumerate(src_node.output):
#                 if output == old_input_name:
#                     src_node.output[i] = new_input_name
#                     print(f"Modified output for source node {src_node.name}: {src_node.output}")
#
#         # # 3. 修改边的名称（即连接这两个节点的名称）
#         # for value_info in graph.value_info:
#         #     if value_info.name == old_input_name:
#         #         value_info.name = new_input_name
#         #         print(f"Modified edge name: {value_info.name}")
#
# # 保存修改后的模型
# onnx.save(model, "modified_model.onnx")
