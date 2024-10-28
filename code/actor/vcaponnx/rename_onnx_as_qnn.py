'''
Author: duanzhijie
Date: 2022-03-25 16:28:34
LastEditTime: 2022-07-12 15:04:09
FilePath: \pytorchquant\vcap_pytorch_quant\QNN\modify_onnx_as_qnn.py
'''
import numpy as np
import onnx
from collections import OrderedDict
from onnx import helper
from onnx import numpy_helper
import argparse

def modify_onnx_for_qnn(onnx_path, output_path):
  input_model = onnx.load(onnx_path)
  #save graph output
  graph_outputlist = []
  graph_connect_out = {}
  swap_graphoutput_dict = {}
  nodes_dict = {}
  for out in input_model.graph.output:
    graph_outputlist.append(out.name)
  for node in input_model.graph.node:
    if node.output:
      nodes_dict[node.output[0]] = node
    if node.output[0] in graph_outputlist:
      graph_connect_out[node.output[0]] = node


  node_len = len(input_model.graph.node)
  swap_dict = {} #The dict for saving layer information
  type_dict = {} #The dict for saving op_type and it's amount
  for i in range(node_len):
    node = input_model.graph.node[i]
    optype = node.op_type
   
    #save output name of the op with quantization parameters but need jumpquantization node
    if optype == 'QuantizeLinear' or optype == 'DequantizeLinear' or optype == 'Constant' \
         or optype == 'Identity' or optype == 'Unsqueeze':
      continue

    if optype not in type_dict:
      type_dict[optype] = 0
    for ind, output in enumerate(node.output):
      swap_dict[output] = optype + '_' + str(type_dict[optype] + ind)
    type_dict[optype] += len(node.output)
    #save output name of weight and bias, you could add other OP_Type which might be with weight and bias
    if (optype in ['Conv', 'Gemm']):
      swap_dict[node.input[1]] = optype + '_' + str(type_dict[optype]) + '.weight'
      if len(node.input) > 2:
        swap_dict[node.input[2]] = optype + '_' + str(type_dict[optype]) + '.bias'

  #add for graph output rename
  for node in input_model.graph.node:
    for output_featuremap in node.output:
      if output_featuremap in graph_connect_out.keys():
        if output_featuremap in swap_dict.keys():
          swap_graphoutput_dict[output_featuremap] = swap_dict[output_featuremap]
  #print(swap_graphoutput_dict)

  #Modify the op output name
  for i in range(node_len):
    node = input_model.graph.node[i]
    #print("**node i=", node)
    input_len = len(node.input)
    output_len = len(node.output)
    for j in range(input_len):
      if node.input[j] in swap_dict.keys():
        #print("%s to %s\n"%(node.input[j], swap_dict[node.input[j]]))
        node.input[j] = swap_dict[node.input[j]]
    for j in range(output_len):
      if node.output[j] in swap_dict.keys():
        #print("%s to %s\n"%(node.output[j], swap_dict[node.output[j]]))
        node.output[j] = swap_dict[node.output[j]]

  #modify the parameter output name, weight/bias
  for m in input_model.graph.initializer:
    if m.name in swap_dict.keys():
      #print("%s to %s in params\n"%(m.name, swap_dict[m.name]))
      for n in input_model.graph.input:
        if m.name == n.name:
          n.name = swap_dict[m.name]
      m.name = swap_dict[m.name]
    else:
      #print("skit rename unsupport node name %s: " % m.name)
      pass

  #add for graph output rename
  for out in input_model.graph.output:
    if out.name in swap_graphoutput_dict.keys():
      new_name = swap_graphoutput_dict[out.name]
      out.name = new_name

  print(input_model.graph.output)
  onnx.save(input_model, output_path)

def main():
  parser = argparse.ArgumentParser()
  required = parser.add_argument_group('required arguments')
  required.add_argument('--input_model', help='Input ONNX model')
  optional = parser.add_argument_group('optional arguments')
  optional.add_argument('--output_model', help='Output ONNX model')
  args = parser.parse_args()
  input_onnx_path = args.input_model
  output_onnx_path = None
  if args.output_model :
    output_onnx_path = args.output_model
  else:
    output_onnx_path = input_onnx_path.split('.onnx')[0] + "_rename.onnx"
  print("input model:", input_onnx_path)
  print("output model:", output_onnx_path)
  modify_onnx_for_qnn(input_onnx_path, output_onnx_path)


if __name__ == "__main__":
  main()