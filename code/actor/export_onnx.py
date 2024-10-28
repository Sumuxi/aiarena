import os

import torch
from model.pytorch.model_v4 import Model
from config.model_config import ModelConfig as Config
from thop import profile

if __name__ == "__main__":
    # instantiate pytorch model
    model = Model(Config)

    base_dir = '/aiarena/output/onnx'
    exp_name = 'f1e1'
    step = 800000
    ckpt_name = 'model_step' + str(step)

    os.makedirs(os.path.join(base_dir, exp_name), exist_ok=True)

    out_onnx_name = base_dir + "/" + exp_name + "/" + exp_name + "s" + str(int(step/10000)) + "w"

    # load checkpoint
    ckpt = torch.load("/aiarena/output/" + exp_name + "/ckpt/" + ckpt_name + ".pth",
                      map_location=torch.device("cpu"))
    print(ckpt.keys())
    model.load_state_dict(ckpt["network_state_dict"], strict=False)

    # make dummy input
    model.build_onnx = True
    dummy_input = [
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        # torch.rand(Config.LSTM_UNIT_SIZE),
        # torch.rand(Config.LSTM_UNIT_SIZE),
    ]
    x = dummy_input

    # measure FLOPs and parameters
    flops, params = profile(model, inputs=(x,))
    print("FLOPs = " + str(flops / 1000 ** 2) + "M")
    print("Params = " + str(params / 1000**2) + "M")

    # dry-run before export
    _ = model(x)

    # for name, module in model.named_modules():
    #    print(name)

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

    # export onnx model
    torch.onnx.export(
        model,
        x,
        out_onnx_name+".onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,  # opset_version 11 is tested to work with VCAP toolchain
    )

    from onnxmltools.utils import float16_converter
    from onnx import load_model, save_model

    onnx_model = load_model(out_onnx_name+".onnx")
    trans_model = float16_converter.convert_float_to_float16(
        onnx_model, keep_io_types=False
    )
    save_model(trans_model, out_onnx_name+".fp16.onnx")
