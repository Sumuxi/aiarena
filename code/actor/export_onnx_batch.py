import os

import torch
from config.model_config import ModelConfig as Config
from thop import profile


def export_onnx(base_dir='',
                model=None,
                exp_name='',
                step=0):
    os.makedirs(os.path.join(base_dir, exp_name), exist_ok=True)

    ckpt_name = 'model_step' + str(step)
    out_onnx_name = base_dir + "/" + exp_name + "/" + exp_name + "s" + str(int(step / 10000)) + "w"

    # load checkpoint
    ckpt_path = "/aiarena/output/" + exp_name + "/ckpt/" + ckpt_name + ".pth"
    print(f"ckpt_path: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        return

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    # print(ckpt.keys())
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["network_state_dict"], strict=False)
    print(f"loaded ckpt, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")

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
    # print("FLOPs = " + str(flops / 1000 ** 2) + "M")
    # print("Params = " + str(params / 1000 ** 2) + "M")
    print(f"exp: {exp_name}, Params: {params / (1000 ** 2)}M, FLOPs: {flops / (1000 ** 2)}M")

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
        out_onnx_name + ".onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,  # opset_version 11 is tested to work with VCAP toolchain
    )

    # from onnxmltools.utils import float16_converter
    # from onnx import load_model, save_model
    #
    # onnx_model = load_model(out_onnx_name + ".onnx")
    # trans_model = float16_converter.convert_float_to_float16(
    #     onnx_model, keep_io_types=False
    # )
    # save_model(trans_model, out_onnx_name + ".fp16.onnx")

    from onnx import load_model, save_model
    onnx_f32 = load_model(out_onnx_name + ".onnx")
    from onnxsim import simplify
    # convert model
    onnx_sim, check = simplify(onnx_f32)
    assert check, "Simplified ONNX model could not be validated"
    save_model(onnx_sim, out_onnx_name + "_sim.onnx")


def create_model(exp_name):
    model = None
    if 'f1' in exp_name:
        from model.pytorch.final_v1 import Model
        model = Model(None)
    elif 'f2' in exp_name:
        from model.pytorch.final_v2 import Model
        model = Model(None)
    elif 'f3' in exp_name:
        from model.pytorch.final_v3 import Model
        model = Model(None)
    elif 'f4' in exp_name:
        from model.pytorch.final_v4 import Model
        model = Model(None)
    elif 'v15e20' in exp_name:
        from model.pytorch.model_v15 import Model
        model = Model(None)
    elif 'v15e30' in exp_name:
        from model.pytorch.model_v15_6 import Model
        model = Model(None)
    elif 'v4e1' in exp_name:
        from model.pytorch.model_v4 import Model
        model = Model(None)
    return model


if __name__ == "__main__":
    base_dir = '/aiarena/output/onnx'
    exps = [
        # 'v4e1',
        # 'v15e20',
        'v15e30',
        # 'f1e1',
        # 'f1e2',
        # 'f2e1',
        # 'f2e2',
        # 'f4e1',
        # 'f4e2',
        # 'f1e3',
        # 'f2e3',

        # 'f3e1',
        # 'f3e2',
        # 'f3e3',
        # 'f4e3',
    ]
    steps = [x for x in range(1400000, 1610000, 50000)]
    for exp in exps:
        model = create_model(exp)
        for step in steps:
            export_onnx(base_dir, model, exp, step)