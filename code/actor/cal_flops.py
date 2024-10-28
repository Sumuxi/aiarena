import importlib

import torch


def profile(model, *input):
    total_params = sum(p.numel() for p in model.parameters())

    # 使用 torch.profiler 进行 FLOPs 计算
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,  # 记录张量形状
            with_modules=True,  # 启用模块级分析
            with_flops=True  # 启用 FLOPs 计算
    ) as prof:
        # 执行一次推理
        outputs = model(*input)

    # # 获取 FLOPs 结果，按总 FLOPs 排序并限制输出数量
    # print(prof.key_averages().table(sort_by="flops", row_limit=10))
    #
    # # 导出为 trace 文件以便后续可视化
    # prof.export_chrome_trace("trace.json")

    # 获取所有操作的 FLOPs 总和
    total_flops = sum([item.flops for item in prof.key_averages() if item.flops != 0])

    return total_flops, total_params


def create_model(model_name):
    model_file = model_name
    module_name = f"model.pytorch.{model_file}"
    class_name = "Model"
    # 动态导入模块
    module = importlib.import_module(module_name)
    # 从模块中获取类
    Net = getattr(module, class_name)
    return Net(None)


if __name__ == "__main__":
    # make dummy input
    dummy_input_5 = [
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        torch.rand(1, 1024),
        torch.rand(1, 1024),
    ]

    dummy_input_3 = [
        torch.rand(1, 4586),
        torch.rand(1, 4586),
        torch.rand(1, 4586),
    ]

    versions = [
        # 'model',
        # 'model_v1',
        # 'model_v4',
        # 'model_v30',
        # 'model_v15',
        # 'model_v15_6',
        # 'model_v16',
        # 'model_v8',
        'final_v1',
        # 'final_v2',
        # 'final_v3',
        # 'final_v4',
    ]

    results = []
    for ver in versions:
        model = create_model(ver)
        model.build_onnx = True
        if ver == 'model':
            flops, params = profile(model, dummy_input_5)
        else:
            flops, params = profile(model, dummy_input_3)
        results.append({
            'model': ver,
            'FLOPs': flops / (1000 ** 2),
            'Params': params / (1000 ** 2),
        })

    from model.pytorch.model import Model, MultiHeadAttention
    from config.model_config import ModelConfig as Config

    model = Model(Config)
    model.build_onnx = True

    x = (torch.rand(1, 19, 224), torch.rand(1, 19, 224), torch.rand(1, 19, 224))
    encoder = MultiHeadAttention(n_head=Config.ATT_HEAD_NUM, d_model=Config.TOKEN_DIM,
                                 d_k=Config.HEAD_DIM, d_v=Config.HEAD_DIM,
                                 hidden_dim=Config.TOKEN_DIM * Config.ATT_HEAD_NUM)
    flops, params = profile(encoder, *x)
    results.append({
        'model': 'encoder',
        'FLOPs': 9 * flops / (1000 ** 2),
        'Params': 3 * params / (1000 ** 2),
    })

    x = [torch.rand(16, 1, 1024), (torch.rand(1, 1, 1024), torch.rand(1, 1, 1024))]
    flops, params = profile(model.lstm, *x)
    results.append({
        'model': 'lstm',
        'FLOPs': 3 * flops / (1000 ** 2),
        'Params': params / (1000 ** 2),
    })

    x = torch.rand(1, 6, 17, 17)
    flops, params = profile(model.conv_layers, x)
    results.append({
        'model': 'conv',
        'FLOPs': 3 * flops / (1000 ** 2),
        'Params': params / (1000 ** 2),
    })

    for item in results:
        print(f"{item['model']}\t{item['FLOPs']}\t{item['Params']}")
