# -*- coding: utf-8 -*-
import torch
import os
from rl_framework.common.logging import logger as LOG


class LocalTorchPredictor(object):
    def __init__(self, net):
        super().__init__()
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.net = net.to(self.device, dtype=self.dtype)

    def load_model(self, model_path):
        model_filename = os.path.join(model_path, "model.pth")
        # 加载baseline模型非严格匹配所有key，加载我们训练好的模型必须严格匹配所有key
        # flag_strict = False if "baseline" in model_path else True
        flag_strict = True
        LOG.info("load model: {}", model_filename)
        LOG.info("flag_strict: {}", flag_strict)
        checkpoint = torch.load(model_filename, map_location=self.device)
        checkpoint["network_state_dict"] = {
            k: v.to(self.dtype) for k, v in checkpoint["network_state_dict"].items()
        }
        self.net.load_state_dict(checkpoint["network_state_dict"], strict=flag_strict)
        LOG.info("step: {}", checkpoint['step'])

    def inference(self, data_list):
        torch_inputs = [
            torch.from_numpy(nparr).to(self.dtype).to(self.device) for nparr in data_list
        ]
        format_inputs = self.net.format_data(torch_inputs, inference=True)
        self.net.eval()
        with torch.no_grad():
            rst_list = self.net(format_inputs, inference=True)
        return rst_list
