import os
import numpy as np

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from model.model import Model


class Predictor:
    def __init__(self) -> None:
        # custom config for the model
        self.lstm_unit_size = 2048

    def reset(self, model_path=None):
        # load ckpt
        if model_path:
            self.device = torch.device("cpu")
            self.model = Model().to(self.device)

            checkpoint = torch.load(
                os.path.join(model_path, "model.pth"), map_location=self.device
            )
            self.model.load_state_dict(checkpoint["network_state_dict"], strict=False)

        self.dtype = np.float32

        # reset lstm info
        self.lstm_hidden = np.zeros([self.lstm_unit_size]).astype(self.dtype)
        self.lstm_cell = np.zeros([self.lstm_unit_size]).astype(self.dtype)

    def inference(self, features):
        """
        Params:
            features: [np.ndarray(1, 4586), np.ndarray(1, 4586), np.ndarray(1, 4586)]

        Return:
            logits: [numpy.ndarray(1, 162), numpy.ndarray(1, 162), numpy.ndarray(1, 162)]
        """

        # prepare data
        # input_tensor = {
        #    "feature_hero0": features[0].astype(self.dtype),
        #    "feature_hero1": features[1].astype(self.dtype),
        #    "feature_hero2": features[2].astype(self.dtype),
        #    "lstm_cell_in": self.lstm_cell.astype(self.dtype),
        #    "lstm_hidden_in": self.lstm_hidden.astype(self.dtype),
        # }

        torch_inputs = [
            features[0].reshape(-1),
            features[1].reshape(-1),
            features[2].reshape(-1),
            self.lstm_cell,
            self.lstm_hidden,
        ]
        torch_inputs = [
            torch.from_numpy(nparr).to(torch.float32) for nparr in torch_inputs
        ]

        format_inputs = self.model.format_data(torch_inputs, inference=True)
        self.model.eval()

        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output_list = []
        for output in output_list:
            np_output_list.append(output.numpy())

        # predict
        logits = np_output_list[:3]

        # save lstm info
        self.lstm_cell, self.lstm_hidden = [x.reshape(-1) for x in np_output_list[-2:]]

        return logits
