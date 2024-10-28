import os
import numpy as np

import onnxruntime as ort


class Predictor:
    def __init__(self) -> None:
        # custom config for the model
        self.lstm_unit_size = 1024

    def reset(self, model_path=None):
        # load ckpt
        if model_path:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1

            self.ort_sess = ort.InferenceSession(
                # os.path.join(model_path, "3v3.onnx"),
                model_path,
                session_options,
                providers=["CPUExecutionProvider"],
            )

        self.dtype=np.float32

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
        input_tensor = {
            "feature_hero0": features[0].astype(self.dtype),
            "feature_hero1": features[1].astype(self.dtype),
            "feature_hero2": features[2].astype(self.dtype),
            # "lstm_cell_in": self.lstm_cell.astype(self.dtype),
            # "lstm_hidden_in": self.lstm_hidden.astype(self.dtype),
        }

        # predict
        np_output_list = self.ort_sess.run(None, input_tensor)
        logits = np_output_list[:3]

        # save lstm info
        self.lstm_cell, self.lstm_hidden = [x.reshape(-1) for x in np_output_list[-2:]]

        return logits
