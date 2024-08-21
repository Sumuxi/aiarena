import numpy as np


class Agent:
    def __init__(self, predictor):
        self._predictor = predictor
        self.last_sgame_id = None

    def reset(self, model_path=None):
        self._predictor.reset(model_path)

    def predict_process(self, features, frame_state):
        """
        params:
            features: features for all heroes
            frame_state: current frame state
        return
            logits: [numpy.ndarray(1, 162), numpy.ndarray(1, 162), numpy.ndarray(1, 162)]
        """
        logits = self._predictor.inference(
            [
                np.array(feature.feature).astype(np.float32).reshape(1, -1)
                for feature in features
            ]
        )
        return logits, None
