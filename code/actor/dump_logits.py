from collections import namedtuple
import os
import struct

from absl import app as absl_app
from absl import flags
import numpy as np

import data_pb2

from agent.agent import Agent
from config.model_config import ModelConfig

from rl_framework.common.logging import setup_logger

if ModelConfig.backend == "tensorflow":
    from model.tensorflow.model import Model
elif ModelConfig.backend == "pytorch":
    from model.pytorch.model import Model
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
else:
    raise NotImplementedError("check ModelConfig, backend=['tensorflow', 'pytorch']")

work_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(work_dir, "model", "init")

flags.DEFINE_string("model_path", DEFAULT_MODEL_PATH, "path to checkpoint")
flags.DEFINE_string("data_bin", "data.bin", "input data file")
flags.DEFINE_string("output_bin", "output.bin", "output logits file")


FakeFeature = namedtuple("FakeFeature", ["feature"])


def dump(_):
    setup_logger()
    FLAGS = flags.FLAGS
    data_bin = FLAGS.data_bin
    output_bin = FLAGS.output_bin

    fin = open(data_bin, "rb")
    fout = open(output_bin, "wb")

    agent = Agent(
        Model(ModelConfig),
        None,
        backend=ModelConfig.backend,
    )

    # load data from file
    while True:
        output_data = data_pb2.GameOutput()
        # load on batch
        data_len_data = fin.read(4)
        if not data_len_data:
            break
        data_len = struct.unpack("<I", data_len_data)[0]

        serialized_feature = fin.read(data_len)
        game_data = data_pb2.GameData()
        game_data.ParseFromString(serialized_feature)

        # reset and inference
        agent.reset(model_path=FLAGS.model_path)
        for feature_pb in game_data.features:
            features = np.array(feature_pb.data).reshape(3, 4586)
            fake_features = [FakeFeature(features[i]) for i in range(3)]
            logits, _ = agent._predict_process_torch(fake_features, None, None)

            output_pb = data_pb2.Array()
            for i in range(3):
                output_pb.data.extend(logits[i].tolist()[0])
            output_data.outputs.append(output_pb)

        serialized_output = output_data.SerializeToString()
        fout.write(struct.pack("<I", len(serialized_output)))
        fout.write(serialized_output)

    fin.close()
    fout.close()


if __name__ == "__main__":
    absl_app.run(dump)
