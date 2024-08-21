import struct

from absl import app as absl_app
from absl import flags
import numpy as np

import data_pb2
from predictor import Predictor

flags.DEFINE_string("model_path", "model", "path to checkpoint directory")
flags.DEFINE_string("data_bin", "data.bin", "input data file")
flags.DEFINE_string("output_bin", "output.bin", "output logits file")


def dump(_):
    FLAGS = flags.FLAGS
    data_bin = FLAGS.data_bin
    output_bin = FLAGS.output_bin

    fin = open(data_bin, "rb")
    fout = open(output_bin, "wb")

    predictor = Predictor()

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
        predictor.reset(FLAGS.model_path)

        for feature_pb in game_data.features:
            features = np.array(feature_pb.data).reshape(3, 4586).astype(np.float32)
            features = [feature.reshape(1, -1) for feature in features]

            logits = predictor.inference(features)

            output_pb = data_pb2.Array()
            output_pb.data.extend(sum([x.tolist()[0] for x in logits], []))

            output_data.outputs.append(output_pb)

        serialized_output = output_data.SerializeToString()
        fout.write(struct.pack("<I", len(serialized_output)))
        fout.write(serialized_output)

    fin.close()
    fout.close()


if __name__ == "__main__":
    absl_app.run(dump)
