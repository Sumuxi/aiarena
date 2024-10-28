import struct

from absl import app as absl_app
from absl import flags
import numpy as np

import data_pb2

flags.DEFINE_string("output_bin_0", "output.bin.android", "output logits file")
flags.DEFINE_string("output_bin_1", "output.bin.linux", "output logits file")


def load_bin(file_path):
    with open(file_path, "rb") as f:
        while True:
            data_len_data = f.read(4)
            if not data_len_data:
                break
            data_len = struct.unpack("<I", data_len_data)[0]

            serialized_data = f.read(data_len)
            game_output = data_pb2.GameOutput()
            game_output.ParseFromString(serialized_data)
            yield np.array([list(output.data) for output in game_output.outputs])


def argmax(logits, split_size):
    split_idx = [sum(split_size[: i + 1]) for i in range(len(split_size) - 1)]
    logits_split = np.split(logits, split_idx, axis=1)

    top_1 = [np.sort(logit.argsort()[:, -1:]) for logit in logits_split]
    top_2 = [np.sort(logit.argsort()[:, -2:]) for logit in logits_split]

    top_1 = np.concatenate(top_1, axis=1)
    top_2 = np.concatenate(top_2, axis=1)

    return top_1, top_2


def same_rate(arr1, arr2):
    num_same = np.sum(np.all(np.equal(arr1, arr2), axis=1))
    same_rate = num_same / arr1.shape[0]
    return same_rate


def check(output_bin_0, output_bin_1):
    loader_0 = load_bin(output_bin_0)
    loader_1 = load_bin(output_bin_1)

    split_size = [13, 25, 42, 42, 1 + 3 + 3 + 1 + 20 + 10 + 1, 1] * 3
    show_first=False
    rates_1 = []
    rates_2 = []
    for out_0, out_1 in zip(loader_0, loader_1):
        if not show_first:
            print(f"out_0: {out_0[0][0:5]}")
            print(f"out_1: {out_1[0][0:5]}")
            show_first = True
        top1_0, top2_0 = argmax(out_0, split_size)
        top1_1, top2_1 = argmax(out_1, split_size)

        rate_1 = same_rate(top1_0, top1_1)
        rate_2 = same_rate(top2_0, top2_1)
        #print("One Same rate: top1({:.2%}) top2({:.2%})".format(rate_1, rate_2))
        rates_1.append(rate_1)
        rates_2.append(rate_2)

    rate_1 = sum(rates_1)/len(rates_1)
    rate_2 = sum(rates_2)/len(rates_2)
    print("top1({:.2%}) top2({:.2%})".format(rate_1, rate_2))
    print(rate_1, rate_2)
    if rate_1 < 0.9 or rate_2 < 0.9:
        return False
    return True


def main(_):
    FLAGS = flags.FLAGS
    ret = check(FLAGS.output_bin_0, FLAGS.output_bin_1)
    if not ret:
        print("Not equal")
        exit(-1)


if __name__ == "__main__":
    absl_app.run(main)
