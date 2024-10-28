import struct
from itertools import accumulate

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

split_size = [13, 25, 42, 42, 39, 1] * 3
split_idx = list(accumulate(split_size[:-1]))

def argmax(logits):
    logits_split = np.split(logits, split_idx, axis=1)
    logits_split = [logits_split[i + 6 * j] for i in range(6) for j in range(3)]
    
    top_1 = [np.sort(logit.argsort()[:, -1:]) for logit in logits_split]
    top_2 = [np.sort(logit.argsort()[:, -2:]) for logit in logits_split]

    top_1 = np.concatenate(top_1, axis=1)
    top_2 = np.concatenate(top_2, axis=1)
    
    top_1_split = np.split(top_1, 6, axis=1)
    top_2_split = np.split(top_2, [6, 12, 18, 24, 30], axis=1)
    top_1_split[-1] = top_1
    top_2_split[-1] = top_2

    return top_1_split, top_2_split


def same_num(arrlist1, arrlist2):
    return [np.all(np.equal(arr1, arr2), axis=1).tolist() for arr1, arr2 in zip(arrlist1, arrlist2)]


def check(output_bin_0, output_bin_1):
    loader_0 = load_bin(output_bin_0)
    loader_1 = load_bin(output_bin_1)

    file_cnt = 0
    total = 0
    top1_same = [[] for i in range(6)]
    top2_same = [[] for i in range(6)]

    for out_0, out_1 in zip(loader_0, loader_1):
        total += out_0.shape[0]
        top1_0, top2_0 = argmax(out_0)
        top1_1, top2_1 = argmax(out_1)

        for x, y in zip(top1_same, same_num(top1_0, top1_1)): x.extend(y)
        for x, y in zip(top2_same, same_num(top2_0, top2_1)): x.extend(y)

        if total >= 5000:
            total -= 5000
            file_cnt += 1
            top1_same_cnt = [sum(x[:5000]) / 50 for x in top1_same]
            top2_same_cnt = [sum(x[:5000]) / 50 for x in top2_same]
            top1_same_cnt = [top1_same_cnt[-1]] + top1_same_cnt[:-1]
            top2_same_cnt = [top2_same_cnt[-1]] + top2_same_cnt[:-1]
            top1_same = [x[5000:] for x in top1_same]
            top2_same = [x[5000:] for x in top2_same]
            print('No %d\tTop1\t%.2f%%(%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f%%)\tTop2\t%.2f%%(%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f%%)' % (file_cnt, *top1_same_cnt, *top2_same_cnt))
    print('Sample total', file_cnt * 5000 + total)
    return True


def main(_):
    FLAGS = flags.FLAGS
    ret = check(FLAGS.output_bin_0, FLAGS.output_bin_1)
    if not ret:
        print("Not equal")
        exit(-1)


if __name__ == "__main__":
    absl_app.run(main)
