import os

from rl_framework.learner.dataset.sample_generation.offline_rlinfo_adapter import (
    OfflineRlInfoAdapter,
)
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.learner.framework.common.log_manager import LogManager

from config.Config import Config

from absl import flags

import torch
import torch.ao.quantization as quantization
from torch.ao.quantization import QConfig, QConfigMapping, FakeQuantize, FusedMovingAvgObsFakeQuantize, MovingAverageMinMaxObserver

FLAGS = flags.FLAGS
flags.DEFINE_boolean("single_test", 0, "test_mode")


def main(argv):
    config_path = os.path.join(os.path.dirname(__file__), "config", "common.conf")

    config_manager = ConfigControl(config_path)
    os.makedirs(config_manager.save_model_dir, exist_ok=True)
    os.makedirs(config_manager.train_dir, exist_ok=True)
    os.makedirs(config_manager.send_model_dir, exist_ok=True)

    use_backend = config_manager.backend

    if use_backend == "pytorch":
        from rl_framework.learner.framework.pytorch.model_manager import ModelManager
        from rl_framework.learner.dataset.network_dataset.pytorch.network_dataset_local import (
            NetworkDataset as NetworkDatasetLocal,
        )
        from rl_framework.learner.dataset.network_dataset.pytorch.network_dataset_random import (
            NetworkDataset as NetworkDatasetRandom,
        )
        from rl_framework.learner.framework.pytorch.apd_benchmark import Benchmark
        from networkmodel.pytorch.model_lyl_4_notarget_qat import NetworkModel
        distributed_backend = config_manager.distributed_backend
        if distributed_backend == "horovod":
            from rl_framework.learner.framework.pytorch.node_info_hvd import NodeInfo
        else:
            from rl_framework.learner.framework.pytorch.node_info_ddp import NodeInfo
    else:
        raise NotImplementedError(
            "Support backend in [pytorch], Check your training backend..."
        )

    # node_info = NodeInfo()
    node_info = NodeInfo(rank=0, rank_size=8, local_rank=0, local_size=8)
    adapter = OfflineRlInfoAdapter(Config.data_shapes)
    config_manager.push_to_modelpool = False
    if FLAGS.single_test:
        dataset = NetworkDatasetRandom(
            config_manager,
            adapter,
        )
    else:
        dataset = NetworkDatasetLocal(
            config_manager,
            adapter,
            npz_directory="/dataset/train",
            file_cnt=Config.FILE_CNT,
        )

    log_manager = LogManager(loss_file_path=config_manager.train_dir + "/loss.txt", backend="pytorch")
    # create a fp32 model instance
    network = NetworkModel()
    # model must be set to eval for fusion to work
    network.eval()
    network.fuse_model()

    # 将自定义的量化配置应用于模型
    # network.qconfig = my_qconfig_dict

    # network.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')

    # network.qconfig = QConfig(
    #     activation=FusedMovingAvgObsFakeQuantize.with_args(
    #         observer=MovingAverageMinMaxObserver,
    #         quant_min=0,
    #         quant_max=255,
    #         dtype=torch.quint8,
    #         qscheme=torch.per_tensor_affine,
    #         reduce_range=False),
    #     weight=FusedMovingAvgObsFakeQuantize.with_args(
    #         observer=MovingAverageMinMaxObserver,
    #         quant_min=0,
    #         quant_max=255,
    #         dtype=torch.quint8,
    #         qscheme=torch.per_tensor_affine,
    #         reduce_range=False)
    # )

    network.qconfig = QConfig(
        activation=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False),
        weight=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine)
    )

    # 准备模型，进行量化感知训练（QAT）
    network_fp32_prepared = torch.ao.quantization.prepare_qat(network.train())


    model_manager = ModelManager(config_manager.push_to_modelpool)
    benchmark = Benchmark(
        network_fp32_prepared,
        dataset,
        log_manager,
        model_manager,
        config_manager,
        node_info,
    )
    benchmark.run()


if __name__ == "__main__":
    from absl import app as absl_app

    absl_app.run(main)
