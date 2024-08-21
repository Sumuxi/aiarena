import os

from rl_framework.learner.dataset.sample_generation.offline_rlinfo_adapter import (
    OfflineRlInfoAdapter,
)
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.learner.framework.common.log_manager import LogManager

from config.Config import Config

from absl import flags

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
        from networkmodel.pytorch.model_v4 import NetworkModel
        distributed_backend = config_manager.distributed_backend
        if distributed_backend == "horovod":
            from rl_framework.learner.framework.pytorch.node_info_hvd import NodeInfo
        else:
            from rl_framework.learner.framework.pytorch.node_info_ddp import NodeInfo
    else:
        raise NotImplementedError(
            "Support backend in [pytorch], Check your training backend..."
        )

    node_info = NodeInfo()
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

    log_manager = LogManager(loss_file_path=os.path.join(config_manager.train_dir, "loss.txt"), backend="pytorch")
    network = NetworkModel()
    model_manager = ModelManager(config_manager.push_to_modelpool, save_interval=0)
    benchmark = Benchmark(
        network,
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
