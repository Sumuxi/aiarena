import importlib
import os

from rl_framework.learner.dataset.sample_generation.offline_rlinfo_adapter import (
    OfflineRlInfoAdapter,
)
from rl_framework.learner.framework.common.config_control import ConfigControl
from rl_framework.learner.framework.common.log_manager import LogManager

from config.Config import Config

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "test", "experiment name")  # 指定本次实验的名称，方便保存相关文件到对应的目录
flags.DEFINE_integer("local_rank", 0, "local_rank")
flags.DEFINE_string("model_file", "NetworkModel", "model file name")  # 指定模型文件名（不带后缀）
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("temperature", 4, "distill_temperature")
flags.DEFINE_float("lambda_weight", 0.5, "distill_lambda_weight")
flags.DEFINE_integer("lr_decay", 0, "whether use learning rate decay")
flags.DEFINE_float("lr_start", 0.0002, "start learning rate")
flags.DEFINE_integer("T_max", 20*10000, "T_max")
flags.DEFINE_float("lr_end", 0.00001, "end learning rate")


def load_class(module_name, class_name):
    # 动态导入模块
    module = importlib.import_module(module_name)
    # 从模块中获取类
    clazz = getattr(module, class_name)
    return clazz


def main(argv):
    project_dir = "/aiarena"
    exp_save_dir = os.path.join(project_dir, "output", FLAGS.exp_name)

    config_path = os.path.join(os.path.dirname(__file__), "config", "common.conf")
    config_manager = ConfigControl(config_path)

    config_manager.exp_save_dir = exp_save_dir
    config_manager.save_model_dir = os.path.join(exp_save_dir, "ckpt")
    config_manager.train_dir = os.path.join(exp_save_dir, "logs/learner")
    config_manager.send_model_dir = os.path.join(exp_save_dir, "backup_model")
    config_manager.batch_size = FLAGS.batch_size
    Config.DISTILL_TEMPERATURE = FLAGS.temperature
    Config.DISTILL_LAMBDA_WEIGHT = FLAGS.lambda_weight
    config_manager.lr_start = FLAGS.lr_start
    config_manager.T_max = FLAGS.T_max
    config_manager.lr_end = FLAGS.lr_end
    config_manager.use_lr_decay = FLAGS.lr_decay

    os.makedirs(config_manager.save_model_dir, exist_ok=True)
    os.makedirs(config_manager.train_dir, exist_ok=True)
    os.makedirs(config_manager.send_model_dir, exist_ok=True)

    model_file = FLAGS.model_file
    use_backend = config_manager.backend

    if use_backend == "pytorch":
        from rl_framework.learner.framework.pytorch.model_manager import ModelManager
        from rl_framework.learner.dataset.network_dataset.pytorch.network_dataset_local import (
            NetworkDataset as NetworkDatasetLocal,
        )
        from rl_framework.learner.framework.pytorch.apd_benchmark import Benchmark
        # from networkmodel.pytorch.model_v4 import NetworkModel
        # 改成动态导入
        module_name = f"networkmodel.pytorch.{model_file}"
        class_name = "NetworkModel"
        NetworkModel = load_class(module_name, class_name)

        distributed_backend = config_manager.distributed_backend
        if distributed_backend == "horovod":
            from rl_framework.learner.framework.pytorch.node_info_hvd import NodeInfo
        else:
            from rl_framework.learner.framework.pytorch.node_info_ddp import NodeInfo
    else:
        raise NotImplementedError(
            "Support backend in [pytorch], Check your training backend..."
        )

    node_info = NodeInfo(rank=FLAGS.local_rank, rank_size=8, local_rank=FLAGS.local_rank, local_size=8)
    adapter = OfflineRlInfoAdapter(Config.data_shapes)
    config_manager.push_to_modelpool = False
    dataset = NetworkDatasetLocal(
        config_manager,
        adapter,
        npz_directory="/aiarena/dataset/train_with_logits",
        file_cnt=Config.FILE_CNT,
    )
    log_manager = LogManager(loss_file_path=os.path.join(config_manager.train_dir, "loss.txt"), backend="pytorch")
    network = NetworkModel()
    network.learning_rate = config_manager.lr_start
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
