import hashlib
import os
import torch
import json
import subprocess
from datetime import datetime, timezone
from multiprocessing import Process, Queue
from rl_framework.common.logging import logger as LOG
import time


class ModelManager(object):
    def __init__(
        self,
        push_to_modelpool,
        remote_addrs=None,
        save_interval=1800,
    ):
        if remote_addrs is None:
            remote_addrs = ["127.0.0.1:10013:10014"]
        self.remote_addrs = remote_addrs
        self._push_to_modelpool = push_to_modelpool
        self.load_optimizer_state = False
        self.start_time = time.time()
        self.last_step = 0
        self.last_save_time = 0
        self.save_interval = save_interval

        if self._push_to_modelpool:
            self.model_queue = Queue(maxsize=100)
            pid = Process(target=self._push_to_model_pool, args=())
            pid.daemon = True
            pid.start()

    def print_variables(self, net, optimizer, step):
        LOG.info(net)

    def send_model(self, save_model_dir, send_model_dir, local_step):
        # if time.time() - self.last_save_time < self.save_interval:
        #     return
        self.last_save_time = time.time()

        os.makedirs(send_model_dir, exist_ok=True)
        tem = str(datetime.now()).replace(" ", "_").replace("-", "").replace(":", "")
        # temp_ckpt = "checkpoints_" + tem
        temp_ckpt = "checkpoints_step" + str(local_step) + "_" + tem

        current_time = datetime.now(timezone.utc)
        rfc3339_time = current_time.isoformat()

        model_info = {
            "train_time": int(time.time() - self.start_time),
            "train_step": int(self.last_step),
            "created_at": rfc3339_time,
        }
        env = os.environ.copy()
        env["OUTPUT_DIR"] = send_model_dir
        env["OUTPUT_FILENAME"] = temp_ckpt + ".zip"

        os.makedirs(send_model_dir, exist_ok=True)
        os.makedirs("/aiarena/code/actor/model/init", exist_ok=True)
        subprocess.run("sh /aiarena/scripts/build_code.sh", env=env, shell=True)

        model_info_file = os.path.join(send_model_dir, temp_ckpt + ".zip.json")
        with open(model_info_file, "w") as f:
            json.dump(model_info, f)
        return 0, ""

    def _push_to_model_pool(self):
        from rl_framework.model_pool import ModelPoolAPIs

        self.model_pool_apis = ModelPoolAPIs(self.remote_addrs)
        self.model_pool_apis.check_server_set_up()
        self.step = 0
        while True:
            model_path = self.model_queue.get()
            if not os.path.exists(model_path):
                LOG.info("[model manager] {} not exists!!".format(model_path))
            else:
                with open(model_path, "rb") as fin:
                    model = fin.read()
                local_md5 = hashlib.md5(model).hexdigest()
                self.model_pool_apis.push_model(
                    model=model,
                    hyperparam=None,
                    key="model_{}".format(self.step),
                    md5sum=local_md5,
                    save_file_name=model_path.split("/")[-1],
                )
                self.step += 1

    def restore_model_and_optimizer(self, net, optimizer, model_path):
        LOG.info(f"Loading checkpoint from {model_path} ...")
        state_dict = torch.load(model_path, map_location="cpu")
        if self.load_optimizer_state:
            if "optimizer_state_dict" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        missing_keys, unexpected_keys = net.load_state_dict(
            state_dict["network_state_dict"], strict=False
        )
        LOG.info(
            f"load ckpt success, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
        )
        return state_dict.get("step", 0)

    def save_checkpoint(self, exp_save_dir, net, optimizer, checkpoint_dir: str, step: int):
        os.makedirs(checkpoint_dir, exist_ok=True)
        step = int(step)
        self.last_step = step
        checkpoint_file = os.path.join(checkpoint_dir, "model.pth")
        torch.save(
            {
                "network_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            },
            checkpoint_file,
        )

        ckpt_file_path = os.path.join(exp_save_dir, "ckpt", f"model_step{step}.pth")
        os.makedirs(os.path.dirname(ckpt_file_path), exist_ok=True)
        torch.save(
            {
                "network_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            },
            ckpt_file_path,
        )

