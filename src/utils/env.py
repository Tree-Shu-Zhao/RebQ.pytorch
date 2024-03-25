import json
import os
import random
import shutil
import time

import nni
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.tensorboard import SummaryWriter

from .hparams import merge_cfg


def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(random.randint(12334,33677))
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

def is_distributed(gpu_info):
    return True if isinstance(gpu_info, list) and len(gpu_info) > 1 else False

def init_env(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Logger
    logger.add(os.path.join(cfg.OUTPUT_DIR, "main.log"))

    # Tensorboard
    tensorboard_logger = SummaryWriter(
        log_dir=os.path.join(cfg.OUTPUT_DIR, "tensorboard"),
        comment=cfg.EXP_NOTE
    )

    # Log experiment note. If it is not be provided, we do not run the experiment.
    assert cfg.EXP_NOTE is not None, "You must provide a experiment note to run the experiment!"
    logger.info(f"Experiment Note: {cfg.EXP_NOTE}")

    # Checkpoint Dir
    checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint Dir: {checkpoint_dir}.")

    # Save hyper-parameters
    parameters_path = os.path.join(cfg.OUTPUT_DIR, "hyper-parameters.json")
    hyper_parameters = get_hyper_parameters(cfg)
    with open(parameters_path, "w") as f:
        json.dump(hyper_parameters, f)
    msg = ""
    for k, v in hyper_parameters.items():
        msg += f"{k}: {v}\n"
    logger.info(f"Save hyper-parameters:\n{msg}to {parameters_path}.")

    # Code Dir
    code_dir = os.path.join(cfg.OUTPUT_DIR, "codes")
    os.makedirs(code_dir, exist_ok=True)
    logger.info(f"Code Dir: {code_dir}")
    save_codes(code_dir)
    logger.info("Codes backup completed.")

    # Enable multiple GPU training
    if is_distributed(cfg.train.GPU):
        logger.info(f"Enable multiple GPU training, GPU: {cfg.train.GPU}.")
        cfg.train.DISTRIBUTED = True

    # Set seed
    random.seed(cfg.SEED)                                      
    torch.manual_seed(cfg.SEED)                                
    torch.cuda.manual_seed(cfg.SEED)                           
    np.random.seed(cfg.SEED)
    logger.info(f"Set seed to {cfg.SEED}.")

    env_info = {
        "is_distributed": is_distributed(cfg.train.GPU),
    }

    # Configure NNI hparams searching
    if cfg.train.HPARAMS_SEARCHING:
        logger.info(f"Enable hparam searching.")
        hparams = nni.get_next_parameter()
        cfg = merge_cfg(cfg, hparams)
        env_info.update({
            "merged_cfg": cfg,
        })
    
    # Set numpy print precision
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)

    return env_info, tensorboard_logger

def save_codes(code_dir):
    VALID_FILE_TYPES = (".py", ".ipynb", ".sh", ".yaml")
    VALID_DIRS = ("configs", "src", "tools")

    logger.info(f"Prepare to backup codes to {code_dir}.\nValid file types: {VALID_FILE_TYPES}.\nValid dirs: {VALID_DIRS}.")

    for valid_dir in VALID_DIRS:
        for root, dirs, files in os.walk(valid_dir):
            for file in files:
                file_type = os.path.splitext(file)[1]
                if file_type in VALID_FILE_TYPES:
                    source_path = os.path.join(root, file)
                    backup_path = os.path.join(code_dir, source_path)
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    shutil.copy2(source_path, backup_path)


def get_hyper_parameters(cfg):
    hyper_parameters = {}
    hyper_parameters["batch_size"] = cfg.train.BATCH_SIZE
    hyper_parameters["learning_rate"] = cfg.train.optimizer.LEARNING_RATE

    return hyper_parameters

class Throughout:
    def __init__(self):
        self.accum_throughout = 0.
        self.cnt = 0
        self.end = False
        self.start = None
    
    def tick(self, batch_size):
        if self.start is None:
            self.start = time.perf_counter()
        else:
            current_throughout = batch_size / (time.perf_counter() - self.start)
            self.accum_throughout += current_throughout
            self.cnt += 1
            self.start = None
            logger.info(f"Current throughout: {current_throughout:.2f}, Avg throughout: {(self.accum_throughout / self.cnt):.2f}")
