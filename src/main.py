import os

import hydra
import nni
import numpy as np
import pyrootutils
import torch
from loguru import logger
from omegaconf import OmegaConf

import utils
from data import build_dataloaders
from engine import Evaluator, build_trainer
from model import build_model

# set root directory
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # Setup logger, code backup, hyper-parameteres records, etc...
    env_info, tensorboard_logger = utils.init_env(cfg)
    if cfg.train.HPARAMS_SEARCHING:
        cfg = env_info["merged_cfg"]
    logger.info(f"Configs: {OmegaConf.to_yaml(cfg)}")

    # Build model
    model = build_model(cfg.model)
    utils.get_number_of_parameters(model)

    # Build dataloader
    dataloaders = build_dataloaders(cfg.data, distributed=env_info["is_distributed"])

    # Build trainer
    trainer = build_trainer(cfg, dataloaders, model, tensorboard_logger, distributed=env_info["is_distributed"])

    if cfg.test.TEST_ONLY:
        # Build evaluator
        logger.info("TEST_ONLY mode.")
        evaluator = Evaluator(cfg.test)

        # The format of results is a matrix
        # For each task, we evaluate the current task and all previous tasks
        result_matrix = np.zeros((cfg.data.NUM_TASKS, cfg.data.NUM_TASKS))
        for task_id in range(cfg.data.NUM_TASKS):
            ckpt_path = os.path.join(cfg.test.CHECKPOINT_DIR, f"task{task_id}_last.pth")
            trainer.load_checkpoint(ckpt_path, task_id=task_id)
            all_task_metrics = trainer.test(dataloaders["test"], task_id)
            for tid, metrics in enumerate(all_task_metrics):
                result_matrix[task_id][tid] = metrics[cfg.test.NAME]
        logger.info(f"Result Matrix: {result_matrix}")
        np.save(os.path.join(cfg.test.CHECKPOINT_DIR, "result_matrix"), result_matrix)

        # Compute matrics
        ap, fg, last = evaluator.compute_cl_metric(result_matrix)
        logger.info(f"AP: {ap:.2f}, Forget: {fg:.2f}, Last: {last:.2f}")
    else:
        the_metric = trainer.train()

    if cfg.train.HPARAMS_SEARCHING:
        if isinstance(the_metric, torch.Tensor):
            the_metric = the_metric.item()
        nni.report_final_result(the_metric)

    tensorboard_logger.flush()
    tensorboard_logger.close()

if __name__ == "__main__":
    main()
