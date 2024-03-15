import os

import torch
from loguru import logger


class Checkpoint:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save(self, **kwargs):
        checkpoint = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.nn.Module):
                checkpoint[key] = value.state_dict()
            else:
                checkpoint[key] = value
        save_path = os.path.join(self.output_dir, "checkpoints", f"task{kwargs['task_id']}_model_best.pth" if kwargs["is_best"] else f"task{kwargs['task_id']}_last.pth")
        logger.info(f"Save checkpoint to {save_path}.")
        torch.save(checkpoint, save_path)
    
    def load(self, checkpoint_pth):
        return torch.load(checkpoint_pth)