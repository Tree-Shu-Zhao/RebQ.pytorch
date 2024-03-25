import warnings

import torch
from transformers import get_cosine_schedule_with_warmup


def build_lr_scheduler(cfg, optimizer, **kwargs):
    lr_scheduler_name = cfg.NAME.lower()
    if lr_scheduler_name == "exponential_with_min":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.GAMMA, cfg.MIN_LR)
    elif lr_scheduler_name == "warmup_cosine":
        num_training_steps = kwargs["num_training_steps"]
        num_warmup_steps = int(cfg.NUM_WARMUP_STEPS * num_training_steps) if isinstance(cfg.NUM_WARMUP_STEPS, float) else cfg.NUM_WARMUP_STEPS
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"LR scheduler should be in ['exponential_with_min', 'warmup_cosine'].")
    return lr_scheduler


class ExponentialLrWithMin(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr, last_epoch=-1, verbose=False):
        super(ExponentialLrWithMin, self).__init__(optimizer, gamma, last_epoch, verbose)
        self.min_lr = min_lr
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        if self.optimizer.param_goups[0]["lr"] <= self.min_lr:
            return [group['lr'] * 1.0
                for group in self.optimizer.param_groups] 
        else:
            return [group['lr'] * self.gamma
                    for group in self.optimizer.param_groups]