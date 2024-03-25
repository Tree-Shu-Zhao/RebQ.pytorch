from .distributed_trainer import DistributedTrainer
from .trainer import Trainer


def build_trainer(cfg, dataloaders, model, tensorboard_logger, **kwargs):
    if kwargs["distributed"]:
        trainer = DistributedTrainer(cfg, dataloaders, model)
    else:
        trainer = Trainer(cfg, dataloaders, model, tensorboard_logger)
    return trainer
