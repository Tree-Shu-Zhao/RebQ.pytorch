import time

import torch
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from optim import build_criterion, build_lr_scheduler, build_optimizer
from utils import AverageMeter, Checkpoint, ProgressMeter

from .trainer import Trainer


class DistributedTrainer(Trainer):
    def __init__(self, cfg, dataloaders, model):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.device = torch.device(f"cuda:{cfg.GPU}")
        self.model = model.to(self.device)

        # Build criterion
        self.criterion = build_criterion(cfg.train.criterion).to(self.device)

        # Build optimizer, lr_scheduler (optinal)
        self.optimizer = build_optimizer(cfg.train.optimizer, filter(lambda p: p.requires_grad, self.model.parameters()))
        self.lr_scheduler = None if "lr_scheduler" not in cfg.train.optimizer else build_lr_scheduler(cfg.train.lr_scheduler)

        # Build evaluator
        #self.evaluator = Evaluator(cfg.test, self.device)

        self.scaler = GradScaler()

        self.epochs = cfg.train.EPOCHS
        self.current_epoch = 0

        self.checkpoint = Checkpoint(cfg.OUTPUT_DIR)

        # Load training checkpoint
        if cfg.train.CHECKPOINT:
            self.load_checkpoint(cfg.train.CHECKPOINT)
        
    def train(self):
        if self.cfg.train.ONLY_VAL:
            logger.info("Debug model! Only run a validation function!")
            metrics, val_score = self.validate()
            self.display_metrics(metrics)
            return

        best_score = 0.
        for epoch in range(self.epochs):
            self.train_data.sampler.set_epoch(epoch)
            self.current_epoch = epoch
            self.train_loop()
            if (epoch + 1) % self.cfg.train.EVAL_FREQ == 0:
                metrics, val_score = self.validate()
            
            self.display_metrics(metrics)

            if best_score < val_score:
                best_score = val_score
                logger.info("New Best score: {:.2f}.".format(best_score))
                self.save_checkpoint(metrics)
    
    def train_loop(self):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':6.3f')
        progress = ProgressMeter(
            len(self.dataloaders["train"]),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(self.current_epoch),
        )

        self.model.train()
        end = time.time()

        for it, (reference_images, target_images, rel_captions) in enumerate(self.dataloaders["train"]):
            data_time.update(time.time() - end)

            reference_images = reference_images.to(self.device, non_blocking=True)
            target_images = target_images.to(self.device, non_blocking=True)

            with autocast():
                outputs = self.model(reference_images, target_images, rel_captions)
                loss = self.criterion(outputs)
            
            losses.update(loss.item(), len(reference_images))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_time.update(time.time() - end)
            end = time.time()

            if (it+1) % self.cfg.train.PRINT_FREQ == 0:
                progress.display(it+1)
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def validate(self):
        self.model.eval()

        index_features = torch.empty((0, self.model.output_dim)).to(self.device, non_blocking=True)
        index_names = []
        for image_names, images in tqdm(self.dataloaders["classic_val"], desc="classic val"):
            images = images.to(self.device, non_blocking=True)
            with torch.no_grad():
                with autocast():
                    image_features = self.model.encode_image(images)
                index_features = torch.vstack((index_features, image_features))
                index_names.extend(image_names)
        
        results = self.evaluator.compute_cirr_val_metrics(
            self.dataloaders["val"], 
            self.model.clip_model, 
            index_features,
            index_names, 
            element_wise_sum,
        )
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

        return {
            'Rs@1': group_recall_at1,
            'Rs@2': group_recall_at2,
            'Rs@3': group_recall_at3,
            'R@1': recall_at1,
            'R@5': recall_at5,
            'R@10': recall_at10,
            'R@50': recall_at50,
            'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
        }, (group_recall_at1 + recall_at5) / 2

    def save_checkpoint(self, metrics):
        ckpt = {
            "model": self.model,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "current_epoch": self.current_epoch,
            "metrics": metrics
        }
        if self.lr_scheduler:
            ckpt.update({"lr_scheduler": self.lr_scheduler.state_dict()})
        self.checkpoint.save(**ckpt)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.current_epoch = checkpoint["current_epoch"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch}).")

    def display_metrics(self, metrics):
        s = "Metrics:\n"
        s += "=" * 50
        s += "\n"
        for k, v in metrics.items():
            s += "{}: {:.2f}\n".format(k, v)
        s += "=" * 50
        logger.info(s)