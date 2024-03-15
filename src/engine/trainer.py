import time

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from optim import build_criterion, build_lr_scheduler, build_optimizer
from utils import AverageMeter, Checkpoint, ProgressMeter, Throughout

from .evaluator import Evaluator


class Trainer:
    def __init__(self, cfg, dataloaders, model, tensorboard_logger):
        super().__init__()

        self.cfg = cfg
        self.task_id = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.epochs = cfg.train.EPOCHS
        self.num_tasks = cfg.model.NUM_TASKS
        # We assume the number of classes per task are the same for simplicity
        # But it is easy to extend it
        self.num_labels_per_task = cfg.model.NUM_LABELS_PER_TASK
        self.dataloaders = dataloaders
        self.tensorboard_logger = tensorboard_logger

        if cfg.train.GPU is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{cfg.train.GPU[0]}") if isinstance(cfg.train.GPU, list) else torch.device(f"cuda:{cfg.train.GPU}")

        self.model = model.to(self.device)

        # Build criterion
        self.criterion = build_criterion(cfg.train.criterion).to(self.device)

        # Build evaluator
        self.evaluator = Evaluator(cfg.test)

        # Fp16 training
        self.scaler = GradScaler()

        # Load training checkpoint
        self.checkpoint = Checkpoint(cfg.OUTPUT_DIR)
        if cfg.train.CHECKPOINT:
            self.load_checkpoint(cfg.train.CHECKPOINT, cfg.train.CHECKPOINT_TASK_ID)

        # Throughout
        if cfg.train.THROUGHOUT:
            self.throughout = Throughout()

    def train(self):
        if self.cfg.train.ONLY_VAL:
            logger.info("Debug model! Only run a validation function!")
            self.prepare_incremental_task(0)
            metrics = self.validate()
            self.display_metrics(metrics)
            return

        current_task_id = self.task_id
        result_matrix = np.zeros((self.num_tasks, self.num_tasks))
        for task_id in range(current_task_id, self.num_tasks):
            best_score = 0.
            # Set task id, set valid class range, reset optimizer, and prepare data for this task
            self.prepare_incremental_task(task_id)
            for epoch in range(self.epochs):
                metrics = None
                self.current_epoch = epoch
                self.train_loop()
                if (epoch + 1) % self.cfg.train.EVAL_FREQ == 0:
                    metrics = self.validate()
                    score = list(metrics[-1].values())[0].item()
                    self.display_metrics(metrics)

                    for task_id, metric in enumerate(metrics):
                        for k, v in metric.items():
                            self.tensorboard_logger.add_scalar(f"metric/task{task_id}_{k}", v, self.task_id)

                    if best_score < score:
                        best_score = score
                        logger.info("New best score: {:.2f}.".format(best_score))
                        self.save_checkpoint(metrics, task_id=task_id, is_best=True)
            if metrics is None:
                metrics = self.validate()
                for task_id, metric in enumerate(metrics):
                    for k, v in metric.items():
                        self.tensorboard_logger.add_scalar(f"metric/task{task_id}_{k}", v, self.task_id)
                score = list(metrics[-1].values())[0].item()
                self.display_metrics(metrics)
            self.save_checkpoint(metrics, task_id=task_id, is_best=False)
            for tid, metric in enumerate(metrics):
                result_matrix[task_id][tid] = metric[self.cfg.test.NAME]
        ap, fg, last = self.evaluator.compute_cl_metric(result_matrix)
        logger.info(f"AP: {ap:.2f}, Forget: {fg:.2f}, Last: {last:.2f}")
        return ap
    
    def train_loop(self):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':6.3f')
        cls_loss = AverageMeter('ClsLoss', ':6.3f')
        rec_loss = AverageMeter('RecLoss', ':6.3f')
        progress = ProgressMeter(
            len(self.train_dataloader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}/{}]".format(self.current_epoch+1, self.epochs),
        )

        self.model.train()
        end = time.perf_counter()
        for it, batch in enumerate(self.train_dataloader):
            if self.cfg.train.THROUGHOUT:
                self.throughout.tick(len(batch["labels"]))
            self.current_iter += 1
            data_time.update(time.perf_counter() - end)

            self.optimizer.zero_grad()
            batch = move_to_device(batch, self.device)
            with autocast():
                outputs = self.model(batch)
                logits = outputs["logits"]
                labels = outputs["labels"] if "labels" in outputs else batch["labels"]
                reconstruction_loss = outputs["reconstruction_loss"] if "reconstruction_loss" in outputs else None

                logits = logits[:, :self.valid_class_range[1]]
                if self.cfg.data.MULTI_LABEL: # One-hot
                    labels = labels[:, :self.valid_class_range[1]]
                if self.cfg.train.FREEZE_LOGITS:
                    # Freeze previous task logits
                    logits[:, :self.valid_class_range[0]] = -100. # -float('inf') will get nan by BCEwithLogits
                loss = self.criterion(logits, labels)

                cls_loss.update(loss.item(), len(batch["labels"]))

                if reconstruction_loss is not None:
                    rec_loss.update(reconstruction_loss.item(), len(batch["labels"]))
                    loss += self.cfg.model.prompt.ALPHA * reconstruction_loss
            
            losses.update(loss.item(), len(batch["labels"]))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.lr_scheduler and self.lr_scheduler_interval == "step":
                self.lr_scheduler.step()

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if (it+1) % self.cfg.train.PRINT_FREQ == 0:
                progress.display(it+1)
                self.tensorboard_logger.add_scalar(f"loss/task{self.task_id}_ce", cls_loss.avg, self.current_iter)
                self.tensorboard_logger.add_scalar(f"loss/task{self.task_id}_rec", rec_loss.avg, self.current_iter)

        if self.lr_scheduler and self.lr_scheduler_interval == "epoch":
            self.lr_scheduler.step()
    
    def prepare_incremental_task(self, task_id):
        #if hasattr(self.model, "prompt") and hasattr(self.model.prompt, "task_count"):
        self.model.process_task_count(task_id)
        self.task_id = task_id
        self.valid_class_range = (task_id * self.num_labels_per_task, (task_id+1) * self.num_labels_per_task)
        self.prepare_dataloaders()
        self.reset_optimizer()
    
    def reset_optimizer(self):
        self.optimizer = build_optimizer(self.cfg.train.optimizer, filter(lambda p: p.requires_grad, self.model.parameters()))
        num_training_steps = len(self.train_dataloader) * self.cfg.train.EPOCHS
        self.lr_scheduler = None if "lr_scheduler" not in self.cfg.train \
            else build_lr_scheduler(
                self.cfg.train.lr_scheduler, 
                self.optimizer,
                num_training_steps=num_training_steps,
            )
        self.lr_scheduler_interval = None if "lr_scheduler" not in self.cfg.train else self.cfg.train.lr_scheduler.LR_SCHEDULER_INTERVAL
    
    def prepare_dataloaders(self):
        self.train_dataloader = self.dataloaders["train"][self.task_id]
        self.val_dataloaders = self.dataloaders["val"][self.task_id]
    
    @torch.no_grad()
    def _validate_one_dataloader(self, val_dataloader, task_id=0):
        logits = []
        labels = []
        for it, batch in tqdm(enumerate(val_dataloader), desc=f"task{task_id}", total=len(val_dataloader)):
            with autocast():
                outputs = self.model(batch)
            logits.append(outputs["logits"].cpu())
            labels.append(batch["labels"].cpu())
        logits = torch.vstack(logits)[:, :self.valid_class_range[1]]
        if self.cfg.data.MULTI_LABEL:
            labels = torch.vstack(labels)
            labels = labels[:, :self.valid_class_range[1]]
        else:
            labels = torch.hstack(labels)
        #labels = F.one_hot(torch.hstack(labels), self.cfg.model.TOTAL_LABELS)[:, :self.valid_class_range[1]]

        metrics = self.evaluator({
            "logits": logits,
            "labels": labels,
        }, num_classes=labels.shape[1] if self.cfg.data.MULTI_LABEL else None)
        return metrics
    
    @torch.no_grad()
    def _validate_tasks(self, dataloaders, num_tasks):
        return [self._validate_one_dataloader(dataloaders[task_id], task_id) for task_id in range(num_tasks)]
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        return self._validate_tasks(self.val_dataloaders, self.task_id+1)
    
    @torch.no_grad()
    def test(self, dataloaders, task_id):
        self.prepare_incremental_task(task_id)
        self.model.eval()
        return self._validate_tasks(dataloaders, task_id+1)

    def save_checkpoint(self, metrics, task_id=None, is_best=False):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),
            "current_epoch": self.current_epoch,
            "metrics": metrics,
            "task_id": task_id,
        }
        if self.lr_scheduler:
            ckpt.update({"lr_scheduler": self.lr_scheduler.state_dict()})
        self.checkpoint.save(**ckpt, is_best=is_best)

    def load_checkpoint(self, checkpoint_path, task_id=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.current_epoch = 0
        if task_id in checkpoint:
            task_id = checkpoint["task_id"]
        else:
            if task_id == None:
                raise ValueError("Must provide a task_id!")
        self.prepare_incremental_task(task_id)
        self.model.load_state_dict(checkpoint["model"])
        #self.optimizer.load_state_dict(checkpoint["optimizer"])
        #if self.lr_scheduler:
        #    self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        logger.info(f"Loaded checkpoint from {checkpoint_path} (task_id: {self.task_id}).")

    def display_metrics(self, metrics):
        s = "Metrics:\n"
        s += "=" * 50
        s += "\n"
        for task_id, metric in enumerate(metrics):
            for k, v in metric.items():
                s += f"Task {task_id} {k}: {v:.2f}\n"
        s += "=" * 50
        logger.info(s)

def move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch