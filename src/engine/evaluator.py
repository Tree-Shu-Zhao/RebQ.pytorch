import numpy as np
from torchmetrics.functional import f1_score


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, data, num_classes=None):
        self.metric_name = self.cfg.NAME.lower()
        if self.metric_name == "accuracy":
            return {
                self.cfg.NAME: accuracy(data["logits"], data["labels"]) * 100.
            }
        elif self.metric_name == "f1-macro":
            return {
                self.cfg.NAME: f1_score(data["logits"].float(), data["labels"], average='macro', num_labels=num_classes, task="multilabel") * 100.
            }
        else:
            raise ValueError(f"Cannot find metric name: {self.metric_name}.")
    
    def compute_cl_metric(self, matrix):
        last_task_performance = matrix[-1, :]
        ap = last_task_performance.mean()
        fg = ((np.diagonal(matrix) - last_task_performance)[:-1]).mean()
        last = matrix[-1, -1]
        return ap, fg, last


def accuracy(logits, labels):
    """
    logits: B, C
    labels: B, C. one-hot
    """
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    preds = logits.argmax(dim=-1)
    corrects = (preds == labels).sum()
    totals = len(logits)
    return corrects * 1.0 / totals
