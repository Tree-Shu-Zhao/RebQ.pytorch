import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(cfg):
    criterion_name = cfg.NAME.lower()
    if criterion_name == "contrastive":
        criterion = ContrastiveLoss()
    elif criterion_name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Criterion should be in ['Constrastive'], found {cfg.NAME}.")
    return criterion


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, similarity_matrix):
        return F.cross_entropy(
            similarity_matrix,
            torch.arange(similarity_matrix.shape[0], dtype=torch.long, device=similarity_matrix.device)
        )