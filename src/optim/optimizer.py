import torch.optim as optim


def build_optimizer(cfg, parameters):
    optimizer_name = cfg.NAME.lower()
    if optimizer_name == "adamw":
       optimizer = optim.AdamW(parameters, lr=cfg.LEARNING_RATE, betas=cfg.BETAS, eps=cfg.EPS)
    else:
        raise ValueError("Optimizer should be in ['AdamW'].")
    return optimizer