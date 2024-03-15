from .rebq import RebQ


def build_model(cfg):
    model_name = cfg.NAME.lower()
    if model_name == "rebq":
        return RebQ(cfg)
    else:
        raise ValueError(f"Cannot find model name: {cfg.NAME}!")
