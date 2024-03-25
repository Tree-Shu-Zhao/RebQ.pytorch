from loguru import logger


def merge_cfg(cfg, params):
    if "cfg.train.optimizer.LEARNING_RATE" in params.keys():
        cfg.train.optimizer.LEARNING_RATE = params["cfg.train.optimizer.LEARNING_RATE"]
        logger.info(f"Reset cfg.train.optimizer.LEARNING_RATE to {params['cfg.train.optimizer.LEARNING_RATE']}")
    if "cfg.train.BATCH_SIZE" in params.keys():
        cfg.train.BATCH_SIZE = params["cfg.train.BATCH_SIZE"]
        logger.info(f"Reset cfg.train.BATCH_SIZE to {params['cfg.train.BATCH_SIZE']}")
    if "cfg.model.prompt.LENGTH" in params.keys():
        cfg.model.prompt.LENGTH = params["cfg.model.prompt.LENGTH"]
        logger.info(f"Reset cfg.model.prompt.LENGTH to {params['cfg.model.prompt.LENGTH']}")
    if "cfg.model.prompt.POOL_SIZE" in params.keys():
        cfg.model.prompt.POOL_SIZE = params["cfg.model.prompt.POOL_SIZE"]
        logger.info(f"Reset cfg.model.prompt.POOL_SIZE to {params['cfg.model.prompt.POOL_SIZE']}")
    if "cfg.model.prompt.LAYERS" in params.keys():
        cfg.model.prompt.LAYERS = params["cfg.model.prompt.LAYERS"]
        logger.info(f"Reset cfg.model.prompt.LAYERS to {params['cfg.model.prompt.LAYERS']}")
    if "cfg.model.prompt.ALPHA" in params.keys():
        cfg.model.prompt.ALPHA = params["cfg.model.prompt.ALPHA"]
        logger.info(f"Reset cfg.model.prompt.ALPHA to {params['cfg.model.prompt.ALPHA']}")

    return cfg
