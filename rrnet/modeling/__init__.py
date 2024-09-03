__all__ = ['build_model']

from .rrnet import RRNet

_MODELS_ = {
    'RRNet': RRNet,
}

def make_model(cfg):
    model = _MODELS_[cfg.METHOD] # 'RRNet'
    try:
        return model(cfg, dataset_name=cfg.DATASET.NAME)
    except:
        return model(cfg.MODEL, dataset_name=cfg.DATASET.NAME)
