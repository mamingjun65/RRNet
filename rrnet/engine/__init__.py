from rrnet.engine.trainer import do_train
from rrnet.engine.trainer import do_val
from rrnet.engine.trainer import inference

ENGINE_ZOO = {
                'RRNet': (do_train, do_val, inference),
                }

def build_engine(cfg):
    return ENGINE_ZOO[cfg.METHOD]
