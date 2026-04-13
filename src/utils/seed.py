"""Global seed setting for reproducibility.

Import and call set_seed() at the top of every script to ensure
deterministic results across runs.

Usage:
    from utils.seed import set_seed
    set_seed()
"""

import os
import random
import numpy as np

GLOBAL_SEED = 42


def set_seed(seed=GLOBAL_SEED):
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except (ImportError, Exception):
        pass
