"""Deterministic seeding across random / numpy / torch."""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 42) -> None:
    """Fix every RNG we touch. Call once at the top of every CLI entry point."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Trade some throughput for determinism (mT5 has no conv layers, but be safe).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
