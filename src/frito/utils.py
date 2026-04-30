import os
from typing import Tuple
from pathlib import Path

import jax
from jax import numpy as np, Array, random as jr
from jax.experimental import checkify

# ============================================================================ #
# Path utilities
# ============================================================================ #

def _resolve(path: str) -> str:
    """Resolve a path to an absolute path, expanding user home and env vars.

    Parameters
    ----------
    path : str
        Path to resolve. Supports ``~``, environment variables, and relative
        paths.

    Returns
    -------
    str
        Absolute path with all expansions applied.
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _makedirs(path: str) -> None:
    """Create parent directories for *path* if they do not already exist.

    Parameters
    ----------
    path : str
        File path whose parent directories should be created.
    """
    os.makedirs(os.path.dirname(_resolve(path)) or ".", exist_ok=True)

def load_disco(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True).item()
    filters = data.keys()
    filter1 = list(filters)[0]
    disco_keys = data[filter1].keys()
    return data, filters, disco_keys

def normalize_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    checkify.check(img.ndim == 2, f'Image must be 2-D, got shape {img.shape}')
    s = img.sum()
    checkify.check(s > 0, "Image has non-positive total flux")
    norm_img = img / s
    norm_img = norm_img.astype(np.float64)
    return norm_img

