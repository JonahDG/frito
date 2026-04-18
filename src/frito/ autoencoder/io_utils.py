import os
from typing import Any, Tuple, Union
import numpy as onp
import jax
import jax.random as jr
import jax.numpy as np
import equinox as eqx

def _resolve(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _makedirs(path: str) -> None:
    os.makedirs(os.path.dirname(_resolve(path)) or ".", exist_ok=True)

def save_model(model: eqx.Module, path: str) -> None:
    _makedirs(path)
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_model(model_like: eqx.Module, path: str) -> eqx.Module:
    with open(path, "rb") as f:
        return eqx.tree_deserialise_leaves(model_like, f)