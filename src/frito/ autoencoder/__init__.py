"""frito.autoencoder
Tools for training and running autoencoders for JWST/AMI image reconstruction.
"""

import importlib
import pkgutil
from pathlib import Path

from frito.autoencoder import io_utils

__all__ = []

for _, module_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if module_name.startswith("_") or module_name == "io_utils":
        continue
    module = importlib.import_module(f"frito.autoencoder.{module_name}")
    globals()[module_name] = module
    __all__.append(module_name)