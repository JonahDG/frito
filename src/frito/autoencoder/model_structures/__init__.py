"""frito.autoencoder.model_structures
Architecture definitions for autoencoders used in JWST/AMI image reconstruction.
"""

import importlib
import pkgutil
from pathlib import Path

__all__ = []

# for _, module_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
#     if module_name.startswith("_"):
#         continue
#     module = importlib.import_module(f"frito.autoencoder.model_structures.{module_name}")
#     globals().update({
#         name: getattr(module, name)
#         for name in vars(module)
#         if not name.startswith("_")
#     })
#  __all__.append(module_name)