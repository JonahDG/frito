"""frito
An extension of the dorito package for machine-learned feature-space reconstruction for JWST/AMI images.
"""

import importlib
import importlib.metadata
from typing import Any

__version__ = importlib.metadata.version("frito")

def __getattr__(name):
    if name == "autoencoder":
        return importlib.import_module(".autoencoder", package="frito")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return ["autoencoder"] + list(globals().keys())

from .dorito_updates import (
    TransformedResolvedDiscoModel,
    AutoencoderBasis,
    TransformedResolvedOIFit,
    PointResolvedOIFit,
    PointResolvedDiscoModel
)

from . import utils

from . import simulate