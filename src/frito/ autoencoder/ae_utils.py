import os
import inspect
import importlib.util
from pathlib import Path
from typing import Any, Tuple, Union

import jax
import jax.numpy as np
import equinox as eqx

from frito.utils import _resolve, _makedirs

from frito.utils import _resolve, _makedirs


def save_model(model: eqx.Module, path: str) -> None:
    _makedirs(path)
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_model(model_like: eqx.Module, path: str) -> eqx.Module:
    with open(path, "rb") as f:
        return eqx.tree_deserialise_leaves(model_like, f)


def load_data(
    path: str,
    train_key: str = "x_train",
    test_key: str = "x_test",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load training and test arrays from an ``.npz`` file.

    Parameters
    ----------
    path : str
        Path to an ``.npz`` file. Supports ``~``, environment variables,
        and relative paths.
    train_key : str, optional
        Key for the training array, by default ``'x_train'``.
    test_key : str, optional
        Key for the test array, by default ``'x_test'``.

    Returns
    -------
    x_train : np.ndarray
        Training data array.
    x_test : np.ndarray
        Test data array. This data is doubly used for validation, but will not
        affect training only as save point for the models.

    Raises
    ------
    ValueError
        If the file extension is not ``.npz``, or if the expected keys
        are absent.
    """
    path = _resolve(path)
    ext = os.path.splitext(path)[1].lower()
    if ext != ".npz":
        raise ValueError(f"Expected an .npz file, got '{ext}'.")
    data = np.load(path)
    keys = list(data.keys())
    if train_key not in keys or test_key not in keys:
        raise ValueError(
            f"Expected keys '{train_key}' and '{test_key}' in .npz file; found: {keys}."
        )
    return data[train_key], data[test_key]


def add_noise(
    array: np.ndarray,
    key: jax.Array,
    noise_factor: Union[float, str, None] = None,
    rms_scale: float = 0.1,
) -> np.ndarray:
    """Add Gaussian noise to a batch of images.

    Parameters
    ----------
    array : np.ndarray
        Image batch of shape ``(N, 1, H, W)`` with values in ``[0, 1]``.
    key : jax.Array
        PRNG key for noise generation.
    noise_factor : float, ``'RMS'``, or None, optional
        Controls the standard deviation of the added noise.

        - ``float`` — a fixed noise scale applied uniformly to all images.
        - ``'RMS'`` — sets the noise scale to ``rms_scale`` % of each image's
          RMS value.
        - ``None`` — draws a per-image scale uniformly from ``[0, 0.5)``
          (default).
    rms_scale : float, optional
        Fraction of each image's RMS value used as the noise scale when
        ``noise_factor='RMS'``. Default is ``0.1`` (10 %).

    Returns
    -------
    np.ndarray
        Noisy image batch clipped to ``[0, 1]``, same shape as ``array``.

    Raises
    ------
    AssertionError
        If ``noise_factor`` is not a float, int, ``'RMS'``, or ``None``.
    """
    normal_key, uniform_key = jax.random.split(key, 2)
    n = array.shape[0]

    if noise_factor is None:
        scale = jax.random.uniform(
            uniform_key, shape=(n,), minval=0.0, maxval=0.5
        )
    elif noise_factor == "RMS":
        scale = rms_scale * np.sqrt(np.mean(array**2, axis=(1, 2, 3)))
    else:
        assert isinstance(noise_factor, (int, float)), (
            "noise_factor must be 'RMS', None, or a numeric value."
        )
        scale = np.full((n,), noise_factor)

    scale = scale.reshape(n, 1, 1, 1)
    return np.clip(
        array + scale * jax.random.normal(normal_key, array.shape), 0.0, 1.0
    )


def preprocess(array: np.ndarray) -> np.ndarray:
    """Cast and reshape a batch of images to ``(N, 1, H, W)`` float64.

    Parameters
    ----------
    array : np.ndarray
        Raw image batch of shape ``(N, H, W)``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(N, 1, H, W)``.
    """
    h, w = array[0].shape
    array = array.astype("float64")
    return np.resize(np.array(array), (len(array), 1, h, w))


def load_classes_from_file(filepath: str) -> dict:
    """Dynamically load a Python file and return all classes defined in it.

    Parameters
    ----------
    filepath : str
        Path to the ``.py`` file to load. Supports ``~``, environment
        variables, and relative paths.

    Returns
    -------
    dict
        A dictionary mapping ``{class_name: class_reference}`` for all
        classes defined in the file, excluding anything imported into it.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a ``.py`` file.
    """
    filepath = Path(_resolve(str(filepath)))

    if not filepath.exists():
        raise FileNotFoundError(f"No such file: {filepath}")
    if filepath.suffix != ".py":
        raise ValueError(f"Expected a .py file, got '{filepath.suffix}'.")

    spec = importlib.util.spec_from_file_location(filepath.stem, str(filepath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    return {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    }
