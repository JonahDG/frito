import os
import inspect
import importlib.util
from pathlib import Path
from typing import Any, Tuple, Union, Optional, Sequence

import jax
from jax import numpy as np, random as jr, Array
import equinox as eqx

from frito.utils import _resolve, _makedirs


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def save_model(model: eqx.Module, path: str) -> None:
    _makedirs(path)
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_model(model_like: eqx.Module, path: str) -> eqx.Module:
    with open(path, "rb") as f:
        return eqx.tree_deserialise_leaves(model_like, f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_4d(array, n_channels: Optional[int] = None):
    """Return ``array`` with a leading channel axis, handling both 3D and 4D.

    ``(N, H, W)`` -> ``(N, 1, H, W)``.
    ``(N, C, H, W)`` -> returned unchanged.

    If ``n_channels`` is given, the resulting channel axis size must match.
    """
    if array.ndim == 3:
        array = array[:, None, :, :]
    elif array.ndim != 4:
        raise ValueError(
            f"Expected array of shape (N, H, W) or (N, C, H, W); got {array.shape}."
        )
    if n_channels is not None and array.shape[1] != n_channels:
        raise ValueError(
            f"Expected {n_channels} channels; got {array.shape[1]} (shape {array.shape})."
        )
    return array


def load_data(
    train_path: str,
    test_path: Optional[str] = None,
    *,
    train_key: str = "x_train",
    test_key: str = "x_test",
    channels: Union[slice, Sequence[int], None] = None,
    mmap: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load training and test arrays.

    Two supported layouts:

    1. **Two ``.npy`` files** (new layout, e.g. ``train_binned.npy`` +
       ``test_binned.npy`` from ``build_data.py``). Pass both paths;
       each file holds one array of shape ``(N, C, H, W)``.
    2. **Single ``.npz`` file** (legacy) containing two keyed arrays
       (default keys ``'x_train'`` and ``'x_test'``). Pass only
       ``train_path``; ``test_path`` is ignored.

    Parameters
    ----------
    train_path : str
        Path to the training data. ``.npy`` (new) or ``.npz`` (legacy).
    test_path : str, optional
        Path to the test data. Required for the ``.npy`` layout, ignored
        for ``.npz``.
    train_key, test_key : str, optional
        Keys inside the ``.npz`` (legacy path only). Defaults
        ``'x_train'`` and ``'x_test'``.
    channels : slice, sequence of int, or None, optional
        Channel selection applied along axis 1 after loading. Useful for
        the 4-channel ``(combined, rings, spirals, planets)`` files:
        e.g. ``slice(1, 4)`` keeps just rings/spirals/planets.
        Default ``None`` returns all channels.
    mmap : bool, optional
        If ``True`` (default), the ``.npy`` files are loaded as
        memory-mapped read-only views (no full RAM load). Has no effect
        on the ``.npz`` legacy path. Set to ``False`` to force a full
        in-memory load.

    Returns
    -------
    train, test : np.ndarray
        Arrays of shape ``(N, C, H, W)`` (or ``(N, len(channels), H, W)``
        if ``channels`` was given).
    """
    train_path = _resolve(train_path)
    ext = os.path.splitext(train_path)[1].lower()

    if ext == ".npz":
        data = np.load(train_path)
        keys = list(data.keys())
        if train_key not in keys or test_key not in keys:
            raise ValueError(
                f"Expected keys '{train_key}' and '{test_key}' in .npz file; "
                f"found: {keys}."
            )
        train = data[train_key]
        test = data[test_key]
    elif ext == ".npy":
        if test_path is None:
            raise ValueError(
                "For the .npy layout, both train_path and test_path are required."
            )
        test_path = _resolve(test_path)
        mmap_mode = "r" if mmap else None
        train = np.load(train_path, mmap_mode=mmap_mode)
        test = np.load(test_path, mmap_mode=mmap_mode)
    else:
        raise ValueError(
            f"Expected .npy or .npz, got '{ext}' (from {train_path})."
        )

    if channels is not None:
        # Works for slice, list, or np.ndarray of indices.
        train = train[:, channels]
        test = test[:, channels]

    return train, test


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
        Image batch of shape ``(N, C, H, W)`` with values in ``[0, 1]``.
        Works for any channel count.
    key : jax.Array
        PRNG key for noise generation.
    noise_factor : float, ``'RMS'``, or None, optional
        Controls the standard deviation of the added noise.

        - ``float`` -- a fixed noise scale applied uniformly.
        - ``'RMS'`` -- sets the noise scale to ``rms_scale`` * each image's
          RMS value (computed across channels, H, W).
        - ``None`` -- draws a per-image scale uniformly from ``[0, 0.5)``
          (default).
    rms_scale : float, optional
        Fraction of each image's RMS value used as the noise scale when
        ``noise_factor='RMS'``. Default ``0.1``.

    Returns
    -------
    np.ndarray
        Noisy image batch clipped to ``[0, 1]``, same shape as ``array``.
    """
    normal_key, uniform_key = jax.random.split(key, 2)
    n = array.shape[0]
    bcast_shape = (n,) + (1,) * (array.ndim - 1)

    if noise_factor is None:
        scale = jax.random.uniform(uniform_key, shape=(n,), minval=0.0, maxval=0.5)
    elif noise_factor == "RMS":
        scale = rms_scale * np.sqrt(
            np.mean(array ** 2, axis=tuple(range(1, array.ndim)))
        )
    else:
        assert isinstance(noise_factor, (int, float)), (
            "noise_factor must be 'RMS', None, or a numeric value."
        )
        scale = np.full((n,), noise_factor)

    scale = scale.reshape(bcast_shape)
    return np.clip(
        array + scale * jax.random.normal(normal_key, array.shape), 0.0, 1.0
    )


def preprocess(
    array: np.ndarray,
    n_channels: Optional[int] = None,
) -> np.ndarray:
    """Cast and ensure batch shape ``(N, C, H, W)`` as float64.

    Auto-detects 3D vs 4D input:

    - ``(N, H, W)`` (single-channel, no channel axis) -> ``(N, 1, H, W)``.
    - ``(N, C, H, W)`` -> returned unchanged (channel axis already present).

    Parameters
    ----------
    array : np.ndarray
        Raw image batch.
    n_channels : int, optional
        If given, assert the channel axis size matches.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(N, C, H, W)``.
    """
    array = np.asarray(array, dtype=np.float64)
    return _ensure_4d(array, n_channels=n_channels)


# ---------------------------------------------------------------------------
# Dynamic class loader
# ---------------------------------------------------------------------------

def load_classes_from_file(filepath: str) -> dict:
    """Dynamically load a Python file and return all classes defined in it.

    Parameters
    ----------
    filepath : str
        Path to the ``.py`` file. Supports ``~``, env vars, relative paths.

    Returns
    -------
    dict
        ``{class_name: class_reference}`` for classes defined in the file
        (excludes anything imported into it).
    """
    filepath = Path(str(filepath))
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


# ---------------------------------------------------------------------------
# SVD (unchanged for now)
# ---------------------------------------------------------------------------

def run_svd(
    model_structure: str | Path,
    trained_model: str | Path,
    data_path: str | Path,
    svd_file_dir: str | Path,
    svd_file_name: str,
    batch_size: int = 250,
    key: Array = jr.key(0),
):
    if model_structure[-3:] != ".py":
        raise NameError(
            f"Model Structure is expected to be stored in a .py script. This ends in {model_structure[-3:]}."
        )
    if trained_model[-4:] != ".eqx":
        raise NameError(
            f"The Trained Model is expected to be a .eqx PyTree. This ends in {trained_model[-4:]}."
        )
    if svd_file_name[-4:] != ".npz":
        raise NameError(
            f'The Name of the SVD File expects to end in ".npz". This ends in {svd_file_name[-4:]}'
        )
    svd_file_path = os.path.join(svd_file_dir, svd_file_name)

    data, _ = load_data(data_path)
    data = np.clip(data, min=1e-8)

    autoencoder_classes = load_classes_from_file(model_structure)
    autoencoder = autoencoder_classes["autoencoder"](key=key)

    trained_autoencoder = eqx.tree_deserialise_leaves(trained_model, autoencoder)
    encoder, decoder = trained_autoencoder.modules

    def encode_batch(batch):
        return jax.vmap(encoder)(batch)

    latents = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch = batch.reshape(-1, 1, batch.shape[-1], batch.shape[-2])
        latents.append(encode_batch(batch))
    train_latents = np.concatenate(latents)

    train_latents_mean = np.mean(train_latents, axis=0)
    train_latents_centered = train_latents - train_latents_mean
    u, s, v = np.linalg.svd(train_latents_centered, full_matrices=False)

    np.savez(svd_file_path, u=u, s=s, v=v, mean=train_latents_mean)


# ---------------------------------------------------------------------------
# Training-loop I/O stubs
# ---------------------------------------------------------------------------

def save_train_losses(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    path: str,
    *,
    train_losses_per_channel: Optional[Sequence[Sequence[float]]] = None,
    val_losses_per_channel: Optional[Sequence[Sequence[float]]] = None,
) -> None:
    """Save training and validation loss histories to an ``.npz``.

    Channel-count agnostic. The per-channel arrays may be omitted (the
    keys simply won't be present in the saved file) or have any channel
    count -- single-channel models pass length-1 vectors, multi-channel
    pass length-C vectors.

    Parameters
    ----------
    train_losses, val_losses : sequence of float
        Per-step (train) and per-validation-pass (val) scalar losses.
    path : str
        Output ``.npz`` path.
    train_losses_per_channel, val_losses_per_channel : sequence of sequence, optional
        Parallel to ``train_losses`` / ``val_losses``, each entry a
        length-C vector of per-channel losses.
    """
    _makedirs(path)
    payload: dict[str, Any] = {
        "train_losses": np.asarray(train_losses),
        "val_losses": np.asarray(val_losses),
    }
    if train_losses_per_channel is not None:
        payload["train_losses_per_channel"] = np.asarray(train_losses_per_channel)
    if val_losses_per_channel is not None:
        payload["val_losses_per_channel"] = np.asarray(val_losses_per_channel)
    np.savez(path, **payload)


def save_embeddings(embeddings, path: str) -> None:
    """Save post-encoder embeddings to disk.

    Shape-agnostic: works for single-channel ``(N, D)`` and multi-channel
    ``(N, C, D)`` latents alike.

    Parameters
    ----------
    embeddings : array-like
        Embeddings to save.
    path : str
        Output ``.npy`` path.
    """
    _makedirs(path)
    np.save(path, np.asarray(embeddings))


def save_checkpoint(
    path: str,
    model: eqx.Module,
    opt_state,
    epoch: int,
    global_step: int,
    key: jax.Array,
) -> None:
    """Save a training checkpoint.

    Serialises the full ``(model, opt_state, epoch, global_step, key)``
    tuple as a single equinox PyTree. Integers are wrapped in JAX arrays
    so they round-trip through the serialisation cleanly.

    Parameters
    ----------
    path : str
        Output path (any extension; ``.eqx`` is conventional).
    model, opt_state : eqx.Module / PyTree
        Model and optimiser state.
    epoch, global_step : int
        Training position.
    key : jax.Array
        PRNG key state.
    """
    _makedirs(path)
    bundle = (
        model,
        opt_state,
        np.asarray(epoch),
        np.asarray(global_step),
        key,
    )
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, bundle)


def load_checkpoint(
    path: str,
    model_like: eqx.Module,
    opt_like,
    key_like: jax.Array,
):
    """Load a checkpoint saved by ``save_checkpoint``.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    model_like, opt_like, key_like : PyTree
        Structural templates matching the saved ones. The model and
        opt-state must be freshly constructed (with any key) so the
        deserialiser knows what shapes to expect.

    Returns
    -------
    model, opt_state, epoch, global_step, key
        Restored values. ``epoch`` and ``global_step`` are returned as
        Python ints.
    """
    bundle_like = (
        model_like,
        opt_like,
        np.asarray(0),
        np.asarray(0),
        key_like,
    )
    with open(path, "rb") as f:
        model, opt_state, epoch, global_step, key = eqx.tree_deserialise_leaves(
            f, bundle_like
        )
    return model, opt_state, int(epoch), int(global_step), key


def norm(x):
    """Placeholder normalisation hook. Returns ``x`` unchanged.

    Reserved for future use; kept so existing imports of ``norm`` from
    this module don't break.
    """
    return x