import os
from typing import Tuple


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