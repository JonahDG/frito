import jax
from jax import numpy as np, tree as jt, random as jr, Array
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def make_coordinate_grid(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a centered coordinate grid in range ``[-1, 1]``.

    Parameters
    ----------
    size : int
        Number of pixels along each axis.

    Returns
    -------
    xx, yy : np.ndarray
        2D arrays of shape ``(size, size)``.
    """
    y = np.linspace(-1, 1, size)
    x = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def elliptical_polar(
    xx: np.ndarray,
    yy: np.ndarray,
    axis_ratio: float = 1.0,
    position_angle: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to elliptical polar coordinates."""
    cos_pa = np.cos(position_angle)
    sin_pa = np.sin(position_angle)
    x_rot = xx * cos_pa + yy * sin_pa
    y_rot = -xx * sin_pa + yy * cos_pa
    y_scaled = y_rot / axis_ratio
    r = np.sqrt(x_rot**2 + y_scaled**2)
    theta = np.arctan2(y_scaled, x_rot)
    return r, theta


def angular_asymmetry(
    theta: np.ndarray,
    a_sin: Optional[List[float]] = None,
    b_cos: Optional[List[float]] = None,
    square: bool = True,
    normalize_mean: bool = True,
    clip_min: float = 0.0,
) -> np.ndarray:
    """Compute an angular modulation map from Fourier coefficients."""
    mod = np.ones_like(theta, dtype=float)

    if a_sin is not None:
        for n, a in enumerate(a_sin):
            if n == 0:
                continue
            mod = mod + a * np.sin(n * theta)

    if b_cos is not None:
        for n, b in enumerate(b_cos):
            if n == 0:
                mod = mod + b
            else:
                mod = mod + b * np.cos(n * theta)

    if clip_min is not None:
        mod = np.clip(mod, clip_min, None)

    if square:
        mod = mod**2

    if normalize_mean:
        mod = mod / (mod.mean() + 1e-8)

    return mod


# ---------------------------------------------------------------------------
# Channel-dict helpers
# ---------------------------------------------------------------------------

CHANNEL_KEYS = ("combined", "rings", "spirals", "planets")


def empty_channels(size: int) -> Dict[str, np.ndarray]:
    """Return a dict of zero arrays for every channel.

    Parameters
    ----------
    size : int
        Image size in pixels.

    Returns
    -------
    dict of str -> np.ndarray
        Keys are ``'combined'``, ``'rings'``, ``'spirals'``, ``'planets'``.
    """
    z = np.zeros((size, size))
    return {k: z for k in CHANNEL_KEYS}


def _ensure_channels(
    x: Union[np.ndarray, Dict[str, np.ndarray]], size: int
) -> Dict[str, np.ndarray]:
    """Coerce input to a channel dict (used internally for flexibility)."""
    if isinstance(x, dict):
        return x
    out = empty_channels(size)
    out["combined"] = x
    return out


# ---------------------------------------------------------------------------
# Ring / spiral builders
# ---------------------------------------------------------------------------


def make_rings(
    size: int = 101,
    ring_radii: Tuple[float, ...] = (0.25, 0.5),
    ring_widths: Tuple[float, ...] = (0.05, 0.05),
    ring_amplitudes: Tuple[float, ...] = (1.0, 0.8),
    axis_ratio: float = 1.0,
    position_angle: float = 0.0,
    disk_scale: float = 1.0,
    ring_a_sin_list: Optional[List[Optional[List[float]]]] = None,
    ring_b_cos_list: Optional[List[Optional[List[float]]]] = None,
    asymmetry_square: bool = True,
) -> np.ndarray:
    """Generate an image of elliptical rings.

    Parameters
    ----------
    size : int, optional
        Image size in pixels.
    ring_radii : tuple of float, optional
        Ring radii in **disk-relative units** ``[0, 1]``. A value of ``1.0``
        sits on the disk edge as defined by ``disk_scale``.
    ring_widths : tuple of float, optional
        Gaussian ring widths in disk-relative units.
    ring_amplitudes : tuple of float, optional
        Peak brightness of each ring.
    axis_ratio : float, optional
        Minor-to-major axis ratio.
    position_angle : float, optional
        Major-axis position angle in radians.
    disk_scale : float, optional
        Normalised radius (in ``[-1, 1]`` image coordinates) at which a
        disk-relative radius of ``1.0`` sits. With ``disk_scale <= 1.0``
        the entire disk lies inside the image. Default is ``1.0``.
    ring_a_sin_list, ring_b_cos_list : list of (list of float or None), optional
        Per-ring Fourier coefficients for angular asymmetry.
    asymmetry_square : bool, optional
        Square the angular modulation map.

    Returns
    -------
    np.ndarray
        Ring image of shape ``(size, size)``.
    """
    xx, yy = make_coordinate_grid(size)
    r, theta = elliptical_polar(xx, yy, axis_ratio, position_angle)
    r = r / disk_scale  # r is now in disk-relative units

    n_rings = len(ring_radii)
    image = np.zeros_like(r)

    if ring_a_sin_list is None:
        ring_a_sin_list = [None] * n_rings
    if ring_b_cos_list is None:
        ring_b_cos_list = [None] * n_rings

    for r0, w, amp, a_sin, b_cos in zip(
        ring_radii,
        ring_widths,
        ring_amplitudes,
        ring_a_sin_list,
        ring_b_cos_list,
    ):
        ring = amp * np.exp(-0.5 * ((r - r0) / w) ** 2)
        mod = angular_asymmetry(
            theta,
            a_sin=a_sin,
            b_cos=b_cos,
            square=asymmetry_square,
            normalize_mean=True,
            clip_min=0.0,
        )
        image = image + ring * mod

    return image


def make_spiral(
    size: int = 101,
    ring_radius: float = 0.4,
    n_arms: int = 2,
    pitch: float = 0.3,
    ring_width: float = 0.05,
    ring_amplitude: float = 1.0,
    arm_width: float = 0.08,
    arm_amplitudes: Optional[List[float]] = None,
    axis_ratio: float = 1.0,
    position_angle: float = 0.0,
    disk_scale: float = 1.0,
    spiral_peak_offset: float = 0.10,
    spiral_radial_sigma: float = 0.20,
    normalize_output: bool = False,
    ring_a_sin: Optional[List[float]] = None,
    ring_b_cos: Optional[List[float]] = None,
    arm_a_sin_list: Optional[List[Optional[List[float]]]] = None,
    arm_b_cos_list: Optional[List[Optional[List[float]]]] = None,
    asymmetry_square: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate a spiral-galaxy image, returning components separately.

    The central ring is placed in the ``'rings'`` channel; the spiral arms
    are placed in the ``'spirals'`` channel; ``'combined'`` is their sum.
    The ``'planets'`` channel is zero.

    Parameters
    ----------
    size : int, optional
        Image size in pixels.
    ring_radius : float, optional
        Central ring radius in disk-relative units. Set to ``0`` for an
        exponential base disk instead.
    n_arms : int, optional
        Number of spiral arms.
    pitch : float, optional
        Pitch angle of the spiral arms.
    ring_width : float, optional
        Gaussian width of the central ring (disk-relative units).
    ring_amplitude : float, optional
        Brightness of the central ring or exponential disk.
    arm_width : float, optional
        Angular width of each spiral arm in radians.
    arm_amplitudes : list of float, optional
        Per-arm brightness amplitudes.
    axis_ratio : float, optional
        Minor-to-major axis ratio.
    position_angle : float, optional
        Major-axis position angle in radians.
    disk_scale : float, optional
        Normalised radius at which disk-relative radius ``1.0`` sits.
    spiral_peak_offset : float, optional
        Radial offset of peak arm brightness (disk-relative units) beyond
        the ring.
    spiral_radial_sigma : float, optional
        Radial Gaussian width of the spiral arm envelope.
    normalize_output : bool, optional
        Normalise the combined channel to a peak of 1.0. The same scale is
        applied to ring and arm channels so they remain consistent.
    ring_a_sin, ring_b_cos : list of float, optional
        Central-ring Fourier coefficients.
    arm_a_sin_list, arm_b_cos_list : list of (list of float or None), optional
        Per-arm Fourier coefficients.
    asymmetry_square : bool, optional
        Square the angular modulation maps.

    Returns
    -------
    dict of str -> np.ndarray
        Channel dict with keys ``'combined'``, ``'rings'``, ``'spirals'``,
        ``'planets'``.
    """
    xx, yy = make_coordinate_grid(size)
    r, theta = elliptical_polar(xx, yy, axis_ratio, position_angle)
    r = r / disk_scale

    # --- central ring (goes in the rings channel) ---
    if ring_radius > 0:
        ring_channel = make_rings(
            size=size,
            ring_radii=[ring_radius],
            ring_widths=[ring_width],
            ring_amplitudes=[ring_amplitude],
            axis_ratio=axis_ratio,
            position_angle=position_angle,
            disk_scale=disk_scale,
            ring_a_sin_list=[ring_a_sin],
            ring_b_cos_list=[ring_b_cos],
            asymmetry_square=asymmetry_square,
        )
        r_ref = ring_radius
    else:
        # exponential disk as the "ring" channel
        ring_channel = ring_amplitude * np.exp(-r / (ring_width + 1e-6))
        ring_channel = ring_channel * angular_asymmetry(
            theta,
            a_sin=ring_a_sin,
            b_cos=ring_b_cos,
            square=asymmetry_square,
            normalize_mean=True,
            clip_min=0.0,
        )
        r_ref = 1e-3

    # --- spiral arms (goes in the spirals channel) ---
    if arm_amplitudes is None:
        arm_amplitudes = [1.0] * n_arms
    if arm_a_sin_list is None:
        arm_a_sin_list = [None] * n_arms
    if arm_b_cos_list is None:
        arm_b_cos_list = [None] * n_arms

    dr = r - r_ref
    # Smooth sigmoid step at the ring edge: transition width matches the
    # ring (or exponential-disk) scale so the spiral envelope joins
    # continuously with the ring profile instead of starting with a hard
    # edge.
    transition_width = 0.5 * (
        ring_width if ring_radius > 0 else spiral_radial_sigma
    )
    transition_width = max(float(transition_width), 1e-3)
    outward_mask = 1.0 / (1.0 + np.exp(-dr / transition_width))
    radial_env = np.exp(
        -0.5 * ((dr - spiral_peak_offset) / (spiral_radial_sigma + 1e-6)) ** 2
    )

    spiral_channel = np.zeros_like(r)
    for i in range(n_arms):
        theta_offset = 2 * np.pi * i / n_arms
        theta_spiral = (1 / pitch) * np.log((r + 1e-6) / r_ref)
        delta_theta = np.arctan2(
            np.sin(theta - theta_spiral - theta_offset),
            np.cos(theta - theta_spiral - theta_offset),
        )
        arm = np.exp(-0.5 * (delta_theta / arm_width) ** 2)
        arm_mod = angular_asymmetry(
            theta,
            a_sin=arm_a_sin_list[i],
            b_cos=arm_b_cos_list[i],
            square=asymmetry_square,
            normalize_mean=True,
            clip_min=0.0,
        )
        spiral_channel = spiral_channel + arm_amplitudes[i] * arm * arm_mod

    spiral_channel = spiral_channel * outward_mask * radial_env

    combined = ring_channel + spiral_channel

    if normalize_output:
        peak = combined.max()
        peak = np.where(peak > 0, peak, 1.0)
        ring_channel = ring_channel / peak
        spiral_channel = spiral_channel / peak
        combined = combined / peak

    return {
        "combined": combined,
        "rings": ring_channel,
        "spirals": spiral_channel,
        "planets": np.zeros_like(combined),
    }


# ---------------------------------------------------------------------------
# Planets
# ---------------------------------------------------------------------------


def _planet_layer(
    size: int,
    planet_radius: float,
    planet_angle: float,
    planet_amplitude: float,
    planet_sigma: float,
    disk_scale: float = 1.0,
) -> np.ndarray:
    """Return a single-planet Gaussian as its own 2D layer."""
    xx, yy = make_coordinate_grid(size)
    pr = (
        planet_radius * disk_scale
    )  # match make_rings/make_spiral: image-coord radius = param * disk_scale
    px = pr * np.cos(planet_angle)
    py = pr * np.sin(planet_angle)
    return planet_amplitude * np.exp(
        -((xx - px) ** 2 + (yy - py) ** 2) / (2 * planet_sigma**2)
    )


def add_planet(
    image: Union[np.ndarray, Dict[str, np.ndarray]],
    planet_radius: float = 0.5,
    planet_angle: float = 0.0,
    planet_amplitude: float = 1.0,
    planet_sigma: float = 1.0,
    disk_scale: float = 1.0,
    cap_to_image_max: bool = True,
    remove: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Add a Gaussian planet to an image or channel dict.

    If a channel dict is passed, the planet is added to the ``'planets'``
    channel **and** the ``'combined'`` channel; the rings and spirals
    channels are unchanged.

    Parameters
    ----------
    image : np.ndarray or dict of str -> np.ndarray
        Existing 2D image, or a channel dict.
    planet_radius : float, optional
        Planet location, multiplied by ``disk_scale`` to get image
        coordinates (same convention as ``make_rings``).
    planet_angle : float, optional
        Planet angular location in radians.
    planet_amplitude : float, optional
        Planet peak brightness.
    planet_sigma : float, optional
        Planet Gaussian width in image-coordinate units.
    disk_scale : float, optional
        Radial scale factor (``planet_radius`` is multiplied by this).
    cap_to_image_max : bool, optional
        Cap the planet contribution so the combined image does not exceed
        its current maximum.
    remove : bool, optional
        Subtract rather than add the planet.

    Returns
    -------
    np.ndarray or dict
        Same type as the input.
    """
    is_dict = isinstance(image, dict)
    if is_dict:
        size = image["combined"].shape[0]
        combined_in = image["combined"]
    else:
        size = image.shape[0]
        combined_in = image

    planet = _planet_layer(
        size=size,
        planet_radius=planet_radius,
        planet_angle=planet_angle,
        planet_amplitude=planet_amplitude,
        planet_sigma=planet_sigma,
        disk_scale=disk_scale,
    )

    if remove:
        planet = planet * -1.0

    if cap_to_image_max:
        max_val = combined_in.max()
        max_val = np.where(max_val == 0.0, 1.0, max_val)
        planet = np.minimum(planet, np.maximum(max_val - combined_in, 0))

    if is_dict:
        out = dict(image)
        out["planets"] = out["planets"] + planet
        out["combined"] = out["combined"] + planet
        return out
    return combined_in + planet


# ---------------------------------------------------------------------------
# Fourier-coefficient sampling
# ---------------------------------------------------------------------------


def sample_fourier_coeffs(
    key: jax.Array,
    max_modes: int = 3,
    coeff_range: Tuple[float, float] = (-0.3, 0.3),
    prob_nonzero: float = 0.7,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Sample random Fourier coefficients for angular asymmetry."""
    k0, k1, k2 = jr.split(key, 3)

    if not bool(jr.bernoulli(k0, p=prob_nonzero)):
        return None, None

    n_modes = int(jr.randint(k1, shape=(), minval=1, maxval=max_modes + 1))
    coeffs = jr.uniform(
        k2,
        shape=(2, n_modes),
        minval=coeff_range[0],
        maxval=coeff_range[1],
    )

    a_sin = [0.0] + [float(x) for x in coeffs[0]]
    b_cos = [0.0] + [float(x) for x in coeffs[1]]
    return a_sin, b_cos


# ---------------------------------------------------------------------------
# Random ring / spiral wrappers (convenience)
# ---------------------------------------------------------------------------


def random_rings(key: jax.Array, size: int = 128) -> np.ndarray:
    """Generate a random ring image (single combined array)."""
    k0, k1, k2, k3, k4, k5, *arm_keys = jr.split(key, 20)

    n_rings = int(jr.randint(k0, shape=(), minval=1, maxval=5))
    ring_radii = np.sort(
        jr.uniform(k1, shape=(n_rings,), minval=0.1, maxval=0.9)
    )
    ring_widths = jr.uniform(k2, shape=(n_rings,), minval=0.02, maxval=0.08)
    ring_amplitudes = jr.uniform(k3, shape=(n_rings,), minval=0.3, maxval=1.0)
    axis_ratio = float(jr.uniform(k4, minval=0.3, maxval=1.0))
    position_angle = float(jr.uniform(k5, minval=0.0, maxval=2 * np.pi))
    disk_scale = float(jr.uniform(arm_keys[0], minval=0.7, maxval=1.0))

    ring_a_sin_list = []
    ring_b_cos_list = []
    for i in range(n_rings):
        a, b = sample_fourier_coeffs(
            arm_keys[i + 1], max_modes=4, prob_nonzero=0.8
        )
        ring_a_sin_list.append(a)
        ring_b_cos_list.append(b)

    return make_rings(
        size=size,
        ring_radii=ring_radii,
        ring_widths=ring_widths,
        ring_amplitudes=ring_amplitudes,
        axis_ratio=axis_ratio,
        position_angle=position_angle,
        disk_scale=disk_scale,
        ring_a_sin_list=ring_a_sin_list,
        ring_b_cos_list=ring_b_cos_list,
    )


def random_spiral(key: jax.Array, size: int = 128) -> Dict[str, np.ndarray]:
    """Generate a random spiral-galaxy channel dict."""
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, *arm_keys = jr.split(key, 30)

    n_arms = int(jr.randint(k0, shape=(), minval=1, maxval=5))
    arm_amplitudes = list(
        jr.uniform(k1, shape=(n_arms,), minval=0.3, maxval=1.5)
    )
    axis_ratio = float(jr.uniform(k2, minval=0.3, maxval=1.0))
    position_angle = float(jr.uniform(k3, minval=0.0, maxval=2 * np.pi))
    disk_scale = float(jr.uniform(k4, minval=0.7, maxval=1.0))

    arm_a_sin_list = []
    arm_b_cos_list = []
    for i in range(n_arms):
        a, b = sample_fourier_coeffs(arm_keys[i], max_modes=3, prob_nonzero=0.7)
        arm_a_sin_list.append(a)
        arm_b_cos_list.append(b)

    ring_a_sin, ring_b_cos = sample_fourier_coeffs(
        arm_keys[n_arms], max_modes=3, prob_nonzero=0.8
    )

    return make_spiral(
        size=size,
        ring_radius=float(jr.uniform(k5, minval=0.2, maxval=0.6)),
        n_arms=n_arms,
        pitch=float(jr.uniform(k6, minval=0.15, maxval=0.5)),
        ring_width=float(jr.uniform(k7, minval=0.03, maxval=0.08)),
        ring_amplitude=float(jr.uniform(k8, minval=0.5, maxval=1.2)),
        arm_width=float(jr.uniform(k9, minval=0.05, maxval=0.15)),
        arm_amplitudes=arm_amplitudes,
        axis_ratio=axis_ratio,
        position_angle=position_angle,
        disk_scale=disk_scale,
        spiral_peak_offset=float(
            jr.uniform(arm_keys[n_arms + 1], minval=0.05, maxval=0.2)
        ),
        spiral_radial_sigma=float(
            jr.uniform(arm_keys[n_arms + 2], minval=0.1, maxval=0.3)
        ),
        ring_a_sin=ring_a_sin,
        ring_b_cos=ring_b_cos,
        arm_a_sin_list=arm_a_sin_list,
        arm_b_cos_list=arm_b_cos_list,
        normalize_output=True,
    )


# ---------------------------------------------------------------------------
# Main random-object factory
# ---------------------------------------------------------------------------


def random_obj(
    key: jax.Array,
    return_parts: bool = False,
    normalize: bool = True,
    power_val: float = 1.25,
    power_start: float = 0.6,
    size: int = 101,
    disc_scale: Tuple[float, float] = (0.7, 1.3),
    position_angle_range: Tuple[float, float] = (0, 2 * np.pi),
    axis_ratio_range: Tuple[float, float] = (0.3, 1.0),
    ring_range: Tuple[int, int] = (1, 6),
    ring_radii_range: Tuple[float, float] = (0.2, 0.9),
    ring_width_range: Tuple[float, float] = (0.05, 0.2),
    ring_amplitude_range: Tuple[float, float] = (0.125, 0.5),
    arm_range: Tuple[int, int] = (1, 5),
    arm_width_range: Tuple[float, float] = (0.05, 0.3),
    arm_amplitude_range: Tuple[float, float] = (0.125, 0.5),
    spiral_peak_offset: Tuple[float, float] = (0.05, 0.6),
    spiral_radial_sigma: Tuple[float, float] = (0.1, 0.4),
    pitch_range: Tuple[float, float] = (-1.0, 1.0),
    planets_range: Tuple[int, int] = (0, 3),
    planet_angular_position_range: Tuple[float, float] = (0, 2 * np.pi),
    planet_orbital_radius_range: Tuple[float, float] = (0.1, 0.9),
    planet_amplitude_range: Tuple[float, float] = (0.0, 2.0),
    planet_sigma_range: Tuple[float, float] = (0.01, 0.0625),
    use_asymmetry: bool = True,
    max_fourier_modes: int = 3,
    fourier_coeff_range: Tuple[float, float] = (-0.1, 0.1),
    per_ring_asymmetry_prob: float = 0.5,
    per_arm_asymmetry_prob: float = 0.5,
    ring_base_asymmetry_prob: float = 0.5,
    asymmetry_square: bool = True,
    max_fov_fill: float = 0.9,
    containment_sigma: float = 3.0,
    planet_brightness_cap: float = 0.5,
    verbose: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Randomly generate a synthetic protoplanetary-disk image.

    Produces either a ringed disk or a spiral disk, optionally with planets.
    Returns either a single combined image or a channel dict suitable for
    multi-channel autoencoder training.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    return_parts : bool, optional
        If ``False`` (default) return only the combined image as a 2D array.
        If ``True`` return a dict with keys ``'combined'``, ``'rings'``,
        ``'spirals'``, ``'planets'``.
    normalize : bool, optional
        Deprecated / unused. The priority-cap combine logic now always
        guarantees the combined image peaks at <= 1.0 without a global
        renormalisation step. Kept for back-compat.
    size : int, optional
        Image size in pixels.
    disc_scale : tuple of float, optional
        Range ``(min, max)`` for the radial scale factor: feature image-coord
        radii are ``param_value * disk_scale``. After sampling, this is
        automatically reduced if needed so the outermost feature stays inside
        ``max_fov_fill`` of the image. Default is ``(0.7, 1.3)``.
    ring_radii_range, ring_width_range, ring_amplitude_range : tuple of float, optional
        Ranges for ring radii, Gaussian widths, and amplitudes. Radii and
        widths are in pre-``disk_scale`` units; the effective image-coord
        extent is the value divided by ``disk_scale``.
    arm_range, arm_width_range, arm_amplitude_range : optional
        Spiral arm count, angular widths (radians), and amplitudes.
    spiral_peak_offset, spiral_radial_sigma : tuple of float, optional
        Ranges for the radial offset and Gaussian width of the spiral arm
        envelope (pre-``disk_scale`` units).
    pitch_range : tuple of float, optional
        Spiral pitch range.
    planets_range, planet_*_range : optional
        Planet count, angular position (radians), orbital radius
        (pre-``disk_scale`` units), amplitude, and sigma (image-coord units).
    use_asymmetry, max_fourier_modes, fourier_coeff_range : optional
        Angular asymmetry controls.
    per_ring_asymmetry_prob, per_arm_asymmetry_prob, ring_base_asymmetry_prob :
        Probabilities of applying asymmetry to rings, arms, and the spiral
        central ring.
    asymmetry_square : bool, optional
        Square the angular modulation map.
    max_fov_fill : float, optional
        Maximum fraction of the image half-width that any feature may occupy.
        The sampled ``disc_scale_val`` is increased post-hoc so that the
        outermost feature (rings + N sigma, spiral envelope + N sigma, or
        planet position + N sigma) lies inside this radius. Default is
        ``0.9``.
    containment_sigma : float, optional
        Multiplier on Gaussian widths used when computing the outermost
        radial extent for containment. Default is ``3.0`` (3-sigma).
    planet_brightness_cap : float, optional
        Peak brightness of the planets channel relative to the disk
        (which is normalised to peak 1.0). Default ``0.5``. Planets are
        added to combined rather than capped against the disk, so a
        planet on top of the disk creates a bright spot; the combined
        image may exceed 1.0 at overlap pixels. Increase for more
        prominent planets; decrease to suppress them.
    verbose : bool, optional
        If ``True``, print all sampled parameters (mode, geometry, ring/arm
        values, planets, asymmetry coefficients, image stats) for this draw.
        The same report is also printed automatically when a blank image is
        detected, regardless of this flag. Default is ``False``.

    Returns
    -------
    np.ndarray or dict of str -> np.ndarray
        See ``return_parts``.

    Notes
    -----
    Not jit-compatible: uses Python-level branching and variable-length lists.
    Ring-mode vs spiral-mode probability is fixed at 30 % / 70 %.
    """
    keys = jr.split(key, 64)

    # --- global disk geometry ---
    disc_scale_val = float(
        jr.uniform(keys[0], minval=disc_scale[0], maxval=disc_scale[1])
    )
    position_angle = float(
        jr.uniform(
            keys[1],
            minval=position_angle_range[0],
            maxval=position_angle_range[1],
        )
    )
    axis_ratio = float(
        jr.uniform(
            keys[2], minval=axis_ratio_range[0], maxval=axis_ratio_range[1]
        )
    )
    rings = bool(jr.bernoulli(keys[3], p=0.3))

    # Defaults used in the blank-image diagnostic and containment calc
    n_rings = 0
    ring_radii = ring_widths = ring_amplitudes = None
    ring_radius = ring_width = ring_amplitude = 0.0
    n_arms = 0
    arm_width = spiral_peak_offset_val = spiral_radial_sigma_val = pitch = 0.0
    arm_amplitudes = None
    ring_a_sin_list = ring_b_cos_list = None
    ring_a_sin = ring_b_cos = None
    arm_a_sin_list = arm_b_cos_list = None

    # =====================================================================
    # Phase 1: sample all radial parameters (rings, spirals, planets)
    # =====================================================================

    # --- ring mode: sample params ---
    if rings:
        n_rings = int(
            jr.randint(
                keys[4],
                shape=(),
                minval=ring_range[0],
                maxval=ring_range[1] + 1,
            )
        )
        ring_radii = np.sort(
            jr.uniform(
                keys[5],
                shape=(n_rings,),
                minval=ring_radii_range[0],
                maxval=ring_radii_range[1],
            )
        )
        ring_widths = jr.uniform(
            keys[6],
            shape=(n_rings,),
            minval=ring_width_range[0],
            maxval=ring_width_range[1],
        )
        ring_amplitudes = jr.uniform(
            keys[7],
            shape=(n_rings,),
            minval=ring_amplitude_range[0],
            maxval=ring_amplitude_range[1],
        )

        if use_asymmetry:
            ring_a_sin_list, ring_b_cos_list = [], []
            asym_keys = jr.split(keys[8], max(n_rings, 1))
            for i in range(n_rings):
                a, b = sample_fourier_coeffs(
                    asym_keys[i],
                    max_modes=max_fourier_modes,
                    coeff_range=tuple(fourier_coeff_range),
                    prob_nonzero=per_ring_asymmetry_prob,
                )
                ring_a_sin_list.append(a)
                ring_b_cos_list.append(b)

    # --- spiral mode: sample params ---
    else:
        ring_radius = float(
            jr.uniform(
                keys[9], minval=ring_radii_range[0], maxval=ring_radii_range[1]
            )
        )
        ring_width = float(
            jr.uniform(
                keys[10], minval=ring_width_range[0], maxval=ring_width_range[1]
            )
        )
        ring_amplitude = float(
            jr.uniform(
                keys[11],
                minval=ring_amplitude_range[0],
                maxval=ring_amplitude_range[1],
            )
        )
        n_arms = int(
            jr.randint(
                keys[12], shape=(), minval=arm_range[0], maxval=arm_range[1] + 1
            )
        )
        arm_width = float(
            jr.uniform(
                keys[13], minval=arm_width_range[0], maxval=arm_width_range[1]
            )
        )
        spiral_peak_offset_val = float(
            jr.uniform(
                keys[14],
                minval=spiral_peak_offset[0],
                maxval=spiral_peak_offset[1],
            )
        )
        spiral_radial_sigma_val = float(
            jr.uniform(
                keys[15],
                minval=spiral_radial_sigma[0],
                maxval=spiral_radial_sigma[1],
            )
        )
        pitch = float(
            jr.uniform(keys[16], minval=pitch_range[0], maxval=pitch_range[1])
        )
        arm_amplitudes = (
            jr.uniform(
                keys[17],
                shape=(n_arms,),
                minval=arm_amplitude_range[0],
                maxval=arm_amplitude_range[1],
            )
            if n_arms > 0
            else np.zeros((0,))
        )

        if use_asymmetry:
            ring_a_sin, ring_b_cos = sample_fourier_coeffs(
                keys[18],
                max_modes=max_fourier_modes,
                coeff_range=tuple(fourier_coeff_range),
                prob_nonzero=ring_base_asymmetry_prob,
            )
            arm_a_sin_list, arm_b_cos_list = [], []
            if n_arms > 0:
                arm_keys = jr.split(keys[19], n_arms)
                for i in range(n_arms):
                    a, b = sample_fourier_coeffs(
                        arm_keys[i],
                        max_modes=max_fourier_modes,
                        coeff_range=tuple(fourier_coeff_range),
                        prob_nonzero=per_arm_asymmetry_prob,
                    )
                    arm_a_sin_list.append(a)
                    arm_b_cos_list.append(b)

    # --- planets: sample params ---
    n_planets = int(
        jr.randint(
            keys[20],
            shape=(),
            minval=planets_range[0],
            maxval=planets_range[1] + 1,
        )
    )
    planet_angular_positions = planet_orbital_radii = planet_amplitudes = (
        planet_sigmas
    ) = None

    if n_planets > 0:
        planet_angular_positions = jr.uniform(
            keys[21],
            shape=(n_planets,),
            minval=planet_angular_position_range[0],
            maxval=planet_angular_position_range[1],
        )
        planet_orbital_radii = jr.uniform(
            keys[22],
            shape=(n_planets,),
            minval=planet_orbital_radius_range[0],
            maxval=planet_orbital_radius_range[1],
        )
        planet_amplitudes = jr.uniform(
            keys[23],
            shape=(n_planets,),
            minval=planet_amplitude_range[0],
            maxval=planet_amplitude_range[1],
        )
        planet_sigmas = jr.uniform(
            keys[24],
            shape=(n_planets,),
            minval=planet_sigma_range[0],
            maxval=planet_sigma_range[1],
        )

    # =====================================================================
    # Phase 2: shrink disc_scale_val if needed to keep features in the FOV
    # =====================================================================
    # In make_rings/make_spiral the image-coord radial extent of a feature
    # is `feature_radius * disk_scale` (because the code does
    # `r = r_image / disk_scale` and then evaluates Gaussians in those
    # units). So to keep a feature inside `max_fov_fill` of the image we
    # need disk_scale <= max_fov_fill / outermost_radius. We compute the
    # outermost (peak + containment_sigma*width) over all features, then
    # clamp disc_scale_val downward if needed.

    max_allowed_disc_scale = float("inf")

    if rings and n_rings > 0:
        r_outer_rings = float(ring_radii.max()) + containment_sigma * float(
            ring_widths.max()
        )
        if r_outer_rings > 0:
            max_allowed_disc_scale = min(
                max_allowed_disc_scale, max_fov_fill / r_outer_rings
            )

    if not rings:
        # central ring of the spiral
        r_outer_central = ring_radius + containment_sigma * ring_width
        # spiral arm envelope: peak at ring_radius + spiral_peak_offset_val
        r_outer_arms = (
            ring_radius
            + spiral_peak_offset_val
            + containment_sigma * spiral_radial_sigma_val
        )
        r_outer_spiral = max(r_outer_central, r_outer_arms)
        if r_outer_spiral > 0:
            max_allowed_disc_scale = min(
                max_allowed_disc_scale, max_fov_fill / r_outer_spiral
            )

    if n_planets > 0:
        # In add_planet the planet is placed at image-coord
        # (planet_radius * disk_scale, planet_angle) with width planet_sigma
        # (already in image-coord units). Containment:
        #   planet_radius * disk_scale + containment_sigma * planet_sigma
        #     <= max_fov_fill
        # => disk_scale <= (max_fov_fill - containment_sigma*planet_sigma)
        #                  / planet_radius
        for radius, sigma in zip(planet_orbital_radii, planet_sigmas):
            r_pl = float(radius)
            s_pl = float(sigma)
            slack = max_fov_fill - containment_sigma * s_pl
            if r_pl > 1e-6 and slack > 0:
                max_allowed_disc_scale = min(
                    max_allowed_disc_scale, slack / r_pl
                )
            # if slack <= 0 the planet itself is wider than the FOV
            # allowance; planet_sigma_range max (0.0625) keeps this safe.

    if max_allowed_disc_scale < disc_scale_val:
        disc_scale_val = max_allowed_disc_scale

    # =====================================================================
    # Phase 3: render each part-channel independently
    # =====================================================================
    # Each of rings/spirals/planets is built as its own 2D array. They are
    # NOT yet summed into the combined image — that happens after per-part
    # normalisation in phase 4, so that no single component dominates the
    # final combined brightness.

    zeros = np.zeros((size, size))
    rings_layer = zeros
    spirals_layer = zeros
    planets_layer = zeros

    if rings:
        rings_layer = make_rings(
            size=size,
            disk_scale=disc_scale_val,
            position_angle=position_angle,
            axis_ratio=axis_ratio,
            ring_radii=ring_radii,
            ring_widths=ring_widths,
            ring_amplitudes=ring_amplitudes,
            ring_a_sin_list=ring_a_sin_list,
            ring_b_cos_list=ring_b_cos_list,
            asymmetry_square=asymmetry_square,
        )
    else:
        spiral_channels = make_spiral(
            size=size,
            disk_scale=disc_scale_val,
            position_angle=position_angle,
            axis_ratio=axis_ratio,
            ring_radius=ring_radius,
            ring_width=ring_width,
            ring_amplitude=ring_amplitude,
            n_arms=n_arms,
            arm_width=arm_width,
            arm_amplitudes=arm_amplitudes,
            spiral_peak_offset=spiral_peak_offset_val,
            spiral_radial_sigma=spiral_radial_sigma_val,
            pitch=pitch,
            ring_a_sin=ring_a_sin,
            ring_b_cos=ring_b_cos,
            arm_a_sin_list=arm_a_sin_list,
            arm_b_cos_list=arm_b_cos_list,
            asymmetry_square=asymmetry_square,
        )
        rings_layer = spiral_channels["rings"]
        spirals_layer = spiral_channels["spirals"]

    if n_planets > 0:
        # Pristine planet Gaussians, summed (no capping).
        planets_layer = np.zeros((size, size))
        for angle, radius, amp, sigma in zip(
            planet_angular_positions,
            planet_orbital_radii,
            planet_amplitudes,
            planet_sigmas,
        ):
            planets_layer = planets_layer + _planet_layer(
                size=size,
                planet_radius=float(radius),
                planet_angle=float(angle),
                planet_amplitude=float(amp),
                planet_sigma=float(sigma),
                disk_scale=disc_scale_val,
            )

    # =====================================================================
    # Phase 4: per-part normalisation with bounded planet brightness, then sum
    # =====================================================================
    # Each part-channel is built independently, then:
    #   * rings   -> normalised to peak 1.0
    #   * spirals -> normalised to peak 1.0
    #   * planets -> normalised to peak ``planet_brightness_cap`` (default 0.5)
    # combined = rings + spirals + planets, with no further renormalisation.
    #
    # Implications:
    #   * Strict additivity holds exactly.
    #   * The disk (rings + spirals) is never dimmed by overlap with a planet.
    #   * Where a planet sits on the disk, the combined image gets BRIGHTER
    #     at that spot (additive, not occluding). Combined.max() may exceed
    #     1.0 in overlap regions; for visualisation use vmax=1 (saturates
    #     overlaps) or per-image normalise at load time for training if
    #     desired.

    def _norm_to(x, target_peak):
        peak = x.max()
        return np.where(
            peak > 1e-8,
            x * (target_peak / np.where(peak > 0, peak, 1.0)),
            x,
        )

    rings_layer = _norm_to(rings_layer, 1.0)
    spirals_layer = _norm_to(spirals_layer, 1.0)
    planets_layer = _norm_to(planets_layer, planet_brightness_cap)

    combined = rings_layer + spirals_layer + planets_layer

    channels = {
        "combined": combined,
        "rings": rings_layer,
        "spirals": spirals_layer,
        "planets": planets_layer,
    }

    # --- parameter report (on verbose, or automatically if blank) ---
    max_val = float(channels["combined"].max())
    min_val = float(channels["combined"].min())

    def _print_params(header):
        print(f"\n--- {header} ---")
        print(f"MODE: {'RING' if rings else 'SPIRAL'}")
        print(
            f"GEOMETRY | disc_scale: {disc_scale_val:.4f} | axis_ratio: {axis_ratio:.4f} | position_angle: {position_angle:.4f}"
        )
        print(
            f"FOV | max_fov_fill: {max_fov_fill} | containment_sigma: {containment_sigma}"
        )
        if rings:
            print(
                f"RINGS | n_rings: {n_rings} | radii: {ring_radii} | widths: {ring_widths} | amplitudes: {ring_amplitudes}"
            )
            print(
                f"RING ASYMMETRY | a_sin: {ring_a_sin_list} | b_cos: {ring_b_cos_list}"
            )
        else:
            print(
                f"BASE RING | ring_radius: {ring_radius:.4f} | ring_width: {ring_width:.4f} | ring_amplitude: {ring_amplitude:.4f}"
            )
            print(
                f"ARMS | n_arms: {n_arms} | arm_width: {arm_width:.4f} | pitch: {pitch:.4f} | amplitudes: {arm_amplitudes}"
            )
            print(
                f"ENVELOPE | spiral_peak_offset: {spiral_peak_offset_val:.4f} | spiral_radial_sigma: {spiral_radial_sigma_val:.4f}"
            )
            print(
                f"BASE-RING ASYMMETRY | a_sin: {ring_a_sin} | b_cos: {ring_b_cos}"
            )
            print(
                f"ARM ASYMMETRY | a_sin: {arm_a_sin_list} | b_cos: {arm_b_cos_list}"
            )
        print(
            f"PLANETS | n_planets: {n_planets} | brightness_cap: {planet_brightness_cap}"
        )
        if n_planets > 0:
            print(
                f"  angles: {planet_angular_positions} | radii: {planet_orbital_radii} | amplitudes: {planet_amplitudes} | sigmas: {planet_sigmas}"
            )
        print(f"IMAGE STATS | min: {min_val:.4f} | max: {max_val:.4f}")
        print("-" * (len(header) + 8) + "\n")

    if max_val < 1e-6:
        _print_params("Blank image detected")
    elif verbose:
        _print_params("random_obj parameters")

    if return_parts:
        return channels
    return channels["combined"]


# ---------------------------------------------------------------------------
# Down-binning utility (unchanged)
# ---------------------------------------------------------------------------


def _bin_2x2_odd(img: np.ndarray) -> np.ndarray:
    """Sum-pool a 2D image of odd size by 2x2 blocks.

    Input shape ``(n, n)`` with ``n`` odd is reduced to
    ``((n+1)//2, (n+1)//2)`` by summing 2x2 blocks in the even-sized
    ``(n-1, n-1)`` core, with the final row, final column, and corner
    pixel carried through unchanged (the final row/col are summed pairwise
    along their length, the corner is the single pixel).

    Parameters
    ----------
    img : np.ndarray
        2D array of shape ``(n, n)`` with ``n`` odd.

    Returns
    -------
    np.ndarray
        Binned 2D array of shape ``((n+1)//2, (n+1)//2)``.
    """
    n = img.shape[0]
    assert img.shape == (n, n), f"expected square input, got {img.shape}"
    assert n % 2 == 1, f"expected odd input size, got {n}"

    core_n = n - 1  # even portion
    out_n = (n + 1) // 2  # = core_n // 2 + 1
    core_out = core_n // 2

    core = (
        img[:core_n, :core_n].reshape(core_out, 2, core_out, 2).sum(axis=(1, 3))
    )
    last_row = img[core_n, :core_n].reshape(core_out, 2).sum(axis=1)
    last_col = img[:core_n, core_n].reshape(core_out, 2).sum(axis=1)

    out = np.zeros((out_n, out_n), dtype=img.dtype)
    out = out.at[:core_out, :core_out].set(core)
    out = out.at[core_out, :core_out].set(last_row)
    out = out.at[:core_out, core_out].set(last_col)
    out = out.at[core_out, core_out].set(img[core_n, core_n])
    return out


def _radial_taper(
    out_size: int, power_val: float, power_start: float
) -> np.ndarray:
    """Build a radial brightness taper map of shape ``(out_size, out_size)``.

    The taper is ``1.0`` inside a normalised radius of ``power_start`` and
    falls smoothly to ``0`` at the corners (normalised radius 1.0) with a
    power-law of exponent ``power_val``. Used to soften the image edge
    after binning.
    """
    xx, yy = make_coordinate_grid(out_size)
    r_edge = np.sqrt(xx**2 + yy**2)
    r_max = r_edge.max()
    t = r_edge / r_max

    taper = np.where(
        t < power_start,
        1.0,
        (1 - (t - power_start) / (1 - power_start)) ** power_val,
    )
    return np.clip(taper, 0.0, 1.0)


def bin_down(
    img,
    power_val: float = 1.25,
    power_start: float = 0.6,
    normalize: bool = True,
):
    """Down-bin a 2x2 image (or channel dict) with a soft radial taper.

    Accepts either a single 2D array of odd square shape, or a dict of
    such arrays (as produced by ``random_obj(return_parts=True)``). Output
    size is ``(n+1)//2`` per axis; e.g. ``101 -> 51``, ``51 -> 26``,
    ``201 -> 101``.

    For a single image the taper is applied and the result is normalised
    to peak 1.0 (when ``normalize`` is true).

    For a channel dict the SAME taper is applied to every channel, and
    every channel is divided by the SAME factor (the post-taper peak of
    ``'combined'``), so the inter-channel brightness ratios produced by
    ``random_obj`` are preserved.

    Parameters
    ----------
    img : np.ndarray or dict of str -> np.ndarray
        Single 2D image or channel dict. All arrays must be square with
        the same odd size.
    power_val : float, optional
        Exponent of the radial taper (steepness of the edge falloff).
        Set to ``0`` to skip the taper entirely. Default ``1.25``.
    power_start : float, optional
        Normalised radius (in ``[0, 1]``) at which the taper begins.
        Set to ``0`` to skip the taper entirely. Default ``0.6``.
    normalize : bool, optional
        If ``True``, divide by the peak so the output (or combined channel)
        peaks at 1.0. Default ``True``.

    Returns
    -------
    np.ndarray or dict
        Binned-down array or channel dict (matches input type).
    """
    # ---- dict path: bin each channel with shared taper + shared norm ----
    if isinstance(img, dict):
        binned = {k: _bin_2x2_odd(v) for k, v in img.items()}
        out_size = next(iter(binned.values())).shape[0]

        if power_val > 0 and power_start > 0:
            taper = _radial_taper(out_size, power_val, power_start)
            binned = {k: v * taper for k, v in binned.items()}

        if normalize:
            # Anchor to the combined channel's peak when available so the
            # relative brightness of rings vs spirals vs planets is
            # preserved; fall back to the per-dict global peak otherwise.
            if "combined" in binned:
                peak = binned["combined"].max()
            else:
                peak = max(float(v.max()) for v in binned.values())
            peak = np.where(peak > 0, peak, 1.0)
            binned = {k: v / peak for k, v in binned.items()}
        return binned

    # ---- single-image path ----
    out = _bin_2x2_odd(img)
    out_size = out.shape[0]

    if power_val > 0 and power_start > 0:
        out = out * _radial_taper(out_size, power_val, power_start)

    if normalize:
        peak = out.max()
        peak = np.where(peak > 0, peak, 1.0)
        out = out / peak
    return out
