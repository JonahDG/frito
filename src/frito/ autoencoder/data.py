import jax
import jax.random as jr
import jax.numpy as np
from math import pi
from typing import List, Optional, Tuple


def make_coordinate_grid(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a centered coordinate grid in range ``[-1, 1]``.

    Parameters
    ----------
    size : int
        Number of pixels along each axis.

    Returns
    -------
    xx : np.ndarray
        2D array of x coordinates, shape ``(size, size)``.
    yy : np.ndarray
        2D array of y coordinates, shape ``(size, size)``.
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
    """Convert Cartesian coordinates to elliptical polar coordinates.

    Parameters
    ----------
    xx : np.ndarray
        2D array of x coordinates.
    yy : np.ndarray
        2D array of y coordinates.
    axis_ratio : float, optional
        Ratio of minor to major axis. Default is ``1.0`` (circular).
    position_angle : float, optional
        Position angle of the major axis in radians. Default is ``0.0``.

    Returns
    -------
    r : np.ndarray
        Elliptical radius at each grid point.
    theta : np.ndarray
        Azimuthal angle at each grid point in radians.
    """
    cos_pa = np.cos(position_angle)
    sin_pa = np.sin(position_angle)
    x_rot = xx * cos_pa + yy * sin_pa
    y_rot = -xx * sin_pa + yy * cos_pa
    y_scaled = y_rot / axis_ratio
    r = np.sqrt(x_rot ** 2 + y_scaled ** 2)
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
    """Compute an angular modulation map from Fourier coefficients.

    Parameters
    ----------
    theta : np.ndarray
        2D array of azimuthal angles in radians.
    a_sin : list of float, optional
        Sine Fourier coefficients. Index ``n`` multiplies ``sin(n * theta)``.
        Index ``0`` is ignored. Default is ``None`` (no sine modes).
    b_cos : list of float, optional
        Cosine Fourier coefficients. Index ``n`` multiplies ``cos(n * theta)``.
        Index ``0`` is added as a constant offset. Default is ``None``
        (no cosine modes).
    square : bool, optional
        If ``True``, square the modulation map before returning. Default is
        ``True``.
    normalize_mean : bool, optional
        If ``True``, divide by the mean of the modulation map so the average
        value is 1. Default is ``True``.
    clip_min : float, optional
        Minimum value to clip the modulation map to before squaring and
        normalising. Default is ``0.0``.

    Returns
    -------
    mod : np.ndarray
        Angular modulation map, same shape as ``theta``.
    """
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
        mod = mod ** 2

    if normalize_mean:
        mod = mod / (mod.mean() + 1e-8)

    return mod


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
    """Generate a synthetic image of elliptical rings.

    Parameters
    ----------
    size : int, optional
        Image size in pixels. Default is ``101``.
    ring_radii : tuple of float, optional
        Normalised radial positions of each ring. Default is ``(0.25, 0.5)``.
    ring_widths : tuple of float, optional
        Gaussian width of each ring. Default is ``(0.05, 0.05)``.
    ring_amplitudes : tuple of float, optional
        Peak brightness of each ring. Default is ``(1.0, 0.8)``.
    axis_ratio : float, optional
        Ratio of minor to major axis. Default is ``1.0`` (circular).
    position_angle : float, optional
        Position angle of the major axis in radians. Default is ``0.0``.
    disk_scale : float, optional
        Global radial scale factor. Default is ``1.0``.
    ring_a_sin_list : list of list of float or None, optional
        Per-ring sine Fourier coefficients for angular asymmetry. Default is
        ``None`` (no asymmetry).
    ring_b_cos_list : list of list of float or None, optional
        Per-ring cosine Fourier coefficients for angular asymmetry. Default is
        ``None`` (no asymmetry).
    asymmetry_square : bool, optional
        If ``True``, square the angular modulation map. Default is ``True``.

    Returns
    -------
    image : np.ndarray
        Synthetic ring image of shape ``(size, size)``.
    """
    xx, yy = make_coordinate_grid(size)
    r, theta = elliptical_polar(xx, yy, axis_ratio, position_angle)
    r = r / disk_scale

    n_rings = len(ring_radii)
    image = np.zeros_like(r)

    if ring_a_sin_list is None:
        ring_a_sin_list = [None] * n_rings
    if ring_b_cos_list is None:
        ring_b_cos_list = [None] * n_rings

    for r0, w, amp, a_sin, b_cos in zip(
        ring_radii, ring_widths, ring_amplitudes,
        ring_a_sin_list, ring_b_cos_list,
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
) -> np.ndarray:
    """Generate a synthetic image of a spiral galaxy disk.

    Parameters
    ----------
    size : int, optional
        Image size in pixels. Default is ``101``.
    ring_radius : float, optional
        Normalised radius of the central ring. Set to ``0`` for an exponential
        disk instead. Default is ``0.4``.
    n_arms : int, optional
        Number of spiral arms. Default is ``2``.
    pitch : float, optional
        Pitch angle of the spiral arms. Default is ``0.3``.
    ring_width : float, optional
        Gaussian width of the central ring or exponential scale length.
        Default is ``0.05``.
    ring_amplitude : float, optional
        Peak brightness of the central ring or disk. Default is ``1.0``.
    arm_width : float, optional
        Angular width of each spiral arm in radians. Default is ``0.08``.
    arm_amplitudes : list of float, optional
        Per-arm brightness amplitudes. Default is ``1.0`` for all arms.
    axis_ratio : float, optional
        Ratio of minor to major axis. Default is ``1.0`` (circular).
    position_angle : float, optional
        Position angle of the major axis in radians. Default is ``0.0``.
    disk_scale : float, optional
        Global radial scale factor. Default is ``1.0``.
    spiral_peak_offset : float, optional
        Radial offset of peak spiral arm brightness beyond the ring. Default
        is ``0.10``.
    spiral_radial_sigma : float, optional
        Radial Gaussian width of the spiral arm envelope. Default is ``0.20``.
    normalize_output : bool, optional
        If ``True``, normalise the output image to a peak of ``1.0``. Default
        is ``False``.
    ring_a_sin : list of float, optional
        Sine Fourier coefficients for the central ring asymmetry. Default is
        ``None``.
    ring_b_cos : list of float, optional
        Cosine Fourier coefficients for the central ring asymmetry. Default is
        ``None``.
    arm_a_sin_list : list of list of float or None, optional
        Per-arm sine Fourier coefficients. Default is ``None``.
    arm_b_cos_list : list of list of float or None, optional
        Per-arm cosine Fourier coefficients. Default is ``None``.
    asymmetry_square : bool, optional
        If ``True``, square the angular modulation map. Default is ``True``.

    Returns
    -------
    image : np.ndarray
        Synthetic spiral image of shape ``(size, size)``.
    """
    xx, yy = make_coordinate_grid(size)
    r, theta = elliptical_polar(xx, yy, axis_ratio, position_angle)
    r = r / disk_scale

    if ring_radius > 0:
        base = make_rings(
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
        base = ring_amplitude * np.exp(-r / (ring_width + 1e-6))
        base = base * angular_asymmetry(
            theta,
            a_sin=ring_a_sin,
            b_cos=ring_b_cos,
            square=asymmetry_square,
            normalize_mean=True,
            clip_min=0.0,
        )
        r_ref = 1e-3

    image = base.copy()

    if arm_amplitudes is None:
        arm_amplitudes = [1.0] * n_arms
    if arm_a_sin_list is None:
        arm_a_sin_list = [None] * n_arms
    if arm_b_cos_list is None:
        arm_b_cos_list = [None] * n_arms

    dr = r - r_ref
    outward_mask = (dr > 0).astype(r.dtype)
    radial_env = np.exp(
        -0.5 * ((dr - spiral_peak_offset) / (spiral_radial_sigma + 1e-6)) ** 2
    )

    spiral_term = np.zeros_like(r)

    for i in range(n_arms):
        theta_offset = 2 * pi * i / n_arms
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
        spiral_term = spiral_term + arm_amplitudes[i] * arm * arm_mod

    image = image + spiral_term * outward_mask * radial_env

    if normalize_output:
        image = image / image.max()

    return image


def add_planet(
    image: np.ndarray,
    planet_radius: float = 0.5,
    planet_angle: float = 0.0,
    planet_amplitude: float = 1.0,
    planet_sigma: float = 1.0,
    disk_scale: float = 1.0,
    cap_to_image_max: bool = True,
    remove: bool = False,
) -> np.ndarray:
    """Add a Gaussian planet to an existing disk image.

    Parameters
    ----------
    image : np.ndarray
        Existing 2D image of shape ``(size, size)``.
    planet_radius : float, optional
        Normalised radial location of the planet. Default is ``0.5``.
    planet_angle : float, optional
        Angular location of the planet in radians. Default is ``0.0``.
    planet_amplitude : float, optional
        Peak brightness of the planet. Default is ``1.0``.
    planet_sigma : float, optional
        Gaussian width of the planet in normalised coordinate units. Default
        is ``1.0``.
    disk_scale : float, optional
        Radial scale factor, must match the disk generator. Default is ``1.0``.
    cap_to_image_max : bool, optional
        If ``True``, prevent the planet from exceeding the current maximum
        brightness in ``image``. Default is ``True``.
    remove : bool, optional
        If ``True``, subtract the planet instead of adding it. Default is
        ``False``.

    Returns
    -------
    image : np.ndarray
        Image with planet added (or removed), same shape as input.
    """
    size = image.shape[0]
    xx, yy = make_coordinate_grid(size)

    pr = planet_radius / disk_scale
    px = pr * np.cos(planet_angle)
    py = pr * np.sin(planet_angle)

    planet = planet_amplitude * np.exp(
        -((xx - px) ** 2 + (yy - py) ** 2) / (2 * planet_sigma ** 2)
    )

    if remove:
        planet = planet * -1.0

    if cap_to_image_max:
        max_val = image.max()
        if max_val == 0.0:
            max_val = 1.0
        planet = np.minimum(planet, np.maximum(max_val - image, 0))

    return image + planet


def sample_fourier_coeffs(
    key: jax.Array,
    max_modes: int = 3,
    coeff_range: Tuple[float, float] = (-0.3, 0.3),
    prob_nonzero: float = 0.7,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Sample random Fourier coefficients for angular asymmetry.

    Parameters
    ----------
    key : jax.Array
        PRNG key for random sampling.
    max_modes : int, optional
        Maximum number of Fourier modes to sample. Default is ``3``.
    coeff_range : tuple of float, optional
        Range ``(min, max)`` for uniform coefficient sampling. Default is
        ``(-0.3, 0.3)``.
    prob_nonzero : float, optional
        Probability that any modes are used at all. Default is ``0.7``.

    Returns
    -------
    a_sin : list of float or None
        Sine coefficients with a dummy ``n=0`` slot, or ``None`` if no modes
        are sampled.
    b_cos : list of float or None
        Cosine coefficients with a dummy ``n=0`` slot, or ``None`` if no
        modes are sampled.

    Notes
    -----
    This function is intended for eager use only and is not compatible with
    ``jax.jit``.
    """
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


def random_rings(key: jax.Array, size: int = 128) -> np.ndarray:
    """Generate a random synthetic ring image.

    Parameters
    ----------
    key : jax.Array
        PRNG key for random parameter sampling.
    size : int, optional
        Image size in pixels. Default is ``128``.

    Returns
    -------
    np.ndarray
        Synthetic ring image of shape ``(size, size)``.
    """
    k0, k1, k2, k3, k4, k5, *arm_keys = jr.split(key, 20)

    n_rings = int(jr.randint(k0, shape=(), minval=1, maxval=5))
    ring_radii = np.sort(jr.uniform(k1, shape=(n_rings,), minval=0.1, maxval=0.9))
    ring_widths = jr.uniform(k2, shape=(n_rings,), minval=0.02, maxval=0.08)
    ring_amplitudes = jr.uniform(k3, shape=(n_rings,), minval=0.3, maxval=1.0)
    axis_ratio = float(jr.uniform(k4, minval=0.3, maxval=1.0))
    position_angle = float(jr.uniform(k5, minval=0.0, maxval=2 * pi))
    disk_scale = float(jr.uniform(arm_keys[0], minval=0.7, maxval=1.3))

    ring_a_sin_list = []
    ring_b_cos_list = []
    for i in range(n_rings):
        a, b = sample_fourier_coeffs(arm_keys[i + 1], max_modes=4, prob_nonzero=0.8)
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


def random_spiral(key: jax.Array, size: int = 128) -> np.ndarray:
    """Generate a random synthetic spiral galaxy image.

    Parameters
    ----------
    key : jax.Array
        PRNG key for random parameter sampling.
    size : int, optional
        Image size in pixels. Default is ``128``.

    Returns
    -------
    np.ndarray
        Synthetic spiral image of shape ``(size, size)``.
    """
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, *arm_keys = jr.split(key, 30)

    n_arms = int(jr.randint(k0, shape=(), minval=1, maxval=5))
    arm_amplitudes = list(jr.uniform(k1, shape=(n_arms,), minval=0.3, maxval=1.5))
    axis_ratio = float(jr.uniform(k2, minval=0.3, maxval=1.0))
    position_angle = float(jr.uniform(k3, minval=0.0, maxval=2 * pi))
    disk_scale = float(jr.uniform(k4, minval=0.7, maxval=1.3))

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
        spiral_peak_offset=float(jr.uniform(arm_keys[n_arms + 1], minval=0.05, maxval=0.2)),
        spiral_radial_sigma=float(jr.uniform(arm_keys[n_arms + 2], minval=0.1, maxval=0.3)),
        ring_a_sin=ring_a_sin,
        ring_b_cos=ring_b_cos,
        arm_a_sin_list=arm_a_sin_list,
        arm_b_cos_list=arm_b_cos_list,
        normalize_output=True,
    )


def random_obj(
    key: jax.Array,
    label: bool = False,
    normalize: bool = True,
    power_val: float = 1.25,
    power_start: float = 0.6,
    size: int = 101,
    disc_scale: Tuple[float, float] = (0.25, 1.0),
    position_angle_range: Tuple[float, float] = (0, 2 * pi),
    axis_ratio_range: Tuple[float, float] = (0.15, 1.0),
    ring_range: Tuple[int, int] = (1, 5),
    ring_radii_range: Tuple[float, float] = (0.25, 1.5),
    ring_width_range: Tuple[float, float] = (0.25, 0.5),
    ring_amplitude_range: Tuple[float, float] = (0.125, 0.5),
    arm_range: Tuple[int, int] = (1, 5),
    arm_width_range: Tuple[float, float] = (0.25, 0.75),
    arm_amplitude_range: Tuple[float, float] = (0.125, 0.5),
    spiral_peak_offset: Tuple[float, float] = (0.0, 1.0),
    spiral_radial_sigma: Tuple[float, float] = (0.0, 1.0),
    pitch_range: Tuple[float, float] = (-1.0, 1.0),
    planets_range: Tuple[int, int] = (0, 3),
    planet_angular_position_range: Tuple[float, float] = (0, 2 * pi),
    planet_orbital_radius_range: Tuple[float, float] = (0.0, 1.0),
    planet_amplitude_range: Tuple[float, float] = (0.0, 2.0),
    planet_sigma_range: Tuple[float, float] = (0.01, 0.0625),
    use_asymmetry: bool = True,
    max_fourier_modes: int = 3,
    fourier_coeff_range: Tuple[float, float] = (-0.1, 0.1),
    per_ring_asymmetry_prob: float = 0.5,
    per_arm_asymmetry_prob: float = 0.5,
    ring_base_asymmetry_prob: float = 0.5,
    asymmetry_square: bool = True,
) -> np.ndarray:
    """Randomly generate a synthetic protoplanetary disk image.

    Produces either a ringed disk or a spiral disk, optionally with planets,
    using fully JAX-based random sampling.

    Parameters
    ----------
    key : jax.Array
        PRNG key for all random sampling.
    label : bool, optional
        Reserved for future label output. Default is ``False``.
    normalize : bool, optional
        If ``True``, normalise the output image to a peak of ``1.0``. Default
        is ``True``.
    power_val : float, optional
        Exponent for the radial power-law taper (currently unused). Default
        is ``1.25``.
    power_start : float, optional
        Normalised radius at which the power-law taper begins (currently
        unused). Default is ``0.6``.
    size : int, optional
        Image size in pixels. Default is ``101``.
    disc_scale : tuple of float, optional
        Range ``(min, max)`` for the global radial scale factor. Default is
        ``(0.25, 1.0)``.
    position_angle_range : tuple of float, optional
        Range ``(min, max)`` for the disk position angle in radians. Default
        is ``(0, 2π)``.
    axis_ratio_range : tuple of float, optional
        Range ``(min, max)`` for the minor-to-major axis ratio. Default is
        ``(0.15, 1.0)``.
    ring_range : tuple of int, optional
        Range ``(min, max)`` for the number of rings in ring mode. Default is
        ``(1, 5)``.
    ring_radii_range : tuple of float, optional
        Range ``(min, max)`` for ring radii in normalised units. Default is
        ``(0.25, 1.5)``.
    ring_width_range : tuple of float, optional
        Range ``(min, max)`` for Gaussian ring widths. Default is
        ``(0.25, 0.5)``.
    ring_amplitude_range : tuple of float, optional
        Range ``(min, max)`` for ring peak brightness. Default is
        ``(0.125, 0.5)``.
    arm_range : tuple of int, optional
        Range ``(min, max)`` for the number of spiral arms. Default is
        ``(1, 5)``.
    arm_width_range : tuple of float, optional
        Range ``(min, max)`` for spiral arm angular widths in radians. Default
        is ``(0.25, 0.75)``.
    arm_amplitude_range : tuple of float, optional
        Range ``(min, max)`` for spiral arm brightness. Default is
        ``(0.125, 0.5)``.
    spiral_peak_offset : tuple of float, optional
        Range ``(min, max)`` for the radial offset of peak arm brightness.
        Default is ``(0.0, 1.0)``.
    spiral_radial_sigma : tuple of float, optional
        Range ``(min, max)`` for the radial Gaussian width of the arm
        envelope. Default is ``(0.0, 1.0)``.
    pitch_range : tuple of float, optional
        Range ``(min, max)`` for the spiral pitch angle. Default is
        ``(-1.0, 1.0)``.
    planets_range : tuple of int, optional
        Range ``(min, max)`` for the number of planets. Default is ``(0, 3)``.
    planet_angular_position_range : tuple of float, optional
        Range ``(min, max)`` for planet angular positions in radians. Default
        is ``(0, 2π)``.
    planet_orbital_radius_range : tuple of float, optional
        Range ``(min, max)`` for planet orbital radii in normalised units.
        Default is ``(0.0, 1.0)``.
    planet_amplitude_range : tuple of float, optional
        Range ``(min, max)`` for planet peak brightness. Default is
        ``(0.0, 2.0)``.
    planet_sigma_range : tuple of float, optional
        Range ``(min, max)`` for planet Gaussian widths. Default is
        ``(0.01, 0.0625)``.
    use_asymmetry : bool, optional
        If ``True``, apply random Fourier angular asymmetry to rings and arms.
        Default is ``True``.
    max_fourier_modes : int, optional
        Maximum number of Fourier modes for angular asymmetry. Default is
        ``3``.
    fourier_coeff_range : tuple of float, optional
        Range ``(min, max)`` for Fourier coefficient sampling. Default is
        ``(-0.1, 0.1)``.
    per_ring_asymmetry_prob : float, optional
        Probability of applying asymmetry to each ring. Default is ``0.5``.
    per_arm_asymmetry_prob : float, optional
        Probability of applying asymmetry to each spiral arm. Default is
        ``0.5``.
    ring_base_asymmetry_prob : float, optional
        Probability of applying asymmetry to the spiral base ring. Default is
        ``0.5``.
    asymmetry_square : bool, optional
        If ``True``, square the angular modulation map. Default is ``True``.

    Returns
    -------
    np.ndarray
        Synthetic disk image of shape ``(size, size)``.

    Notes
    -----
    This function is intended for eager execution only and is not compatible
    with ``jax.jit`` due to Python-level branching and variable-length lists.
    The probability of ring mode vs spiral mode is fixed at 30 % / 70 %.
    """
    keys = jr.split(key, 64)

    # Global disk geometry
    disc_scale_val = float(jr.uniform(keys[0], minval=disc_scale[0], maxval=disc_scale[1]))
    position_angle = float(jr.uniform(keys[1], minval=position_angle_range[0], maxval=position_angle_range[1]))
    axis_ratio = float(jr.uniform(keys[2], minval=axis_ratio_range[0], maxval=axis_ratio_range[1]))
    rings = bool(jr.bernoulli(keys[3], p=0.3))

    # Defaults
    n_rings = 0
    ring_radii = ring_widths = ring_amplitudes = None
    ring_radius = ring_width = ring_amplitude = 0.0
    n_arms = 0
    arm_width = spiral_peak_offset_val = spiral_radial_sigma_val = pitch = 0.0
    arm_amplitudes = None
    ring_a_sin_list = ring_b_cos_list = None
    ring_a_sin = ring_b_cos = None
    arm_a_sin_list = arm_b_cos_list = None

    # Ring mode
    if rings:
        n_rings = int(jr.randint(keys[4], shape=(), minval=ring_range[0], maxval=ring_range[1] + 1))
        ring_radii = np.sort(jr.uniform(keys[5], shape=(n_rings,), minval=ring_radii_range[0], maxval=ring_radii_range[1]))
        ring_widths = jr.uniform(keys[6], shape=(n_rings,), minval=ring_width_range[0], maxval=ring_width_range[1])
        ring_amplitudes = jr.uniform(keys[7], shape=(n_rings,), minval=ring_amplitude_range[0], maxval=ring_amplitude_range[1])

        if use_asymmetry:
            ring_a_sin_list, ring_b_cos_list = [], []
            asym_keys = jr.split(keys[8], max(n_rings, 1))
            for i in range(n_rings):
                a, b = sample_fourier_coeffs(asym_keys[i], max_modes=max_fourier_modes, coeff_range=tuple(fourier_coeff_range), prob_nonzero=per_ring_asymmetry_prob)
                ring_a_sin_list.append(a)
                ring_b_cos_list.append(b)

        image = make_rings(
            size=size, disk_scale=disc_scale_val, position_angle=position_angle,
            axis_ratio=axis_ratio, ring_radii=ring_radii, ring_widths=ring_widths,
            ring_amplitudes=ring_amplitudes, ring_a_sin_list=ring_a_sin_list,
            ring_b_cos_list=ring_b_cos_list, asymmetry_square=asymmetry_square,
        )

    # Spiral mode
    else:
        ring_radius = float(jr.uniform(keys[9], minval=ring_radii_range[0], maxval=ring_radii_range[1]))
        ring_width = float(jr.uniform(keys[10], minval=ring_width_range[0], maxval=ring_width_range[1]))
        ring_amplitude = float(jr.uniform(keys[11], minval=ring_amplitude_range[0], maxval=ring_amplitude_range[1]))
        n_arms = int(jr.randint(keys[12], shape=(), minval=arm_range[0], maxval=arm_range[1] + 1))
        arm_width = float(jr.uniform(keys[13], minval=arm_width_range[0], maxval=arm_width_range[1]))
        spiral_peak_offset_val = float(jr.uniform(keys[14], minval=spiral_peak_offset[0], maxval=spiral_peak_offset[1]))
        spiral_radial_sigma_val = float(jr.uniform(keys[15], minval=spiral_radial_sigma[0], maxval=spiral_radial_sigma[1]))
        pitch = float(jr.uniform(keys[16], minval=pitch_range[0], maxval=pitch_range[1]))
        arm_amplitudes = jr.uniform(keys[17], shape=(n_arms,), minval=arm_amplitude_range[0], maxval=arm_amplitude_range[1]) if n_arms > 0 else np.zeros((0,))

        if use_asymmetry:
            ring_a_sin, ring_b_cos = sample_fourier_coeffs(keys[18], max_modes=max_fourier_modes, coeff_range=tuple(fourier_coeff_range), prob_nonzero=ring_base_asymmetry_prob)
            arm_a_sin_list, arm_b_cos_list = [], []
            if n_arms > 0:
                arm_keys = jr.split(keys[19], n_arms)
                for i in range(n_arms):
                    a, b = sample_fourier_coeffs(arm_keys[i], max_modes=max_fourier_modes, coeff_range=tuple(fourier_coeff_range), prob_nonzero=per_arm_asymmetry_prob)
                    arm_a_sin_list.append(a)
                    arm_b_cos_list.append(b)

        image = make_spiral(
            size=size, disk_scale=disc_scale_val, position_angle=position_angle,
            axis_ratio=axis_ratio, ring_radius=ring_radius, ring_width=ring_width,
            ring_amplitude=ring_amplitude, n_arms=n_arms, arm_width=arm_width,
            arm_amplitudes=arm_amplitudes, spiral_peak_offset=spiral_peak_offset_val,
            spiral_radial_sigma=spiral_radial_sigma_val, pitch=pitch,
            ring_a_sin=ring_a_sin, ring_b_cos=ring_b_cos,
            arm_a_sin_list=arm_a_sin_list, arm_b_cos_list=arm_b_cos_list,
            asymmetry_square=asymmetry_square,
        )

    # Planets
    n_planets = int(jr.randint(keys[20], shape=(), minval=planets_range[0], maxval=planets_range[1] + 1))
    planet_angular_positions = planet_orbital_radii = planet_amplitudes = planet_sigmas = None

    if n_planets > 0:
        planet_angular_positions = jr.uniform(keys[21], shape=(n_planets,), minval=planet_angular_position_range[0], maxval=planet_angular_position_range[1])
        planet_orbital_radii = jr.uniform(keys[22], shape=(n_planets,), minval=planet_orbital_radius_range[0], maxval=planet_orbital_radius_range[1])
        planet_amplitudes = jr.uniform(keys[23], shape=(n_planets,), minval=planet_amplitude_range[0], maxval=planet_amplitude_range[1])
        planet_sigmas = jr.uniform(keys[24], shape=(n_planets,), minval=planet_sigma_range[0], maxval=planet_sigma_range[1])

        for angle, radius, amp, sigma in zip(planet_angular_positions, planet_orbital_radii, planet_amplitudes, planet_sigmas):
            image = add_planet(
                image=image, planet_radius=float(radius), planet_angle=float(angle),
                planet_amplitude=float(amp), planet_sigma=float(sigma),
                disk_scale=disc_scale_val, cap_to_image_max=True,
            )

    # Blank image debug
    max_val = float(image.max())
    min_val = float(image.min())

    if max_val < 1e-6:
        print("\n--- Blank image detected ---")
        print(f"disc_scale: {disc_scale_val}, axis_ratio: {axis_ratio}, position_angle: {position_angle}")
        if rings:
            print(f"RING MODE | n_rings: {n_rings} | radii: {ring_radii} | widths: {ring_widths} | amplitudes: {ring_amplitudes}")
            print(f"ring_a_sin_list: {ring_a_sin_list} | ring_b_cos_list: {ring_b_cos_list}")
        else:
            print(f"SPIRAL MODE | ring_radius: {ring_radius} | ring_width: {ring_width} | ring_amplitude: {ring_amplitude}")
            print(f"n_arms: {n_arms} | arm_width: {arm_width} | arm_amplitudes: {arm_amplitudes}")
            print(f"spiral_peak_offset: {spiral_peak_offset_val} | spiral_radial_sigma: {spiral_radial_sigma_val} | pitch: {pitch}")
            print(f"ring_a_sin: {ring_a_sin} | ring_b_cos: {ring_b_cos}")
            print(f"arm_a_sin_list: {arm_a_sin_list} | arm_b_cos_list: {arm_b_cos_list}")
        print(f"PLANETS | n_planets: {n_planets}")
        if n_planets > 0:
            print(f"angles: {planet_angular_positions} | radii: {planet_orbital_radii} | amplitudes: {planet_amplitudes} | sigmas: {planet_sigmas}")
        print(f"IMAGE STATS | min: {min_val} | max: {max_val}")
        print("----------------------------\n")

    if normalize:
        image = image / image.max()

    return image