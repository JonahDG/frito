import os
from pathlib import Path

import jax
from jax import numpy as np, random as jr, Array
from jax.experimental import checkify

from .utils import mas2rad, normalize_image, load_disco

__all__ = [
    image2ComplexVis,
    observables_from_image,
    inject_image
]

def _image_pixel_grid_rad(n_rows: int, n_cols: int, pscale_mas: float=16.4) -> tuple[np.ndarray, np.ndarray]:
    def half(n: int):
        if n % 2 == 0:
            return n / 2.0
        else:
            return (n - 1.0) / 2.0
    alpha = (np.arange(n_cols) - half(n_cols)) * pscale_mas * mas2rad
    delta = (np.arange(n_rows) - half(n_rows)) * pscale_mas * mas2rad
    return alpha, delta

def image2ComplexVis(img: np.ndarray, u: np.ndarray, v: np.ndarray, wavel: float, pscale_mas: float=16.4) -> np.ndarray:
    checkify.check(img.ndim == 2, f"Image must be 2-D, got shape {img.shape}")

    n_rows, n_cols = img.shape
    alpha, delta = _image_pixel_grid_rad(n_rows, n_cols, pscale_mas)

    wavenum = 2 * np.pi / wavel
    phase_alpha = np.exp(-1j * wavenum * np.outer(u, alpha))
    phase_delta = np.exp(-1j * wavenum * np.outer(v, delta))

    V = np.einsum("ki, ij, kj->k", phase_delta, img, phase_alpha)
    return V

def observables_from_image(img: np.ndarray, filter_block: dict, pscale_mas: float = 16.4, flux_scale: float=1.0):
    u = filter_block['u']
    v = filter_block['v']
    wavel = float(filter_block['wavel'])

    img = np.asarray(img, dtype=np.float64) * float(flux_scale)

    V = image2ComplexVis(img, u, v, wavel, pscale_mas)

    amplitudes = np.abs(V)                  # (1300,)
    phases = np.angle(V)                    # (1300,)

    vis_mat = filter_block['vis_mat']       # (420, 1300)
    phi_mat = filter_block['phi_mat']       # (420, 1300)
    vis = vis_mat @ amplitudes              # (420,)
    phi = phi_mat @ phases                  # (420,)

    K_vis_mat = filter_block["K_vis_mat"]   # (349, 420)
    K_phi_mat = filter_block["K_phi_mat"]   # (349, 420)
    K_vis = K_vis_mat @ vis                 # (349,)
    K_phi = K_phi_mat @ phi                 # (349,)

    O_vis_mat = filter_block["O_vis_mat"]   # (349, 349)
    O_phi_mat = filter_block["O_phi_mat"]   # (349, 349)
    O_vis = O_vis_mat @ K_vis               # (349,)     
    O_phi = O_phi_mat @ K_phi               # (349,)
    
    return dict(vis=vis, phi=phi, K_vis=K_vis, K_phi=K_phi, O_vis=O_vis, O_phi=O_phi)

def _draw_correlated_noise(cov: np.ndarray, key: Array=jr.key(0)) -> np.ndarray:
    C = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(C)
    w_clipped = np.clip(w, 0.0, None)
    L = V * np.sqrt(w_clipped)
    z = jr.normal(key=key, shape=C.shape[0])
    return L @ z

def inject_image(disco_template: dict, img: np.ndarray, pscale_mas: float=16.4, flux_scale: float=1.0, add_noise: bool=True, scale_noise_w_flux: bool=False, key: Array=jr.key(0)):
    norm_img = normalize_image(img)
    cov_factor = 1.0 / (flux_scale ** 2)
    
    if scale_noise_w_flux and flux_scale == 1.0:
        print("Scaling noise with flux, but the flux scale is 1.0, so no scaling happens")
    elif scale_noise_w_flux and flux_scale != 1.0:
        print(f'Scaling noise by {cov_factor:.3f}')

    cov_keys = ["vis_cov", "phi_cov", "K_vis_cov", "K_phi_cov", "O_vis_cov", "O_phi_cov"]
    eigv_keys = ["O_vis_eigv", "O_phi_eigv"]
    
    out = {}
    for filter, disco_filter in disco_template.items():
        new_disco_filter = dict(disco_filter)
        observables = observables_from_image(norm_img, disco_filter, pscale_mas, flux_scale)
        new_disco_filter.update(observables)

        if scale_noise_w_flux and cov_factor != 1.0:
            for ck in cov_keys:
                if ck in new_disco_filter:
                    new_disco_filter[ck] = new_disco_filter[ck] * cov_factor
            for ek in eigv_keys:
                if ek in new_disco_filter:
                    new_disco_filter[ek] = new_disco_filter[ek] * cov_factor
        if add_noise:
            for disco_key, disco_cov_key in [
                ("vis", "vis_cov"), ("phi", "phi_cov"),
                ("K_vis", "K_vis_cov"), ("K_phi", "K_phi_cov"),
                ("O_vis", "O_vis_cov"), ("O_phi", "O_phi_cov")
            ]:
                cov = new_disco_filter[disco_cov_key]
                noise = _draw_correlated_noise(cov, key)
                new_disco_filter[disco_key] = new_disco_filter[disco_key] + noise
            
        out[filter] = new_disco_filter
    return out
