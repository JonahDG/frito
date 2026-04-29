import equinox as eqx
from jax import Array, numpy as np, tree as jtu
from zodiax import Base

from dorito.model_fits import ResolvedOIFit, _OIFit
from dorito.models import ResolvedDiscoModel
from dorito.bases import ImageBasis

from amigo.core_models import BaseModeller

class TransformedResolvedOIFit(ResolvedOIFit):

    def initialise_params(self, model, coeffs, basis):

        params = _OIFit.initialise_params(self, model)

        # log distribution stored directly as coefficients
        params["log_dist"] = (self.get_key("log_dist"), coeffs)

        # decode once to get a reference distribution for base_uv normalisation
        ref_dist = basis.from_basis(coeffs)
        params["base_uv"] = (
            self.get_key("base_uv"),
            self.get_base_uv(model, ref_dist.shape[0]),
        )

        return params
    
class AutoencoderBasis(Base):
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, autoencoder: eqx.Module):
        self.encoder = autoencoder.modules[0]
        self.decoder = autoencoder.modules[1]

    def to_basis(self, img: Array) -> Array:
        return self.encoder(img[None, ...])

    def from_basis(self, coeffs: Array) -> Array:
        return self.decoder(coeffs)[0]
    
class TransformedResolvedDiscoModel(ResolvedDiscoModel):
    basis: None
    window: Array

    def __init__(
            self,
            ois: list,
            distribution: Array,
            basis: ImageBasis,
            uv_npixels: int,
            uv_pscale: float,
            oversample: float = 1.0,
            psf_pixel_scale: float = 0.065524085,
            rotate: bool = False,
            window: Array = None,
    ):
        def fn(x):
            if isinstance(x, Array):
                if 'i' in x.dtype.str:
                    return x
                return np.array(x, dtype=float)
            return x
        
        self.basis = jtu.map(lambda x: fn(x), basis)
        self.window = window

        self.uv_npixels = uv_npixels
        self.uv_pscale = uv_pscale
        self.oversample = oversample
        self.psf_pixel_scale = psf_pixel_scale
        self.rotate = rotate

        init_dist = distribution
        init_coeffs = self.basis.to_basis(init_dist)

        params = {}
        for oi in ois:
            param_dict = oi.initialise_params(self, distribution)
            key, _ = param_dict['log_dist']
            param_dict['log_dist'] = (key, init_coeffs)

            for param, (key, value) in param_dict.items():
                if param not in params:
                    params[param] = {}
                param[param][key] = value
        BaseModeller.__init__(self, params)
    
    def get_distribution(
        self,
        exposure,
        rotate: bool = None,
        exponentiate: bool = True,
        window: bool = True,
    ):

        coeffs = self.params["log_dist"][exposure.get_key("log_dist")]

        if exponentiate:
            distribution = 10 ** self.basis.from_basis(coeffs)
        else:
            distribution = self.basis.from_basis(coeffs)

        if self.window is not None and window:
            distribution *= self.window

        if rotate is None:
            rotate = self.rotate
        if rotate:
            distribution = exposure.rotate(distribution)

        return distribution