"""Wrappers of dorito basis, models, and model fits

This module includes classes that work with an autoencoder basis with
OI fits (discos). It wraps model_fits and models from dorito to do this.
"""
from pytest import approx

import equinox as eqx

from jax import Array, numpy as np, tree as jtu, debug
from zodiax import Base

from dorito.model_fits import ResolvedOIFit, _OIFit
from dorito.models import ResolvedDiscoModel
from dorito.bases import ImageBasis

from amigo.core_models import BaseModeller


class AutoencoderBasis(Base):
    """Image basis backed by a trained autoencoder.

    Wraps an Equinox autoencoder so that its encoder and decoder act as the
    forward and inverse transforms of an image basis. Images are projected
    into the latent (coefficient) space via the encoder, and reconstructed
    via the decoder.

    Parameters
    ----------
    autoencoder : eqx.Module
        A trained autoencoder whose ``modules`` attribute is a sequence
        ``(encoder, decoder)``. The encoder must accept a batched image of
        shape ``(1, H, W)`` and the decoder must accept a coefficient
        vector and return a batched image of shape ``(1, H, W)``.

    Attributes
    ----------
    encoder : eqx.Module
        The encoder network mapping images to latent coefficients.
    decoder : eqx.Module
        The decoder network mapping latent coefficients back to images.
    """

    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, autoencoder: eqx.Module):
        self.encoder = autoencoder.modules[0]
        self.decoder = autoencoder.modules[1]

    def to_basis(self, img: Array) -> Array:
        """Encode an image into latent coefficients.

        Parameters
        ----------
        img : Array
            Image array of shape ``(H, W)``. A leading batch dimension is
            added internally before passing to the encoder.

        Returns
        -------
        Array
            Latent coefficient representation of ``img``.
        """
        return self.encoder(img[None, ...])

    def from_basis(self, coeffs: Array) -> Array:
        """Decode latent coefficients back into an image.

        Parameters
        ----------
        coeffs : Array
            Latent coefficient array compatible with the decoder's input.

        Returns
        -------
        Array
            Reconstructed image of shape ``(H, W)``. The leading batch
            dimension produced by the decoder is squeezed out.
        """
        decoded_img = self.decoder(coeffs)[0]
        pix_sum = np.sum(decoded_img)
        debug.callback(lambda x: print(f"sum={x}") if abs(x - 1.0) > 1e-3 else None, pix_sum)
        return decoded_img


class TransformedResolvedOIFit(ResolvedOIFit):
    """Resolved OI fit that parameterises the brightness distribution by
    basis coefficients rather than by the image itself.

    The parent :class:`ResolvedOIFit` expects ``log_dist`` to be a log
    brightness distribution. This subclass instead stores basis
    coefficients in ``log_dist``; the distribution is recovered on demand
    by passing the coefficients through ``basis.from_basis``. This lets the
    parameter vector live in a (typically lower-dimensional) latent space
    while the rest of the fitting machinery is unchanged.
    """

    def initialise_params(self, model, coeffs, basis):
        """Initialise fit parameters from basis coefficients.

        Decodes ``coeffs`` into a reference image so the parent
        initialiser can set up everything that depends on image shape
        (notably ``base_uv``), then overwrites ``log_dist`` with the raw
        coefficients so subsequent fitting operates in coefficient space.

        Parameters
        ----------
        model : object
            The model owning these parameters; passed through to the
            parent initialiser.
        coeffs : Array
            Basis coefficients to use as the initial value of
            ``log_dist``.
        basis : object
            Object exposing a ``from_basis`` method that converts
            coefficients into an image.

        Returns
        -------
        dict
            Parameter dictionary with ``log_dist`` set to ``coeffs`` and
            all other entries inherited from the parent initialiser.
        """
        # Decode coeffs to get a reference distribution, then let the parent
        # do its normal init (which sets log_dist and base_uv).
        ref_dist = basis.from_basis(coeffs)
        params = ResolvedOIFit.initialise_params(self, model, ref_dist)

        # Overwrite log_dist with the actual coefficients; base_uv is already
        # correct because it only depends on ref_dist's shape.
        params["log_dist"] = (self.get_key("log_dist"), coeffs)

        return params


class PointResolvedOIFit(TransformedResolvedOIFit):
    """OI fit modelling a point source plus a resolved component.

    Extends :class:`TransformedResolvedOIFit` with a per-filter ``contrast``
    parameter in ``[0, 1]`` that mixes a point-source complex visibility
    (constant, equal to one) with the visibility of the resolved
    distribution. ``contrast = 0`` is a pure point source and
    ``contrast = 1`` is the pure resolved model.
    """

    def get_key(self, param):
        """Return the parameter dictionary key for ``param``.

        Parameters
        ----------
        param : str
            Name of the parameter.

        Returns
        -------
        str
            The filter name when ``param == "contrast"`` (so contrast is
            shared per filter), otherwise whatever the parent class
            returns.
        """
        match param:
            case "contrast":
                return self.filter
        return super().get_key(param)

    def map_param(self, param):
        """Map a parameter name to its dotted ``"name.key"`` form.

        Parameters
        ----------
        param : str
            Parameter name.

        Returns
        -------
        str
            ``"contrast.<filter>"`` for the contrast parameter, otherwise
            the parent mapping.
        """
        if param in ["contrast"]:
            return f"{param}.{self.get_key(param)}"
        return super().map_param(param)

    def initialise_params(self, model, coeffs, basis, contrast):
        """Initialise parameters including the contrast term.

        Parameters
        ----------
        model : object
            The model owning these parameters.
        coeffs : Array
            Initial basis coefficients for ``log_dist``.
        basis : object
            Basis object with a ``from_basis`` method.
        contrast : float or Array
            Initial value of the contrast parameter, expected in
            ``[0, 1]``.

        Returns
        -------
        dict
            Parameter dictionary containing the parent's parameters plus
            an entry for ``contrast`` keyed by filter.
        """
        params = super().initialise_params(model, coeffs, basis)
        params["contrast"] = (self.get_key("contrast"), np.array(contrast))
        return params

    def to_cvis(self, model, distribution):
        """Compute complex visibilities for the point + resolved model.

        Combines a unit (point-source) visibility with the resolved
        visibility produced by the parent class via the contrast
        parameter:

        ``cvis = (1 - contrast) * 1 + contrast * cvis_resolved``.

        Parameters
        ----------
        model : object
            The model providing the ``contrast`` parameter under
            ``model.params["contrast"]``.
        distribution : Array
            Brightness distribution passed through to the parent's
            ``to_cvis``.

        Returns
        -------
        Array
            Complex visibilities of the combined point + resolved model,
            same shape as the resolved visibilities.
        """
        contrast = model.params["contrast"][self.get_key("contrast")]

        resolved_cvis = super().to_cvis(model, distribution)
        point_cvis = np.ones_like(resolved_cvis)

        return (1 - contrast) * point_cvis + contrast * resolved_cvis


class TransformedResolvedDiscoModel(ResolvedDiscoModel):
    """Resolved disco model parameterised in a basis coefficient space.

    Stores a basis (e.g. an :class:`AutoencoderBasis`) and represents the
    brightness distribution as latent coefficients. The image is
    reconstructed on the fly via ``basis.from_basis`` whenever a
    distribution is needed. An optional spatial ``window`` mask can be
    applied to the reconstructed distribution.

    Parameters
    ----------
    ois : list
        List of OI fit objects whose ``initialise_params`` method is used
        to build the parameter dictionary.
    distribution : Array
        Initial brightness distribution. Encoded once at construction
        time to seed the latent coefficients.
    basis : ImageBasis
        Basis providing ``to_basis`` and ``from_basis``. All
        floating-point leaves are cast to ``float`` to ensure a
        consistent dtype; integer-typed leaves are left untouched.
    uv_npixels : int
        Number of pixels along one axis of the UV-plane sampling grid.
    uv_pscale : float
        Pixel scale of the UV-plane sampling grid.
    oversample : float, optional
        Oversampling factor for the model image, by default ``1.0``.
    psf_pixel_scale : float, optional
        Pixel scale of the PSF in the same units as ``uv_pscale``,
        by default ``0.065524085``.
    rotate : bool, optional
        Default rotation behaviour for :meth:`get_distribution`. If
        ``True``, distributions are rotated to the exposure frame unless
        explicitly overridden, by default ``False``.
    window : Array, optional
        Optional multiplicative spatial window applied to the
        distribution after decoding, by default ``None`` (no window).

    Attributes
    ----------
    basis : ImageBasis
        The stored basis used to convert between coefficients and images.
    window : Array or None
        The optional spatial window mask.
    """

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
                if "i" in x.dtype.str:
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
            param_dict = oi.initialise_params(self, init_coeffs, self.basis)
            key, _ = param_dict["log_dist"]
            param_dict["log_dist"] = (key, init_coeffs)

            for param, (key, value) in param_dict.items():
                if param not in params:
                    params[param] = {}
                params[param][key] = value
        BaseModeller.__init__(self, params)

    def get_distribution(
        self,
        exposure,
        rotate: bool = None,
        exponentiate: bool = False,
        window: bool = False,
    ):
        """Reconstruct the brightness distribution for an exposure.

        Looks up the coefficient vector for ``exposure`` in
        ``self.params["log_dist"]``, decodes it through the basis, and
        optionally exponentiates, applies the spatial window, and rotates
        the result.

        Parameters
        ----------
        exposure : object
            Exposure object providing ``get_key("log_dist")`` to select
            the appropriate coefficient vector and (if rotation is
            enabled) a ``rotate`` method.
        rotate : bool, optional
            Whether to rotate the distribution to the exposure frame.
            If ``None`` (default), falls back to ``self.rotate``.
        exponentiate : bool, optional
            If ``True`` (False by Default), interpret the decoded values as
            ``log10`` brightness and return ``10 ** decoded``. If
            ``False``, return the decoded values directly.
        window : bool, optional
            If ``True`` (False by default) and ``self.window`` is not ``None``,
            multiply the distribution by the stored window mask.

        Returns
        -------
        Array
            The reconstructed brightness distribution.
        """
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


class PointResolvedDiscoModel(TransformedResolvedDiscoModel):
    """Disco model combining a point source with a resolved component.

    Same as :class:`TransformedResolvedDiscoModel`, but each OI fit is
    initialised with an additional ``contrast`` parameter (typically via
    :class:`PointResolvedOIFit`) that controls the relative contribution
    of an unresolved point source and the resolved distribution.

    Parameters
    ----------
    ois : list
        List of OI fit objects whose ``initialise_params`` accepts the
        signature ``(model, coeffs, basis, contrast)``.
    distribution : Array
        Initial brightness distribution; encoded once to seed the latent
        coefficients.
    basis : ImageBasis
        Basis providing ``to_basis`` and ``from_basis``. Floating-point
        leaves are cast to ``float``.
    contrast : float
        Initial contrast value passed to each OI fit's parameter
        initialiser. Expected in ``[0, 1]``, with ``0`` meaning pure
        point source and ``1`` meaning pure resolved.
    uv_npixels : int
        Number of pixels along one axis of the UV-plane sampling grid.
    uv_pscale : float
        Pixel scale of the UV-plane sampling grid.
    oversample : float, optional
        Oversampling factor, by default ``1.0``.
    psf_pixel_scale : float, optional
        PSF pixel scale, by default ``0.065524085``.
    rotate : bool, optional
        Default rotation behaviour for :meth:`get_distribution`,
        by default ``False``.
    window : Array, optional
        Optional spatial window applied after decoding, by default
        ``None``.
    """

    def __init__(
        self,
        ois: list,
        distribution: Array,
        basis: ImageBasis,
        contrast: float,
        uv_npixels: int,
        uv_pscale: float,
        oversample: float = 1.0,
        psf_pixel_scale: float = 0.065524085,
        rotate: bool = False,
        window: Array = None,
    ):
        def fn(x):
            if isinstance(x, Array):
                if "i" in x.dtype.str:
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

        init_coeffs = self.basis.to_basis(distribution)

        params = {}
        for oi in ois:
            param_dict = oi.initialise_params(
                self, init_coeffs, self.basis, contrast
            )
            for param, (key, value) in param_dict.items():
                if param not in params:
                    params[param] = {}
                params[param][key] = value
        BaseModeller.__init__(self, params)
