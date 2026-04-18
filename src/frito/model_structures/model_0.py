import jax
import jax.random as jr
import equinox as eqx


class encoder(eqx.Module):
    """Convolutional encoder network that maps images to a latent representation.

    Consists of 9 convolutional layers with decreasing kernel sizes, followed
    by a flatten operation and a linear projection to the latent space.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key used to initialise layer weights.

    Attributes
    ----------
    layers : list
        Sequential list of convolutional layers, activation functions, and a
        final linear layer.
    """

    layers: list

    def __init__(self, *, key):
        """Initialise the encoder.

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key used to initialise layer weights. Split internally into
            10 subkeys, one per learnable layer.
        """
        keys = jr.split(key, 10)

        self.layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3, key=keys[2]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, key=keys[4]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, key=keys[5]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[6]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[7]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[8]),
            jax.nn.relu,
            eqx.nn.Lambda(lambda z: z.reshape(*z.shape[:-3], -1)),
            eqx.nn.Linear(20_000, 2048, key=keys[9]),
        ]

    def __call__(self, X):
        """Encode an input image into a latent vector.

        Parameters
        ----------
        X : jax.Array
            Input image tensor of shape ``(1, H, W)``, where ``H`` and ``W``
            are the spatial dimensions.

        Returns
        -------
        jax.Array
            Latent representation of shape ``(2048,)``.

        Notes
        -----
        The method name in the original source is ``__cal__`` (a typo);
        rename to ``__call__`` for the module to be callable.
        """
        for layer in self.layers:
            X = layer(X)
        return X


class decoder(eqx.Module):
    """Convolutional decoder network that reconstructs images from a latent vector.

    Mirrors the encoder: a linear projection expands the latent vector back to
    a feature map, which is then upsampled and refined by convolutional layers
    before a sigmoid activation produces the final pixel values.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key used to initialise layer weights.

    Attributes
    ----------
    layers : list
        Sequential list of linear, reshape, convolutional, transposed-
        convolutional layers, and activation functions.
    """

    layers: list

    def __init__(self, *, key):
        """Initialise the decoder.

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key used to initialise layer weights. Split internally into
            6 subkeys, one per learnable layer.
        """
        keys = jax.random.split(key, 6)
        self.layers = [
            jax.nn.relu,
            eqx.nn.Linear(2048, 20_000, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Lambda(lambda z: z.reshape(*z.shape[:-1], 32, 25, 25)),
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3, key=keys[1]),
            jax.nn.relu,
            eqx.nn.ConvTranspose(2, in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, output_padding=0, key=keys[2]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[4]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, key=keys[5]),
            jax.nn.sigmoid,
        ]

    def __call__(self, x):
        """Decode a latent vector into a reconstructed image.

        Parameters
        ----------
        x : jax.Array
            Latent vector of shape ``(2048,)``.

        Returns
        -------
        jax.Array
            Reconstructed image tensor of shape ``(1, H, W)`` with values in
            ``[0, 1]`` produced by a sigmoid activation.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class autoencoder(eqx.Module):
    """Full convolutional autoencoder combining an encoder and a decoder.

    Passes input images through the encoder to obtain a compressed latent
    representation, then through the decoder to reconstruct the original image.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key used to initialise both the encoder and decoder weights.

    Attributes
    ----------
    modules : list
        Two-element list ``[encoder, decoder]`` applied sequentially.
    """

    modules: list

    def __init__(self, *, key):
        """Initialise the autoencoder.

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key split into two subkeys: one for the encoder and one for
            the decoder.
        """
        enc_key, dec_key = jax.random.split(key, 2)
        enc = encoder(key=enc_key)
        dec = decoder(key=dec_key)
        self.modules = [enc, dec]

    def __call__(self, x):
        """Encode then decode an input image.

        Parameters
        ----------
        x : jax.Array
            Input image tensor of shape ``(1, H, W)``.

        Returns
        -------
        jax.Array
            Reconstructed image tensor of shape ``(1, H, W)`` with values in
            ``[0, 1]``.
        """
        for layer in self.modules:
            x = layer(x)
        return x