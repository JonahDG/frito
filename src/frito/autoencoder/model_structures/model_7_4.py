import jax
import equinox as eqx
# ============================================================================ #
# Encode
class encoder(eqx.Module):
    """Encoder module for autoencoder

    Args:
        eqx.Module: eequinox base class

    Returns:
        x: returns layer by layer encoder
    """

    layers: list

    def __init__(self, *, key):
        keys = jax.random.split(key, 10)
        self.layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3, key=keys[2]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5, stride=1, padding=2, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5, stride=2, padding=1, key=keys[4]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5, stride=1, padding=2, key=keys[5]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[6]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[7]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[8]),
            jax.nn.relu,
            eqx.nn.Lambda(lambda z:z.reshape(*z.shape[:-3], -1)),
            eqx.nn.Linear(20_000,2048,key=keys[9]),
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================================ #
# Decode
class decoder(eqx.Module):
    """Decoder module for autoencoder

    Args:
        eqx.Module: Equinox base class

    Returns:
        x: layer by layer decoder
    """

    layers: list
    
    def __init__(self, *, key):
        keys = jax.random.split(key,6)
        self.layers = [
            jax.nn.relu,
            eqx.nn.Linear(2048,20_000,key=keys[0]),
            jax.nn.relu,
            eqx.nn.Lambda(lambda z:z.reshape(*z.shape[:-1],32,25,25)),
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3, key=keys[1]),
            jax.nn.relu,
            eqx.nn.ConvTranspose(2, in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, output_padding=0, key=keys[2]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, key=keys[4]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, key=keys[5]),
            jax.nn.sigmoid
        ]
    def __call__(self, x):
        for layer in self.layers:
            # print(f'{type(layer).__name__}')
            x = layer(x)
        return x

# ============================================================================ #
# Combined
class autoencoder(eqx.Module):
    """Combined autoencoder pulls in encoder and decoders from above classes

    Args:
        eqx (Module): base equinox class

    Returns:
        x: loops from encoder to decoder, each will go through their own layers
    """
    
    modules: list
    def __init__(self, *, key):
        enc_key, dec_key = jax.random.split(key, 2)
        enc = encoder(key=enc_key)
        dec = decoder(key=dec_key)
        self.modules = [enc, dec]
    def __call__(self, x):
        for layer in self.modules:
            x = layer(x)
        return x