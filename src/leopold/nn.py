"""
Definition of the LEOPOLD ML model using cuequivariance

creation: 2025-03-05 16:46:15
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Math
import jax.numpy as jnp
from jax import Array

# Flax
import flax.linen as nn

from flax.linen.initializers import Initializer

# Cuequivariance
import cuequivariance as cue
import cuequivariance_jax as cuex

from cuequivariance_jax import RepArray
from cuequivariance import Irreps

# ==== UTILITIES ==== #

NONLINEARITY = {
    "none": lambda x: x,
    "relu": nn.relu,
    "swish": nn.swish,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
}

# ==== OBJECTS ==== #


class MLP(nn.Module):
    """Implementation of a general MLP in flax

    Attributes:
        dimensions: tuple with dimensions of the layers
        gate_names: names of the linearities to use after each linear layer
        use_biases: tell if linear layers should use biases
        init_funct: initializer kernel functions for the weights
    """

    dimensions: tuple[int, ...]
    gate_names: tuple[str, ...] | str

    use_biases: bool = True
    init_funct: Initializer = nn.initializers.normal()

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # See dimensionality of gates
        if isinstance(self.gate_names, tuple):
            # Check dimensions and gates have same number
            if len(self.dimensions) != len(self.gate_names):
                raise ValueError(
                    "must have same number of dimensions and gates to construct an MLP!"
                )

            # Check all gates exist
            for gate in self.gate_names:
                if gate not in NONLINEARITY.keys():
                    raise KeyError(
                        f"non linearity {gate} does not exist, possibles are: {NONLINEARITY.keys()}!"
                    )

            layers = zip(self.dimensions, self.gate_names)
        else:
            # Check if wanted gate exist
            if self.gate_names not in NONLINEARITY.keys():
                raise KeyError(
                    f"non linearity {self.gate_names} does not exist, possibles are: {NONLINEARITY.keys()}!"
                )

            layers = zip(self.dimensions, [self.gate_names for _ in self.dimensions])

        # Perform operations
        for d, g in layers:
            x = nn.Dense(d, use_bias=self.use_biases, kernel_init=self.init_funct)(x)
            x = NONLINEARITY[g](x)

        return x


class FullyConnectedTensorProduct(nn.Module):
    """Cuequivariance implementation of a fully connected tensor product module

    Attributes:
        irr_out: Irreps of output
    """

    irr_out: Irreps

    @nn.compact
    def __call__(self, x1: RepArray, x2: RepArray) -> RepArray:
        e = cue.descriptors.fully_connected_tensor_product(
            x1.irreps, x2.irreps, self.irr_out
        )
        w = self.param("weights", lambda x: cuex.randn(x, e.operands[0]))

        return cuex.equivariant_polynomial(e, [w, x1, x2])  # pyright: ignore


class Linear(nn.Module):
    """Cuequivariance implementation of a linear layer

    Attributes:
        irr_out: Irreps of output
    """

    irr_out: Irreps

    @nn.compact
    def __call__(self, x: RepArray) -> RepArray:
        e = cue.descriptors.linear(x.irreps, self.irr_out)

        # TODO: This can be enhanced with a better initialization dependent on the dim
        #       so that the std of the normal is 1 / sqrt(dim), but dim is different
        #       for weights acting on scalars and the one acting on other parts.
        #       Thus, the implementation would need to do something like what is done
        #       inside line 72 of linear.py in e3nn
        w = self.param("weights", lambda key: cuex.randn(key, e.operands[0]))

        return cuex.equivariant_polynomial(e, [w, x])  # pyright: ignore


class BesselEmbedding(nn.Module):
    """Flax module to embed a distance in a high dimensional space

    Takes a distance :math:`r` and embeds it in a higher space defined by a
    set of bessel functions :math:`[J_0(\omgega_1 r), ..., J_0(\omega_n r)]`.
    The different functions are then multiplied by a smooth envelope function
    defined by a polynomial relation that smoothly sets to zeros the entries
    given by :math:`r > r_C` with :math:`r_C` being a outer cutoff radius,
    while the envelope is set to be equal to 1 for :math:`r < r_c` with
    :math:`r_c` being an inner cutoff radius.

    Attributes:
        count: number of frequencies to use for the embedding
        inner_cutoff: inner cutoff of the envelope function
        outer_cutoff: outer cutoff of the envelope function
    """

    count: int
    inner_cutoff: float
    outer_cutoff: float

    @nn.compact
    def __call__(self, r: Array) -> Array:
        # Define frequencies of Bessel functions
        w = self.param("frequences", lambda _: jnp.arange(1, self.count + 1) * jnp.pi)

        # Broadcast (N,) to (N, 1)
        r = r[:, jnp.newaxis]

        # Compute bessel functions
        b = w[jnp.newaxis, :] * r
        b = 2 * jnp.sin(b / self.outer_cutoff) / (r * self.outer_cutoff)

        # Compute envelop to make them smooth
        r2, rc, ro = r * r, self.outer_cutoff**2, self.inner_cutoff**2

        envelop = jnp.where(
            r < self.outer_cutoff,
            (rc - r2) ** 2 * (rc + 2 * r2 - 3 * ro) / (rc - ro) ** 3,
            0,
        )
        envelop = jnp.where(r < self.inner_cutoff, 1, envelop)

        # Final multiplication
        return jnp.where(r > 1e-5, b, 0) * envelop


class NequIPConvolution(nn.Module):
    hidden_irreps: Irreps

    radial_mlp_layers: int = 2
    radial_mlp_hidden: int = 64
    radial_mlp_initia: Initializer = nn.initializers.normal(4.0)

    n_neighbours: float = 1

    @nn.compact
    def __call__(
        self,
    ):
        pass
