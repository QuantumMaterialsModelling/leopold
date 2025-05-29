"""
Definition of the GENERAL ML components for flax

creation: 2025-05-28 15:54:15
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Math
import jax.numpy as jnp

# Flax
import flax.linen as nn
from flax.linen.initializers import Initializer

# Typing
from jax.typing import ArrayLike
from typing import Dict, Callable

# ==== UTILITIES ==== #

NONLINEARITY: Dict[str, Callable[[ArrayLike], ArrayLike]] = {
    "none": lambda x: x,
    "relu": nn.relu,
    "raw_swish": nn.swish,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
    "gelu": nn.gelu,
    "soft": lambda x: -1.0 * x * jnp.expm1(-1.0 * (x * x)),
    "seagul": lambda x: jnp.log1p(x * x),
}

EVEN_NONLINEARITY = ["seagul", "raw_swish"]
ODD_NONLINEARITY = ["none", "soft", "tanh"]


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
    def __call__(self, x: ArrayLike) -> ArrayLike:
        # See dimensionality of gates
        if isinstance(self.gate_names, tuple):
            # Check dimensions and gates have same number
            assert len(self.dimensions) == len(self.gate_names), (
                "must have same number of dimensions and gates to construct an MLP!"
            )

            # Check all gates exist
            for gate in self.gate_names:
                assert gate in NONLINEARITY.keys(), (
                    f"non linearity {gate} does not exist, possibles are: {NONLINEARITY.keys()}!"
                )

            layers = zip(self.dimensions, self.gate_names)
        else:
            # Check if wanted gate exist
            assert self.gate_names in NONLINEARITY.keys(), (
                f"non linearity {self.gate_names} does not exist, possibles are: {NONLINEARITY.keys()}!"
            )

            layers = zip(self.dimensions, [self.gate_names for _ in self.dimensions])

        # Perform operations
        for d, g in layers:
            x = nn.Dense(d, use_bias=self.use_biases, kernel_init=self.init_funct)(x)
            x = NONLINEARITY[g](x)

        return x


class BesselEmbedding(nn.Module):
    """Flax module to embed a distance in a high dimensional space

    Takes a distance :math:`r` and embeds it in a higher space defined by a
    set of bessel functions :math:`[J_0(\\omega_1 r), ..., J_0(\\omega_n r)]`.
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
    def __call__(self, R: ArrayLike) -> ArrayLike:
        # Take it as array
        R = jnp.asarray(R)

        # Define frequencies of Bessel functions
        w = self.param("frequences", lambda _: jnp.arange(1, self.count + 1) * jnp.pi)

        # Avoid infinities for distances that are too small
        r = jnp.where(R > 1e-5, R, 1.0)

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
        return jnp.where(R[:, jnp.newaxis] > 1e-5, b, 0) * envelop
