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

# Cuequivariance
import cuequivariance as cue
import cuequivariance_jax as cuex

# Types
from cuequivariance_jax import RepArray
from cuequivariance import Irreps

# ==== OBJECTS ==== #


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
