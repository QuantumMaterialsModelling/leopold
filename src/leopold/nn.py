"""
Definition of the LEOPOLD ML model using cuequivariance

creation: 2025-03-05 16:46:15
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Math
import jax.numpy as jnp
from jax import Array, ensure_compile_time_eval
from jax.scipy.special import erfinv

# Flax
import flax.linen as nn

from flax.linen.initializers import Initializer

# Cuequivariance
import cuequivariance as cue
import cuequivariance_jax as cuex

from cuequivariance_jax import RepArray
from cuequivariance import Irreps

# Typing
from typing import Callable

# ==== UTILITIES ==== #

NONLINEARITY = {
    "none": lambda x: x,
    "relu": nn.relu,
    "swish": nn.swish,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
    "gelu": nn.gelu,
    "soft": lambda x: x * (1 - jnp.exp(-(x * x))),
}

EVEN_NONLINEARITY = ["none", "gelu", "sigmoid"]
ODD_NONLINEARITY = ["none", "soft", "tanh"]


def _normalize(f: Callable[[Array], Array]) -> Callable[[Array], Array]:
    with ensure_compile_time_eval():
        x = jnp.sqrt(2) * erfinv(jnp.linspace(-1.0, 1.0, 1_000_000 + 2)[1:-1])
        C = jnp.sqrt(jnp.mean(f(x) ** 2))

        if jnp.allclose(C, 1):
            return f
        else:
            return lambda x: f(x) / C


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


class Gate(nn.Module):
    even_gate: str = "gelu"
    even_act: str = "sigmoid"

    odd_gate: str = "soft"
    odd_act: str = "tanh"

    normalize: bool = True

    @nn.compact
    def __call__(self, x: RepArray) -> RepArray:
        # Sanity checks
        assert x.irreps.irrep_class is cue.O3, (
            "Gate module implemented only for O3 groups for now!"
        )

        assert self.even_gate in EVEN_NONLINEARITY, (
            f"{self.even_gate} is not even, cannot be used for even channel"
        )
        assert self.odd_gate in ODD_NONLINEARITY, (
            f"{self.odd_gate} is not odd, cannot be used for odd channel"
        )
        assert self.even_act in EVEN_NONLINEARITY, (
            f"{self.even_act} is not even, cannot be used for even channel"
        )
        assert self.odd_act in ODD_NONLINEARITY, (
            f"{self.odd_act} is not odd, cannot be used for odd channel"
        )

        # Get the correct functions
        even_gate = NONLINEARITY[self.even_gate]
        even_act = NONLINEARITY[self.even_act]
        odd_gate = NONLINEARITY[self.odd_gate]
        odd_act = NONLINEARITY[self.odd_act]

        # Noralize if needed
        even_gate = _normalize(even_gate) if self.normalize else even_gate
        odd_gate = _normalize(odd_gate) if self.normalize else odd_gate

        # Divide scalars from vectors
        scalars = x.filter(keep=[cue.O3(0, 0), cue.O3(0, 1)])
        vectors = x.filter(drop=[cue.O3(0, 0), cue.O3(0, 1)])

        # Check scalars are enough
        final_scalars = scalars.irreps.dim - vectors.irreps.num_irreps
        assert final_scalars > 0, "not enough scalars for Gate operation!"

        # Divide gated scalars and coefficient for vectors
        gated_scalars = scalars.slice_by_mul[:final_scalars]
        coeff_scalars = scalars.slice_by_mul[final_scalars:]

        # Apply gate to gated scalars
        arrs = []
        for (_, irr), arr in zip(gated_scalars.irreps, gated_scalars.segments):
            assert isinstance(irr, cue.O3)  # Just for correct linting

            if irr.p == 1:
                arrs.append(even_gate(arr))
            else:
                arrs.append(odd_gate(arr))

        gated_scalars = cuex.from_segments(
            gated_scalars.irreps, arrs, gated_scalars.shape, cue.ir_mul
        )

        # Apply act to coeff scalars
        arrs = []
        for (_, irr), arr in zip(coeff_scalars.irreps, coeff_scalars.segments):
            assert isinstance(irr, cue.O3)  # Just for correct linting

            if irr.p == 1:
                arrs.append(even_act(arr))
            else:
                arrs.append(odd_act(arr))

        coeff_scalars = cuex.from_segments(
            coeff_scalars.irreps, arrs, coeff_scalars.shape, cue.ir_mul
        )

        # Multiply vectors for coeff scalars
        (num_ng, num_mul), num_ir = vectors.shape, vectors.irreps.num_irreps
        values = vectors.array.reshape(num_ng, num_ir, num_mul // num_ir)
        values = values * coeff_scalars.array[:, :, jnp.newaxis]

        vectors = RepArray(vectors.irreps, values.reshape(vectors.shape), cue.ir_mul)

        # Concatenate and return
        return cuex.concatenate([gated_scalars, vectors])
