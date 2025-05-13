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

# Jraph
from jraph import GraphsTuple

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
    "seagul": lambda x: jnp.log1p(x * x),
}

EVEN_NONLINEARITY = ["seagul"]
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
    """Cuequivariance implementation of a non linear equivariant gate

    Apply non linearities to scalar and vector quantites inside a cuequivariance RepArray
    in a way that continues to respect the O3 equivariance. In particular the way in which
    this is done follows exactly the [gate function of e3nn.](https://e3nn-jax.readthedocs.io/en/latest/api/nn.html#e3nn_jax.gate)
    Thus, apply selected gates to scalars inside the Array where gates for even scalars
    can have whatever symmetry while for odd scalars must be even or odd. Then for vectors
    no real non linearity can be applied directly, therefore the scalars are divided so that::

        gated_s, coeff_s = s[num_irreps_vect:], s[:num_irreps_vect],

    where a possibly different non linearity is applied to the seconds and the final output becomes::

        output = [f(gated_s), g(coeff_s) * vectors]

    The list of possibles activation functions can be found as keys of the NONLINEARITY constant.

    Attributes:
        even_gate: name of the gate function to apply at the even scalars
        even_act: name of the gate function to apply at the even coefficients
        odd_gate: name of the gate funciton to apply at the odd scalars
        odd_act: name of the gate function to apply at the odd coefficients
        normalize: True if wanted the activation function to be normalized
    """

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

        assert (
            self.odd_gate in ODD_NONLINEARITY or self.odd_gate in EVEN_NONLINEARITY
        ), f"{self.odd_gate} has no parity, cannot be used for odd channel"
        assert self.odd_act in ODD_NONLINEARITY or self.odd_act in EVEN_NONLINEARITY, (
            f"{self.odd_act} has no parity, cannot be used for odd channel"
        )

        # Get the correct functions
        even_gate = NONLINEARITY[self.even_gate]
        even_act = NONLINEARITY[self.even_act]
        odd_gate = NONLINEARITY[self.odd_gate]
        odd_act = NONLINEARITY[self.odd_act]

        # Get parity of odd functions
        odd_gate_p = 2 * int(self.odd_gate in ODD_NONLINEARITY) - 1
        odd_act_p = 2 * int(self.odd_act in ODD_NONLINEARITY) - 1

        # Noralize if needed
        even_gate = _normalize(even_gate) if self.normalize else even_gate
        odd_gate = _normalize(odd_gate) if self.normalize else odd_gate

        # Divide scalars from vectors
        scalars = x.filter(keep=[cue.O3(0, 0), cue.O3(0, 1)])
        vectors = x.filter(drop=[cue.O3(0, 0), cue.O3(0, 1)])

        # Check scalars are enough
        final_scalars = scalars.irreps.dim - vectors.irreps.num_irreps
        assert final_scalars > 0, "not enough scalars for Gate operation!"

        # Divide gated scalars
        gated_scalars = scalars.slice_by_mul[:final_scalars]

        # Apply gate to gated scalars
        arrs, irrs = [], Irreps("O3", "")
        for (mul, irr), arr in zip(gated_scalars.irreps, gated_scalars.segments):
            assert isinstance(irr, cue.O3)  # Just for correct linting

            if irr.p == 1:
                arrs.append(even_gate(arr))
                irrs += Irreps("O3", [(mul, irr)])
            else:
                arrs.append(odd_gate(arr))

                # Parity of output in odd channel depends on nonlinearity
                irrs += Irreps("O3", [(mul, (0, odd_gate_p))])

        gated_scalars = cuex.from_segments(irrs, arrs, gated_scalars.shape, cue.ir_mul)

        # If no vector part is present then return already
        if vectors.irreps.num_irreps == 0:
            return gated_scalars

        # Get Coefficient scalars for vector multiplication
        coeff_scalars = scalars.slice_by_mul[final_scalars:]

        # Apply act to coeff scalars
        arrs, irrs = [], Irreps("O3", "")
        for (mul, irr), arr in zip(coeff_scalars.irreps, coeff_scalars.segments):
            assert isinstance(irr, cue.O3)  # Just for correct linting

            if irr.p == 1:
                arrs.append(even_act(arr))
                irrs += Irreps("O3", [(mul, irr)])
            else:
                arrs.append(odd_act(arr))

                # Parity of output in odd channel depends on nonlinearity
                irrs += Irreps("O3", [(mul, (0, odd_act_p))])

        coeff_scalars = cuex.from_segments(irrs, arrs, coeff_scalars.shape, cue.ir_mul)

        # Multiply vectors for coeff scalars
        (num_ng, num_mul), num_ir = vectors.shape, vectors.irreps.num_irreps
        values = vectors.array.reshape(num_ng, num_ir, num_mul // num_ir)
        values = values * coeff_scalars.array[:, :, jnp.newaxis]

        vectors = RepArray(vectors.irreps, values.reshape(vectors.shape), cue.ir_mul)

        # Concatenate and return
        return cuex.concatenate([gated_scalars, vectors])


class Leopold(nn.Module):
    """Leopold interatomic potential structure

    This is basically a NequIP architecture following the original implementation by [Batzner et al.](nature.com/articles/s41467-022-29939-5)
    and the reimplementation present inside the [jax-md](https://github.com/jax-md/jax-md) package.

    Attributes:
        n_elems: number of elements that the model handles
        n_basis: number of bessel basis to use in the embedding of the edges
        n_harmo: maximum l of the spherical harmonics to evaluate on the edges
        n_convo: number of convolutions
        hidden_irr: irreducible representations used in the convolutional layers
        r_cutof: cutoff radius of the message passing convolution
        n_neighbour: average number of neighbour used to average the convolution sum
        energy_scale: scaling factor for the energy (can be a float or a vector changing for every species)
        energy_shift: shifting factor for the energy (can be a float or a vector changing for every species)
        magchg_scale: scaling factor for the magnetizations and charges (can be a float or a vector changing for every species)
        magchg_shift: shifting factor for the magnetizations and charges (can be a float or a vector changing for every species)
        radial_mlp_layers: number of layers for the MLP giving the weights for the convolutions
        radial_mlp_hidden: number of neurons for the MLP giving the weights for the convolutions
        radial_mlp_activa: activation function for the MLP giving the weights for the convolutions
        even_gate: activation function to use for equivariant scalar even gate
        even_act: activation function to use for equivariant vector even gate
        odd_gate: activation function to use for equivariant scalar odd gate
        odd_act: activation function to use for equivariant vector odd gate
    """

    n_elems: int
    n_basis: int = 8
    n_harmo: int = 2

    n_convo: int = 2
    hidden_irr: Irreps = Irreps("O3", "48x0e + 8x1e")
    r_cutof: float = 3.5
    n_neighbour: float = 1.0

    energy_scale: Array | float = 1.0
    energy_shift: Array | float = 0.0
    magchg_scale: Array | float = 1.0
    magchg_shift: Array | float = 0.0

    radial_mlp_layers: int = 2
    radial_mlp_hidden: int = 64
    radial_mlp_activa: str = "swish"

    even_gate: str = "gelu"
    even_act: str = "sigmoid"
    odd_gate: str = "soft"
    odd_act: str = "tanh"

    @nn.compact
    def __call__(self, graph: GraphsTuple):
        # Get edges
        dR = jnp.asarray(graph.edges)
        R = jnp.linalg.norm(dR, axis=-1)

        # Transform edges in RepArray
        dR = cuex.RepArray(cue.Irreps("O3", "1e"), dR, cue.ir_mul)

        # Embed edges
        dR = cuex.spherical_harmonics([i for i in range(self.n_harmo + 1)], dR)
        R = BesselEmbedding(self.n_basis, self.r_cutof - 0.5, self.r_cutof)(R)

        # Transform nodes in Rep Array
        nodes = jnp.asarray(graph.nodes)
        nodes = cuex.RepArray(cue.Irreps("O3", f"{self.n_elems}x0e"), nodes, cue.ir_mul)

        # Take senders and recievers
        sender = jnp.asarray(graph.senders)
        reciev = jnp.asarray(graph.receivers)

        # Perform convolution
        conv = Linear(self.hidden_irr)(nodes)
        for _ in range(self.n_convo):
            # Get the output irreps of the tensor product
            scalar_irr = self.hidden_irr.filter(keep=[cue.O3(0, 1), cue.O3(0, -1)])
            vector_irr = self.hidden_irr.filter(drop=[cue.O3(0, 1), cue.O3(0, -1)])

            hidden_irr = scalar_irr + vector_irr.new_scalars(vector_irr.num_irreps)
            hidden_irr += vector_irr

            # Construct the tensor product descriptor
            e = cue.descriptors.fully_connected_tensor_product(
                conv.irreps, dR.irreps, hidden_irr
            )

            # Get dimensions for MLP and non linearities
            mlp_dims = (self.radial_mlp_hidden,) * self.radial_mlp_layers
            mlp_dims += (e.operands[0].dim,)
            mlp_gate = (self.radial_mlp_activa,) * self.radial_mlp_layers + ("none",)

            # First linear layer
            conv = Linear(conv.irreps)(conv)

            # Create the self connection
            self_conn = FullyConnectedTensorProduct(
                Irreps("O3", str(e.outputs[0])).simplify(),
            )(conv, nodes)

            # Get weights and perform convolution
            w = MLP(mlp_dims, mlp_gate, False)(R)
            edge_feat = cuex.equivariant_polynomial(e, [w, conv[sender], dR])

            # Check no problem arised and simplify
            assert not isinstance(edge_feat, list)
            edge_feat = edge_feat.simplify()

            # Perform a scatter sum averaged beetween neighbours
            res = jnp.zeros((conv.shape[0], edge_feat.shape[1]))
            res = res.at[reciev].add(edge_feat.array) / self.n_neighbour

            conv = cuex.RepArray(edge_feat.irreps, res, cue.ir_mul)

            # Second linear layer and self connection
            conv = Linear(conv.irreps)(conv) + self_conn

            # Gate the convolution results
            conv = Gate(self.even_gate, self.even_act, self.odd_gate, self.odd_act)(
                conv
            )

        # Get final layers
        second_irreps = conv.irreps.filter(keep=[cue.O3(0, 1)]).dim // 2
        second_irreps = Irreps("O3", f"{second_irreps}x0e")

        # Energy calculation
        energy = Linear(second_irreps)(conv)
        energy = Gate(self.even_gate, self.even_act, self.odd_gate, self.odd_act)(
            energy
        )
        energy = Linear(Irreps("O3", "1x0e"))(energy).array

        # Magnetization and Charge calculation
        magchg = Linear(second_irreps)(conv)
        magchg = Gate(self.even_gate, self.even_act, self.odd_gate, self.odd_act)(
            magchg
        )
        magchg = Linear(Irreps("O3", "2x0e"))(magchg).array

        # Scale and Shift energy
        scale = self.energy_scale
        if jnp.isscalar(self.energy_scale):
            scale = jnp.full((self.n_elems - 1, 1), self.energy_scale)

        shift = self.energy_shift
        if jnp.isscalar(self.energy_shift):
            shift = jnp.full((self.n_elems - 1, 1), self.energy_shift)

        scale = nodes.array[:, :-1] @ scale
        shift = nodes.array[:, :-1] @ shift

        energy = jnp.sum(scale * energy + shift, axis=0)

        # Scale and Shift magchg
        scale = self.magchg_scale
        if jnp.isscalar(self.magchg_scale):
            scale = jnp.full((self.n_elems - 1, 2), self.magchg_scale)

        shift = self.magchg_shift
        if jnp.isscalar(self.magchg_shift):
            shift = jnp.full((self.n_elems - 1, 2), self.magchg_shift)

        scale = nodes.array[:, :-1] @ scale
        shift = nodes.array[:, :-1] @ shift

        magchg = jnp.sum(scale * magchg + shift, axis=0)

        return energy, magchg
