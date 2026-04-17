"""
Definition of the LEOPOLD ML model using cuequivariance

creation: 2025-05-28 15:52:11
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Math
import jax.numpy as jnp

# Flax
import flax.linen as nn

# NN generals
from leopold.nn.generals import BesselEmbedding
from jax_md.nn import nequip

# e3nn
from e3nn_jax import Irreps, IrrepsArray
from e3nn_jax import spherical_harmonics, FunctionalLinear
from e3nn_jax.utils import vmap

# Functools
from functools import partial

# Typing
from jax.typing import ArrayLike


# ==== OBJECTS ==== #


class Linear(nn.Module):
    irreps_out: Irreps

    @nn.compact
    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        irreps_out = Irreps(self.irreps_out)

        x = x.remove_nones().simplify()

        lin = FunctionalLinear(x.irreps, irreps_out, instructions=None, biases=None)

        w = [
            self.param(
                f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                nn.initializers.normal(stddev=ins.weight_std, dtype=x.dtype),
                ins.path_shape,
            )
            if ins.i_in == -1
            else self.param(
                f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},"
                f"{lin.irreps_out[ins.i_out]}",
                nn.initializers.normal(stddev=ins.weight_std, dtype=x.dtype),
                ins.path_shape,
            )
            for ins in lin.instructions
        ]

        f = partial(lin, w)
        for _ in range(x.ndim - 1):
            f = vmap(f)
        return f(x)


class Leopold(nn.Module):
    """Leopold interatomic potential structure

    This is basically a NequIP architecture following the original implementation by [Batzner et al.](nature.com/articles/s41467-022-29939-5)
    and the reimplementation present inside the [jax-md](https://github.com/jax-md/jax-md) package.

      Args:
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
    hidden_irr: str = "48x0e + 48x1e"
    r_cutoff: float = 3.5
    n_neighbour: float = 1.0

    energy_scale: ArrayLike = 1.0
    energy_shift: ArrayLike = 0.0
    magchg_scale: ArrayLike = 1.0
    magchg_shift: ArrayLike = 0.0

    radial_mlp_layers: int = 2
    radial_mlp_hidden: int = 64
    radial_mlp_activa: str = "raw_swish"

    even_gate: str = "raw_swish"
    even_act: str = "raw_swish"
    odd_gate: str = "tanh"
    odd_act: str = "tanh"

    self_connection: bool = False

    @nn.compact
    def __call__(self, graph):
        # Set non linearities
        non_lin = {"e": self.even_act, "o": self.odd_act}

        # Convert hidden_irr to Irreps
        hidden_irr = Irreps(self.hidden_irr)

        # Get edges
        dR = IrrepsArray(Irreps("1e"), jnp.asarray(graph.edges))
        R = jnp.linalg.norm(graph.edges, axis=-1)

        # Embed edges
        dR_sh = spherical_harmonics([i for i in range(self.n_harmo + 1)], dR, True)
        R = BesselEmbedding(self.n_basis, self.r_cutoff - 0.5, self.r_cutoff)(R)

        # Take senders and recievers
        sender = jnp.asarray(graph.senders)
        reciev = jnp.asarray(graph.receivers)

        # Transform nodes in Rep Array
        nodes = IrrepsArray(Irreps(f"{self.n_elems}x0e"), graph.nodes)

        # Perform convolution
        conv = Linear(hidden_irr)(nodes)
        for _ in range(self.n_convo):
            conv = nequip.NequIPConvolution(
                hidden_irreps=hidden_irr,
                use_sc=self.self_connection,
                nonlinearities=non_lin,
                radial_net_nonlinearity=self.radial_mlp_activa,
                radial_net_n_hidden=self.radial_mlp_hidden,
                radial_net_n_layers=self.radial_mlp_layers,
                num_basis=self.n_basis,
                n_neighbors=self.n_neighbour,
                scalar_mlp_std=4.0,
            )(conv, nodes, dR_sh, sender, reciev, R)

        # Get final layers
        second_irreps = conv.irreps.filter(keep=Irreps("0e")).dim // 2
        second_irreps = Irreps(f"{second_irreps}x0e")

        # Energy calculation
        energy = Linear(second_irreps)(conv)
        energy = Linear(Irreps("1x0e"))(energy).array

        # Magnetization and Charge calculation
        magchg = Linear(second_irreps)(conv)
        magchg = Linear(Irreps("2x0e"))(magchg).array

        # Scale and Shift energy
        scale = self.energy_scale
        if jnp.isscalar(self.energy_scale):
            scale = jnp.full((self.n_elems - 1, 1), self.energy_scale)

        shift = self.energy_shift
        if jnp.isscalar(self.energy_shift):
            shift = jnp.full((self.n_elems - 1, 1), self.energy_shift)

        scale = nodes.array[:, :-1] @ scale
        shift = nodes.array[:, :-1] @ shift

        energy = jnp.sum(scale * energy + shift)

        # Scale and Shift magchg
        scale = self.magchg_scale
        if jnp.isscalar(self.magchg_scale):
            scale = jnp.full((self.n_elems - 1, 2), self.magchg_scale)

        shift = self.magchg_shift
        if jnp.isscalar(self.magchg_shift):
            shift = jnp.full((self.n_elems - 1, 2), self.magchg_shift)

        scale = nodes.array[:, :-1] @ scale
        shift = nodes.array[:, :-1] @ shift

        magchg = scale * magchg + shift

        return energy, magchg


class LeopoldDescriptor(nn.Module):
    """Leopold interatomic potential structure

    This is basically a NequIP architecture following the original implementation by [Batzner et al.](nature.com/articles/s41467-022-29939-5)
    and the reimplementation present inside the [jax-md](https://github.com/jax-md/jax-md) package.

      Args:
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
    hidden_irr: str = "48x0e + 48x1e"
    r_cutoff: float = 3.5
    n_neighbour: float = 1.0

    energy_scale: ArrayLike = 1.0
    energy_shift: ArrayLike = 0.0
    magchg_scale: ArrayLike = 1.0
    magchg_shift: ArrayLike = 0.0

    radial_mlp_layers: int = 2
    radial_mlp_hidden: int = 64
    radial_mlp_activa: str = "raw_swish"

    even_gate: str = "raw_swish"
    even_act: str = "raw_swish"
    odd_gate: str = "tanh"
    odd_act: str = "tanh"

    self_connection: bool = False

    @nn.compact
    def __call__(self, graph):
        # Set non linearities
        non_lin = {"e": self.even_act, "o": self.odd_act}

        # Convert hidden_irr to Irreps
        hidden_irr = Irreps(self.hidden_irr)

        # Get edges
        dR = IrrepsArray(Irreps("1e"), jnp.asarray(graph.edges))
        R = jnp.linalg.norm(graph.edges, axis=-1)

        # Embed edges
        dR_sh = spherical_harmonics([i for i in range(self.n_harmo + 1)], dR, True)
        R = BesselEmbedding(self.n_basis, self.r_cutoff - 0.5, self.r_cutoff)(R)

        # Take senders and recievers
        sender = jnp.asarray(graph.senders)
        reciev = jnp.asarray(graph.receivers)

        # Transform nodes in Rep Array
        nodes = IrrepsArray(Irreps(f"{self.n_elems}x0e"), graph.nodes)

        # Perform convolution
        conv = Linear(hidden_irr)(nodes)
        for _ in range(self.n_convo):
            conv = nequip.NequIPConvolution(
                hidden_irreps=hidden_irr,
                use_sc=self.self_connection,
                nonlinearities=non_lin,
                radial_net_nonlinearity=self.radial_mlp_activa,
                radial_net_n_hidden=self.radial_mlp_hidden,
                radial_net_n_layers=self.radial_mlp_layers,
                num_basis=self.n_basis,
                n_neighbors=self.n_neighbour,
                scalar_mlp_std=4.0,
            )(conv, nodes, dR_sh, sender, reciev, R)

        return conv


# ==== FUNCTIONS ==== #
