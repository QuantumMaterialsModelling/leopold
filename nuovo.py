"""
DESCRIPTION

creation: 2025-05-07 16:22:28
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Functional
from functools import partial

from jax.numpy.linalg import vector_norm

# LEOPOLD
import leopold.nn as nn
from leopold.dataset import leopold_load_datasets
from old.leopold import NequIPEnergyModel

# Math
import jax.numpy as jnp
import jax.random as jrn
from jax import Array, vmap, jit, tree_util

# GRaph
from jraph import segment_sum

# NN
from flax import linen

# JAX-MD
from jax_md import space, partition

# CUEQUI
import cuequivariance as cue
import cuequivariance_jax as cuex

# Types
from jraph import GraphsTuple
from cuequivariance import Irreps
from cuequivariance_jax import RepArray

# ==== FUNCTIONS ==== #


def get_graph_constructor(pos: Array, box: Array, r_cutoff: float):
    d, _ = space.periodic_general(box)

    neighbor_fns = partition.neighbor_list(
        d,
        box,
        r_cutoff,
        format=partition.Sparse,
        fractional_coordinates=True,
    )
    neighbor = neighbor_fns.allocate(pos)

    def get_graph(position: Array, elements: Array, box: Array):
        neigh = neighbor.update(position, box=box)
        mask = partition.neighbor_list_mask(neigh, True)

        sender, reciver = neigh.idx

        dR = vmap(partial(d, box=box))(position[sender], position[reciver])
        jnp.where(mask[:, None], dR, 1)

        return partition.to_jraph(neigh, nodes=elements, edges=dR)

    return get_graph


# ==== MAIN ==== #


class Leopold(linen.Module):
    r_cutof: float
    n_basis: int
    n_harmo: int
    n_elems: int

    hidden_irr: Irreps
    n_convo: int = 2

    radial_mlp_layers: int = 2
    radial_mlp_hidden: int = 64

    n_neighbour: int = 32

    energy_scale: Array | float = 1.0
    energy_shift: Array | float = 0.0
    magchg_scale: Array | float = 1.0
    magchg_shift: Array | float = 0.0

    @linen.compact
    def __call__(self, graph: GraphsTuple):
        # Get edges
        dR = jnp.asarray(graph.edges)
        R = jnp.linalg.norm(dR, axis=-1)

        # Transform edges in RepArray
        dR = cuex.RepArray(cue.Irreps("O3", "1e"), dR, cue.ir_mul)

        # Embed edges
        dR = cuex.spherical_harmonics([i for i in range(self.n_harmo + 1)], dR)
        R = nn.BesselEmbedding(self.n_basis, self.r_cutof - 0.5, self.r_cutof)(R)

        # Transform nodes in Rep Array
        nodes = jnp.asarray(graph.nodes)
        nodes = cuex.RepArray(cue.Irreps("O3", f"{self.n_elems}x0e"), nodes, cue.ir_mul)

        # Take senders and recievers
        sender = jnp.asarray(graph.senders)
        reciev = jnp.asarray(graph.receivers)

        # Perform convolution
        conv = nn.Linear(self.hidden_irr)(nodes)
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
            mlp_nnln = ("swish",) * self.radial_mlp_layers + ("none",)

            # First linear layer
            conv = nn.Linear(conv.irreps)(conv)

            # Create the self connection
            self_conn = nn.FullyConnectedTensorProduct(
                Irreps("O3", str(e.outputs[0])).simplify(),
            )(conv, nodes)

            # Get weights and perform convolution
            w = nn.MLP(mlp_dims, mlp_nnln, False)(R)
            edge_feat = cuex.equivariant_polynomial(e, [w, conv[sender], dR])

            # Check no problem arised and simplify
            assert not isinstance(edge_feat, list)
            edge_feat = edge_feat.simplify()

            # Perform a scatter sum averaged beetween neighbours
            res = jnp.zeros((conv.shape[0], edge_feat.shape[1]))
            res = res.at[reciev].add(edge_feat.array) / self.n_neighbour

            conv = cuex.RepArray(edge_feat.irreps, res, cue.ir_mul)

            # Second linear layer and self connection
            conv = nn.Linear(conv.irreps)(conv) + self_conn

            # Gate the convolution results
            conv = nn.Gate()(conv)

        # Get final layers
        second_irreps = conv.irreps.filter(keep=[cue.O3(0, 1)]).dim // 2
        second_irreps = Irreps("O3", f"{second_irreps}x0e")

        # Energy calculation
        energy = nn.Linear(second_irreps)(conv)
        energy = nn.Linear(Irreps("O3", "1x0e"))(conv).array

        # Magnetization and Charge calculation
        magchg = nn.Linear(second_irreps)(conv)
        magchg = nn.Linear(Irreps("O3", "2x0e"))(conv).array

        # Scale and Shift energy
        scale = self.energy_scale
        if jnp.isscalar(self.energy_scale):
            scale = jnp.full((self.n_elems - 1,), self.energy_scale)

        shift = self.energy_shift
        if jnp.isscalar(self.energy_shift):
            shift = jnp.full((self.n_elems - 1,), self.energy_shift)

        scale = nodes.array[:, :-1] @ scale
        shift = nodes.array[:, :-1] @ shift

        energy = jnp.sum(shift * energy + scale, axis=0)

        print(energy)

        # Scale and Shift magchg
        scale = self.magchg_scale
        if jnp.isscalar(self.magchg_scale):
            scale = jnp.full((self.n_elems - 1, 2), self.magchg_scale)

        shift = self.magchg_shift
        if jnp.isscalar(self.magchg_shift):
            shift = jnp.full((self.n_elems - 1, 2), self.magchg_shift)

        scale = nodes.array[:, :-1] @ scale
        shift = nodes.array[:, :-1] @ shift

        magchg = jnp.sum(shift * magchg + scale, axis=0)

        return energy, magchg


def main():
    # Get data loader
    data = leopold_load_datasets({"dataset": {"test": "train.xyz"}}, batch_size=1)
    data = data["test"]

    # take first entry
    (pos, ele, box), _ = data[0]

    # Construct graph
    get_graph = get_graph_constructor(pos[0], box[0], 3.5)

    graph = get_graph(pos[0], ele[0], box[0])

    # Construct model
    hidden_irr = cue.Irreps("O3", "42x0e + 8x1e")
    model = Leopold(3.5, 8, 3, 4, hidden_irr, n_convo=1)

    # old_m = NequIPEnergyModel(
    #     1, False, {"e": "raw_swish", "o": "tanh"}, 3, str(hidden_irr), "1x0e + 1x1e"
    # )

    param = model.init(jrn.PRNGKey(0), graph)
    # opara = old_m.init(jrn.PRNGKey(0), graph)

    jitted_model = jit(model.apply)

    res, _ = jitted_model(param, graph)
    res, _ = jitted_model(param, graph)
    # ore = old_m.apply(opara, graph)

    print(res.shape)
    # print(ore.shape)


if __name__ == "__main__":
    main()
