"""
DESCRIPTION

creation: 2025-05-07 16:22:28
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Functional
from functools import partial

# LEOPOLD
import leopold.nn as nn
from leopold.dataset import leopold_load_datasets
from old.leopold import NequIPEnergyModel

# Math
import jax.numpy as jnp
import jax.random as jrn
from jax import Array, vmap

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

    @linen.compact
    def __call__(self, graph: GraphsTuple):
        # Get edges
        dR = jnp.asarray(graph.edges)
        R = jnp.linalg.norm(dR, axis=-1)

        # Transform edges in RepArray
        dR = cuex.RepArray(cue.Irreps("O3", "1e"), dR, cue.mul_ir)

        # Embed edges
        dR = cuex.spherical_harmonics([i for i in range(self.n_harmo + 1)], dR)
        R = nn.BesselEmbedding(self.n_basis, self.r_cutof - 0.5, self.r_cutof)(R)

        # Transform nodes in Rep Array
        nodes = jnp.asarray(graph.nodes)
        nodes = cuex.RepArray(cue.Irreps("O3", f"{self.n_elems}x0e"), nodes, cue.mul_ir)

        # Take senders and recievers
        sender = jnp.asarray(graph.senders)
        reciev = jnp.asarray(graph.receivers)

        # Define the convolution
        for _ in range(self.n_convo):
            # Construct the tensor product descriptor
            e = cue.descriptors.channelwise_tensor_product(
                nodes.irreps, dR.irreps, self.hidden_irr
            )

            # Get dimensions for MLP and non linearities
            mlp_dims = (self.radial_mlp_hidden,) * self.radial_mlp_layers
            mlp_dims += (e.operands[0].dim,)
            mlp_nnln = ("swish",) * self.radial_mlp_layers + ("none",)

            # First linear layer
            nodes = nn.Linear(nodes.irreps)(nodes)

            # Get weights
            w = nn.MLP(mlp_dims, mlp_nnln, False)(R)
            edge_feat = cuex.equivariant_polynomial(e, [w, nodes[sender], dR])

            # Check no problem arised and simplify
            assert not isinstance(edge_feat, list)
            edge_feat = edge_feat.simplify()

            # Perform a scatter sum
            res = jnp.zeros((nodes.shape[0], edge_feat.shape[1]))
            res = res.at[reciev].add(edge_feat.array)

            nodes = cuex.RepArray(edge_feat.irreps, res, cue.mul_ir)

            # Second linear layer
            nodes = nn.Linear(nodes.irreps)(nodes)

            # TODO: Self connection

            # TODO: gate function

        return nodes


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
    model = Leopold(4, 8, 1, 3, hidden_irr, n_convo=1)

    old_m = NequIPEnergyModel(
        1, False, {"e": "raw_swish", "o": "tanh"}, 3, "42x0e + 8x1e", "1x0e + 1x1e"
    )

    param = model.init(jrn.PRNGKey(0), graph)
    # opara = old_m.init(jrn.PRNGKey(0), graph)

    # print(param)
    # print(
    #     opara["params"]["NequIPConvolution_0"]["Linear_0"]["w[0,0] 42x0e,42x0e"].shape
    # )

    # res = model.apply(param, graph)
    # ore = old_m.apply(opara, graph)

    # print(res)
    # print(ore)


if __name__ == "__main__":
    main()
