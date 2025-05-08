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

    @linen.compact
    def __call__(self, graph: GraphsTuple):
        # Get edges
        dR = jnp.asarray(graph.edges)
        R = jnp.linalg.norm(dR, axis=-1)

        # Transform edges in RepArray
        dR = cuex.RepArray(cue.Irreps("O3", "1e"), dR, cue.mul_ir)

        # Embed edges
        dR = cuex.spherical_harmonics([i for i in range(self.n_harmo + 1)], dR)
        R = nn.BesselEmbedding(self.n_basis, self.r_cutof - 0.5, self.r_cutof)(R[0])

        print(dR)

        # Transform nodes in Rep Array
        nodes = jnp.asarray(graph.nodes)
        nodes = cuex.RepArray(cue.Irreps("O3", f"{self.n_elems}x0e"), nodes, cue.mul_ir)

        nodes = nn.Linear(self.hidden_irr)(nodes)

        return nodes


def main():
    # Get data loader
    data = leopold_load_datasets({"dataset": {"test": "test.xyz"}}, batch_size=1)
    data = data["test"]

    # take first entry
    (pos, ele, box), _ = data[0]

    # Construct graph
    get_graph = get_graph_constructor(pos[0], box[0], 3.5)

    graph = vmap(get_graph)(pos, ele, box)

    # Construct model
    hidden_irr = cue.Irreps("O3", "42x0e + 8x1e")
    model = Leopold(4, 8, 1, 3, hidden_irr)

    old_m = NequIPEnergyModel(
        8, True, {"e": "raw_swish", "o": "tanh"}, 3, "42x0e + 8x1e", "1x0e + 1x1e"
    )

    param = model.init(jrn.PRNGKey(0), graph)
    opara = old_m.init(jrn.PRNGKey(0), graph)

    res = model.apply(param, graph)


if __name__ == "__main__":
    main()
