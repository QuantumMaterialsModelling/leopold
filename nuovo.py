"""
DESCRIPTION

creation: 2025-05-07 16:22:28
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# LEOPOLD
from functools import partial
import leopold.nn as nn
from leopold.dataset import leopold_load_datasets

# Math
import jax.numpy as jnp
from jax import Array, vmap

# JAX-MD
from jax_md import space, partition

# CUEQUI
import cuequivariance as cuex
import cuequivariance_jax as cujx

# ASE
from ase.build import bulk

# Types
from argparse import ArgumentParser, Namespace

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


# ==== PARSER ==== #


def parse_args() -> Namespace:
    parser = ArgumentParser("nuovo")

    return parser.parse_args()


# ==== MAIN ==== #


def main():
    args = parse_args()

    # Get data loader
    data = leopold_load_datasets({"dataset": {"test": "test.xyz"}}, batch_size=1)
    data = data["test"]

    # take first entry
    (pos, ele, box), _ = data[0]

    # Construct metric
    get_graph = get_graph_constructor(pos[0], box[0], 3.5)

    graph = vmap(get_graph)(pos, ele, box)

    r = jnp.linalg.norm(graph.edges, axis=-1)
    print(r)


if __name__ == "__main__":
    main()
