"""
Module with Leopold observables functions

creation: 2025-05-19 11:57:59
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Partial
from functools import partial

# JAX
import jax.numpy as jnp
from jax import Array, vmap

# JAX-MD
from jax_md import space, partition

# Leopold
from leopold.nn import Leopold

# Jraph
from jraph import GraphsTuple

# Typing
from typing import Callable

# ==== FUNCTIONS ==== #


def leopold_graph_constructor(
    pos: Array, box: Array, r_cutoff: float
) -> Callable[[Array, Array, Array], GraphsTuple]:
    # Get the space metric
    d, _ = space.periodic_general(box)

    # Allocate sparse neighbor list
    neighbor_fns = partition.neighbor_list(
        d,
        box,
        r_cutoff,
        format=partition.Sparse,
        fractional_coordinates=True,
    )
    neighbor = neighbor_fns.allocate(pos)

    def get_graph(position: Array, elements: Array, box: Array):
        # Update neighbor list with new positions and box
        neigh = neighbor.update(position, box=box)

        # Create mask to avoid account self and padding
        mask = neigh.idx[0] < (elements == 1).any(1).sum()
        mask = mask & (neigh.idx[0] != neigh.idx[1])

        # Get sender and reciver
        sender, reciver = neigh.idx

        # Compute edges and set masked ones to zero
        dR = vmap(partial(d, box=box))(position[sender], position[reciver])  # pyright: ignore
        dR = jnp.where(mask[:, None], dR, 0)

        # Construct jax graph
        return partition.to_jraph(neigh, nodes=elements, edges=dR)

    return get_graph


def leopold_model(conf: dict, pos: Array, elem: Array, box: Array):
    f = leopold_graph_constructor(pos, box, conf["r_cutoff"])
    model = Leopold(**conf)

    def apply(param, pos, elem, box):
        graph = f(pos, elem, box)
        return model.apply(param, graph)

    def init(key: Array):
        graph = f(pos, elem, box)
        return model.init(key, graph)

    return apply, init
