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
from jax import Array, vmap, value_and_grad
from jax.tree_util import tree_map

# JAX-MD
from jax_md import space, partition
from jax_md.partition import NeighborList, is_sparse, neighbor_list_mask
from jax_md.util import PyTree

# Leopold
from leopold.nn import Leopold

# Jraph
from jraph import GraphsTuple

# Typing
from typing import Callable, Optional

# ==== FUNCTIONS ==== #


def custom_to_jraph(
    neighbor: NeighborList,
    mask: Optional[Array] = None,
    nodes: Optional[PyTree] = None,
    edges: Optional[PyTree] = None,
    globals: Optional[PyTree] = None,
) -> GraphsTuple:
    """Convert a sparse neighbor list to a `jraph.GraphsTuple`.

    Customized so that no padding is added since in Leopold padding is done previously.

    Args:
      neighbor: A neighbor list that we will convert to the jraph format. Must be
        sparse.
      mask: An optional mask on the edges.

    Returns:
      A `jraph.GraphsTuple` that contains the topology of the neighbor list.
    """
    if not is_sparse(neighbor.format):
        raise ValueError(
            "Cannot convert a dense neighbor list to jraph format. "
            "Please use either NeighborListFormat.Sparse or "
            "NeighborListFormat.OrderedSparse."
        )

    receivers, senders = neighbor.idx
    N = len(neighbor.reference_position)

    _mask = neighbor_list_mask(neighbor)

    # If there is an additional mask, reorder the edges.
    if mask is not None:
        _mask = _mask & mask
        cumsum = jnp.cumsum(_mask)
        index = jnp.where(_mask, cumsum - 1, len(receivers))
        ordered = N * jnp.ones((len(receivers) + 1,), jnp.int32)
        receivers = ordered.at[index].set(receivers)[:-1]
        senders = ordered.at[index].set(senders)[:-1]

        def reorder_edges(x):
            return jnp.zeros_like(x).at[index].set(x)

        edges = tree_map(reorder_edges, edges)
        mask = receivers < N

    return GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=globals,
        n_node=jnp.array([N, 1]),
        n_edge=jnp.array([jnp.sum(_mask), jnp.sum(~_mask)]),
    )


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
        return custom_to_jraph(neigh, nodes=elements, edges=dR)

    return get_graph


def leopold_model(conf: dict, pos: Array, elem: Array, box: Array):
    f = leopold_graph_constructor(pos, box, conf["r_cutoff"])
    model = Leopold(**conf)

    def apply_fn(param, pos, elem, box):
        graph = f(pos, elem, box)
        return model.apply(param, graph)

    def init_fn(key: Array):
        graph = f(pos, elem, box)
        return model.init(key, graph)

    return value_and_grad(apply_fn, argnums=1, has_aux=True), init_fn
