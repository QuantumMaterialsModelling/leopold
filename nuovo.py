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
from leopold.observables import leopold_graph_constructor
from leopold.dataset import (
    LeopoldData,
    leopold_load_datasets,
    LeopoldDataLoader,
    leopold_dataloader_to_hdf5,
)

# HDF5
from h5py import File, Group

# Math
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import Array, value_and_grad, vmap, jit

# JAX-MD
from jax_md import space, partition

# CUEQUI
import cuequivariance as cue
import cuequivariance_jax as cuex

from cuequivariance_jax import RepArray

# Dataclass
from dataclasses import asdict, dataclass, field

# Types
from jraph import GraphsTuple
from cuequivariance import Irreps
from flax.linen import FrozenDict

# ==== FUNCTIONS ==== #


# ==== MAIN ==== #


def main():
    data = leopold_load_datasets(
        {"test": "train.xyz"}, batch_size=1, shuffle=False, r_cutoff=3.5
    )
    data = data["test"]

    # take first entry
    (pos, ele, box), _ = data[0]

    # Construct graph
    get_graph = leopold_graph_constructor(pos, box, 3.5)

    graph = get_graph(pos, ele, box)

    R = jnp.linalg.norm(graph.edges, axis=-1)
    test = nn.BesselEmbedding(8, 3.0, 3.5)
    param = test.init(jrn.PRNGKey(0), R)

    # R = jnp.where(R == 0, 1, R)

    @jit
    def loss(params, r):
        res = test.apply(params, r)

        return res.sum()

    val, grad = value_and_grad(loss)(param, R)

    print(val)
    print(grad)

    # Construct model
    # model = nn.Leopold(len(data.info.species) + len(data.info.pol_types))
    #
    # param = model.init(jrn.PRNGKey(0), graph)
    #
    # @jit
    # def loss(params, batch: LeopoldData):
    #     graph = get_graph(*batch.config)
    #     energy, _ = model.apply(params, graph)
    #
    #     return jnp.square(energy - batch.labels.energy).mean()
    #
    # val, grad = value_and_grad(loss)(param, data[0])
    #
    # print(grad)


if __name__ == "__main__":
    main()
