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
from leopold.dataset import (
    leopold_load_datasets,
    LeopoldDataLoader,
    leopold_dataloader_to_hdf5,
)

# HDF5
from h5py import File, Group

# Math
import numpy as np
import jax.numpy as jnp
import jax.random as jrn
from jax import Array, vmap, jit

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


def get_graph_constructor(pos: Array, box: Array, r_cutoff: float):
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


def save_params_to_hdf5(group: Group, params: FrozenDict | dict, compression: int = 9):
    for key, value in params.items():
        if isinstance(value, dict):
            save_params_to_hdf5(group.require_group(key), value)
        else:
            # Handle RepArray
            if isinstance(value, RepArray):
                data = np.asarray(value.array)

                d = group.require_dataset(
                    key, data.shape, data.dtype, compression=compression
                )

                # Save irreps as attributes
                d.attrs["Irreps"] = str(value.irreps)
                d[:] = data
            else:
                # Handle scalars
                if np.isscalar(value):
                    data = np.asarray([value])
                else:
                    data = np.asarray(value)

                group.require_dataset(
                    key, data.shape, data.dtype, compression=compression
                )[:] = data


def save_data_to_hdf5(
    group: Group,
    loader: LeopoldDataLoader,
    save_data: bool = True,
    compression: int = 9,
):
    # Save info on the dataset
    g = group.require_group("info")

    prova = asdict(loader.info)
    save_params_to_hdf5(g, prova, compression)

    # If requested save the whole dataset
    if save_data:
        # Gather config and labels data to GPU
        confis = {key: [] for key in loader[0].config._asdict().keys()}
        labels = {key: [] for key in loader[0].labels._asdict().keys()}
        for batch in loader:
            for key, value in batch.config._asdict().items():
                confis[key].extend(value)

            for key, value in batch.labels._asdict().items():
                if value is None:
                    continue
                labels[key].extend(value)

        confis = {key: np.asarray(value) for key, value in confis.items()}
        labels = {key: np.asarray(value) for key, value in labels.items()}

        # Save it on HDF5
        save_params_to_hdf5(group.require_group("configurations"), confis)
        save_params_to_hdf5(group.require_group("labels"), labels)


@dataclass
class Occhi:
    colore: int = 0
    sopracciglia: str = "test"


@dataclass
class Barba:
    lunghezza: float = 1.0


@dataclass
class Faccia:
    barba: Barba = field(default_factory=lambda: Barba())
    occhi: Occhi = field(default_factory=lambda: Occhi())


# ==== MAIN ==== #


def main():
    faccia = Faccia()

    print(asdict(faccia))

    # Get data loader
    # data = leopold_load_datasets(
    #     {"dataset": {"test": "train.xyz"}}, batch_size=5, shuffle=False
    # )
    # data = data["test"]
    #
    # # take first entry
    # (pos, ele, box), _ = data[0]
    #
    # # Construct graph
    # get_graph = get_graph_constructor(pos[0], box[0], 3.5)
    #
    # graph = get_graph(pos[1], ele[1], box[1])
    #
    # # Construct model
    # model = nn.Leopold(len(data.info.species) + len(data.info.pol_types))
    #
    # param = model.init(jrn.PRNGKey(0), graph)
    #
    # # Save parameters to HDF5
    # with File("test.h5", "a") as f:
    #     save_params_to_hdf5(f.require_group("models"), param)
    #     leopold_dataloader_to_hdf5(f.require_group("datasets"), data)


if __name__ == "__main__":
    main()
