"""
Module for the creation of the atomistic graph

creation: 2025-04-30 15:40:27
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== IMPORTS ==== #

# WARNING
import warnings

# ASE
from warnings import WarningMessage
from ase import Atoms
from ase.io import read

# JAX
import jax.numpy as jnp
from jax import Array

# JAX-MD
from jax_md.partition import neighbor_list, Sparse
from jax_md.space import periodic_general

# JAX-GRAPH
from jraph import GraphsTuple

# Types
from typing import NamedTuple

# ==== DATA ==== #

# TODO: in future add stress
scalar_labels = {"energy": "energy"}
vector_labels = {
    "forces": "forces",
    "charges": "decomposed_charge",
    "magmoms": "decomposed_magmoms",
}


# ==== FUNCTIONS ==== #


def get_graphs_from_atoms(atoms: list[Atoms]):
    pass


def get_labels_from_atoms(file: str, pol_character: int) -> dict[str, Array]:
    atoms = read(file, ":", "extxyz")
    if not isinstance(atoms, list):
        atoms = [atoms]

    # First construct the data dictionary
    res = {key: [] for key, _ in (scalar_labels | vector_labels).items()}

    for i, atom in enumerate(atoms):
        # Retrieve scalar
        for label, name in scalar_labels.items():
            if getattr(atom.info, name, None) is None:
                warnings.warn(
                    f"Configuration {i} does not provide {label} at {name} position!",
                    RuntimeWarning,
                )

                res[label].append(jnp.inf)
            res[label].append(atom.info[name])

        # Retrieve vector
        for label, name in vector_labels.items():
            if getattr(atom.arrays, name, None) is None:
                warnings.warn(
                    f"Configuration {i} does not provide {label} at {name} position!",
                    RuntimeWarning,
                )
                res[label].append(jnp.full_like(res[label][-1], jnp.inf))

            # For charges and magmoms get only polaron character, if
            # the latter is -1 then assume the file contained already
            # the right component of the charge and magmom
            if label in ["charges", "magmoms"] and pol_character > -1:
                res[label].append(atom.arrays[name][:, pol_character])
            else:
                res[label].append(atom.arrays[name])

    return {key: jnp.array(arr) for key, arr in res.items()}


# ==== TEST ==== #

if __name__ == "__main__":
    atoms = read("test.xyz", ":100", "extxyz")
    if not isinstance(atoms, list):
        atoms = [atoms]

    box = jnp.array(atoms[0].cell.array)
    d, _ = periodic_general(box, wrapped=False)

    sneighbor_fn = neighbor_list(d, box, 3.5, format=Sparse)
    dneighbor_fn = neighbor_list(d, box, 3.5)

    sneighbor = sneighbor_fn.allocate(atoms[0].get_scaled_positions())
    dneighbor = dneighbor_fn.allocate(atoms[0].get_scaled_positions())

    print(sneighbor.idx[0, :20])
    print(sneighbor.idx[1, :40])
    print(dneighbor.idx.shape)
