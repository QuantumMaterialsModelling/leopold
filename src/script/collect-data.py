"""
Script to gather data from a vaspout.h5

creation: 2025-05-02 11:44:15
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Math
import numpy as np

# H5
import h5py as h5
from h5py import File, Dataset, Group

# ASE
from ase import Atoms
from ase.io import write

# OS
import os

# Parser
from argparse import ArgumentParser, Namespace

# ==== GLOBAL VARIABLES ==== #

scalar_quantities = {"energy": "energy", "stress": "stress"}
vector_quantities = {"forces": "forces", "decomposed_magmoms": ""}

# ==== FUNCTIONS ==== #


def get_atoms_from_file(file: str) -> list[Atoms]:
    # Control the file
    if not h5.is_hdf5(file):
        raise RuntimeError(f"{file} is not a valide hdf5 file!")

    # Gather data
    scalar_quantities = {}
    vector_quantities = {}

    with File(file, "r") as f:
        # Gets atomic species
        types = Dataset(f["input/poscar/ion_types"].id)[:]
        numbers = Dataset(f["input/poscar/number_ion_types"].id)[:]
        elements = [
            t.decode("utf-8").split("_")[0]
            for t, n in zip(types, numbers)
            for _ in range(n)
        ]

        # See if NBLOCK was different from 1
        step = 1
        if "NBLOCK" in Group(f["input/incar"].id).keys():
            step = Dataset(f["input/incar/NBLOCK"].id)[()]

        g = Group(f["intermediate/ion_dynamics"].id)

        # Get structure informations
        positions = Dataset(g["position_ions"].id)[:]
        lattices = Dataset(g["lattice_vectors"].id)[:]

        # Get energy and forces
        scalar_quantities["energy"] = Dataset(g["energies"].id)[::step, 0]
        if "stress" in g.keys():
            scalar_quantities["stress"] = Dataset(g["stress"].id)[::step]
        vector_quantities["forces"] = Dataset(g["forces"].id)[::step]

        # Get magmoms and charges
        d = Dataset(g["magnetism/spin_moments/values"].id)
        vector_quantities["decomposed_magmoms"] = d[::step, 1]
        vector_quantities["decomposed_charges"] = d[::step, 0]

        # See if something was not converged
        not_converged = Dataset(g["electronic_step_converged"].id)[::step]
        not_converged = np.array(not_converged, dtype=np.bool).flatten()

    # Create the Atoms
    data: list[Atoms] = []
    for i, bad in enumerate(not_converged):
        if bad:
            continue

        atoms = Atoms(elements, positions[i], cell=lattices[i], pbc=True)

        # Set scalar quantities
        for key, item in scalar_quantities.items():
            atoms.info[key] = item[i]

        # Set vector quantities
        for key, item in vector_quantities.items():
            atoms.arrays[key] = item[i]

        data.append(atoms)

    return data


def get_atoms_from_folder(path: str) -> list[Atoms]:
    data = []
    for file in os.listdir(path):
        if os.path.isdir(file):
            data.extend(get_atoms_from_folder(file))
        else:
            try:
                atoms = get_atoms_from_file(file)
            except Exception:
                atoms = []

            data.extend(atoms)

    return data


# ==== PARSER ==== #


def parse_args() -> Namespace:
    parser = ArgumentParser("collect-data")

    parser.add_argument("path", help="Path to the file or folder of the calculation")

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search for vaspout.h5 inside the folder",
    )

    return parser.parse_args()


# ==== MAIN ==== #


def main():
    args = parse_args()

    if not args.recursive:
        data = get_atoms_from_file(args.path)
    else:
        if not os.path.isdir(args.path):
            data = get_atoms_from_file(args.path)
        else:
            data = get_atoms_from_folder(args.path)

    write("data.xyz", data, format="extxyz")


if __name__ == "__main__":
    main()
