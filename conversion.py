"""
Roba veloce

creation: 2025-05-09 13:57:24
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== IMPORTS ==== #

# OS
import os

# Math
import numpy as np

# ASE
from ase.io import read, write

# Types
from argparse import ArgumentParser, Namespace

# ==== FUNCTIONS ==== #


# ==== PARSER ==== #


def parse_args() -> Namespace:
    parser = ArgumentParser("conversion")

    parser.add_argument("path")

    return parser.parse_args()


# ==== MAIN ==== #


def main():
    args = parse_args()

    data = read(args.path, index=":", format="extxyz")
    if not isinstance(data, list):
        data = [data]

    for atoms in data:
        atoms.arrays.pop("pol_state")

        toccup = atoms.arrays.pop("toccup")

        atoms.arrays["decomposed_charges"] = np.sum(toccup, axis=-1, keepdims=True)
        atoms.arrays["decomposed_magmoms"] = np.diff(toccup, axis=-1)

    write(os.path.basename(args.path), data, format="extxyz")


if __name__ == "__main__":
    main()
