"""
Script for evaluating a LEOPOLD model on a set of structures

creation: 2026-03-24 11:26:34
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== IMPORTS ==== #

# MATH
import numpy as np

# JAX
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import jit, vmap

# System
from os.path import basename, join, dirname

# Leopold
from leopold.dataset import (
    leopold_load_datasets,
    leopold_data_desctructor,
    DEFAULT_SCALAR_LABELS,
    DEFAULT_VECTOR_LABELS,
)
from leopold.observables import leopold_model
from leopold.hdf5 import LeopoldCheckpointFile
from leopold import __version__

# ASE
from ase.io import write

# ArgumentParser
from argparse import ArgumentParser, Namespace

# PreattyTables
from prettytable import PrettyTable

# ==== CONSTANTS ==== #

GREETING = f"""
                              ░░▒▒▒▒  ░░          
                          ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒        
    LEOPOLD             ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒    
    version: {__version__}      ▒▒▒▒▒▒▒▒░░░░░░░░░░░░▒▒    
                      ▒▒▒▒░░▒▒░░░░░░░░░░░░░░▒▒    
                      ▒▒░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒  
                      ▒▒▒▒░░░░░░░░░░░░▒▒░░░░▒▒▒▒  
                      ▒▒▒▒▒▒░░░░██░░░░██░░░░▒▒▒▒  
                    ░░▒▒▒▒▒▒░░░░▒▒░░▓▓░░░░░░▒▒░░  
                      ▒▒▒▒▒▒░░░░░░░░██▒▒░░░░▒▒▒▒  
░░▒▒░░                ▒▒▒▒▒▒░░░░░░▒▒░░░░░░▒▒▒▒▒▒  
  ▒▒▒▒          ░░░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▒▒▒▒    
      ░░      ░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░    
            ░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░▒▒░░░░░░░░░░
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░      ░░░░░░
        ░░░░░░░░░░░░░░░░      ░░░░░░░░░░          
        ░░░░░░░░░░░░░░            ░░░░░░░░        
        ░░░░  ░░░░░░                  ░░░░        
        ░░░░  ░░░░                                
"""

# ==== PARSER ==== #


def parse_args() -> Namespace:
    parser = ArgumentParser("leopold_eval")

    parser.add_argument("model", help="Path to the leopold checkpoint file")

    parser.add_argument(
        "data",
        nargs="+",
        help=".xyz files containing the configurations to evaluate",
    )

    parser.add_argument(
        "--label",
        default="LEO",
        help="Prefix used for energy, forces and charges in the output .xyz file",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name used for the output .xyz file",
    )

    parser.add_argument(
        "--use_float32",
        action="store_true",
        help="Use float32 for the evaluation",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["gpu", "cpu"],
        help="Accelerator device for the model",
    )

    return parser.parse_args()


# ==== MAIN ==== #


def main():
    print(GREETING)

    # Argument parsing
    args = parse_args()

    # ---- SETUP

    # Retrieve training group
    gtrain = LeopoldCheckpointFile(args.model, "r").get_training()

    # Select tag
    tag = gtrain.conf.general.tag
    if args.name is not None:
        tag = args.name

    # Set the default precision
    if not args.use_float32:
        jax.config.update("jax_enable_x64", True)

    # Select the devices
    device = jax.devices(args.device)[0]

    # ---- LOAD DATA
    data_path = {}
    for data in args.data:
        name = basename(data).split(".")[0]
        data_path[name] = data

    data = leopold_load_datasets(
        data_path,
        gtrain.conf.datasets.labels,
        batch_size=gtrain.conf.training.batch_size,
        device=device,
        r_cutoff=gtrain.conf.model["r_cutoff"],
    )

    # ---- INITIALIZE MODEL ---- #

    # Get first data
    first_data = data[next(iter(data_path.keys()))]

    # Construct the model
    example_data = first_data[first_data.info.max_neigh_idx].config
    apply_fn, init_fn = leopold_model(
        gtrain.model_conf,
        example_data.positions,
        example_data.ones_hot,
        example_data.box,
    )

    # Compile the model
    comp_vect_model = jit(vmap(apply_fn, (None, 0, 0, 0)))

    # Initialize model
    key = jrn.PRNGKey(gtrain.conf.general.seed)
    params = init_fn(key)

    # Load from checkpoint
    params = gtrain.load_model(params)
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params)

    # ---- EVALUATE MODEL ---- #

    # Prepare RMSE table
    # Collect results
    table = PrettyTable(
        ["", "energy (meV / Atoms)", "forces (meV/A)", "charges (me)", "magmoms (mµ)"]
    )

    # Go around the dataloaders
    for name in data_path.keys():
        # Get user labels
        labels = gtrain.conf.datasets.labels

        # Modify labels as user requested
        scalar_labels = DEFAULT_SCALAR_LABELS.copy()
        vector_labels = DEFAULT_VECTOR_LABELS.copy()

        if labels is not None:
            if "scalar" in labels.keys():
                scalar_labels.update(labels["scalar"])
            if "vector" in labels.keys():
                vector_labels.update(labels["vector"])

        # Get data destructor
        destr = leopold_data_desctructor(data[name].info, scalar_labels, vector_labels)

        # Gather on traj
        traj, rmse = [], {"energy": 0.0, "forces": 0.0, "charges": 0.0, "magmoms": 0.0}
        for batch in data[name]:
            # Evaluate model
            (energy, magchg), forces = comp_vect_model(params, *batch.config)
            magmoms, charges = jnp.split(magchg, 2, axis=-1)

            # Compute RMSE
            N = jnp.any(batch.config.ones_hot == 1, axis=-1, keepdims=True).sum(
                -2, keepdims=True
            )

            # We want energy per atom
            rmse["energy"] += np.square((energy - batch.labels.energy) / N).sum()
            # Forces are 3D vector so needs to divide by N and 3
            rmse["forces"] += np.sum(jnp.square(forces + batch.labels.forces) / N) / 3
            rmse["charges"] += np.sum(jnp.square(charges - batch.labels.charges) / N)
            rmse["magmoms"] += np.sum(jnp.square(magmoms - batch.labels.magmoms) / N)

            # Create atoms and add evaluation
            atoms = destr(batch)
            for i, atom in enumerate(atoms):
                atom.info[args.label + "_" + scalar_labels["energy"]] = float(energy[i])
                atom.arrays[args.label + "_" + vector_labels["forces"]] = forces[i]
                atom.arrays[args.label + "_" + vector_labels["magmoms"]] = magmoms[i]
                atom.arrays[args.label + "_" + vector_labels["charges"]] = charges[i]

            # Extend traj
            traj.extend(atoms)

        # perform mean and root
        for key, val in rmse.items():
            rmse[key] = np.sqrt(val / len(data))

        # Add to RMSE table
        vals = [v * 1e3 for v in rmse.values()]
        vals = [name] + vals
        table.add_row(vals)

        # Write traj to file
        path = join(dirname(data_path[name]), tag + "_" + name + ".xyz")
        write(path, traj, format="extxyz")

    # Print it
    for tline in table.get_string().split("\n"):
        print(tline)


if __name__ == "__main__":
    main()
