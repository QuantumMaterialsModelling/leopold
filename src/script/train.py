"""
Training script for LEOPOLD model

creation: 2025-05-16 12:49:30
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== IMPORTS ==== #

# Math
import numpy as np

# JAX
import jax
import jax.numpy as jnp
import jax.random as jrn

# Logging
import logging
from logging import INFO

# System
import sys
import os

# YAML
import yaml

# Leopold
from leopold.dataset import leopold_load_datasets
from leopold.config import read_leopold_configuration, LeopoldConfiguration
from leopold.observables import leopold_model
from leopold import __version__

# ArgumentParser
from argparse import ArgumentParser, Namespace

# Dataclass
from dataclasses import asdict

# Typing
from typing import Union, Optional, Any

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

# ==== FUNCTIONS ==== #


def setup_logger(
    level: Union[int, str] = INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create general formatting
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Create standard output Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler
    if (directory is not None) and (tag is not None):
        os.makedirs(directory, exist_ok=True)

        fh = logging.FileHandler(os.path.join(directory, tag + ".log"))
        fh.setFormatter(formatter)

        logger.addHandler(fh)


# ==== PARSER ==== #


def parse_args() -> Namespace:
    parser = ArgumentParser("leopold_train")

    parser.add_argument(
        "configuration",
        help="YAML configuration file containing all the informations",
        nargs="?",
        default=None,
    )

    return parser.parse_args()


# ==== MAIN ==== #


def main():
    print(GREETING)

    # Argument parsing
    args = parse_args()

    # If no configuration is given create one
    # with standard arguments and finish!
    # TODO: create a way to print a default minimal
    # configuration with some description of the options
    if args.configuration is None:
        conf = asdict(LeopoldConfiguration())
        with open("leopold_default_cong.yaml", "w") as f:
            yaml.safe_dump(conf, f)

        return

    # Read configuration
    conf = read_leopold_configuration(args.configuration)

    # ---- SETUP
    tag = f"{conf.general.name}-{conf.general.seed}"

    setup_logger(conf.general.logs_lev, tag, conf.general.logs_dir)

    # ---- DATASET
    logging.info("Reading dataset")
    dconf = conf.datasets
    mconf = conf.model

    # See if training exist
    if "train" not in dconf.data_paths:
        raise KeyError("a traing dataset must be given inside the configuration!")

    # Create the dataloaders
    data = leopold_load_datasets(
        dconf.data_paths,
        dconf.labels,
        batch_size=dconf.batch_size,
        r_cutoff=mconf["r_cutoff"],
    )

    # Perform the validation splitting if needed
    if "validation" not in data.keys():
        logging.info(
            f"No validation set present, perform a {int(dconf.val_split * 100)}% split"
        )
        # Compute the splitting idx
        idx = int(len(data["train"]) * dconf.val_split)
        data["validation"], data["train"] = data["train"].split([idx])

    # Tell the user what we know about this
    logging.info("Uploaded succesfully the following datasets:")
    for key, loader in data.items():
        logging.info(f"\t-{key}: {len(loader)} configurations")

    # Tell about other quantities
    logging.info(
        f"Average number of neighbours in dataset: {data['train'].info.average_neigh:.2f}"
    )

    # ---- MODEL
    logging.info("Constructing model")

    # Take number of elements in the dataset
    mconf["n_elems"] = len(data["train"].info.species) + 1

    # Set the average number of neighbours
    mconf["n_neighbour"] = data["train"].info.average_neigh

    # Get energy scale and shift
    mconf["energy_shift"] = data["train"].get_mean("energy")
    mconf["energy_scale"] = data["train"].get_std("energy")

    # Get species dependent scale and shift
    mconf["magchg_shift"] = jnp.append(
        data["train"].get_mean("magmoms"), data["train"].get_mean("charges"), 1
    )
    mconf["magchg_shift"] = jnp.append(
        data["train"].get_std("magmoms"), data["train"].get_std("charges"), 1
    )

    # TODO:
    # tell about average energy and stuff

    # Construct the model
    example_data = data["train"][data["train"].info.max_neigh_idx]
    apply, init = leopold_model(mconf, *example_data.config)

    # Initialize model
    key = jrn.PRNGKey(conf.general.seed)
    param = init(key)


if __name__ == "__main__":
    main()
