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
import jax.numpy as jnp

# Plot
import matplotlib.pyplot as plt

# Logging
import logging
from logging import INFO

# System
import sys
import os

# YAML
import yaml

# Leopold
from leopold.dataset import DEFAULT_SCALAR_LABELS, DEFAULT_VECTOR_LABELS, leopold_load_datasets
from leopold.nn import Leopold
from leopold.config import read_leopold_configuration, LeopoldConfiguration
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
        with open("leopold_default_cong.yaml", "w")as f:
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

    # See if training exist
    if "train" not in dconf.data_paths:
        raise KeyError("a traing dataset must be given inside the configuration!")

    # Create the dataloaders
    data = leopold_load_datasets(dconf.data_paths, dconf.labels, batch_size=dconf.batch_size)

    # Perform the validation splitting if needed
    if "validation" not in data.keys():
        pass

    # Get the training set means
    # TODO: make so that in the info of the dataset the mean and std of labels are 
    #       shown in a species dependent manner. Then add them to the model config


    print(data["train"][0:3])
    # ---- MODEL



if __name__ == "__main__":
    main()
