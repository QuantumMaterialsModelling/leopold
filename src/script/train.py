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
from jax import jit, vmap, value_and_grad

# Logging
import logging
from logging import INFO

# System
import sys
import os

# YAML
import yaml

# Leopold
from leopold.dataset import leopold_load_datasets, LeopoldData, leopold_data_from_hdf5
from leopold.config import read_leopold_configuration, LeopoldConfiguration
from leopold.observables import leopold_model
from leopold.hdf5 import LeopoldCheckpointFile
from leopold import __version__

# Optax
import optax

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
    logger = logging.getLogger("leopold")
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

    return logger


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

    # Read configuration and divide
    conf = read_leopold_configuration(args.configuration)

    dconf = conf.datasets
    gconf = conf.general
    mconf = conf.model
    tconf = conf.training

    # ---- SETUP
    # Set the logger
    logger = setup_logger(gconf.logs_lev, gconf.tag, gconf.logs_dir)

    # Set the default precision
    if conf.general.use_float64:
        jax.config.update("jax_enable_x64", True)

    # Search for existing Checkpoint
    os.makedirs(gconf.models_dir, exist_ok=True)

    fcheck = os.path.join(gconf.models_dir, gconf.checkpoint_file)
    if os.path.isfile(fcheck):
        fcheck = LeopoldCheckpointFile(fcheck, "a")
    else:
        fcheck = LeopoldCheckpointFile(fcheck, "w")

    # ---- DATASET
    logger.info("Reading dataset")

    # if data_paths is empty read from HDF5
    if len(dconf.data_paths) == 0:
        data = fcheck.get_datasets()

        # See if training exist and is not empty
        if "train" not in data.keys():
            raise KeyError(
                "a traing dataset must be given, and was not present either in configuration or checkpoint!"
            )

        # Check training dataset is not empty
        if len(data["train"]) == 0:
            raise ValueError("training dataset inside checkpoint was empty!")

    # Read from configuration
    else:
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

        # Save dataset
        for key, loader in data.items():
            fcheck.write_dataset(key, loader, dconf.save_data)

    # Perform the validation splitting if needed
    if "validation" not in data.keys():
        logger.info(
            f"No validation set present, perform a {int(dconf.val_split * 100)}% split"
        )
        # Compute the splitting idx
        idx = int(len(data["train"]) * dconf.val_split)
        data["validation"], data["train"] = data["train"].split([idx])

    # Tell the user what we know about this
    logger.info("Uploaded succesfully the following datasets:")
    for key, loader in data.items():
        logger.info(f"\t-{key}: {len(loader)} configurations")

    # Tell about other quantities
    logger.info(
        f"Average number of neighbours in dataset: {data['train'].info.average_neigh:.2f}"
    )

    # ---- MODEL
    if tconf.restart:
        logger.info("Taking model from checkpoint")

    else:
        logger.info("Constructing model")

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
    model, init = leopold_model(mconf, *example_data.config)

    # Initialize model
    key = jrn.PRNGKey(conf.general.seed)
    params = init(key)

    # Jit the function
    comp_vect_model = jit(vmap(model, (None, 0, 0, 0)))

    # ---- TRAINING
    tconf = conf.training

    opt = optax.chain(
        optax.adam(tconf.learning_rate),
        optax.contrib.reduce_on_plateau(
            0.5, patience=5, accumulation_size=len(data["train"])
        ),  # Use the learning rate from the scheduler.
    )

    opt_state = opt.init(params)

    # Set the weight for forces, magmom and charges
    tconf.forces_weight *= data["train"][0].config.ones_hot.shape[-2] ** 2
    tconf.magmom_weight *= data["train"][0].config.ones_hot.shape[-2] ** 2
    tconf.charge_weight *= data["train"][0].config.ones_hot.shape[-2] ** 2

    @jit
    def loss_fn(params, batch: LeopoldData):
        (energy, magchg), forces = comp_vect_model(params, *batch.config)

        magmom, charge = jnp.split(magchg, 2, -1)

        # Loss of all observables
        e_loss = jnp.square(energy - batch.labels.energy).mean()
        f_loss = jnp.square(forces + batch.labels.forces).mean()
        m_loss = jnp.square(magmom - batch.labels.magmoms).mean()
        c_loss = jnp.square(charge - batch.labels.charges).mean()

        # Sum of all magnetizations
        s_loss = jnp.square(magmom.sum(-1) - batch.labels.magmoms.sum(-1)).mean()

        loss = (
            tconf.energy_weight * e_loss
            + tconf.forces_weight * f_loss
            + tconf.magmom_weight * m_loss
            + tconf.charge_weight * c_loss
            + tconf.smagmo_weight * s_loss
        )

        return loss, (e_loss, f_loss, m_loss, c_loss, s_loss)

    # Define the training function
    @jit
    def update(params, opt_state, batch: LeopoldData):
        grad_fn = value_and_grad(loss_fn, has_aux=True)

        (loss, aux), params_grad = grad_fn(params, batch)

        updates, opt_state = opt.update(
            params_grad, opt_state, params, value=jnp.float32(loss)
        )

        return (
            optax.apply_updates(params, updates),
            opt_state,
            (loss, *aux),
        )

    # ---- START TRAINING LOOP
    logger.info("Starting training")

    logger.info(
        f"{'Total':>35s} {'Energy':>11s} {'Forces':>12s} {'Magmoms':>12s} {'Charges':>12s} {'Magmom Sum':>12s}  {'Total':>27s} {'Energy':>11s} {'Forces':>12s} {'Toccup':>12s} {'Magmom Sum':>12s} | {'Learning rate':>12s}"
    )

    lowest_loss, patience_count = jnp.inf, 0
    for i in range(tconf.max_epoch):
        if patience_count == tconf.patience:
            logger.info("Too many iterations without improvement, stopping training")
            break

        # Train loop and loss
        train_loss = {
            "total": jnp.array([]),
            "energy": jnp.array([]),
            "forces": jnp.array([]),
            "magmom": jnp.array([]),
            "charge": jnp.array([]),
            "magsum": jnp.array([]),
        }
        for batch in data["train"]:
            params, opt_state, losses = update(params, opt_state, batch)

            for key, loss in zip(train_loss.keys(), losses):
                train_loss[key] = jnp.append(train_loss[key], loss)

        for key in train_loss.keys():
            train_loss[key] = train_loss[key].mean()

        # Validation loss
        valid_loss = {
            "total": jnp.array([]),
            "energy": jnp.array([]),
            "forces": jnp.array([]),
            "magmom": jnp.array([]),
            "charge": jnp.array([]),
            "magsum": jnp.array([]),
        }
        for batch in data["validation"]:
            losses = loss_fn(params, batch)

            # Flatten the tuple
            losses = (losses[0], *losses[1])

            for key, loss in zip(valid_loss.keys(), losses):
                valid_loss[key] = jnp.append(valid_loss[key], loss)

        for key in valid_loss.keys():
            valid_loss[key] = valid_loss[key].mean()

        # Loss logger
        train_log, valid_log = "", ""
        for (key, tval), (_, vval) in zip(train_loss.items(), valid_loss.items()):
            train_log += f"{tval:>12.8f} " if key != "total" else f"{tval:>13.8f}"
            valid_log += f"{vval:>12.8f} " if key != "total" else f"{vval:>13.8f}"

        logger.info(
            f"Epoch {i:4} ==> Train: {train_log}   Validation: {valid_log}| {tconf.learning_rate * optax.tree_utils.tree_get(opt_state, 'scale'):12.8f}"
        )

        # Saving checkpoints
        # os.makedirs(args.checkpoints_dir, exist_ok=True)
        # with open(check_path + "L.pkl", "wb") as f:
        #     pickle.dump([config, params, opt_state], f)

        if valid_loss["total"] < lowest_loss:
            lowest_loss = valid_loss["total"]
            patience_count = 0

            # with open(check_path + ".pkl", "wb") as f:
            #     pickle.dump([config, params, opt_state], f)
        else:
            patience_count += 1


if __name__ == "__main__":
    main()
