"""
Training script for LEOPOLD model

creation: 2025-05-16 12:49:30
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
import jax.tree_util as tree
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
from leopold.dataset import leopold_load_datasets, Configuration, Labels
from leopold.config import read_leopold_configuration, LeopoldConfiguration
from leopold.observables import leopold_model, evaluate_model
from leopold.hdf5 import LeopoldCheckpointFile, LeopoldState
from leopold import __version__

# Optax
import optax

# ArgumentParser
from argparse import ArgumentParser, Namespace

# Dataclass
from dataclasses import asdict

# TQDM
from tqdm import tqdm

# PreattyTables
from prettytable import PrettyTable

# Typing
from typing import Union, Optional

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
        with open("leopold_default_conf.yaml", "w") as f:
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

    # Get device
    device = jax.devices(gconf.device)[0]

    # Checkpoint setup
    os.makedirs(gconf.models_dir, exist_ok=True)

    fcheck = os.path.join(gconf.models_dir, gconf.checkpoint_file)
    if os.path.isfile(fcheck) and tconf.restart:
        fcheck = LeopoldCheckpointFile(fcheck, "a")
    else:
        fcheck = LeopoldCheckpointFile(fcheck, "w")

    # Get new or existing training in the checkpoint
    gtrain = fcheck.create_training(conf)

    # ---- DATASET

    # if data_paths is empty read from HDF5
    if len(dconf.data_paths) == 0:
        logger.info("Reading dataset from checkpoint")
        data = gtrain.get_datasets()

        # See if training exist
        if "train" not in data.keys():
            raise KeyError(
                "a traing dataset must be given, and was not present either in configuration or checkpoint!"
            )

        # Construct data loader
        data = {k: v.get_dataloader(tconf.batch_size, device) for k, v in data.items()}

    # Read from configuration
    else:
        logger.info("Reading dataset from given paths")

        # See if training exist
        if "train" not in dconf.data_paths:
            raise KeyError("a traing dataset must be given inside the configuration!")

        # Create the dataloaders
        data = leopold_load_datasets(
            dconf.data_paths,
            dconf.labels,
            batch_size=tconf.batch_size,
            device=device,
            r_cutoff=mconf["r_cutoff"],
        )

        # Save dataset
        for key, loader in data.items():
            gtrain.attach_dataset(key, loader, save_data=dconf.save_data)

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
    info = data["train"].info
    logger.info(
        f"Using following energy per atom scale and shift: {data['train'].info.deviations['energy']:.2f} {data['train'].info.averages['energy']:.2f}"
    )
    for key in ["magmoms", "charges"]:
        logger.info(
            f"Using following species dependent values for {key} scale and shift:"
        )
        for s, m, d in zip(info.species, info.averages[key], info.deviations[key]):
            logger.info(f"\t*{s:<2d} -> {d} {m}")
    logger.info(
        f"Average number of neighbours in dataset: {data['train'].info.average_neigh:.2f}"
    )

    # ---- MODEL / OPTIMIZER
    logger.info("Constructing model")

    # Construct the model
    example_data = data["train"][data["train"].info.max_neigh_idx].config
    apply_fn, init_fn = leopold_model(
        gtrain.model_conf,
        example_data.positions,
        example_data.ones_hot,
        example_data.box,
    )

    # Jit the function
    comp_vect_model = jit(vmap(apply_fn, (None, 0, 0, 0)))

    # Construct the optimizer
    opt = optax.chain(
        optax.adam(tconf.learning_rate),
        optax.contrib.reduce_on_plateau(
            0.5, patience=5, accumulation_size=len(data["train"])
        ),  # Use the learning rate from the scheduler.
    )

    # ---- INITIALIZATION
    # Initialize model
    key = jrn.PRNGKey(conf.general.seed)
    params = init_fn(key)

    # Initialize Optimizer
    opt_state = opt.init(params)

    # Load state if restarted
    if tconf.restart and gtrain.step != 0:
        logger.info(f"Restarting from existing training at step {gtrain.step}")

        # Get params and optimizer state
        params, opt_state, _, _ = gtrain.load_state(LeopoldState(params, opt_state, 0))
    # If restarted anew load only model
    elif tconf.restart and fcheck.n_train > 1:
        logger.info("Reading best model from previous existing checkpoint")

        params = gtrain.load_model(params, "best")
        params = tree.tree_map(lambda x: jax.device_put(x, device), params)

    # Give information on total number of parameters
    nparams = np.sum([np.prod(arr.shape) for arr in tree.tree_leaves(params)])
    logger.info(f"Total number of parameters in model {nparams}")

    # ---- LOSS DEFINITION

    # Set the weight for forces, magmom and charges
    tconf.forces_weight *= data["train"][0].config.ones_hot.shape[-2] ** 2
    tconf.magmom_weight *= data["train"][0].config.ones_hot.shape[-2] ** 2
    tconf.charge_weight *= data["train"][0].config.ones_hot.shape[-2] ** 2

    @jit
    def loss_fn(params, conf: Configuration, labels: Labels):
        (energy, magchg), forces = comp_vect_model(
            params, conf.positions, conf.ones_hot, conf.box
        )
        magmom, charge = jnp.split(magchg, 2, axis=-1)

        # Loss of all observables
        e_loss = jnp.square(energy - labels.energy).mean()
        f_loss = jnp.square(forces + labels.forces).mean()
        m_loss = jnp.square(magmom - labels.magmoms).mean()
        c_loss = jnp.square(charge - labels.charges).mean()

        # Sum of all magnetizations
        s_loss = jnp.square(magmom.sum(1) - labels.magmoms.sum(1)).mean()

        loss = (
            tconf.energy_weight * e_loss
            + tconf.forces_weight * f_loss
            + tconf.magmom_weight * m_loss
            + tconf.charge_weight * c_loss
            + tconf.smagmo_weight * s_loss
        )

        return loss, (e_loss, f_loss, m_loss, c_loss, s_loss)

    grad_fn = value_and_grad(loss_fn, has_aux=True)

    # ---- UPDATE DEFINITION
    @jit
    def update_fn(params, opt_state, conf: Configuration, labels: Labels):
        (loss, aux), params_grad = grad_fn(params, conf, labels)

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
        f"{'Total':>35s} {'Energy':>11s} {'Forces':>12s} {'Magmoms':>12s} {'Charges':>12s} {'Magmom Sum':>12s}  {'Total':>27s} {'Energy':>11s} {'Forces':>12s} {'Magmoms':>12s} {'Charges':>12s} {'Magmom Sum':>12s} | {'Learning rate':>12s}"
    )

    for i in range(gtrain.step, tconf.max_epoch):
        if tconf.patience == gtrain.impatience:
            logger.info("Too many iterations without improvement, stopping training")
            break

        # Train loop and loss
        train_loss = {
            "total": 0.0,
            "energy": 0.0,
            "forces": 0.0,
            "magmom": 0.0,
            "charge": 0.0,
            "magsum": 0.0,
        }

        for batch in tqdm(data["train"]):
            params, opt_state, losses = update_fn(
                params, opt_state, batch.config, batch.labels
            )

            for key, loss in zip(train_loss.keys(), losses):
                train_loss[key] += loss

        for key in train_loss.keys():
            train_loss[key] /= len(data["train"])

        # Validation loss
        valid_loss = {
            "total": 0.0,
            "energy": 0.0,
            "forces": 0.0,
            "magmom": 0.0,
            "charge": 0.0,
            "magsum": 0.0,
        }
        for batch in tqdm(data["validation"]):
            losses = loss_fn(params, *batch)

            # Flatten the tuple
            losses = (losses[0], *losses[1])

            for key, loss in zip(valid_loss.keys(), losses):
                valid_loss[key] += loss

        for key in valid_loss.keys():
            valid_loss[key] /= len(data["validation"])

        # Loss logger
        train_log, valid_log = "", ""
        for (key, tval), (_, vval) in zip(train_loss.items(), valid_loss.items()):
            train_log += f"{tval:>12.8f} " if key != "total" else f"{tval:>13.8f}"
            valid_log += f"{vval:>12.8f} " if key != "total" else f"{vval:>13.8f}"

        logger.info(
            f"Epoch {i:4} ==> Train: {train_log}   Validation: {valid_log}| {tconf.learning_rate * optax.tree_utils.tree_get(opt_state, 'scale'):12.8f}"
        )

        # Saving State
        gtrain.update_state(LeopoldState(params, opt_state, float(valid_loss["total"])))

    # ---- FINAL EVALUATION
    logger.info("Finished training, evaluating model...")

    # Get best model
    params = gtrain.load_model(params, "best")
    params = tree.tree_map(lambda x: jax.device_put(x, device), params)

    # Collect results
    table = PrettyTable(
        ["", "energy (meV)", "forces (meV/A)", "charges (me)", "magmoms (mµ)"]
    )

    # Evaluate the model
    for name in data.keys():
        vals = evaluate_model(params, comp_vect_model, data[name])
        vals = [v * 1e3 for v in vals.values()]
        vals = [name] + vals
        table.add_row(vals)

    # Print it
    logger.info(table)


if __name__ == "__main__":
    main()
