"""
MD script for the Leopoldo package

creation: 2025-05-26 11:26:07
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== IMPORTS ==== #

# JAX
import jax
import jax.random as jrn
import jax.numpy as jnp
from jax import Array, jit

# JAX-MD
from jax_md import simulate, quantity, space

# MATH
import numpy as np

# Logging
import logging
from logging import INFO

# System
import sys
import os

# Leopold
from leopold.hdf5 import LeopoldCheckpointFile
from leopold.observables import leopold_model
from leopold.dataset import leopold_data_constructor, read, Configuration
from leopold import __version__

# Argument Parser
from argparse import ArgumentParser, Namespace

# Functools
from functools import partial

# ASE
from ase.io import read as ase_read
from ase.units import kB, fs

# Typing
from typing import Union, Optional, Callable

# TESTING
from tqdm import tqdm

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


def nvt_nose_hoover(
    model: Callable,
    shift_fn: Callable,
    dt: float,
    kT: float,
    chain_length: int = 5,
    chain_steps: int = 2,
    sy_steps: int = 3,
    tau: Optional[float] = None,
    **sim_kwargs,
):
    dt = jnp.float32(dt)
    dt_2 = jnp.float32(dt / 2)
    if tau is None:
        tau = dt * 100

    thermostat = simulate.nose_hoover_chain(
        dt, chain_length, chain_steps, sy_steps, jnp.float32(tau)
    )

    def init_fn(
        key: Array,
        R: Array,
        ones_hot: Array,
        box: Array,
        mass=jnp.float32(1.0),
        **kwargs,
    ):
        _kT = kT if "kT" not in kwargs else kwargs["kT"]

        dof = quantity.count_dof(R)

        (_, _), forces = model(R, ones_hot, box)
        state = simulate.NVTNoseHooverState(R, None, -forces, mass, None)
        state = simulate.canonicalize_mass(state)
        state = simulate.initialize_momenta(state, key, _kT)
        KE = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
        return state.set(chain=thermostat.initialize(dof, KE, _kT))

    @jit
    def apply_fn(state, ones_hot: Array, box: Array, **kwargs):
        _kT = kT if "kT" not in kwargs else kwargs["kT"]

        chain = state.chain

        chain = thermostat.update_mass(chain, _kT)

        p, chain = thermostat.half_step(state.momentum, chain, _kT)
        state = state.set(momentum=p)

        # ---- VERLET
        state = simulate.momentum_step(state, dt_2)
        state = simulate.position_step(state, shift_fn, dt, **kwargs)

        # Compute forces
        aux, forces = model(state.position, ones_hot, box)
        state = state.set(force=-forces)

        state = simulate.momentum_step(state, dt_2)

        chain = chain.set(
            kinetic_energy=quantity.kinetic_energy(
                momentum=state.momentum, mass=state.mass
            )
        )

        p, chain = thermostat.half_step(state.momentum, chain, _kT)
        state = state.set(momentum=p, chain=chain)

        return state, aux

    return init_fn, apply_fn


# ==== PARSER ==== #


def parse_args() -> Namespace:
    parser = ArgumentParser("run_md")

    # Main argument
    parser.add_argument("conf_path", type=str)
    parser.add_argument("model_path", type=str)

    # Simulation options
    parser.add_argument("--time_step", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--num_steps", type=int, default=10_000_000)

    # Logging options
    parser.add_argument("--logs_int", type=int, default=5)
    parser.add_argument("--logs_lev", help="log level", type=str, default="INFO")
    parser.add_argument("--logs_dir", type=str, default="logs")

    # General options
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", type=str, choices=["cpu", "gpu", "tpu"], default="gpu"
    )
    parser.add_argument("--use_float64", action="store_true")

    return parser.parse_args()


# ==== MAIN ==== #


def main():
    print(GREETING)

    # Argument parsing
    args = parse_args()

    # ---- SETUP

    # Retrieve training group
    gtrain = LeopoldCheckpointFile(args.model_path, "r").get_training()

    # Select tag
    tag = gtrain.conf.general.tag
    if args.name is not None:
        tag = args.name

    # Set the logger
    logger = setup_logger(args.logs_lev, tag, args.logs_dir)

    # Set the default precision
    if args.use_float64:
        jax.config.update("jax_enable_x64", True)

    # Select the device
    device = jax.devices()[0]

    # Get data constructor
    info = gtrain.get_dataset("train").info
    info.max_num_atoms = len(ase_read(args.conf_path))

    f = leopold_data_constructor(
        info,
        {},
        {"magmoms": gtrain.conf.datasets.labels["vector"]["magmoms"]},
        device,
    )

    # ---- READ ATOMS ---- #
    atoms = read(
        args.conf_path,
        {},
        {"magmoms": gtrain.conf.datasets.labels["vector"]["magmoms"]},
        "0",
    )

    # Save masses
    mass = atoms[0].get_masses()

    # Get information on the configuration
    positions, ones_hot, box = Configuration(
        *jax.tree_util.tree_map(lambda x: x[0], f(atoms).config)
    )

    # ---- INITIALIZE MODEL ---- #
    mconf = gtrain.model_conf
    apply_fn, init_fn = leopold_model(mconf, positions, ones_hot, box)

    # initialize
    key, rng = jrn.split(jrn.PRNGKey(args.seed))
    params = init_fn(rng)

    # Load from checkpoint
    params = gtrain.load_model(params)
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params)

    # Compile the model
    model = jax.jit(partial(apply_fn, params))

    # Create metric
    _, shift = space.periodic_general(box)

    # Create state
    dy_init_fn, step_fn = nvt_nose_hoover(
        model, shift, args.time_step * fs, args.temperature * kB
    )
    state = dy_init_fn(key, positions, ones_hot, box, mass=mass)

    # Test steps
    for i in range(args.num_steps):
        state, (energy, magchg) = step_fn(state, ones_hot, box)

        magmoms, _ = jnp.split(magchg, 2, axis=-1)
        magmoms = magmoms.flatten()
        pol_state = jnp.argsort(jnp.abs(magmoms))

        ones_hot = ones_hot.at[:, -1].set(0)
        ones_hot = ones_hot.at[pol_state[-1], -1].set(1)

        temp = quantity.temperature(momentum=state.momentum, mass=state.mass) / kB

        # Log results
        if i % args.logs_int == 0:
            logger.info(
                f"{i * args.time_step * 1e-3:13.3f} "
                f"{quantity.kinetic_energy(momentum=state.momentum, mass=state.mass) + energy:12.3f} "
                f"{energy:12.3f} "
                f"{temp:12.3f} "
                f"{int(pol_state[-1]):14d} {magmoms[pol_state[-1]]:17.3f}"
                f"{int(pol_state[-2]):15d} {magmoms[pol_state[-2]]:17.3f}"
                f"{magmoms.sum():17.3f}"
            )


if __name__ == "__main__":
    main()
