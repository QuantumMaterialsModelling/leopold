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
from leopold.hdf5 import LeopoldCheckpointFile, LeopoldH5MDState, LeopoldH5MDWriter
from leopold.observables import leopold_model
from leopold.dataset import leopold_data_constructor, read, Configuration
from leopold.config import read_leopold_md_options
from leopold import __version__

# Argument Parser
from argparse import ArgumentParser, Namespace

# Functools
from functools import partial

# ASE
from ase.io import read as ase_read
from ase.units import kB, fs
from ase.symbols import Symbols

# Typing
from typing import Union, Optional, Callable

# DATACLASS
from jax_md import dataclasses

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

# ==== CLASSES ==== #


@dataclasses.dataclass
class NVTNoseHooverState:
    """State information for an NVT system with a Nose-Hoover chain thermostat.

    Attributes:
      position: The current position of particles. An ndarray of floats
        with shape `[n, spatial_dimension]`.
      momentum: The momentum of particles. An ndarray of floats
        with shape `[n, spatial_dimension]`.
      force: The current force on the particles. An ndarray of floats with shape
        `[n, spatial_dimension]`.
      mass: The mass of the particles. Can either be a float or an ndarray
        of floats with shape `[n]`.
      chain: The variables describing the Nose-Hoover chain.
    """

    position: Array
    momentum: Array
    force: Array
    mass: Array
    ones_hot: Array
    chain: simulate.NoseHooverChain

    @property
    def velocity(self):
        return self.momentum / self.mass


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

        fh = logging.FileHandler(os.path.join(directory, tag + ".log"), "w")
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger


def nvt_nose_hoover(
    model: Callable,
    shift_fn: Callable,
    dt: float,
    kT: float,
    npol: int,
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
        state = NVTNoseHooverState(R, None, -forces, mass, ones_hot, None)
        state = simulate.canonicalize_mass(state)
        state = simulate.initialize_momenta(state, key, _kT)
        KE = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
        return state.set(chain=thermostat.initialize(dof, KE, _kT))

    @jit
    def apply_fn(state, box: Array, **kwargs):
        _kT = kT if "kT" not in kwargs else kwargs["kT"]

        chain = state.chain

        chain = thermostat.update_mass(chain, _kT)

        p, chain = thermostat.half_step(state.momentum, chain, _kT)
        state = state.set(momentum=p)

        # ---- VERLET
        state = simulate.momentum_step(state, dt_2)
        state = simulate.position_step(state, shift_fn, dt, **kwargs)

        # Compute forces
        (energy, magchg), forces = model(state.position, state.ones_hot, box)
        state = state.set(force=-forces)

        state = simulate.momentum_step(state, dt_2)

        chain = chain.set(
            kinetic_energy=quantity.kinetic_energy(
                momentum=state.momentum, mass=state.mass
            )
        )

        p, chain = thermostat.half_step(state.momentum, chain, _kT)
        state = state.set(momentum=p, chain=chain)

        # Set new polaron position
        magmoms, charges = jnp.split(magchg, 2, axis=-1)
        magmoms = magmoms.flatten()
        pol_state = jnp.argsort(jnp.abs(magmoms))

        ones_hot = state.ones_hot.at[:, -1].set(0)
        ones_hot = ones_hot.at[pol_state[-npol:], -1].set(1)

        state = state.set(ones_hot=ones_hot)

        # Return everithing
        return state, (energy, magmoms, charges, ones_hot[:, -1])

    return init_fn, apply_fn


# ==== PARSER ==== #


def parse_args() -> Namespace:
    parser = ArgumentParser("run_md")

    parser.add_argument("input")

    return parser.parse_args()


# ==== MAIN ==== #


def main():
    print(GREETING)

    # Argument parsing
    args = parse_args()

    # read options
    opts = read_leopold_md_options(args.input)

    # ---- SETUP

    # Retrieve training group
    gtrain = LeopoldCheckpointFile(opts.model_path, "r").get_training()

    # Select tag
    tag = gtrain.conf.general.tag
    if opts.name is not None:
        tag = opts.name

    # Set the logger
    logger = setup_logger(opts.logs_level, tag, opts.logs_dir)

    # Set the default precision
    if opts.use_float64:
        jax.config.update("jax_enable_x64", True)

    # Select the devices
    device = jax.devices(opts.device)[0]

    # Get data constructor
    info = gtrain.get_dataset("train").info
    info.max_num_atoms = len(ase_read(opts.start_path))

    f = leopold_data_constructor(
        info,
        {},
        {"magmoms": gtrain.conf.datasets.labels["vector"]["magmoms"]},
        device,
    )

    logger.info("LEOPOLD molecular dynamic script")

    # ---- READ ATOMS ---- #
    logger.info(f"Reading starting configuration from {opts.start_path}")

    atoms = read(
        opts.start_path,
        {},
        {"magmoms": gtrain.conf.datasets.labels["vector"]["magmoms"]},
        "0",
    )

    # Save info
    mass = atoms[0].get_masses()
    spec = atoms[0].get_atomic_numbers()
    elem, nele = np.unique(spec, return_counts=True)

    # Get information on the configuration
    positions, ones_hot, box = Configuration(
        *jax.tree_util.tree_map(lambda x: x[0], f(atoms).config)
    )

    logger.info(f"Read configuration with {len(atoms[0])} ions of types:")
    for e, n in zip(elem, nele):
        logger.info(f"\t*{Symbols([e]).get_chemical_formula():<2s} -> {n}")
    logger.info(f"Configuration contains {int(ones_hot[:, -1].sum())} polarons")

    # ---- INITIALIZE MODEL ---- #
    logger.info("Initializing model")
    mconf = gtrain.model_conf
    apply_fn, init_fn = leopold_model(mconf, positions, ones_hot, box)

    # initialize
    key, rng = jrn.split(jrn.PRNGKey(opts.rng_seed))
    params = init_fn(rng)

    # Load from checkpoint
    params = gtrain.load_model(params)
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params)

    # Compile the model
    model = jax.jit(partial(apply_fn, params))

    # Create metric
    _, shift = space.periodic_general(box, wrapped=False)

    # Create state
    dy_init_fn, step_fn = nvt_nose_hoover(
        model,
        shift,
        opts.time_step * fs,
        opts.temperature * kB,
        int(ones_hot[:, -1].sum()),
    )
    state = dy_init_fn(key, positions, ones_hot, box, mass=mass)

    # Setup HDF5
    os.makedirs(opts.traj_dir, exist_ok=True)

    writer = LeopoldH5MDWriter(
        os.path.join(opts.traj_dir, tag + "-traj.h5"),
        spec,
        n_frames=opts.max_steps // opts.traj_int,
        compression=opts.traj_compr_backend,
        compression_opts=opts.traj_compr_level,
        velocities=opts.traj_write_velocity,
        forces=opts.traj_write_forces,
        author=opts.traj_author_name,
        author_email=opts.traj_author_mail,
    )

    # ---- RUN
    logger.info("Run MD")
    for i in range(opts.max_steps):
        # perform the step
        state, (energy, magmoms, charges, pol_state) = step_fn(state, box)

        # Collect temperature
        temp = quantity.temperature(momentum=state.momentum, mass=state.mass) / kB

        # Just checking for divergences
        if jnp.isnan(energy) or jnp.isnan(temp):
            break

        # Log results
        if i % opts.logs_int == 0:
            if i == 0:
                logger.info(
                    f"{'MD Time[ps]':>13s} "
                    f"{'Etot[eV]':>12s} {'Epot[eV]':>12s} {'Temp[K]':>12s} "
                    f"{'Polaron[idx]':>14s} {'Pol. Mag.[μ]':>17s} "
                    f"{'2˚ Pol.[idx]':>14s} {'2˚ Mag.[μ]':>17s}"
                    f"{'Tot. Mag.[μ]':>17s}"
                )

            pol = jnp.argsort(jnp.abs(magmoms))

            logger.info(
                f"{i * opts.time_step * 1e-3:13.3f} "
                f"{quantity.kinetic_energy(momentum=state.momentum, mass=state.mass) + energy:12.3f} "
                f"{energy:12.3f} "
                f"{temp:12.3f} "
                f"{int(pol[-1]):14d} {magmoms[pol[-1]]:17.3f}"
                f"{int(pol[-2]):15d} {magmoms[pol[-2]]:17.3f}"
                f"{magmoms.sum():17.3f}"
            )

        # Write frame to trajectory
        if i % opts.traj_int == 0:
            # Save positions in Angstrom
            position = jnp.matmul(state.position, box.T)

            # Create the state
            h5state = LeopoldH5MDState(
                i, opts.time_step, np.asarray(position), np.asarray(box)
            )

            # Update with requested info
            if opts.traj_write_velocity:
                h5state.__setattr__("velocities", np.asarray(state.velocity))

            if opts.traj_write_forces:
                h5state.__setattr__("forces", np.asarray(state.force))

            # Create observables
            observables = {
                "energy": float(energy),
                "temperature": float(temp),
                "magmoms": np.asarray(magmoms),
                "charges": np.asarray(charges),
                "polarons": np.asarray(pol_state),
            }

            # Write stuff
            writer(h5state, observables)
    logger.info("DONE")


if __name__ == "__main__":
    main()
