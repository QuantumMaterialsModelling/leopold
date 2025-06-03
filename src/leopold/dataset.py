"""
Module containing the objects and funciton that allows to load datasets

creation: 2025-05-05 11:46:40
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# MATH
import numpy as np
from numpy.typing import NDArray

# ASE
from ase import Atoms
from ase.io import read as ase_read

# JAX
import jax
import jax.random as jrn
import jax.numpy as jnp
from jax import Array, Device, vmap
from jax.typing import ArrayLike

# JAX_MD
from jax_md.space import periodic_general

# DATACLASS
from dataclasses import dataclass

# Typing
from typing import Callable, NamedTuple, Optional

# ==== DATA DEFAULTS ==== #

DEFAULT_SCALAR_LABELS = {"energy": "energy"}
DEFAULT_VECTOR_LABELS = {
    "forces": "forces",
    "charges": "decomposed_charges",
    "magmoms": "decomposed_magmoms",
}


# ==== OBJECTS ==== #


class Configuration(NamedTuple):
    """Defines a configuration in the Leopold model

    Attributes:
        position: Position of the ions
        ones_hot: Ones-hot encoding of the species with polaronic character
        box: Simulation box of the configuration
    """

    positions: Array  # [.., N, 3]
    ones_hot: Array  # [.., N, species]
    box: Array  # [.., 3, 3]


class Labels(NamedTuple):
    """Defines labels associated to a configuration

    Attributes:
        energy: energy of the configuration
        forces: forces acting on the ions
        magmoms: s, p, d or f decomposed magmom of every ion based on polaron character
        charges: s, p, d or f decomposed charge of every ion based on polaron character
        stress: stress applied to the configuration
    """

    energy: Array  # [..,]
    forces: Array  # [.., N, 3]
    magmoms: Array  # [.., N, 1]
    charges: Array  # [.., N, 1]
    # stress: Array | None = None  # [.., N]


@dataclass
class LeopoldDataInfo:
    """Info on the Leopold dataset

    Attributes:
        species: species present in the dataset
        pol_types: character of the polaron present (s, p, d or f)
        max_num_atoms: maximum number of atoms in dataset, used for padding
        average_neigh: average number of neighbours inside the dataset
        max_neigh_idx: index of the structure that will require the larger memory to allocate the neighbour list
        averages: dictionary with the averages of the label quantities inside the dataset
        deviations: dictionary witht he standard deviations of the label quantities inside the dataset
    """

    species: NDArray
    pol_types: NDArray
    max_num_atoms: int
    average_neigh: float
    max_neigh_idx: int

    averages: dict[str, NDArray]
    deviations: dict[str, NDArray]


class LeopoldData(NamedTuple):
    """Description of a Leopold data entry

    Attributes:
        config: atomic and polaronic configuration
        labels: labels related to the configuration
    """

    config: Configuration
    labels: Labels

    def __len__(self) -> int:
        return self.config.box.shape[0]


class LeopoldDataLoader:
    """Data loader for the Leopold model

    Data loader that works as a iterator uploading one batch of data at a time
    on the GPU. Allows also for the possibility of loading everithing in one
    go to the GPU and make it return the batches without further overhead.

    Attributes:
        info: general information on the dataset
        data: list LeopoldData object loaded on the device
        nbatches: number of batches present
        batch_size: size of every batch
        device: jax device on which to upload the data
    """

    info: LeopoldDataInfo
    data: LeopoldData

    batch_size: int
    device: Device

    def __init__(
        self,
        raw_data: list[Atoms] | LeopoldData,
        batch_size: int = 1,
        device: Device = jax.devices("cuda")[0],
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        # Set the device
        self.device = device

        # If LeopoldData are directly given simply accept them
        if isinstance(raw_data, LeopoldData):
            self.data = raw_data

            if "info" not in kwargs:
                raise ValueError(
                    "when LeopoldData are given to data loader then info on the dataset must be given by the user (cannot infer types)"
                )
            self.info = kwargs["info"]
        # Real raw_data were given
        else:
            # Get kwargs
            scalar_labels = kwargs.get("scalar_labels", DEFAULT_SCALAR_LABELS)
            vector_labels = kwargs.get("vector_labels", DEFAULT_VECTOR_LABELS)

            # See if info was given
            if "info" not in kwargs:
                # Get the cutoff radius
                r_cutoff = kwargs.get("r_cutoff", None)

                # Load data from list of ASE atoms
                self.info = leopold_data_info(
                    raw_data, scalar_labels, vector_labels, r_cutoff
                )
            else:
                self.info = kwargs["info"]

            # Get the data constructor function
            cpu_device = jax.devices("cpu")[0]
            f = leopold_data_constructor(
                self.info, scalar_labels, vector_labels, self.device
            )

            # Preload the data in batches
            self.data = f(raw_data)

        # Compute the number of batches
        self.batch_size = batch_size

        # Shuffle if wanted
        if shuffle:
            self.shuffle(jrn.PRNGKey(0))

    @property
    def nbatches(self) -> int:
        n = len(self.data) // self.batch_size

        if len(self.data) % self.batch_size != 0:
            return n + 1
        return n

    def shuffle(self, rng_key: Array) -> None:
        perm = jrn.permutation(rng_key, len(self))

        conf = self.data.config._asdict()
        labe = self.data.labels._asdict()

        new_conf = {k: v[perm] for k, v in conf.items()}
        new_labe = {k: v[perm] if v is not None else None for k, v in labe.items()}

        self.data = LeopoldData(Configuration(**new_conf), Labels(**new_labe))  # pyright: ignore

    def split(self, idx):
        conf = self.data.config._asdict()
        labe = self.data.labels._asdict()

        new_conf = {k: jnp.split(v, idx, axis=0) for k, v in conf.items()}
        new_labe = {
            k: jnp.split(v, idx, axis=0) if v is not None else None
            for k, v in labe.items()
        }

        new_conf = [{k: v[i] for k, v in new_conf.items()} for i in range(len(idx) + 1)]
        new_labe = [
            {k: v[i] if v is not None else None for k, v in new_labe.items()}
            for i in range(len(idx) + 1)
        ]

        data = [
            LeopoldData(Configuration(**co), Labels(**la))  # pyright: ignore
            for co, la in zip(new_conf, new_labe)
        ]

        return [
            LeopoldDataLoader(
                d, info=self.info, batch_size=self.batch_size, device=self.device
            )
            for d in data
        ]

    def __iter__(self):
        self.idx = 0

        return self

    def __next__(self) -> LeopoldData:
        # See if we hit the end
        if self.idx >= self.nbatches:
            raise StopIteration()

        # Set idx and increase internal
        idx = self.idx
        self.idx += 1

        # Get beginning and end
        beg = idx * self.batch_size
        end = beg + self.batch_size

        return self[beg:end]

    def __getitem__(self, idx) -> LeopoldData:
        conf = self.data.config._asdict()
        labe = self.data.labels._asdict()

        new_conf = {k: v[idx] for k, v in conf.items()}
        new_labe = {k: v[idx] if v is not None else None for k, v in labe.items()}

        return LeopoldData(Configuration(**new_conf), Labels(**new_labe))  # pyright: ignore

    def __len__(self) -> int:
        return self.data.config.box.shape[0]


# ==== GENERAL FUNCTIONS ==== #


def read(
    path: str, scalar_labels: dict[str, str], vector_labels: dict[str, str]
) -> list[Atoms]:
    """Custom data reader

    Read raw data from file and return a list of ASE Atoms object containing everithing.
    The output Atoms have the energy and forces put in the info and arrays variable respectively,
    so that is compatible wiht the rest of the code standards. Also, if a single vector is
    given for magmoms and charges of dimension (N_atoms,) then it is rearranged to be of size
    (N_atoms, 1) compatible with other methods.

    Args:
        path: path to the raw data file

    Returns:
        list of ASE Atoms
    """
    data = ase_read(path, index=":", format="extxyz")
    if not isinstance(data, list):
        data = [data]

    # Rearrange some entries
    for atoms in data:
        if scalar_labels["energy"] == "energy":
            atoms.info["energy"] = atoms.get_potential_energy()

        if vector_labels["forces"] == "forces":
            atoms.arrays["forces"] = atoms.get_forces()

        magmom_label = vector_labels["magmoms"]
        charge_label = vector_labels["charges"]
        if atoms.arrays[magmom_label].ndim == 1:
            atoms.arrays[magmom_label] = atoms.arrays[magmom_label][:, np.newaxis]
            atoms.arrays[charge_label] = atoms.arrays[charge_label][:, np.newaxis]

    return data


# ==== INFO FUNCTIONS ==== #


def _get_average_num_neighbour(cell: ArrayLike, positions: ArrayLike, r_max) -> float:
    c, p = jnp.asarray(cell), jnp.asarray(positions)
    distance, _ = periodic_general(c)

    dist_matrix = vmap(vmap(distance, (None, 0)), (0, None))(p, p)
    dist_matrix = jnp.linalg.norm(dist_matrix, axis=-1)

    return (dist_matrix < r_max).sum() - len(p)


def leopold_data_info(
    raw_data: list[Atoms],
    scalar_labels: dict[str, str],
    vector_labels: dict[str, str],
    r_cutoff: Optional[float] = None,
) -> LeopoldDataInfo:
    """Get Leopold data info from raw data

    Get object containing general info about the loaded dataset.

    Args:
        raw_data: raw list of ASE atoms containg the data
        scalar_labels: labels for the scalar quantities
        vector_labels: labels for the vectorial quantities
        r_cutoff: cutoff radius used to perform a neighbour analysis

    Returns:
        LeopoldDataInfo object

    Raises:
        NotImplementedError: Leopold is not yet capable of handling polarons of multiple type (s, p, d or f) present at the same time!
    """
    # ---- Collect all informations

    # Definition of storing variables
    elements, neighs, max_num_atoms = {}, [], 0
    mean_scalar = {k: 0 for k in scalar_labels.keys()}
    std_scalar = {k: 0 for k in scalar_labels.keys()}
    mean_vector = {k: {} for k in vector_labels.keys()}
    std_vector = {k: {} for k in vector_labels.keys()}

    # Main loop over the dataset
    for atoms in raw_data:
        # See if this is the frame with maximum atoms
        max_num_atoms = max([max_num_atoms, len(atoms)])

        # Get species and add possible new elements
        atomic_num = atoms.get_atomic_numbers()
        elements_a, counts_e = np.unique(atomic_num, return_counts=True)

        for e in elements_a:
            if e not in elements.keys():
                elements[e] = 0
                for key in vector_labels.keys():
                    mean_vector[key][e] = 0
                    std_vector[key][e] = 0

        # Add counting of elements
        for e, count in zip(elements_a, counts_e):
            elements[e] += count

        # Get scalar labels
        for key, name in scalar_labels.items():
            val = atoms.info[name]

            # Energy is stored as per atom
            if key == "energy":
                val /= len(atoms)

            mean_scalar[key] += val
            std_scalar[key] += val * val

        # Get vector labels
        for key, name in vector_labels.items():
            for e, count in zip(elements_a, counts_e):
                # Get atoms of that species
                idxs = atomic_num == e

                # Do calculations
                mean_vector[key][e] += np.sum(atoms.arrays[name][idxs], axis=0)
                std_vector[key][e] += np.sum(atoms.arrays[name][idxs] ** 2, axis=0)

        # Neighbour analysis
        if r_cutoff is not None:
            neighs.append(
                _get_average_num_neighbour(
                    atoms.cell.array, atoms.get_scaled_positions(), r_cutoff
                )
            )

    # ---- Analysis

    # Get species in order
    species = np.sort([k for k in elements.keys()])

    # perform final average
    averages: dict[str, NDArray] = {
        k: np.asarray(v / len(neighs)) for k, v in mean_scalar.items()
    }
    deviations = {
        k: np.sqrt(v / len(neighs) - averages[k] ** 2) for k, v in std_scalar.items()
    }

    for k, v in mean_vector.items():
        mean, std = [], []
        for e in species:
            m = v[e] / elements[e]
            s = std_vector[k][e] / elements[e]

            mean.append(m)
            std.append(np.sqrt(s - m**2))
        averages[k] = np.asarray(mean)
        deviations[k] = np.asarray(std)

    # Final neighbour quantities
    if len(neighs) != 0:
        avg_neigh = np.sum(neighs) / sum(elements.values())
        max_neigh = int(np.argmax(neighs))
    else:
        avg_neigh, max_neigh = 1.0, -1

    # ---- Search polaron character

    # see total electron minus average
    pol_character = []
    for e in species:
        val = np.abs(np.round(mean_vector["magmoms"][e] / len(raw_data)))
        pol_character.append(np.arange(len(val))[val > 0])
    pol_character = np.unique(np.concatenate(pol_character))

    # Exctract only component of mag and chg related to polaronic character
    for key in ["magmoms", "charges"]:
        averages[key] = averages[key][:, pol_character]
        deviations[key] = deviations[key][:, pol_character]

    # TODO: Find a way to deal with multicharacter polaron systems (HARD!)
    if len(pol_character) > 1:
        raise NotImplementedError(
            "Leopold is not yet capable of handling polarons of multiple type (s, p, d or f) present at the same time!"
        )

    return LeopoldDataInfo(
        species,
        pol_character,
        max_num_atoms,
        avg_neigh,
        max_neigh,
        averages,
        deviations,
    )


# ==== DATA CONSTRUCTOR ==== #


def leopold_data_constructor(
    info: LeopoldDataInfo,
    scalar_labels: dict[str, str],
    vector_labels: dict[str, str],
    device: Device = jax.default_device,
) -> Callable[[list[Atoms]], LeopoldData]:
    """Construct the data constructor function

    Defines the way in which the LeopoldData object can be constructed based
    on the general info of the dataset and the labels given by the user. The
    final result is a function able to take a list of ASE Atoms and return a
    LeopoldData object containing all the informations about the configurations
    and labels.

    Args:
        info: general info on the dataset
        scalar_labels: labels related to the scalar observables
        vector_labels: labels related to the vector observables

    Returns:
        function that takes as input a list of ASE Atoms and return a LeopoldData
    """

    def data_constructor(raw_data: list[Atoms]) -> LeopoldData:
        confs = {key: [] for key in ["positions", "ones_hot", "box"]}
        labels = {key: [] for key in (scalar_labels | vector_labels).keys()}

        for atoms in raw_data:
            # First search for needed padding
            atom_padding = info.max_num_atoms - len(atoms)

            # Labels information
            for observable, name in scalar_labels.items():
                labels[observable].append(atoms.info[name])
            for observable, name in vector_labels.items():
                if observable in ["charges", "magmoms"]:
                    value = atoms.arrays[name][:, info.pol_types]
                else:
                    value = atoms.arrays[name]

                labels[observable].append(np.pad(value, ((0, atom_padding), (0, 0))))

            # Configuration information
            box = atoms.cell.array
            pos = np.pad(atoms.get_scaled_positions(), ((0, atom_padding), (0, 0)))
            ele = atoms.get_atomic_numbers()

            # Polaron state
            pol = np.abs(labels["magmoms"][-1])
            pol = np.argsort(pol, axis=0)[-np.round(pol.sum()).astype(np.int32) :]

            pol_state = np.zeros((len(atoms), 1))
            pol_state[pol] = 1

            # Create ones-hot encoding
            ones_hot = (ele[:, np.newaxis] == info.species).astype(np.int32)
            ones_hot = np.concatenate((ones_hot, pol_state), axis=1)
            ones_hot = np.pad(ones_hot, ((0, atom_padding), (0, 0)))

            confs["positions"].append(pos)
            confs["ones_hot"].append(ones_hot)
            confs["box"].append(box)

        # Load on GPU memory
        confs = Configuration(
            **{key: jnp.array(value, device=device) for key, value in confs.items()}
        )
        labels = Labels(
            **{key: jnp.array(value, device=device) for key, value in labels.items()}
        )

        return LeopoldData(confs, labels)

    return data_constructor


# ==== GENERAL LOAD DATASET ==== #


def leopold_load_datasets(
    data_paths: dict[str, str],
    labels: dict[str, dict] | None = None,
    share_info: bool = True,
    batch_size: int = 1,
    device: Device = jax.devices("cpu")[0],
    shuffle: bool = False,
    r_cutoff: Optional[float] = None,
) -> dict[str, LeopoldDataLoader]:
    """Load the Leopold dataset inside a configuration

    Read and load the datasets specified in a dictionary object containign the datasets
    you want to read as follows:
        data_paths = {"dataset1": "path1", "dataset2": "path2" ...}.
    Every dataset will be read using ASE and the different labels are collected searching
    inside the info and arrays objects of ASE Atoms using the labels provided in the
    second variable or the ones given by default:
        labels = {"scalar": {"energy": "name_energy"}, "vector": {...}},
    scalars are searched in Atoms.info, whiel vectors in Atoms.arrays.

    Args:
        data_paths: dictionary containing info on datasets path
        labels: dictionary with informations related to how process labels
        share_info: share the datasets info between them instead of searching one for each of them
        batch_size: size of the batches
        preload: preload all the data on the GPU to reduce possible overhead of computations
        shuffle: randomly shuffle the data in the dataset
        r_cutoff: if given, perform a neighbour analysis for maximum performance in graph construction

    Returns:
        dictionary with keys the name of the dataset and values the associated dataloader
    """
    # Modify labels as user requested
    scalar_labels = DEFAULT_SCALAR_LABELS.copy()
    vector_labels = DEFAULT_VECTOR_LABELS.copy()

    if labels is not None:
        if "scalar" in labels.keys():
            scalar_labels.update(labels["scalar"])
        if "vector" in labels.keys():
            vector_labels.update(labels["vector"])

    # Get dataset files path and read on CPU
    raw_data: dict[str, list[Atoms]] = {}
    for name, path in data_paths.items():
        raw_data[name] = read(path, scalar_labels, vector_labels)

    # Collect info on datasets
    if share_info:
        # Find larger dataset
        larger = max(raw_data.keys(), key=lambda x: len(raw_data[x]))

        # Copy the info of the larger dataset to all others
        infos = [
            leopold_data_info(raw_data[larger], scalar_labels, vector_labels, r_cutoff)
        ] * len(raw_data)
    else:
        # Get info for every dataset
        infos = [
            leopold_data_info(value, scalar_labels, vector_labels, r_cutoff)
            for value in raw_data.values()
        ]

    # Construct the final dataloader
    datasets = {}
    for info, (key, value) in zip(infos, raw_data.items()):
        datasets[key] = LeopoldDataLoader(
            value,
            batch_size,
            device,
            shuffle,
            scalar_labels=scalar_labels,
            vector_labels=vector_labels,
            info=info,
        )

    return datasets


# ==== TEST ==== #
if __name__ == "__main__":
    atoms = read("test.xyz", DEFAULT_SCALAR_LABELS, DEFAULT_VECTOR_LABELS)

    info = leopold_data_info(atoms, DEFAULT_SCALAR_LABELS, DEFAULT_VECTOR_LABELS, 3.5)

    print(info)
    pass
