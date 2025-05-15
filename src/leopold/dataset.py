"""
Module containing the objects and funciton that allows to load datasets

creation: 2025-05-05 11:46:40
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# RANDOM
from random import shuffle as rng_shuffle

# MATH
import numpy as np

# HDF5
from h5py import Group, Dataset

# CUEQUIVARIANCE
from cuequivariance_jax import RepArray

# ASE
from ase import Atoms
from ase.io import read as ase_read

# JAX
import jax.numpy as jnp
from jax import Array

# YAML
import yaml

# DATACLASS
from dataclasses import dataclass, asdict

# Typing
from typing import Callable, NamedTuple

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
    stress: Array | None = None  # [.., N]


@dataclass
class LeopoldDataInfo:
    """Info on the Leopold dataset

    Attributes:
        species: species present in the dataset
        pol_types: character of the polaron present (s, p, d or f)
        max_num_atoms: maximum number of atoms in dataset, used for padding
    """

    species: Array
    pol_types: Array
    max_num_atoms: int


class LeopoldData(NamedTuple):
    """Description of a Leopold data entry

    Attributes:
        config: atomic and polaronic configuration
        labels: labels related to the configuration
    """

    config: Configuration
    labels: Labels


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
    """

    info: LeopoldDataInfo
    data: list[LeopoldData]

    nbatches: int
    batch_size: int = 1

    def __init__(
        self,
        raw_data: list[Atoms] | Group,
        scalar_labels: dict[str, str] = DEFAULT_SCALAR_LABELS,
        vector_labels: dict[str, str] = DEFAULT_VECTOR_LABELS,
        info: LeopoldDataInfo | None = None,
        batch_size: int = 1,
        shuffle: bool = True,
    ) -> None:
        # If an HDF5 group is given everithing is easy
        if isinstance(raw_data, Group):
            # Collect metadata
            self.info = leopold_info_from_hdf5(raw_data)
            self.nbatches = raw_data.attrs.get("nbatches", 0)
            self.batch_size = raw_data.attrs.get("batch_size", 0)

            if self.nbatches == 0 or self.batch_size == 0:
                raise KeyError(f"group {raw_data.name} is not a Leopold dataset")

            # Collect data
            self.data = leopold_data_from_hdf5(raw_data)

            # Finish here
            return

        # Load data from list of ASE atoms
        self.info = leopold_data_info(raw_data, vector_labels) if info is None else info

        # Get the data constructor function
        f = leopold_data_constructor(self.info, scalar_labels, vector_labels)

        # Compute the number of batches
        self.batch_size = batch_size
        self.nbatches = len(raw_data) // batch_size
        if len(raw_data) % batch_size != 0:
            self.nbatches += 1

        # Randomly shuffle the data if requested
        if shuffle and isinstance(raw_data, list):
            rng_shuffle(raw_data)

        # Preload the data in batches
        self.data = []
        for i in range(self.nbatches):
            data = f(raw_data[i * batch_size : (i + 1) * batch_size])

            self.data.append(data)

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

        # If Configurations are already loaded send the right one
        return self.data[idx]

    def __getitem__(self, idx: int) -> LeopoldData:
        return self.data[idx]

    def __len__(self) -> int:
        return sum([d.config.box.shape[0] for d in self.data])


# ==== FUNCTIONS ==== #


# TODO: in future make it possible to read also hdf5 dataset
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


def leopold_data_constructor(
    info: LeopoldDataInfo, scalar_labels: dict[str, str], vector_labels: dict[str, str]
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
            **{key: jnp.asarray(value) for key, value in confs.items()}
        )
        labels = Labels(**{key: jnp.asarray(value) for key, value in labels.items()})

        return LeopoldData(confs, labels)

    return data_constructor


def leopold_data_info(
    raw_data: list[Atoms],
    vector_labels: dict[str, str],
) -> LeopoldDataInfo:
    """Get Leopold data info from raw data

    Get object containing general info about the loaded dataset.

    Args:
        raw_data: raw list of ASE atoms containg the data
        vector_labels: labels for the vectorial quantities, only "magmoms" one is used

    Returns:
        LeopoldDataInfo object

    Raises:
        NotImplementedError: Leopold is not yet capable of handling polarons of multiple type (s, p, d or f) present at the same time!
    """
    # First get species
    element = [atoms.get_atomic_numbers() for atoms in raw_data]
    species = jnp.unique(jnp.concatenate(element))

    # Search maximum number of atoms
    max_num_atoms = max([len(atoms) for atoms in raw_data])

    # Search maximum number of components in magmoms
    mag_label = vector_labels["magmoms"]

    max_num_compon = [atoms.arrays[mag_label].shape[1] for atoms in raw_data]
    max_num_compon = max(max_num_compon)

    # Collect all magnetization and ones_hot encoding.
    # To allow vectorization entries will be padded to maximum n_atoms
    magmoms, ones_hot = [], []
    for atoms in raw_data:
        element = atoms.get_atomic_numbers()
        magmom = atoms.arrays[mag_label]

        atom_padding = max_num_atoms - len(atoms)
        comp_padding = max_num_compon - magmom.shape[1]

        one_hot = (element[:, jnp.newaxis] == species).astype(jnp.int32)

        ones_hot.append(np.pad(one_hot, ((0, atom_padding), (0, 0))))
        magmoms.append(np.pad(magmom, ((0, atom_padding), (0, comp_padding))))
    magmoms, ones_hot = np.array(magmoms), np.array(ones_hot)

    # Get polaronic character
    pol_character = []
    for s in ones_hot.T:
        # get magnetization of the species
        magmom = jnp.abs(magmoms[s.T == 1])

        # Take the one above the mean (a polaron is present)
        pol_mag = jnp.where(magmom > magmom.mean(0), magmom, 0)
        pol_mag = jnp.round(pol_mag).sum(0)

        # The non zero entry tells us that polarons of that character are present
        pol_character.append(*jnp.where(pol_mag))

    pol_character = jnp.unique(jnp.concatenate(pol_character))

    # TODO: Find a way to deal with multicharacter polaron systems (HARD!)
    if len(pol_character) > 1:
        raise NotImplementedError(
            "Leopold is not yet capable of handling polarons of multiple type (s, p, d or f) present at the same time!"
        )

    return LeopoldDataInfo(species, pol_character, max_num_atoms)


def leopold_load_datasets(
    conf_file: str | dict,
    share_info: bool = True,
    batch_size: int = 1,
    shuffle: bool = True,
) -> dict[str, LeopoldDataLoader]:
    """Load the Leopold dataset inside a configuration

    Read the configuration file and load the datasets with the specifics present
    on it. It's also possible to load them from a dictionary that posses a similar
    sintax, thus {"dataset": {"dataset1": "path1", "dataset2": "path2" ...}}

    Args:
        conf_file: path to the configuration file or dictionary with configuration
        share_info: share the datasets info between them instead of searching one for each of them
        batch_size: size of the batches
        preload: preload all the data on the GPU to reduce possible overhead of computations
        shuffle: randomly shuffle the data in the dataset

    Returns:
        dictionary with keys the name of the dataset and values the associated dataloader
    """
    # if was not read just read it
    if isinstance(conf_file, str):
        with open(conf_file, "r") as f:
            conf = yaml.safe_load(f)
    else:
        conf = conf_file.copy()

    # Modify labels as user requested
    scalar_labels = DEFAULT_SCALAR_LABELS.copy()
    vector_labels = DEFAULT_VECTOR_LABELS.copy()

    if conf["dataset"].get("labels") is not None:
        if conf["dataset"]["labels"].get("scalar") is not None:
            scalar_labels.update(conf["dataset"]["labels"].get("scalar"))
        if conf["dataset"]["labels"].get("vector") is not None:
            vector_labels.update(conf["dataset"]["labels"].get("vector"))

    # Get dataset files path and read on CPU
    raw_data: dict[str, list[Atoms]] = {}
    for name in conf["dataset"].keys():
        # Avoid try to read the leables as datasets
        if name == "labels":
            continue

        raw_data[name] = read(conf["dataset"][name], scalar_labels, vector_labels)

    # Collect info on datasets
    if share_info:
        # Find larger dataset
        larger = max(raw_data.keys(), key=lambda x: len(raw_data[x]))

        # Copy the info of the larger dataset to all others
        infos = [leopold_data_info(raw_data[larger], vector_labels)] * len(raw_data)
    else:
        # Get info for every dataset
        infos = [leopold_data_info(value, vector_labels) for value in raw_data.values()]

    # Construct the final dataloader
    datasets = {}
    for info, (key, value) in zip(infos, raw_data.items()):
        datasets[key] = LeopoldDataLoader(
            value, scalar_labels, vector_labels, info, batch_size, shuffle
        )

    return datasets


# ==== HDF5 ==== #


def save_dictionary_to_hdf5(group: Group, data: dict, compression: int = 9) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            save_dictionary_to_hdf5(group.require_group(key), value)
        else:
            # Handle RepArray
            if isinstance(value, RepArray):
                arr = np.asarray(value.array)

                d = group.require_dataset(
                    key, arr.shape, arr.dtype, compression=compression
                )

                # Save irreps as attributes
                d.attrs["Irreps"] = str(value.irreps)
                d[:] = arr
            else:
                # Handle scalars
                if np.isscalar(value):
                    value = np.asarray([value])
                else:
                    value = np.asarray(value)

                group.require_dataset(
                    key, value.shape, value.dtype, compression=compression
                )[:] = value


def leopold_dataloader_to_hdf5(
    group: Group,
    loader: LeopoldDataLoader,
    save_data: bool = True,
    compression: int = 9,
) -> None:
    # Save general attributes
    group.attrs["batch_size"] = loader.batch_size
    group.attrs["nbatches"] = loader.nbatches

    # Save info on the dataset
    g = group.require_group("info")

    prova = asdict(loader.info)
    save_dictionary_to_hdf5(g, prova, compression)

    # If requested save the whole dataset
    if save_data:
        # Gather config and labels data to GPU
        confis = {key: [] for key in loader[0].config._asdict().keys()}
        labels = {key: [] for key in loader[0].labels._asdict().keys()}
        for batch in loader:
            for key, value in batch.config._asdict().items():
                confis[key].extend(value)

            for key, value in batch.labels._asdict().items():
                if value is None:
                    continue
                labels[key].extend(value)

        confis = {key: np.asarray(value) for key, value in confis.items()}
        labels = {key: np.asarray(value) for key, value in labels.items()}

        # Save it on HDF5
        save_dictionary_to_hdf5(group.require_group("configurations"), confis)
        save_dictionary_to_hdf5(group.require_group("labels"), labels)


def leopold_info_from_hdf5(group: Group) -> LeopoldDataInfo:
    # See if group has info subgroup
    if "info" not in group.keys():
        raise KeyError(f"the group {group.name} has no info subgroup")

    # Retrive info
    info, g = {}, Group(group["info"].id)

    for key in g.keys():
        info[key] = Dataset(g[key].id)[:]

    return LeopoldDataInfo(**info)


def leopold_data_from_hdf5(group: Group) -> list[LeopoldData]:
    # Get metadata
    nbatches = group.attrs.get("nbatches", 0)
    batch_size = group.attrs.get("batch_size", 0)

    if nbatches == 0 or batch_size == 0:
        raise KeyError(f"group {group.name} is not a Leopold dataset")

    # Get the groups for easy use
    config_g = Group(group["configurations"].id)
    labels_g = Group(group["labels"].id)

    # Collect real data
    data = []
    for i in range(nbatches):
        config, labels = {}, {}
        beg, end = i * batch_size, (i + 1) * batch_size

        for key in config_g.keys():
            config[key] = jnp.asarray(Dataset(config_g[key].id)[beg:end])

        for key in labels_g.keys():
            value = jnp.asarray(Dataset(labels_g[key].id)[beg:end])

            # Check everithing has value
            if len(value) > 0:
                labels[key] = value
            else:
                labels[key] = None

        config = Configuration(**config)
        labels = Labels(**labels)

        data.append(LeopoldData(config, labels))

    return data


# ==== TEST ==== #
if __name__ == "__main__":
    pass
