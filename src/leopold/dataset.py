"""
Module containing the objects and funciton that allows to load datasets

creation: 2025-05-05 11:46:40
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #


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
import jax
import jax.random as jrn
import jax.numpy as jnp
from jax import Array, Device

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
    """

    info: LeopoldDataInfo
    data: LeopoldData

    nbatches: int
    batch_size: int = 1
    device: Device = jax.devices("cuda")[0]

    def __init__(
        self,
        raw_data: list[Atoms] | Group | LeopoldData,
        scalar_labels: dict[str, str] = DEFAULT_SCALAR_LABELS,
        vector_labels: dict[str, str] = DEFAULT_VECTOR_LABELS,
        info: LeopoldDataInfo | None = None,
        batch_size: int = 1,
        device: Device = jax.devices("cuda")[0],
        shuffle: bool = True,
    ) -> None:
        # Set the device
        self.device = device

        # If LeopoldData are directly given simply accept them
        if isinstance(raw_data, LeopoldData):
            self.data = raw_data

            if info is None:
                raise ValueError("when LeopoldData are given to data loader then info on the dataset must be given by the user (cannot infer types)")

            self.info = info
            self.batch_size = batch_size
            self.nbatches = len(raw_data) // batch_size
            if len(raw_data) % batch_size != 0:
                self.nbatches += 1

        # If an HDF5 group is given everithing is easy
        elif isinstance(raw_data, Group):
            # Collect metadata
            self.info = leopold_info_from_hdf5(raw_data)
            self.nbatches = raw_data.attrs.get("nbatches", 0)
            self.batch_size = raw_data.attrs.get("batch_size", 0)

            if self.nbatches == 0 or self.batch_size == 0:
                raise KeyError(f"group {raw_data.name} is not a Leopold dataset")

            # Collect data
            self.data = leopold_data_from_hdf5(raw_data)

        # Real raw_data were given
        else:
            # Load data from list of ASE atoms
            self.info = leopold_data_info(raw_data, vector_labels) if info is None else info

            # Get the data constructor function
            cpu_device = jax.devices("cpu")[0]
            f = leopold_data_constructor(self.info, scalar_labels, vector_labels, cpu_device)

            # Compute the number of batches
            self.batch_size = batch_size
            self.nbatches = len(raw_data) // batch_size
            if len(raw_data) % batch_size != 0:
                self.nbatches += 1

            # Preload the data in batches
            self.data = f(raw_data)

        # Shuffle if wanted
        if shuffle:
            self.shuffle(jrn.PRNGKey(0))

    def shuffle(self, rng_key: Array) -> None:
        perm = jrn.permutation(rng_key, len(self))

        conf = self.data.config._asdict()
        labe = self.data.labels._asdict()

        new_conf = {k: v[perm] for k, v in conf.items()}
        new_labe = {k: v[perm] if v is not None else None for k, v in labe.items()}

        self.data = LeopoldData(Configuration(**new_conf), Labels(**new_labe)) # pyright: ignore

    def split(self, idx):
        conf = self.data.config._asdict()
        labe = self.data.labels._asdict()

        new_conf = {k: jnp.split(v, [idx], axis=0) for k, v in conf.items()}
        new_labe = {k: jnp.split(v, [idx], axis=0) if v is not None else None for k, v in labe.items()}

        new_conf = [{k: v[i] for k, v in new_conf.items()} for i in range(len(idx))]
        new_labe = [{k: v[i] if v is not None else None for k, v in new_labe.items()} for i in range(len(idx))]

        data = [LeopoldData(Configuration(**c), Labels(**l)) for c, l in zip(new_conf, new_labe)] # pyright: ignore

        return [LeopoldDataLoader(d, info=self.info, batch_size=self.batch_size, device=self.device) for d in data]

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

        # If Configurations are already loaded send the right one
        return self[beg:end]

    def __getitem__(self, idx) -> LeopoldData:
        conf = self.data.config._asdict()
        labe = self.data.labels._asdict()

        new_conf = {k: jax.device_put(v[idx], self.device) for k, v in conf.items()}
        new_labe = {k: jax.device_put(v[idx], self.device) if v is not None else None for k, v in labe.items()}

        return LeopoldData(Configuration(**new_conf), Labels(**new_labe)) # pyright: ignore

    def __len__(self) -> int:
        return self.data.config.box.shape[0]


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
    info: LeopoldDataInfo, scalar_labels: dict[str, str], vector_labels: dict[str, str], device: Device = jax.default_device
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
            **{key: jnp.asarray(value, device=device) for key, value in confs.items()}
        )
        labels = Labels(**{key: jnp.asarray(value, device=device) for key, value in labels.items()})

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
    data_paths: dict[str, str],
    labels: dict[str, dict] | None = None,
    share_info: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
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


def leopold_data_from_hdf5(group: Group) -> LeopoldData:
    # Get the groups for easy use
    config_g = Group(group["configurations"].id)
    labels_g = Group(group["labels"].id)

    # Collect real data
    config, labels = {}, {}

    for key in config_g.keys():
        config[key] = jnp.asarray(Dataset(config_g[key].id)[:])

    for key in labels_g.keys():
        value = jnp.asarray(Dataset(labels_g[key].id)[:])

        # Check everithing has value
        if len(value) > 0:
            labels[key] = value
        else:
            labels[key] = None

    config = Configuration(**config)
    labels = Labels(**labels)

    return LeopoldData(config, labels)


# ==== TEST ==== #
if __name__ == "__main__":
    pass
