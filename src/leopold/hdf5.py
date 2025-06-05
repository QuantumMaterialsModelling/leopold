"""
Module containing the HDF5 file specifications for the LEOPOLD module

creation: 2025-03-05 15:37:29
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Math
import numpy as np

# ASE
from ase import Atoms

# H5
from h5py import File, Group, Dataset

# Cuequivariance
from cuequivariance import Irrep, IrrepsLayout, Irreps
from cuequivariance_jax import RepArray

# e3nn
import e3nn_jax as e3
from e3nn_jax import IrrepsArray

# Jax
import jax
import jax.numpy as jnp
from jax import Device, tree_util

# Leopold
from leopold.config import LeopoldConfiguration
from leopold.dataset import LeopoldData, LeopoldDataInfo, LeopoldDataLoader
from leopold.dataset import Configuration, Labels

# Flax
from flax.core import FrozenDict
from flax.serialization import to_state_dict, from_state_dict
from flax.serialization import register_serialization_state

# Optax
from optax import OptState

# Pickle
import pickle

# Types
from dataclasses import asdict
from enum import Enum, unique
from typing import Optional, Union, NamedTuple, Any, Dict

# Warnings
import warnings

# Version
from leopold.__init__ import __version__


# ==== SERIALIZATION ==== #


def _reparr_state_dict(x: RepArray) -> Dict[str, Any]:
    return {"irreps": str(x.irreps), "array": x.array, "layout": x.layout.name}


def _restore_reparr(_: RepArray, x: Dict[str, Any]) -> RepArray:
    return RepArray(
        Irreps("O3", x["irreps"]), x["array"], IrrepsLayout.as_layout(x["layout"])
    )


register_serialization_state(RepArray, _reparr_state_dict, _restore_reparr)


def _irreparr_state_dict(x: IrrepsArray) -> Dict[str, Any]:
    return {"irreps": str(x.irreps), "array": x.array}


def _restore_irreparr(_: IrrepsArray, x: Dict[str, Any]) -> IrrepsArray:
    return IrrepsArray(e3.Irreps(x["irreps"]), x["array"])


register_serialization_state(IrrepsArray, _irreparr_state_dict, _restore_irreparr)

# ==== FUNCTIONS ==== #


def save_dict_to_hdf5(group: Group, state: Any, compression: int = 9) -> None:
    # Walk inside the dictionary
    for name, data in state.items():
        # If nested dictionary do it recursivelly
        if isinstance(data, Dict):
            subgroup = group.require_group(name)
            save_dict_to_hdf5(subgroup, data, compression)
        # Here we have data
        else:
            if np.isscalar(data) or np.ndim(data) == 0:
                # If it's a string handle it specially
                if type(data) is str:
                    group.require_dataset(name, (), f"S{len(data)}")[()] = data
                elif data is None:
                    group.require_dataset(name, None, "f")
                else:
                    group.require_dataset(name, (), np.asarray(data).dtype)[()] = data
            # These are vector data
            # so data can be compressed
            else:
                data = np.asarray(data)

                group.require_dataset(
                    name, data.shape, data.dtype, compression=compression
                )[:] = data


def save_hdf5_to_dict(
    group: Group, device: Device = jax.devices("cpu")[0]
) -> Dict[str, Any]:
    state = {}

    # Go trough the group
    for name in group:
        child = group[name]

        # If another group is met go recursive
        if type(child) is Group:
            state[name] = save_hdf5_to_dict(child, device)
        # If dataset then read data
        elif type(child) is Dataset:
            data = child[()]

            # If scalar return the scalar version
            if np.isscalar(data) or np.ndim(data) == 0:
                # If string then decode it
                if type(data) is np.bytes_:
                    state[name] = data.decode()
                else:
                    state[name] = data
            # The rest should be arrays
            else:
                state[name] = jnp.asarray(data, device=device)

    return state


def save_tree_to_hdf5(group: Group, tree: Any, compression: int = 9) -> None:
    # Flatten the tree
    leaves, struct = tree_util.tree_flatten(tree)

    # Save leafs as datasets
    for i, leave in enumerate(leaves):
        # Name of dataset
        name = f"leave{i}"

        # Deal with scalars first
        if np.isscalar(leave) or np.ndim(leave) == 0:
            # If it's a string handle it specially
            if type(leave) is str:
                group[name] = leave
            else:
                group.require_dataset(name, (), np.asarray(leave).dtype)[()] = leave
        # These are vector data
        # so data can be compressed
        else:
            data = np.asarray(leave)

            group.require_dataset(
                name, data.shape, data.dtype, compression=compression
            )[:] = data

    # Save the structure as bytes
    struct = pickle.dumps(struct)
    group.require_dataset("struct", (1,), f"S{len(struct)}")[()] = struct


def save_hdf5_to_tree(group: Group, device: Device = jax.devices("gpu")[0]) -> Any:
    # Collect the struct of the tree
    struct = pickle.loads(Dataset(group["struct"].id)[()])

    # Collect all the leafs
    leaves, names = [], [i for i, n in enumerate(group.keys()) if "leave" in n]
    for name in names:
        data = Dataset(group[f"leave{name}"].id)[()]

        # If data are bytes then is a string
        if type(data) is bytes:
            leaves.append(data.decode())
        # If scalar return the scalar version
        elif np.isscalar(data) or np.ndim(data) == 0:
            leaves.append(data)
        # The rest should be numbers
        else:
            data = jnp.asarray(data, device=device)
            leaves.append(data)

    return tree_util.tree_unflatten(struct, leaves)


# ==== ARCHETIPE FILE ==== #


class LeopoldHDF5(File):
    def __init__(
        self, name, mode, author_name: str = "N/A", email: str = "N/A", *args, **kwargs
    ):
        super(LeopoldHDF5, self).__init__(name, mode, *args, **kwargs)

        if "w" in mode:
            # Specifics for saing it's a LEOPOLD HDF5 file
            g = self.create_group("creator")
            g.attrs["name"] = "leopold"
            g.attrs["version"] = __version__
            g.attrs["date"] = str(np.datetime64("today"))

            # Author of the document
            g = self.create_group("author")
            g.attrs["name"] = author_name
            g.attrs["email"] = email
        else:
            if "creator" not in self:
                raise KeyError("creator group not found in file")
            g = Group(self["creator"].id)

            # Look if created through LEOPOLD
            if not g.attrs["name"] == "leopold":
                raise ValueError("this hdf5 file was not generated by Leopold")

            # Look version mistmatch
            if g.attrs["version"] != __version__:
                warnings.warn("this hdf5 leopold file and module versions mismatch")

        # Add last modification attributes
        if "w" in mode or "a" in mode:
            self["creator"].attrs["modified"] = str(np.datetime64("now"))


# ==== General Group ==== #


class LeopoldDatasetGroup:
    group: Group

    def __init__(self, group: Group):
        # Set group
        self.group = group

        # Set variables
        self.__data = group.require_group("data")
        self.__info = group.require_group("info")

        # See if empty or full
        self.__empty = "struct" in self.__data

    def __getitem__(self, idx) -> LeopoldData:
        assert not self.__empty, "tried to index an empty LeopoldDataGroup!"

        # Get config and labels
        config, labels = {}, {}

        for name in Group(self.__data["config"].id):
            config[name] = Dataset(self.__data[f"config/{name}"].id)[idx]
        for name in Group(self.__data["labels"].id):
            # Deal with possible empty dataset
            if Dataset(self.__data[f"labels/{name}"].id).shape is not None:
                labels[name] = Dataset(self.__data[f"labels/{name}"].id)[idx]
            else:
                labels[name] = None

        return LeopoldData(Configuration(**config), Labels(**labels))

    @property
    def info(self) -> LeopoldDataInfo:
        info = save_hdf5_to_dict(self.__info)

        return LeopoldDataInfo(**info)

    def write_dataloader(
        self,
        loader: LeopoldDataLoader,
        save_data: bool = True,
        compression: int = 9,
    ) -> None:
        # First write down the info
        save_dict_to_hdf5(self.__info, asdict(loader.info))

        # Save data
        if save_data:
            # If wanted save all
            save_dict_to_hdf5(self.__data, to_state_dict(loader.data), compression)

            # Set as not empty
            self.__empty = False

    def get_dataloader(
        self,
        batch_size: int = 1,
        device: Device = jax.devices("cpu")[0],
    ) -> LeopoldDataLoader:
        assert not self.__empty, (
            "tried to create a data loader from an empty LeopoldDataGroup!"
        )

        return LeopoldDataLoader(
            self[()], info=self.info, batch_size=batch_size, device=device
        )


class LeopoldState(NamedTuple):
    params: Union[FrozenDict, dict]
    opt_state: OptState
    loss: float
    observables: Optional[dict[str, float]] = None


class LeopoldTrainingGroup:
    group: Group
    impatience: int = 0

    def __init__(
        self,
        group: Group,
        dataset: Optional[Group] = None,
        start_model: Optional[Group] = None,
    ) -> None:
        # Save the group
        self.group = group

        # Set dataset group link if present
        if dataset is not None:
            self.group["datasets"] = dataset

        # If starting model exist copy it
        if start_model is not None:
            self.group.copy(start_model, f"{group.name}/models")

        # Create the subgroup we are working with
        self.__inputs = group.require_group("inputs")
        self.__datasets = group.require_group("datasets")
        self.__models = group.require_group("models")
        self.__state = group.require_group("state")
        self.__observables = group.require_group("observables")

        # If group already existing set some stuff
        if len(self.__inputs) != 0:
            self.__max_epoch = self.conf.training.max_epoch

        # Get best loss if present
        if self.step != 0:
            best_step = self.__models["best_model"].attrs.get("step", None)
            if best_step is None:
                raise RuntimeError("Leopold internal error in retriving best step!")

            self.__best_loss = Dataset(self.__observables["loss"].id)[best_step]
        else:
            self.__best_loss = np.inf

    @property
    def step(self) -> int:
        return self.group.attrs.get("step", 0)

    @property
    def model_conf(self) -> dict:
        mconf = self.conf.model
        info = self.get_dataset("train").info

        # Take number of elements in the dataset
        mconf["n_elems"] = len(info.species) + 1

        # Set the average number of neighbours
        mconf["n_neighbour"] = info.average_neigh

        # Get energy scale and shift
        mconf["energy_shift"] = info.averages["energy"]
        mconf["energy_scale"] = info.deviations["energy"]

        # Get species dependent scale and shift
        vals = [info.averages["magmoms"], info.averages["charges"]]
        mconf["magchg_shift"] = np.concat(vals, axis=1)

        vals = [info.deviations["magmoms"], info.deviations["charges"]]
        mconf["magchg_shift"] = np.concat(vals, axis=1)

        return mconf

    @property
    def conf(self) -> LeopoldConfiguration:
        # See if inputs is empty
        if len(self.__inputs) == 0:
            raise ValueError(
                "tried to reference configurations of Leopold training before assigning them!"
            )

        conf = save_hdf5_to_dict(self.__inputs)
        return LeopoldConfiguration(**conf)

    @conf.setter
    def conf(self, conf: LeopoldConfiguration) -> None:
        save_dict_to_hdf5(self.__inputs, asdict(conf))

        # Save it for future convenience
        self.__max_epoch = conf.training.max_epoch

    def load_state(self, state: LeopoldState) -> LeopoldState:
        params = self.load_model(state.params, "last")
        g = self.__state.require_group("optax_state")
        opt_state = from_state_dict(state.opt_state, save_hdf5_to_dict(g))

        # NOTE: meaby better output the best loss so far?
        g = self.__observables.require_dataset(
            "loss", shape=self.__max_epoch, dtype=np.float32
        )
        loss = g[self.step - 1]

        return LeopoldState(params, opt_state, loss)

    def update_state(self, values: LeopoldState) -> None:
        params, opt_state, loss, others = values

        # Save params of last model
        g = self.__models.require_group("last_model")
        save_dict_to_hdf5(g, to_state_dict(params))

        # Save last state of optimizer
        g = self.__state.require_group("optax_state")
        save_dict_to_hdf5(g, to_state_dict(opt_state))

        # Save the loss
        g = self.__observables.require_dataset(
            "loss", shape=self.__max_epoch, dtype=np.float64
        )
        g[self.step] = loss

        # Save other observables if present
        if others is not None:
            for key, value in others.items():
                g = self.__observables.require_dataset(
                    key, shape=self.__max_epoch, dtype=np.float64
                )
                g[self.step] = value

        # Controll if loss is the best in case save best model
        if self.__best_loss > loss:
            g = self.__models.require_group("best_model")
            save_dict_to_hdf5(g, to_state_dict(params))

            # Save also the step of the best model
            g.attrs["step"] = self.step

            # Reset loss and impatience
            self.__best_loss = loss
            self.impatience = 0
        else:
            # Increase impatience
            self.impatience += 1

        # Add one to step
        self.group.attrs["step"] = self.step + 1

    def load_model(self, params: Union[FrozenDict, Dict], which: str = "best"):
        if which == "best":
            g = self.__models.require_group("best_model")
            return from_state_dict(params, save_hdf5_to_dict(g))
        elif which == "last":
            g = self.__models.require_group("last_model")
            return from_state_dict(params, save_hdf5_to_dict(g))
        else:
            raise KeyError(f"no model named {which} exist!")

    def attach_dataset(
        self, name: str, data: Union[LeopoldDataLoader, LeopoldDatasetGroup], **kwargs
    ) -> None:
        if isinstance(data, LeopoldDatasetGroup):
            data = data.get_dataloader()
        g = LeopoldDatasetGroup(self.__datasets.require_group(name))
        g.write_dataloader(data, **kwargs)

    def get_dataset(self, name: str) -> LeopoldDatasetGroup:
        g = Group(self.__datasets[name].id)
        return LeopoldDatasetGroup(g)

    def get_datasets(self) -> dict[str, LeopoldDatasetGroup]:
        data = {}
        for name in self.__datasets.keys():
            data[name] = self.get_dataset(name)

        return data


# ==== ML STATE ==== #


class LeopoldCheckpointFile(LeopoldHDF5):
    def __init__(
        self,
        name,
        mode,
        author_name: str = "N/A",
        email: str = "N/A",
        *args,
        **kwargs,
    ):
        super(LeopoldCheckpointFile, self).__init__(
            name, mode, author_name, email, *args, **kwargs
        )

        # Create training group with specifics
        self.__trainings = self.require_group("trainings")

    @property
    def n_train(self) -> int:
        return len(self.__trainings)

    def create_training(self, conf: LeopoldConfiguration) -> LeopoldTrainingGroup:
        # If no trainings are present just create the first
        if len(self.__trainings) == 0:
            g = LeopoldTrainingGroup(self.__trainings.create_group("training1"))

            # Set configuration
            g.conf = conf
            return g

        # See if one restarts
        if conf.training.restart:
            last_train = Group(self.__trainings[f"training{self.n_train}"].id)
            last_train = LeopoldTrainingGroup(last_train)

            # Model specifics needs to be equal
            for key, val in last_train.conf.model.items():
                if key in ["averages", "deviations"]:
                    continue

                assert np.all(conf.model[key] == val), (
                    "tried restart training but with different model setup"
                )

            # If train specifics changes crete a new training
            for key, val in asdict(last_train.conf.training).items():
                # Changes in these does not matter
                if key in ["max_epoch", "restart", "patience"]:
                    continue

                # Some important variable has changed
                if val != asdict(conf.training)[key]:
                    # Create new training but with same dataset
                    g = LeopoldTrainingGroup(
                        self.__trainings.create_group(f"training{self.n_train + 1}"),
                        Group(last_train.group["datasets"].id),
                        Group(last_train.group["models"].id),
                    )

                    g.conf = conf

                    return g

            # Compatible return last train
            return last_train
        else:
            g = LeopoldTrainingGroup(
                self.__trainings.create_group(f"training{self.n_train + 1}")
            )

            # Set Configuration
            g.conf = conf
            return g

    def get_training(self, which: Optional[int] = None) -> LeopoldTrainingGroup:
        if which is None:
            which = self.n_train

        g = Group(self.__trainings[f"training{which}"].id)
        return LeopoldTrainingGroup(g)


# ==== H5MD FILE ==== #


class LeopoldH5MD(File):
    particles: Group
    observables: Group

    def __init__(
        self, name, mode, author_name: str = "N/A", email: str = "N/A", *args, **kwargs
    ):
        super(LeopoldH5MD, self).__init__(name, mode, *args, **kwargs)

        if "w" in mode:
            # Construct metadata

            # Version and Creation
            g = self.create_group("h5md")
            g.attrs["version"] = "1.1"
            g.attrs["creation"] = str(np.datetime_as_string(np.datetime64("now"), "s"))

            # Specifics for saing it's a LEOPOLD H5MD file
            g = self.create_group("h5md/creator")
            g.attrs["name"] = "leopold"
            g.attrs["version"] = __version__

            # Author of the document
            g = self.create_group("h5md/author")
            g.attrs["name"] = author_name
            g.attrs["email"] = email
        else:
            # Look if h5md file
            if "h5md" not in self:
                raise KeyError("h5md group not found in file")
            g = Group(self["h5md"])

            # Look if created through LEOPOLD
            if not g["creator"].attrs["name"] == "leopold":
                raise ValueError("This h5md file was not generated by Leopold")

    def attach_simulation(self, start_config: Atoms, compression: int = 5) -> None:
        pass


# ==== MAIN ==== #
if __name__ == "__main__":
    IrrepsLayout.as_layout("ir_mul")
