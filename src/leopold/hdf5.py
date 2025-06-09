"""
Module containing the HDF5 file specifications for the LEOPOLD module

creation: 2025-03-05 15:37:29
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Math
import numpy as np
from numpy.typing import NDArray

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
from dataclasses import asdict, dataclass
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


@dataclass
class LeopoldH5MDState:
    step: int
    dt: float
    positions: NDArray
    box: NDArray
    forces: Optional[NDArray] = None
    velocities: Optional[NDArray] = None

    @property
    def dimensions(self) -> int:
        return self.positions.shape[-1]

    @property
    def time(self) -> float:
        return self.step * self.dt


class LeopoldH5MDWriter:
    format = "H5MD"
    multiframe = True
    #: These variables are not written from :attr:`Timestep.data`
    #: dictionary to the observables group in the H5MD file
    data_blacklist = ["step", "time", "dt"]

    #: currently written version of the file format
    H5MD_VERSION = (1, 1)

    # This dictionary is used to translate MDAnalysis units to H5MD units.
    # (https://nongnu.org/h5md/modules/units.html)
    _unit_translation_dict = {
        "time": "fs",
        "length": "Angstrom",
        "velocity": "Angstrom fs-1",
        "force": "Electronvolt Angstrom-1",
    }

    def __init__(
        self,
        filename,
        n_atoms,
        n_frames=None,
        driver=None,
        chunks=None,
        compression=None,
        compression_opts=None,
        velocities=True,
        forces=True,
        author="N/A",
        author_email=None,
        creator="Leopold",
        creator_version=__version__,
    ):
        self.filename = filename
        if n_atoms == 0:
            raise ValueError("H5MDWriter: no atoms in output trajectory")
        self._driver = driver
        if self._driver == "mpio":
            raise ValueError(
                "H5MDWriter: parallel writing with MPI I/O is not currently supported."
            )
        self.n_atoms = n_atoms
        self.n_frames = n_frames
        self.chunks = (1, n_atoms, 3) if chunks is None else chunks
        if self.chunks is False and self.n_frames is None:
            raise ValueError(
                "H5MDWriter must know how many frames will be "
                "written if ``chunks=False``."
            )
        self.contiguous = self.chunks is False and self.n_frames is not None
        self.compression = compression
        self.compression_opts = compression_opts
        self.h5md_file = None

        # The writer defaults to writing all data from the parent Timestep if
        # it exists. If these are True, the writer will check each
        # Timestep.has_*  value and fill the self._has dictionary accordingly
        # in _initialize_hdf5_datasets()
        self._write_velocities = velocities
        self._write_forces = forces

        # Pull out various keywords to store metadata in 'h5md' group
        self.author = author
        self.author_email = author_email
        self.creator = creator
        self.creator_version = creator_version

    def __call__(self, state: LeopoldH5MDState, observables: Dict):
        """Write information associated with ``ag`` at current frame
        into trajectory

        Parameters
        ----------
        ag : AtomGroup or Universe

        """
        if state.positions.shape[0] != self.n_atoms:
            raise IOError(
                "H5MDWriter: Timestep does not have the correct number of atoms"
            )

        # This should only be called once when first timestep is read.
        if self.h5md_file is None:
            self._open_file()
            self._initialize_hdf5_datasets(state, observables)

        return self._write_next_timestep(state, observables)

    def _open_file(self):
        """Opens file with `H5PY`_ library and fills in metadata from kwargs.

        :attr:`self.h5md_file` becomes file handle that links to root level.

        """

        self.h5md_file = File(name=self.filename, mode="w", driver=self._driver)

        # fill in H5MD metadata from kwargs
        root = self.h5md_file.require_group("h5md")
        root.attrs["version"] = np.array(self.H5MD_VERSION)
        g = root.require_group("author")
        g.attrs["name"] = self.author
        if self.author_email is not None:
            g.attrs["email"] = self.author_email
        g = root.require_group("creator")
        g.attrs["name"] = self.creator
        g.attrs["version"] = self.creator_version

    def _initialize_hdf5_datasets(self, state: LeopoldH5MDState, observables: Dict):
        """initializes all datasets that will be written to by
        :meth:`_write_next_timestep`

        Note
        ----
        :exc:`NoDataError` is raised if no positions, velocities, or forces are
        found in the input trajectory. While the H5MD standard allows for this
        case, :class:`H5MDReader` cannot currently read files without at least
        one of these three groups. A future change to both the reader and
        writer will allow this case.


        """

        # for keeping track of where to write in the dataset
        self._counter = 0

        # Check if state has forces and velocities
        self._has = {}
        self._has["forces"] = state.forces is not None
        self._has["velocities"] = state.velocities is not None

        # initialize trajectory group
        self._traj = self.h5md_file.require_group("particles").require_group(
            "trajectory"
        )

        # box group is required for every group in 'particles'
        self._traj.require_group("box")
        self._traj["box"].attrs["dimension"] = 3
        if state.dimensions is not None and np.all(state.dimensions > 0):
            self._traj["box"].attrs["boundary"] = 3 * ["periodic"]
            self._traj["box"].require_group("edges")
            self._edges = self._traj.require_dataset(
                "box/edges/value",
                shape=(0, 3, 3),
                maxshape=(None, 3, 3),
                dtype=np.float32,
            )
            self._step = self._traj.require_dataset(
                "box/edges/step", shape=(0,), maxshape=(None,), dtype=np.int32
            )
            self._time = self._traj.require_dataset(
                "box/edges/time", shape=(0,), maxshape=(None,), dtype=np.float32
            )
            self._set_attr_unit(self._edges, "length")
            self._set_attr_unit(self._time, "time")
        else:
            # if no box, boundary attr must be "none" according to H5MD
            self._traj["box"].attrs["boundary"] = 3 * ["none"]
            self._create_step_and_time_datasets()

        # Positions always present
        self._create_trajectory_dataset("position")
        self._pos = self._traj["position/value"]
        self._set_attr_unit(self._pos, "length")
        if self.has_velocities:
            self._create_trajectory_dataset("velocity")
            self._vel = self._traj["velocity/value"]
            self._set_attr_unit(self._vel, "velocity")
        if self.has_forces:
            self._create_trajectory_dataset("force")
            self._force = self._traj["force/value"]
            self._set_attr_unit(self._force, "force")

        # intialize observable datasets from ts.data dictionary that
        # are NOT in self.data_blacklist
        if len(observables) > 0:
            self._obsv = self.h5md_file.require_group("observables")
            for key, val in observables.items():
                self._create_observables_dataset(key, val)

    def _create_step_and_time_datasets(self):
        """helper function to initialize a dataset for step and time

        Hunts down first available location to create the step and time
        datasets. This should only be called if the trajectory has no
        dimension, otherwise the 'box/edges' group creates step and time
        datasets since 'box' is the only required group in 'particles'.

        :attr:`self._step` and :attr`self._time` serve as links to the created
        datasets that other datasets can also point to for their step and time.
        This serves two purposes:
            1. Avoid redundant writing of multiple datasets that share the
               same step and time data.
            2. In HDF5, each chunked dataset has a cache (default 1 MiB),
               so only 1 read is required to access step and time data
               for all datasets that share the same step and time.

        """

        for group, value in self._has.items():
            if value:
                self._step = self._traj.require_dataset(
                    f"{group}/step", shape=(0,), maxshape=(None,), dtype=np.int32
                )
                self._time = self._traj.require_dataset(
                    f"{group}/time", shape=(0,), maxshape=(None,), dtype=np.float32
                )
                self._set_attr_unit(self._time, "time")
                break

    def _create_trajectory_dataset(self, group):
        """helper function to initialize a dataset for
        position, velocity, and force"""

        if self.n_frames is None:
            shape = (0, self.n_atoms, 3)
            maxshape = (None, self.n_atoms, 3)
        else:
            shape = (self.n_frames, self.n_atoms, 3)
            maxshape = None

        chunks = None if self.contiguous else self.chunks

        self._traj.require_group(group)
        self._traj.require_dataset(
            f"{group}/value",
            shape=shape,
            maxshape=maxshape,
            dtype=np.float32,
            chunks=chunks,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        if "step" not in self._traj[group]:
            self._traj[f"{group}/step"] = self._step
        if "time" not in self._traj[group]:
            self._traj[f"{group}/time"] = self._time

    def _create_observables_dataset(self, group, data):
        """helper function to initialize a dataset for each observable"""

        self._obsv.require_group(group)
        # guarantee ints and floats have a shape ()
        data = np.asarray(data)
        self._obsv.require_dataset(
            f"{group}/value",
            shape=(0,) + data.shape,
            maxshape=(None,) + data.shape,
            dtype=data.dtype,
        )
        if "step" not in self._obsv[group]:
            self._obsv[f"{group}/step"] = self._step
        if "time" not in self._obsv[group]:
            self._obsv[f"{group}/time"] = self._time

    def _set_attr_unit(self, dset, unit):
        """helper function to set a 'unit' attribute for an HDF5 dataset"""
        dset.attrs["unit"] = self._unit_translation_dict[unit]

    def _write_next_timestep(self, state: LeopoldH5MDState, observables: Dict):
        """Write coordinates and unitcell information to H5MD file.

        Do not call this method directly; instead use
        :meth:`write` because some essential setup is done
        there before writing the first frame.

        The first dimension of each dataset is extended by +1 and
        then the data is written to the new slot.

        Note
        ----
        Writing H5MD files with fancy trajectory slicing where the Timestep
        does not increase monotonically such as ``u.trajectory[[2,1,0]]``
        or ``u.trajectory[[0,1,2,0,1,2]]`` raises a :exc:`ValueError` as this
        violates the rules of the step dataset in the H5MD standard.

        """

        i = self._counter

        # H5MD step refers to the integration step at which the data were
        # sampled, therefore ts.data['step'] is the most appropriate value
        # to use. However, step is also necessary in H5MD to allow
        # temporal matching of the data, so ts.frame is used as an alternative
        self._step.resize(self._step.shape[0] + 1, axis=0)
        self._step[i] = state.step
        if len(self._step) > 1 and self._step[i] < self._step[i - 1]:
            raise ValueError(
                "The H5MD standard dictates that the step "
                "dataset must increase monotonically in value."
            )

        # the dataset.resize() method should work with any chunk shape
        self._time.resize(self._time.shape[0] + 1, axis=0)
        self._time[i] = state.time

        if "edges" in self._traj["box"]:
            self._edges.resize(self._edges.shape[0] + 1, axis=0)
            self._edges.write_direct(state.box, dest_sel=np.s_[i, :])
        # These datasets are not resized if n_frames was provided as an
        # argument, as they were initialized with their full size.
        if self.n_frames is None:
            self._pos.resize(self._pos.shape[0] + 1, axis=0)
        self._pos.write_direct(state.positions, dest_sel=np.s_[i, :])
        if self.has_velocities:
            if self.n_frames is None:
                self._vel.resize(self._vel.shape[0] + 1, axis=0)
            self._vel.write_direct(state.velocities, dest_sel=np.s_[i, :])
        if self.has_forces:
            if self.n_frames is None:
                self._force.resize(self._force.shape[0] + 1, axis=0)
            self._force.write_direct(state.forces, dest_sel=np.s_[i, :])

        if len(observables) > 0:
            for key, val in observables.items():
                obs = self._obsv[f"{key}/value"]
                obs.resize(obs.shape[0] + 1, axis=0)
                obs[i] = val

        self._counter += 1

    @property
    def has_velocities(self):
        """``True`` if writer is writing velocities from Timestep."""
        return self._has["velocities"]

    @property
    def has_forces(self):
        """``True`` if writer is writing forces from Timestep."""
        return self._has["forces"]


# ==== MAIN ==== #
if __name__ == "__main__":
    IrrepsLayout.as_layout("ir_mul")
