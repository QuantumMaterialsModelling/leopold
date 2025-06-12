"""
Module defining the configuration file manager for Leopold

creation: 2025-05-16 17:17:00
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== IMPORTS ==== #

# Yaml
import yaml

# Logging
from logging import INFO

# Dataclass
from dataclasses import asdict, dataclass, field

# Leopold model
from leopold.nn.e3nn_imp import Leopold
from leopold.dataset import DEFAULT_SCALAR_LABELS, DEFAULT_VECTOR_LABELS

# Typing
from typing import Union, Optional

# ==== OBJECTS ==== #


@dataclass
class LeopoldMDOptions:
    # Mandatory
    start_path: str
    model_path: str

    # Simulation options
    temperature: int = 300  # K
    time_step: float = 1.0  # fs
    max_steps: int = 1_000_000

    # Logging options
    logs_int: int = 5
    logs_dir: str = "logs"
    logs_level: int = INFO

    # Trajectory options
    traj_dir: str = "traj"
    traj_int: int = 10
    traj_write_forces: bool = False
    traj_write_velocity: bool = False
    traj_compr_backend: str = "gzip"
    traj_compr_level: int = 5
    traj_author_name: str = "N/A"
    traj_author_mail: Optional[str] = None

    # General options
    use_float64: bool = True
    rng_seed: int = 42
    device: str = "gpu"
    name: Optional[str] = None


@dataclass
class LeopoldGeneralOptions:
    name: str = "LEOPOLD"
    seed: int = 42
    result_dir: str = "results"
    models_dir: str = "checkpoints"
    logs_dir: str = "logs"
    logs_lev: Union[int, str] = INFO
    use_float64: bool = True
    checkpoint_name: Optional[str] = None
    device: str = "gpu"

    @property
    def checkpoint_file(self) -> str:
        if self.checkpoint_name is None:
            return self.tag + ".h5"
        else:
            return self.checkpoint_name

    @property
    def tag(self) -> str:
        return f"{self.name}-{self.seed}"


@dataclass
class LeopoldTrainingOptions:
    learning_rate: float = 5e-4
    max_epoch: int = 1000
    patience: int = 200
    restart: bool = False
    batch_size: int = 2

    # Options for the loss function
    energy_weight: float = 1.0
    smagmo_weight: float = 1.0
    forces_weight: float = 1.0
    magmom_weight: float = 1.0
    charge_weight: float = 1.0


@dataclass
class LeopoldDatasetsOptions:
    data_paths: dict = field(default_factory=lambda: {})
    val_split: float = 0.05
    save_data: bool = True

    labels: dict = field(
        default_factory=lambda: {
            "scalar": DEFAULT_SCALAR_LABELS,
            "vector": DEFAULT_VECTOR_LABELS,
        }
    )

    def __post_init__(self) -> None:
        default_lables = {
            "scalar": DEFAULT_SCALAR_LABELS,
            "vector": DEFAULT_VECTOR_LABELS,
        }

        self.labels = {k: default_lables[k] | v for k, v in self.labels.items()}


@dataclass
class LeopoldConfiguration:
    general: LeopoldGeneralOptions = field(
        default_factory=lambda: LeopoldGeneralOptions()
    )

    training: LeopoldTrainingOptions = field(
        default_factory=lambda: LeopoldTrainingOptions()
    )

    datasets: LeopoldDatasetsOptions = field(
        default_factory=lambda: LeopoldDatasetsOptions()
    )

    # ---- Model
    # Model standard hyper parameters taken from the model
    # directly, so that it changes as I change the model.
    model: dict = field(default_factory=lambda: _get_default_hyperparams())

    def __post_init__(self) -> None:
        if not isinstance(self.general, LeopoldGeneralOptions):
            self.general = LeopoldGeneralOptions(**self.general)
        if not isinstance(self.training, LeopoldTrainingOptions):
            self.training = LeopoldTrainingOptions(**self.training)
        if not isinstance(self.datasets, LeopoldDatasetsOptions):
            self.datasets = LeopoldDatasetsOptions(**self.datasets)


# ==== FUNCTIONS ==== #


def _get_default_hyperparams() -> dict:
    # Get dictionaty with everithing
    model = Leopold(1).__dict__

    # Remove flax added stuff
    for key in model.copy().keys():
        if key[0] == "_":
            model.pop(key)

    model.pop("name")

    # Remove stuff infered from dataset
    for key in model.copy().keys():
        if "shift" in key or "scale" in key:
            model.pop(key)
    model.pop("n_neighbour")
    model.pop("n_elems")

    return model


def read_leopold_configuration(path: str) -> LeopoldConfiguration:
    with open(path, "r") as f:
        conf = yaml.safe_load(f)

    default = LeopoldConfiguration()
    default = asdict(default)

    for key in default.keys():
        if key in conf.keys():
            default[key].update(conf[key])

    return LeopoldConfiguration(**default)


def read_leopold_md_options(path: str) -> LeopoldMDOptions:
    with open(path, "r") as f:
        conf = yaml.safe_load(f)

    # Check the configuration
    model_path = conf.pop("model_path", None)
    if model_path is None:
        raise KeyError(
            f"No model path was specified inside MD configuration file: {path}"
        )
    start_path = conf.pop("start_path", None)
    if start_path is None:
        raise KeyError(
            f"No starting configuration path was specified inside MD configuration file: {path}"
        )

    # Create object
    default = LeopoldMDOptions(start_path, model_path)
    default = asdict(default)

    # update
    default.update(conf)

    return LeopoldMDOptions(**default)


if __name__ == "__main__":
    read_leopold_configuration("default_config.yaml")
