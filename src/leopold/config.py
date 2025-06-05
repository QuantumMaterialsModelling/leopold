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
from typing import Union

# ==== OBJECTS ==== #


@dataclass
class LeopoldMDOptions:
    checkpoint_file: str
    temperature: int = 0
    use_float64: bool = True
    seed: int = 42


@dataclass
class LeopoldGeneralOptions:
    name: str = "LEOPOLD"
    seed: int = 42
    result_dir: str = "results"
    models_dir: str = "checkpoints"
    logs_dir: str = "logs"
    logs_lev: Union[int, str] = INFO
    use_float64: bool = True
    checkpoint_file: str = ""
    device: str = "gpu"

    def __post_init__(self) -> None:
        # If checkpoint file not given then set it to tag name
        if self.checkpoint_file == "":
            self.checkpoint_file = self.tag + ".h5"

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


if __name__ == "__main__":
    read_leopold_configuration("default_config.yaml")
