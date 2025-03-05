"""
Definition of the LEOPOLD ML model using cuequivariance

creation: 2025-03-05 16:46:15
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== DEPENDENCIES ==== #

# Math
import jax.numpy as jnp

# Flax
import flax.linen as nn

# Cuequivariance
import cuequivariance as cue
import cuequivariance_jax as cuex

# Types
from typing import Dict, Union, Optional
from dataclasses import field
from jax import Array
from cuequivariance_jax import IrrepsArray
from cuequivariance import Irreps, EquivariantTensorProduct

# ==== OBJECTS ==== #


class FullyConnectedTensorProduct(nn.Module):
    """Flax module of an equivariant Fully-Connected Tensor Product."""

    descriptor: EquivariantTensorProduct

    def __init__(
        self,
        irreps_out: Irreps,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
    ):
        self.descriptor = cue.descriptors.fully_connected_tensor_product(
            irreps_in1, irreps_in2, irreps_out
        )

    def __call__(self, x1: IrrepsArray, x2: IrrepsArray) -> IrrepsArray:
        x1 = cuex.as_irreps_array(x1)
        x2 = cuex.as_irreps_array(x2)

        return cuex.equivariant_tensor_product(self.descriptor, x1, x2)


class Leopold(nn.Module):
    graph_net_steps: int
    use_sc: bool
    nonlinearities: Union[str, Dict[str, str]]
    n_elements: int

    hidden_irreps: str
    sh_irreps: str

    num_basis: int = 8
    r_max: float = 4.0

    radial_net_nonlinearity: str = "raw_swish"
    radial_net_n_hidden: int = 64
    radial_net_n_layers: int = 2

    shift: float = 0.0
    scale: float = 1.0
    shift_occ: Optional[Array] = field(default=None)
    scale_occ: Optional[Array] = field(default=None)
    n_neighbors: float = 1.0
    scalar_mlp_std: float = 4.0

    learn_energy: bool = True
    occup_clipping: bool = False


# ==== TEST ==== #
if __name__ == "__main__":
    in1 = Irreps(cue.O3, "10x0e + 3x1e")

    print(in1)
