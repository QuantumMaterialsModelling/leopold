# LEOPOLD: LEarning Of POLaron Dynamics

## Overview

LEOPOLD is a machine learning package designed to model small polaron dynamics at the accuracy of density functional theory (DFT). By leveraging a message-passing neural network (MPNN) trained on first-principles molecular dynamics (FPMD) data, LEOPOLD enables nanosecond-scale simulations of polaron hopping dynamics, overcoming the timescale limitations of traditional FPMD approaches.

## Features

**Message-Passing Neural Network (MPNN):** Based on a modified version of the Neural Equivariant Interatomic Potential (NequIP) architecture as implemented in the [JAX-MD repository](https://github.com/google/jax-md).

- **Polaron Encoding:** Explicit charge state encoding to ensure charge conservation.
- **Occupation Prediction:** Direct prediction of site occupation to track polaron hopping.
- **Implemented in JAX:** Optimized for high-performance machine learning training and inference.

## Installation

LEOPOLD requires Python 3.10+ and to set up the environment, run:

```sh
pip install .
```

Alternatively, install dependencies directly using:

```sh
pip install -e .
```

To create a virtual environment and install the package:

```sh
python -m venv venv
source venv/bin/activate
pip install .
```

For optimal performances we suggest to install the JAX package before installing LEOPOLD to ensure its correct functionality.

## Usage

### Training the Model

To train a model you can use the `leopold-train` command that is installed alongside the LEOPOLD package.
A LEOPOLD YAML configuration with the training specifics needs to be passed to the command as `leopold-train conf.yaml`, and a default configuration can be obtained by calling the command with no arguments.
Thus, a general workflow to train a LEOPOLD model would be:

1. Generate the default YAML configuration by calling `leopold-train`.
2. Modify the configurations with the specifics of your case.
3. Run the training `leopold-train conf.yaml`.

An example of standard configuration to train a model is present in the `examples` folder.

### Running ML-Based MD Simulations

To run a polaron dynamics simulation using the trained model use the `leopold-md` command that is installed alongside the LEOPOLD package.
The behaviour is analogous to the `leopold-train` command, using a YAML configuration to store the specifics.

An example of standard configuration to run polaron molecular dynamics is present in the `examples` folder.

### Evaluation using LEOPOLD

Is possible to use a LEOPOLD checkpoint to evaluate the structures contained inside a XYZ file and obtain an output XYZ with the predictions. That can be done with the command `leopold-eval data.xyz`.

For further information on this feature just look at the help message:

```bash
leopold-eval --help
```

## Data

The dataset consists of first-principles molecular dynamics (FPMD) data generated using VASP containing informations on energies, forces, orbital decomposed charge and magnetization. Examples on how the data has to be formatted in order to LEOPOLD to use it are present in the `example` folder.

## Citation

If you use LEOPOLD in your research, please cite our papers:

```
@article{PhysRevLett.134.216301,
  title = {Machine Learning Small Polaron Dynamics},
  author = {Birschitzky, Viktor C. and Leoni, Luca and Reticcioli, Michele and Franchini, Cesare},
  journal = {Phys. Rev. Lett.},
  volume = {134},
  issue = {21},
  pages = {216301},
  numpages = {8},
  year = {2025},
  month = {May},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.134.216301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.134.216301}
}

@misc{https://doi.org/10.48550/arxiv.2606.13833,
  doi = {10.48550/ARXIV.2606.13833},
  url = {https://arxiv.org/abs/2606.13833},
  author = {Leoni,  Luca and Franchini,  Cesare},
  title = {Machine-learned dynamics of surface polarons at reduced oxide surfaces},
  publisher = {arXiv},
  year = {2026},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Contact

For questions and contributions, please open an issue or contact the authors.
