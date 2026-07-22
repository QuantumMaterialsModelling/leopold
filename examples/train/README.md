# Example training of LEOPOLD

The folder `data` contains a training set for the hole polaron in MgO composed of the training data `train.xyz` and the validation data `valid.xyz`. Every datapoint in those file contains the following informations

```xyz
64
Lattice="8.449313 0.0 0.0 0.0 8.449313 0.0 0.0 0.0 8.449313" Properties=species:S:1:pos:R:3:decomposed_charges:R:1:decomposed_magmoms:R:1:forces:R:3:magmoms:R:4 energy=-148.41823872 pbc="T T T"
Mg       6.33265000       6.25395000       4.27976000       6.12380000       0.00000000       0.06411900       0.37184900      -0.59447600      -0.00000000      -0.00000000      -0.00000000      -0.00000000
Mg       8.37318000       2.12865000       2.02936000       6.12580000       0.00000000       0.78817800      -0.19846800       1.53909600      -0.00000000      -0.00000000      -0.00000000      -0.00000000
Mg       2.06730000       8.44382000       2.07628000       6.11620000       0.00020000       0.93348700       0.23651200       0.39196100      -0.00000000      -0.00000000      -0.00000000      -0.00100000
Mg       2.06468000       2.13668000       0.05168000       6.12190000       0.00010000       0.79962300      -0.11493700      -0.65935400      -0.00000000      -0.00000000      -0.00000000      -0.00000000
```

Where is possible to see that the properties reported are the following:

- `energy`, energy of the given structure.
- `forces`, forces on every atoms.
- `decomposed_charges`, atomic charges projected on the orbital where the polaron is present.
- `decomposed_magmoms`, atomic magnetic moments projected on the orbital where the polaron is present.
- `magmoms`, present in this case but are not used by LEOPOLD.

The training command will read those information along side the structure in order to train the model.

## Run the example

In order to run this example as it is one can simply run the command `leopold-train conf.yaml` once the LEOPOLD package is installed correctly.
