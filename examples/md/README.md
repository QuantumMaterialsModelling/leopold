# Example molecular dynamics run with LEOPOLD

This folder contains the minimum requirements to perform a molecular dynamics run with LEOPOLD:

1. `start.xyz`, initial configuration to start the dynamic.
2. `model.h5`, model checkpoint to use in the run.
3. `conf.yaml`,  leopold configuration containing the specifics of the run.

In this example we run a 300 K hole polaron dynamics in MgO for 1 ps using a really small and quickly trained model just to show how to start the dynamic.

## Run the example

In order to run this example as it is one can simply run the command `leopold-md conf.yaml` once the LEOPOLD package is installed correctly.
