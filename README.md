# Modeling of opioids in the Pre-Bötzinger complex

This repository uses the [Brian 2](https://brian2.readthedocs.io/en/stable/) neural simulator (Python) to run computational experiments on the neuronal networks of the Pre-Bötzinger complex. Also included are Python scripts for data postprocessing, Jupyter Notebooks for analyzing burst statistics, graph metrics, and network structure, as well as various figures resulting from experiments and analysis.

## Root directory
- **brian_utils/postproc.py** - functions that generate burst statistics and classifies neurons as tonic, bursting, or quiescent.
- **damgo_ramp_sim/** - DAMGO ramping experiment
- **gnap_mod_sim/** - G_NaP modulation experiment
- **synblock_sim/** - Synaptic block experiment (for creating T/B/Q phase diagrams)
- **.gitignore** - prevents excessively large or unecessary files from being pushed to GitHub 
- **harris_eqs_oprmv1_gnap.yaml** - contains Hodgkin-Huxley style equations to be parsed by the experiment scripts 
- **harris_ns_oprmv1.yaml** - contains run parameters

## Creating a new experiment (general workflow)
1. In the root directory, make a new subdirectory for the new experiment. Within this subdirectory, set up folders for where .pkl files and figures outputted by the simulation will go.
2. To implement a new experiment, start with a copy of the script that runs a similar experiment. Modify perturbations, G_NaP, G_Leak, T/B/Q, etc. as necessary. This might be easier to debug when the script is in notebook format or if using an editor that can run the code in chunks (e.g. Spyder).
3. Create a postprocessing script that can process the .pkl files outputted by the simulations. We want to export the data into a format that is clean and easy to handle (.csv, networkx object, numpy array, etc.). Note that not all the data we want will be in the .pkl files, but we can use the existing data to do other computations.
4. Write a bash script that runs the experiment for many different network seeds, with varying parameters if necessary, and then runs the postprocessing script(s).
5. Once the bash script finishes running, we should have our data fully prepped. Create a Jupyter Notebook (or use your preferred analysis tool) and analyze the data accordingly (make plots, run regressions, compute connectivity, etc).



