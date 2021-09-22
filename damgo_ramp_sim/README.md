## damgo_ramp_sim/
- **cell_classes/** - .npy files (numpy arrays) that tell which neurons are tonic, bursting, and quiescent
- **damgo_ramp_figs/** - various figures from each network simulation
- **networkx_objects/** - .pkl files that contain networkx objects describing the structure of each network (T/B/Q, inhibitory/excitatory/oprm1+, synaptic connections)
- **create_graphs.py** - code for creating networkx DiGraph (directed graph) objects for each network and saving them as .pkl files in networkx_objects/ 
- **damgo_ramp.csv** - dataset containing count of each synapse type, network shutdown values via firing rate threshold, and LD50s of each network
- **damgo_ramp_analysis.ipynb** - notebook for analyzing results of the experiment
- **damgo_ramp_postproc.py** - postprocessing script that counts how many of each type of synapse are in each network and computes network shutdown values by taking the average DAMGO dosage across multiple firing rate thresholds
- **graph_metrics.ipynb** - notebook for analyzing the algebraic connectivity of each network and its various subpopulations, as well as the centrality of the individual neurons in each network.
- **plot_max_fr_ld50.py** - plots the max firing rate in a sliding 2 second window throughout the simulation against the corresponding DAMGO dosage. A logistic curve is then fitted onto the data, which can then be used to compute the LD50 of each network.
- **run_damgo_ramp.py** - code for running the experiment, outputs traces and saves data into .pkl files
- **run_damgo_ramps** - bash script for sweeping across many different network seeds

### Running the experiment
run_damgo_ramp.py arguments:
- ```prefix```: the name of the run, which is used for data/figure saving purposes. (This is always the first argument)
- ```run_seed```: network seed to be simulated
- ```hyp_opioid```: the maximum opioid current (pA), default value is 8 pA
- ```syn_shut```: the maximum decrease in synaptic strength [0,1], default value is 1 (meaning that synapses are completely shut down by the end of the simulation)
#### Example invocations:
```bash
# default parameters
python run_damgo_ramp.py seed1-damgo_ramp --run_seed 1  

# if we want to increase max opioid current to 10 pA and only decrease synaptic strength by half
python run_damgo_ramp.py seed1-damgo_ramp --run_seed 1 --hyp_opioid 10 --syn_shut 0.5
```
Edit/run the executable run_damgo_ramps script in order to sweep across many different network seeds and different parameters. Once the .pkl data has been generated, the script will run damgo_ramp_postproc.py, plot_max_fr_ld50.py, and create_graphs.py in that order. The arguments for each of these three scripts is the range of network seeds we want to use.

```bash
# postprocessing the data for network seeds 1-40
python damgo_ramp_postproc.py 1 40
python plot_max_fr_ld50.py 1 40
python create_graphs.py 1 40
```
Now we should have data ready to be used in damgo_ramp_analysis.ipynb and graph_metrics.ipynb.