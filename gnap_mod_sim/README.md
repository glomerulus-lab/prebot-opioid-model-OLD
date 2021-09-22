## gnap_mod_sim/
- **gnap_mod_figs/** - various figures outputted from each G_NaP modulation simulation
- **burst_analysis.ipynb** - notebook for analyzing burst statistics
- **explore_network.ipynb** - notebook for exploratory analysis and regression of period, amplitude, width mean and irregularity of every network in each phase of the experiment against the count of excitation and inhibition
- **get_burst_stats.py** - postprocessing script that reads every .pkl file and outputs a .csv file containing burst statistics 
- **gnap_mod_burst_stats.csv** - dataset of burst statistics for network seeds 1-40 with G_NaP increasing/decreasing by 10%, 30%, and 50% (to be used in burst_analysis.ipynb)
- **gnap_mod_network_data.csv** - dataset containing number of excitatory and inhibitory synapses, exc/inh-exc/inh connection counts, number of T/B/Q neurons with exc or inh inputs, mean period/amplitude/width and period/amplitude/width irregularity for each network
- **quickstart.ipynb** - notebook for stepping through the experiment code (run_gnap_mod.py), useful for debugging
- **run_gnap_mod.py** - code for running the experiment, outputs traces and saves data into .pkl files
- **run_model.py** - same as run_gnap_mod.py, except with shorter transient periods.

### Running the experiment
run_damgo_ramp.py arguments:
- ```prefix```: the name of the run, which is used for data/figure saving purposes. (This is always the first argument)
- ```run_seed```: network seed to be simulated
- ```g_nap_str```: The amount to increase or decrease G_NaP by (e.g. pass in 0.3 for a 30% increase, -0.3 for a 30% decrease)
- ```hyp_opioid```: the opioid dosage given when DAMGO is turned on (pA), default value is 4 pA
- ```syn_shut```: the maximum decrease in synaptic strength [0,1] when DAMGO is turned on, default value is 0.5 (meaning that synaptic strength decreases by half)
#### Example invocations:
```bash
# increasing G_NaP by 30%
python run_gnap_mod.py seed1-gnap-03 --run_seed 1 --g_nap_str 0.3

# decreasing G_NaP by 30% (note: this effect is clearer when the opioid and synaptic shutdown isn't as strong)
python run_gnap_mod.py seed1-gnap-03 --run_seed 1 --g_nap_str -0.3 --hyp_opioid 3 --syn_shut 0.3
```
Edit/run the executable run_gnap_mods script in order to sweep across many different network seeds and different parameters. Once the .pkl data has been generated, the script will run get_burst_stats.py in order to generate the burst statistics from each simulation to be analyzed in burst_analysis.ipynb

```bash
# generating the burst statistics for all sims for network seeds 1-40
python get_burst_stats.py 1 40
```