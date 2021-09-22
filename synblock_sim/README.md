## synblock_sim/
- **synblock_data/** - .csv files for each network seed that contain the T/B/Q classification of each neuron in each phase of the G_NaP modulation experiment
- **synblock_figs/** - traces of each network simulation
- **phase_diagram.ipynb** - notebook for processing .pkl files outputted from simulation in order to create T/B/Q phase diagrams.
- **run_synblock.py** - code for running the experiment, command line arguments specify the network seed, phase of the experiment and whether to use the G_NaP/G_Leak two cloud population or the grid,  traces are outputted and data is saved into .pkl files
- **run_synblock_clouds** - bash script for running synaptic block for multiple network seeds and experiment phases using the two cloud G_NaP/G_Leak
- **run_synblock_grid** - bash script for running synaptic block for multiple network seeds and experiment phases using the grid of G_NaP/G_Leak values (0.2 - 1.5)

### Running the experiment
run_synblock.py arguments:
- ```prefix```: the name of the run, which is used for data/figure saving purposes. (This is always the first argument)
- ```run_seed```: network seed to be simulated
- ```condition```: the perturbation to be simulated (Control, DAMGO, G_NaP or DAMGO+G_NaP)
- ```g_nap_str```: The amount to increase or decrease G_NaP by (e.g. pass in 0.3 for a 30% increase, -0.3 for a 30% decrease)
- ```struct```: the structure of the assigned G_NaP and G_Leak values (two cloud population - 'clouds' or the grid from 0.2-1.5 nS - 'grid')
#### Example invocation:
```bash
python run_synblock.py seed1-damgo_gnap03_grid --run_seed 1 --g_nap_str 0.3 --condition DAMGO+G_NaP --struct grid
```
Note that executing the example above will only run one simulation. Edit/run the executable run_synblock_clouds and run_synblock_grid scripts in order to sweep across many different network seeds and different parameters. Once the data has been generated, use the phase_diagram.ipynb notebook to create the phase diagrams. 

