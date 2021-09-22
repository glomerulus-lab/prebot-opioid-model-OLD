#%%
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')
import matplotlib
import matplotlib.pyplot as plt
from brian2 import *
import brian_utils.postproc as bup
import pickle
import click
#%%
@click.command()
@click.argument('start_seed', type = int)
@click.argument('end_seed', type = int)
def main(start_seed, end_seed):
    
    # initialize dataframe
    df_data = {'run_seed': range(start_seed,end_seed+1), 'i_opioid_shutdown_val': 0,

               'oprm1_T_oprm1_T': 0, 'oprm1_T_oprm1_B': 0, 'oprm1_T_oprm1_Q': 0,
               'oprm1_B_oprm1_T': 0, 'oprm1_B_oprm1_B': 0, 'oprm1_B_oprm1_Q': 0, 
               'oprm1_Q_oprm1_T': 0, 'oprm1_Q_oprm1_B': 0, 'oprm1_Q_oprm1_Q': 0,

               'exc_T_oprm1_T': 0, 'exc_T_oprm1_B': 0, 'exc_T_oprm1_Q': 0,
               'exc_B_oprm1_T': 0, 'exc_B_oprm1_B': 0, 'exc_B_oprm1_Q': 0, 
               'exc_Q_oprm1_T': 0, 'exc_Q_oprm1_B': 0, 'exc_Q_oprm1_Q': 0,

               'inh_T_oprm1_T': 0, 'inh_T_oprm1_B': 0, 'inh_T_oprm1_Q': 0,
               'inh_B_oprm1_T': 0, 'inh_B_oprm1_B': 0, 'inh_B_oprm1_Q': 0, 
               'inh_Q_oprm1_T': 0, 'inh_Q_oprm1_B': 0, 'inh_Q_oprm1_Q': 0,

               'oprm1_T_exc_T': 0, 'oprm1_T_exc_B': 0, 'oprm1_T_exc_Q': 0,
               'oprm1_B_exc_T': 0, 'oprm1_B_exc_B': 0, 'oprm1_B_exc_Q': 0, 
               'oprm1_Q_exc_T': 0, 'oprm1_Q_exc_B': 0, 'oprm1_Q_exc_Q': 0,

               'oprm1_T_inh_T': 0, 'oprm1_T_inh_B': 0, 'oprm1_T_inh_Q': 0,
               'oprm1_B_inh_T': 0, 'oprm1_B_inh_B': 0, 'oprm1_B_inh_Q': 0, 
               'oprm1_Q_inh_T': 0, 'oprm1_Q_inh_B': 0, 'oprm1_Q_inh_Q': 0,

               'exc_T_exc_T': 0, 'exc_T_exc_B': 0, 'exc_T_exc_Q': 0,
               'exc_B_exc_T': 0, 'exc_B_exc_B': 0, 'exc_B_exc_Q': 0, 
               'exc_Q_exc_T': 0, 'exc_Q_exc_B': 0, 'exc_Q_exc_Q': 0,

               'exc_T_inh_T': 0, 'exc_T_inh_B': 0, 'exc_T_inh_Q': 0,
               'exc_B_inh_T': 0, 'exc_B_inh_B': 0, 'exc_B_inh_Q': 0, 
               'exc_Q_inh_T': 0, 'exc_Q_inh_B': 0, 'exc_Q_inh_Q': 0,

               'inh_T_exc_T': 0, 'inh_T_exc_B': 0, 'inh_T_exc_Q': 0,
               'inh_B_exc_T': 0, 'inh_B_exc_B': 0, 'inh_B_exc_Q': 0, 
               'inh_Q_exc_T': 0, 'inh_Q_exc_B': 0, 'inh_Q_exc_Q': 0,

               'inh_T_inh_T': 0, 'inh_T_inh_B': 0, 'inh_T_inh_Q': 0,
               'inh_B_inh_T': 0, 'inh_B_inh_B': 0, 'inh_B_inh_Q': 0, 
               'inh_Q_inh_T': 0, 'inh_Q_inh_B': 0, 'inh_Q_inh_Q': 0,
              }
    df = pd.DataFrame(df_data)
    
    burst_stats_list = [] # Initialize list that will contain burst statistic dataframes of each network (to be concatenated into one dataframe later)
    
    for seed in range(start_seed,end_seed+1):
        print(f'Processing seed {seed}...')
        
        # change filepath and/or naming accordingly (this can be a common cause of bugs)
        with open(f'damgo_ramp_pkls/batch2_pkls/seed{seed}-damgo_ramp_vars.pkl','rb') as fid:
            ramp_data = pickle.load(fid)
            
        rate = ramp_data['ratemonitor']
        binsize = 10 * ms
        smoothed_pop_rate = bup.smooth_saved_rate(rate, binsize)
        burst_stats = bup.pop_burst_stats(rate['t'], smoothed_pop_rate, height = 4, prominence = 6)
        burst_stats_list.append(burst_stats)
        
#         uncomment lines 77-87 and comment out line 90 if the network's cell classification hasn't already been processed into npy files
        
#         with open(f'pkl_files/synblock_batch/seed{seed}-control_pop_vars.pkl','rb') as fid:
#             synblock_data = pickle.load(fid)

#         ts = synblock_data['spikemonitor']['t']
#         spike_idx = synblock_data['spikemonitor']['i']
#         train = bup.create_train(ts,spike_idx)
#         cell_int, cell_class = bup.find_bursters_pk_ISI(train,300,)
        
#         print('Saving cell classes...')
#         with open(f'npy_files/seed{seed}_cell_class.npy', 'wb') as f:
#             np.save(f, cell_class)

        # load T/B/Q classification of neurons 
        cell_class = np.load(f'cell_classes/seed{seed}_cell_class.npy')

        # count connection types and update dataframe accordingly
        # NOTE: neurons 0-59 are inhibitory, 60-179 are oprm1+, 180-299 are oprm1- (exc)
        # i is the source neuron, j is the target neuron (i.e. inh_j contains the target neurons of the inhibitory neurons)
        print('Counting connections...')
        damgo_i = ramp_data['oprm1_i'] + 60
        damgo_j = ramp_data['oprm1_j']
        for i in range(len(damgo_i)):
            if damgo_j[i] >= 60 and damgo_j[i] < 180:
                df.loc[df['run_seed'] == seed, f'oprm1_{cell_class[damgo_i[i]][0].upper()}_oprm1_{cell_class[damgo_j[i]][0].upper()}'] += 1
        
            elif damgo_j[i] < 60:
                df.loc[df['run_seed'] == seed, f'oprm1_{cell_class[damgo_i[i]][0].upper()}_inh_{cell_class[damgo_j[i]][0].upper()}'] += 1
                
            elif damgo_j[i] >= 180 and damgo_j[i] < 300:
                df.loc[df['run_seed'] == seed, f'oprm1_{cell_class[damgo_i[i]][0].upper()}_exc_{cell_class[damgo_j[i]][0].upper()}'] += 1

        inh_i = ramp_data['inh_i']
        inh_j = ramp_data['inh_j']
        for i in range(len(inh_i)):
            if inh_j[i] >= 60 and inh_j[i] < 180:
                df.loc[df['run_seed'] == seed, f'inh_{cell_class[inh_i[i]][0].upper()}_oprm1_{cell_class[inh_j[i]][0].upper()}'] += 1
         
            elif inh_j[i] < 60:
                df.loc[df['run_seed'] == seed, f'inh_{cell_class[inh_i[i]][0].upper()}_inh_{cell_class[inh_j[i]][0].upper()}'] += 1
            
            elif inh_j[i] >= 180 and inh_j[i] < 300:
                df.loc[df['run_seed'] == seed, f'inh_{cell_class[inh_i[i]][0].upper()}_exc_{cell_class[inh_j[i]][0].upper()}'] += 1
        
        exc_i = ramp_data['exc_i'] + 180
        exc_j = ramp_data['exc_j']
        for i in range(len(exc_i)):
            if exc_j[i] >= 60 and exc_j[i] < 180:
                df.loc[df['run_seed'] == seed, f'exc_{cell_class[exc_i[i]][0].upper()}_oprm1_{cell_class[exc_j[i]][0].upper()}'] += 1
           
            elif exc_j[i] < 60:
                df.loc[df['run_seed'] == seed, f'exc_{cell_class[exc_i[i]][0].upper()}_inh_{cell_class[exc_j[i]][0].upper()}'] += 1
                
            elif exc_j[i] >= 180 and exc_j[i] < 300:
                df.loc[df['run_seed'] == seed, f'exc_{cell_class[exc_i[i]][0].upper()}_exc_{cell_class[exc_j[i]][0].upper()}'] += 1
    
        # Compute network shutdown dosage using firing rate thresholds
        peaks = smoothed_pop_rate[burst_stats['Peak Samples']] / Hz        
        damgo_conc = ramp_data['vm_opioid']/pA
        
        # For more robustness, we get the dosage from the last burst at multiple thresholds (10-15 Hz) and take the average 
        i_opioid_peak10 = damgo_conc[int(burst_stats[burst_stats['Peaks'] >= 10].iloc[-1]['Offset Times'] / 3)]
        i_opioid_peak11 = damgo_conc[int(burst_stats[burst_stats['Peaks'] >= 11].iloc[-1]['Offset Times'] / 3)]
        i_opioid_peak12 = damgo_conc[int(burst_stats[burst_stats['Peaks'] >= 12].iloc[-1]['Offset Times'] / 3)]
        i_opioid_peak13 = damgo_conc[int(burst_stats[burst_stats['Peaks'] >= 13].iloc[-1]['Offset Times'] / 3)]
        i_opioid_peak14 = damgo_conc[int(burst_stats[burst_stats['Peaks'] >= 14].iloc[-1]['Offset Times'] / 3)]
        i_opioid_peak15 = damgo_conc[int(burst_stats[burst_stats['Peaks'] >= 15].iloc[-1]['Offset Times'] / 3)]
        shutdown_val = (i_opioid_peak10 + i_opioid_peak11 + i_opioid_peak12 + i_opioid_peak13 + i_opioid_peak14 + i_opioid_peak15) / 6
        df.loc[df['run_seed'] == seed, 'I_opioid_shutdown'] = shutdown_val
        
    # save data
    df.to_csv(f'seed{start_seed}to{end_seed}_damgo_ramp.csv')
    seed1to40_burst_stats = pd.concat(burst_stats_list)
    seed1to40_burst_stats.to_csv(f'seed{start_seed}to{end_seed}_damgo_ramp_burst_stats.csv')

if __name__ == "__main__":
    main()
