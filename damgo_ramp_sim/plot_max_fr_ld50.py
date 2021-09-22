import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import re
import io
import os
import sys 
import pickle
sys.path.append('..')
from brian2 import *
import brian_utils.postproc as bup
from scipy import stats
from scipy.optimize import curve_fit
import click

# Define logistic equation to be used in scipy's curve_fit function
def func(x,L,x0,k,b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return y

@click.command()
@click.argument('start_seed', type = int)
@click.argument('end_seed', type = int)
def main(start_seed, end_seed):
    df = pd.read_csv(f'seed{start_seed}to{end_seed}_damgo_ramp.csv')
    ld50s = []
    for seed in range(1,41):
        # change filepath and/or naming accordingly (this can be a common cause of bugs)
        with open(f'damgo_ramp_pkls/batch2_pkls/seed{seed}-damgo_ramp_vars.pkl', 'rb') as fid:
            data = pickle.load(fid)
        rate = data['ratemonitor']
        binsize = 10*ms
        smoothed_pop_rate = bup.smooth_saved_rate(rate, binsize)
        pop_rate = smoothed_pop_rate / Hz

        ts = rate['t'] 
        ts = (ts / ms) / 1000
        
        # initialize two second window (avoid setting a window that would include the exact moment of a perturbation step)
        start = 0.98
        end = 2.98

        rate_df = {'time': ts, 'rate' : pop_rate}
        rate_df = pd.DataFrame(rate_df)
        op_vals = np.array(data['vm_opioid']/pA)

        X = op_vals[op_vals>=0]
        Y = []

        plt.figure(figsize=(10,10))
        for i in range(len(X)):
            window = np.logical_and(rate_df['time']>=start,rate_df['time']<=end)
            window_df = rate_df[window]
            Y.append(window_df['rate'].max())
            plt.plot(X[i], window_df['rate'].max(),'.', color='k', alpha=0.3)
            plt.ylim(0,40)
            
            # slide the window (3 sec because each dosage level lasts 3 seconds before ramping up again)
            start += 3 
            end += 3
        
        # compute LD50
        p0 = [max(Y),np.median(X),1,min(Y)]
        popt, pcov = curve_fit(func, X, Y,p0, method='dogbox')    
        ld50s.append(popt[1])
        print(f'Seed {seed} LD50: {popt[1]}')
        
        # fit logistic curve onto opioid dosage and max FR data
        plt.plot(X,func(X, *popt))
        plt.xlabel(r'$I_{opioid}$ (pA)')
        plt.ylabel('FR \n(Hz/cell)')
        plt.title('Max firing rates of each dosage level')
        plt.savefig(f'damgo_ramp_figs/batch2_figs/seed{seed}-damgo_ramp/seed{seed}_max_fr.png')
    
    # add LD50s to dataset
    df['LD50'] = ld50s
    df.to_csv(f'seed{start_seed}to{end_seed}_damgo_ramp.csv')

    
if __name__ == "__main__":
    main()