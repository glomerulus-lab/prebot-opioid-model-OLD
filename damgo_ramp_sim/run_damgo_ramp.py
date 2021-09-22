#!/usr/bin/env python
# coding: utf-8

# DAMGO ramping experiment

### Import needed packages
import os
from tqdm import tqdm
import sys
import seaborn as sns
import click

sys.path.append('..')

from brian2 import * # This is the meaty import
import brian_utils.postproc as bup # Import the postprocessing module
import brian_utils.model_IO as bio # Import the postprocessing module
import numpy as np
import pandas as pd
import pickle# For saving the runs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ### Create convinience functions

def set_ics(neurons):
    '''
    Set the initial conditions of the state variables
    '''
    neurons.v = -58*mV
    neurons.h = 0.1
    neurons.n = 0.1
    neurons.g_syne = 0 * nS
    neurons.g_syni = 0 * nS
    neurons.g_synopioid = 0 * nS

# This function is not critical to understand at this point.
def remap_inhibitory_tonic_to_quiescent(gl_vals,n_inh,t_val=0.5,q_val=1.2):
    '''
    Finds all the inhibitory tonic neurons and turns them into quiescent neurons. Takes the same number of
    excitatory quiescent neurons and turns them into tonic neurons

    This does not change the number of B/T/Q neurons in the overall population
    :param gl_vals:
    :param n_inh:
    :param t_val:
    :param q_val:
    :return:
    '''
    mask = np.where(gl_vals[:n_inh]==t_val*nS)[0]
    n_switch = len(mask)
    gl_vals[:n_inh][mask] = q_val*nS
    idx = np.where(gl_vals[n_inh:]==q_val*nS)[0]
    sub_idx = np.random.choice(idx,n_switch,replace=False)
    gl_vals[n_inh:][sub_idx] = t_val * nS

    return(gl_vals)

@click.command()
@click.argument('prefix', type = str) # 1st command line argument: naming the run (for saving purposes)
@click.option('--run_seed', type = int, default = 0, show_default = True) # 2nd command line argument: setting the seed to initialize different random networks
@click.option('--g_nap_str', type = float, default = 1, show_default = True) 
@click.option('--idx', type = int, default = 0, show_default = True)
@click.option('--hyp_opioid', type = int, default = 8, show_default = True) # Opioid hyperpolarization magnitude (pA)
@click.option('--syn_shut', type = float, default = 1, show_default = True) # Opioid synaptic shutdown effect [0,1]
def main(prefix, run_seed, g_nap_str, idx, hyp_opioid, syn_shut):
    # Set some run parameters
    # Typically these will be set as command line arguments (that is, this are commonly swept variables).
    # Units get added later as these values will be used in string operations (unimportant)
    os.mkdir(f'damgo_ramp_figs/batch2_figs/{prefix}')
    basename='test'
    savename= f'damgo_ramp_pkls/batch2_pkls/{prefix}_vars.pkl'

    N = 300 # Keep this small for testing purposes
    k_avg = 6 # Average synapses/neuron

    pt = 0.35 # Percent intrinsically tonic neurons [0,1]
    pb = 0.1 # Percent intrincially bursting neurons [0,1]
    pQ = 1-pt-pb # Quiescent neurons are left over

    frac_oprm1 = 0.5 # Fraction of excitatory neurons that are OPRM1+ [0,1]
    frac_inh = 0.2 # Fraction of neurons that are inhibitory [0,1]

    we_max = 3.5 # Excitatory synaptic strength (nS)
    wi_max = 3.5 # Inhibitory synaptic strength (nS)

    perturbation_dt = 3*second # Time between between perturbation steps

    # Set the runname and seeds, initialize a helper variable

    # set random seeds for Brian's code generator, and for numpy
    seed(run_seed)
    np.random.seed(run_seed)

    synblock = 1 # binary used to block the synapses at the end
    pct_connect = (k_avg / 2) / (N - 1) # Calculate the connection probability based on the average connection number

    # Map fractions of population to integers
    n_inh = int(np.ceil(N*frac_inh))
    n_oprm1 = int((N-n_inh)*frac_oprm1)
    n_excit = N-n_inh
    print(f'Saving to: {prefix}')

    # The leak conductance (gl) sets the neurons' identity as (tonic, bursting, quiescent) according to the fraction of neurons for each group
    gl_vals = np.random.choice([0.5, 0.7, 1.2], N, p=[pt, pb, pQ]) * nS
    # Make all inhibitory tonic neurons quiescent
    gl_vals = remap_inhibitory_tonic_to_quiescent(gl_vals,n_inh,t_val=0.5,q_val = 1.2)
    plt.hist(gl_vals/nS)
    plt.xlabel('$G_{leak}$ (nS)')
    plt.ylabel('Number of Neurons')

    # ### Use the model IO module to load in the equations and parameters
    #
    # "eqs" and "ns" get parsed into dicts

    eqs_yaml = '../harris_eqs_oprmv1_gnap.yaml'
    ns_yaml = '../harris_ns_oprmv1.yaml'
    eqs = bio.import_eqs_from_yaml(eqs_yaml)
    ns = bio.import_namespace_from_yaml(ns_yaml)
    print(eqs.keys())

    # For illustration, these are the imported equations that define the neuron model
    print(eqs['neuron_eqs'])

    # And the equations that define the namespace, ie. the shared parameters:
    for k,val in ns.items():
        print(f'{k} : {val}')

    # ## Set up the model
    defaultclock.dt = 0.05*ms # set the timestep -- if you get NaNs in your runs, decrease this (smaller timesteps)

    # the cpp_standalone mode gives you massive speed up, no interactivity. Use it when working, but it won't work here
    run_cpp = True
    if run_cpp:
        print('Setting the device')
        set_device('cpp_standalone',clean=True,build_on_run=False)
        prefs.devices.cpp_standalone.openmp_threads = 4 # For multithreading

    # Set up the full neuron population

    neurons = NeuronGroup(N, eqs['neuron_eqs'],
                        threshold='v>=-20*mV',
                        refractory=2 * ms,
                        method='rk4',
                        namespace=ns)

    # ### Set up synapses.
    print('Setting up synapses...')

    # Set subpopulations
    Pi = neurons[:n_inh] # inhibitory neurons
    Pe_oprm1 = neurons[n_inh:n_oprm1+n_inh] # Excitatory oprm1+ neurons
    Pe = neurons[n_oprm1+n_inh:] # Excitatory oprm1- neurons

    # Connect the subpopulations to the entire population - note, the different synapses affect different variables in the post-synaptic neuron
    # con_oprm1 affects the gsyn_opioid in the post-synaptic cell and con_e affects gsyn_e
    con_e = Synapses(Pe, neurons, model=eqs['excitatory_synapse_eqs'], method='exponential_euler', namespace=ns)
    con_oprm1 = Synapses(Pe_oprm1, neurons, model=eqs['opioid_synapse_eqs'], method='exponential_euler', namespace=ns)
    con_i = Synapses(Pi, neurons, model=eqs['inhibitory_synapse_eqs'], method='exponential_euler', namespace=ns)

    # Use the run seed to reset the random number generator and create the network topology -
    # this allows for repetition of the same network topology based on the run seed provided
    np.random.seed(run_seed)
    seed(run_seed)
    con_i.connect(p=pct_connect)
    con_e.connect(p=pct_connect)
    con_oprm1.connect(p=pct_connect)

    # ### Set up integrator (for use in the phase-driven optogenetic stimulus)
    # Kept for posterity, not used, but you need to run this...
    sensor_int = NeuronGroup(1, eqs['sensor_eqs'], method='rk4')

    # incoming connections
    con_in = Synapses(neurons, sensor_int, on_pre='v += 1')
    con_in.connect()  # fully connected

    # outgoing connections
    neurons.sensor = linked_var(sensor_int, 'v')
    sensor_int.v = 0.

    #
    # Set up cell specific parameters
    #

    set_ics(neurons)

    # again reinit the random seed to make the runs repeatable
    seed(run_seed)
    np.random.seed(run_seed)

    # Add some randomness to the gl,gnap values
    gl_vals = gl_vals + np.random.normal(0,0.05,N) * nS
    gnap_vals = 0.8 * nS + np.random.normal(0,.05,N) * nS

    # Assign the values to the neurons
    neurons.g_l = gl_vals
    neurons.g_k = 11.2 * nsiemens
    neurons.g_na = 28. * nsiemens
    neurons.g_nap = gnap_vals
    # Make all neurons damgo-insensitive, then make the OPRM1 neurons damgo sensitive (these are multipliers on the gsynopioid variable in the neuron equations)
    neurons.damgo_sensitivity = 0
    Pe_oprm1.damgo_sensitivity = 1

    # ### Set up monitors for saving the data

    statemon = StateMonitor(neurons, variables=['v','h','n','g_syne','g_syni','g_synopioid'],
                            record=np.arange(0, N).astype('int'), dt=1 * ms)

    # Record the population spike rate
    ratemon = PopulationRateMonitor(neurons)
    # Record the population spike times
    spikemon = SpikeMonitor(neurons)


    # Set up perturbations
    # This is a little confusing. Here we are using timed-arrays to turn on and off different perturbations.
    # Each value will be applied for the duration of perturbation_dt. (Change these sequences to modify the pertubation timing and duration)
    
    # increasing the opioid dosage by 5 percent each perturbation
    op_vals = np.arange(0,1.005,0.005)
    
    # no g_nap perturbation in this experiment
    g_nap_mult = np.zeros(len(op_vals)) + 1

    we = np.ones(len(op_vals)) # When are excitatory synapses active? (always)
    wi = np.ones(len(op_vals)) # When are inhibitory synapses active
    
    # Map to a Timed Array with the magnitude scaled by the run parameter (e.g. we_max)
    we = TimedArray(we*we_max*nS,dt=perturbation_dt)
    wi = TimedArray(wi*wi_max*nS,dt=perturbation_dt)

    # Map the opioid effects to timed arrays
    g_nap_drug = TimedArray(g_nap_mult, dt = perturbation_dt)
    vm_opioid = TimedArray(op_vals*hyp_opioid*pA,dt = perturbation_dt) # The hyperpolarization effect
    we_opioid = TimedArray((1-op_vals*syn_shut)*we_max*nS,dt = perturbation_dt) # The synaptic shutdown effect

    # make the runtime as long as all the perturbations
    runtime = len(op_vals)*perturbation_dt

    #plot perturbed gnap vs gleak
    modified_gnap = (g_nap_str + 1) * gnap_vals
    plt.figure(figsize=(4,4))
    plt.plot(gl_vals/nS, gnap_vals/nS, 'k.', alpha = 0.4)
    #plt.plot(gl_vals/nS, modified_gnap/nS, 'r.', alpha=0.4)
    plt.tight_layout()
    plt.xlabel('$g_{leak} (nS)$')
    plt.ylabel('$g_{NaP} (nS)$')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'damgo_ramp_figs/batch2_figs/{prefix}/{prefix}_mod_gnap_gleak.png')


    # Visualize the perturbations
    tt = np.arange(0,runtime,1)*second
    tlines = np.arange(0,runtime,perturbation_dt) # these show you every "perturbation_dt" time period

    f,ax = plt.subplots(nrows=5,sharex=True)
    ax[0].plot(tt,we(tt)/nS)
    ax[1].plot(tt,wi(tt)/nS)
    ax[2].plot(tt,vm_opioid(tt)/pA)
    ax[3].plot(tt,we_opioid(tt)/nS)
    ax[4].plot(tt,g_nap_drug(tt)/nS)

    ax[0].set_ylabel('E Syn')
    ax[1].set_ylabel('I Syn')
    ax[2].set_ylabel('Hyp')
    ax[3].set_ylabel('Opioid Syn')
    ax[4].set_ylabel('Drug')

    plt.xlabel('time(s)')
    for ii in range(5):
        for ll in tlines:
            ax[ii].axvline(ll,color='k',alpha=0.2)
    sns.despine()
    plt.savefig(f'damgo_ramp_figs/batch2_figs/{prefix}/{prefix}_gnap_pert.png')


    # ## Run the network!

    seed(run_seed)
    np.random.seed(run_seed)
    # lets shorten the runtime so we can get to the end
    # runtime = 60*second
    # =============== #
    # Run baseline
    # =============== #
    print('Setting up run...may take some time')
    net = Network(collect())
    net.run(runtime, report='text')

    # =============== #
    # Run intrinsic (i.e. block all synapses)
    # =============== #
#     synblock = 0
#     net.run(20*second,report='text')

    #====================== #
    # Build
    #====================== #
    if run_cpp:
        device.build(directory=f'./{idx:04.0f}', compile=True, run=True, debug=False, clean=True)
        # device.build(directory=None, compile=True, run=True, debug=False, clean=True)

    # ## Save the results

    # Standardizing this could be good?
    save_tgl = True
    if save_tgl:
        states = net.get_states()
        states['ns'] = ns
        states['eqs'] = eqs
        states['n_inh'] = n_inh
        states['n_oprm1'] = n_oprm1
        states['hyp'] = hyp_opioid
        states['syn_shut'] = syn_shut
        states['run_seed'] = run_seed
        states['perturbation_dt'] = perturbation_dt
        states['we_opioid'] = we_opioid.values
        states['vm_opioid'] = vm_opioid.values
        states['we'] = we.values
        states['wi'] = wi.values
        states['exc_i'] = con_e.i[0:len(con_e.i)]
        states['exc_j'] = con_e.j[0:len(con_e.j)]
        states['inh_i'] = con_i.i[0:len(con_i.i)]
        states['inh_j'] = con_i.j[0:len(con_i.j)]
        states['oprm1_i'] = con_oprm1.i[0:len(con_oprm1.i)]
        states['oprm1_j'] = con_oprm1.j[0:len(con_oprm1.j)]
        states['int_i'] = con_in.i[0:len(con_in.i)]
        states['int_j'] = con_in.j[0:len(con_in.j)]

        with open(savename,'wb') as fid:
            pickle.dump(states,fid)
    

    # this deletes all the temporary files if using CPP mode
    if run_cpp:
        device.delete


    # ## Do some preliminary plotting
    #### Compute the binned spike trains and population rates
    raster, cell_id, bins = bup.bin_trains(spikemon.t, spikemon.i, neurons.N, max_time=statemon.t[-1])
    binsize = 10 * ms
    smoothed_pop_rate = ratemon.smooth_rate('gaussian', binsize)

    # =================== #
    # Plotting the experiment trace
    # =================== #
    f = plt.figure(figsize=(14, 16))
    g = f.add_gridspec(12, 1)

    # Plot raster
    ax0 = f.add_subplot(g[:6, 0])
    plt.pcolormesh(bins, cell_id, raster / (binsize / second), cmap='gray_r')
    plt.ylabel('Neuron')

    # Plot population rate
    ax1 = f.add_subplot(g[7, 0], sharex=ax0)
    plt.plot(ratemon.t, smoothed_pop_rate, 'k', linewidth=1,alpha=0.5)
    plt.ylabel('FR\n(Hz/cell)')
    burst_stats = bup.pop_burst_stats(ratemon.t, smoothed_pop_rate, height = 4, prominence = 6)
    #burst_stats.to_csv(f'csv_files/{prefix}_burst_stats.csv')

    # Plot DAMGO
    ax1 = f.add_subplot(g[8, 0], sharex=ax0)
    tvec = np.arange(0,runtime/second,.1)*second
    ax1.plot(tvec,we_opioid(tvec)/nS, 'tab:blue', linewidth=1,alpha=0.5)
    ax1.set_ylim(0,we_max+1)
    # plt.ylabel('DAMGO (percent max block)')
    ax1.set_ylabel('$g_{syn}^{opioid}$',fontsize=8)

    ax2 = f.add_subplot(g[9, 0], sharex=ax0)
    tvec = np.arange(0,runtime/second,.1)*second
    ax2.plot(tvec,vm_opioid(tvec)/pA, 'tab:blue', linewidth=1,alpha=0.5)
    ax2.set_ylim(0,hyp_opioid)
    # plt.ylabel('DAMGO (percent max block)')
    ax2.set_ylabel('$I_{opioid}$ (pA)',fontsize=8)
    ax2.set_xlabel('Time (s)')

    # Plot we/wi
    ax3 = f.add_subplot(g[10, 0], sharex=ax0)
    tvec = np.arange(0,runtime/second,.1)*second
    ax3.plot(tvec,we(tvec)/nS, 'tab:blue', linewidth=1,alpha=0.5)
    ax3.plot(tvec,wi(tvec)/nS, 'tab:red', linewidth=1,alpha=0.5)
    ax3.set_ylim(-.1,we_max+1)
    # plt.ylabel('DAMGO (percent max block)')
    ax3.set_ylabel('Syn str (nS)',fontsize=8)
    ax3.set_xlabel('Time (s)')

    # Plot drug perturbation
    ax4 = f.add_subplot(g[11,0], sharex=ax0)
    ax4.plot(tt,g_nap_drug(tt)/nS)
    tvec = np.arange(0,runtime/second,.1)*second
    ax4.set_ylabel(r'$G_NaP (nS)$',fontsize=8)
    ax4.set_xlabel('Time (s)')

    # Cleanup and save
    sns.despine()
    # plt.xlim(5,13)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=1.0)
    plt.xlim(10,runtime/second)
    plt.savefig(f'damgo_ramp_figs/batch2_figs/{prefix}/{prefix}_trace.png',dpi=300)
    
#     plt.figure(figsize=(4,4))
#     for ii in range(15):
#         plt.plot(statemon.t,statemon.v[ii]/mV+70*ii,'k',lw=0.5)
#     sns.despine()
#     plt.axvline(runtime/second,c='tab:red',ls='--')
#     plt.text(runtime/second,1000,'Syn. Block',c='tab:red')
#     plt.savefig(f'damgo_ramp_figs/batch2_figs/{prefix}/{prefix}_synblock.png')
    
    f = plt.figure(figsize=(20, 10))
    g = f.add_gridspec(3, 1, height_ratios = [2,1,1])
    plt.tight_layout()

    # Plot zoomed in, higher resolution trace
    ax0 = f.add_subplot(g[0,0])
    ax0.plot(ratemon.t, smoothed_pop_rate, 'k', linewidth=1, alpha=0.7)
    ax0.set_xlabel('Time (s)')
    ax0.set_xlim(0,600)
    ax0.set_ylabel('FR\n(Hz/s)')
    ax0.plot(burst_stats['Peak Times'], smoothed_pop_rate[burst_stats['Peak Samples']], ".", c = "tab:red")
    for index, row in burst_stats.iterrows():
        ax0.axvspan(row['Onset Times'], row['Offset Times'], color='tab:blue', alpha=0.3, lw=0)
    
    ax1 = f.add_subplot(g[1,0], sharex=ax0)
    tvec = np.arange(0,runtime/second,.1)*second
    ax1.plot(tvec,we_opioid(tvec)/nS, 'tab:blue', linewidth=1,alpha=0.7)
    ax1.set_ylim(0,we_max+1)
    ax1.set_ylabel('$G_{syn}^{opioid}$',fontsize=8)

    ax2 = f.add_subplot(g[2,0], sharex=ax0)
    tvec = np.arange(0,runtime/second,.1)*second
    ax2.plot(tvec,vm_opioid(tvec)/pA, 'tab:blue', linewidth=1,alpha=0.7)
    ax2.set_ylim(0,hyp_opioid)
    ax2.set_ylabel('$I_{opioid}$ (pA)',fontsize=8)
    ax2.set_xlabel('Time (s)')
    
    plt.savefig(f'damgo_ramp_figs/batch2_figs/{prefix}/{prefix}_high_res_trace.png', dpi=300)


if __name__ == "__main__":
    main()
