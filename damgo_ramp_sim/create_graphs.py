#!/usr/bin/env python
# coding: utf-8

# Use this script to create networkx DiGraph objects for each network and store them in pkl files

import pandas as pd
import numpy as np
import pickle
import networkx as nx
import brian2
import matplotlib
import matplotlib.pyplot as plt
import glob
import re
import io
import os
import sys 

@click.command()
@click.argument('first_seed', type = int)
@click.argument('last_seed', type = int)
def main(first_seed, last_seed):
    for seed in range(first_seed,last_seed+1):
        # change filepath and/or naming accordingly (this can be a common cause of bugs)
        with open(f'damgo_ramp_pkls/batch1_pkls/seed{seed}-damgo_ramp_hyp8_vars.pkl', 'rb') as fid:
            data = pickle.load(fid)
    
        network = nx.DiGraph()
        network.add_nodes_from(np.arange(0,60,1), neuron_type = 'inh')
        network.add_nodes_from(np.arange(60,180,1), neuron_type = 'oprm1')
        network.add_nodes_from(np.arange(180,300,1), neuron_type = 'exc')
        
        cell_class = np.load(f'cell_classes/seed{seed}_cell_class.npy')
        for i in range(len(cell_class)):
            network.nodes[i]['spike_type'] = cell_class[i] 
        
        exc_i = data['exc_i'] + 180
        exc_j = data['exc_j']
        for i in range(len(exc_i)):
            network.add_edge(exc_i[i], exc_j[i])
            
        oprm1_i = data['oprm1_i'] + 60
        oprm1_j = data['oprm1_j']
        for i in range(len(oprm1_i)):
            network.add_edge(oprm1_i[i], oprm1_j[i])
            
        inh_i = data['inh_i']
        inh_j = data['inh_j']
        for i in range(len(inh_i)):
            network.add_edge(inh_i[i], inh_j[i])
            
        for u, v, attr in network.edges(data=True):
            network[u][v]['con_type'] = f'{network.nodes[u]["neuron_type"]}_{network.nodes[u]["spike_type"][0].upper()}_{network.nodes[v]["neuron_type"]}_{network.nodes[v]["spike_type"][0].upper()}'    
            
        nx.write_gpickle(network, f'networkx_objects/seed{seed}graph.pkl')
        print(f'Seed {seed} graph saved.')

if __name__ == "__main__":
    main()
