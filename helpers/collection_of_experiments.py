import xarray as xr
import os
from helpers.experiment import Experiment
from helpers.computational_tools import remesh, Lk_error
from helpers.plot_helpers import *
import cmocean
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class CollectionOfExperiments:
    '''
    This class automatically reads and initialized 
    all experiments in the given folder
    '''
    def __init__(self, exps, experiments_dict, names_dict):
        '''
        experiments_dict - "experiment" objects labeled by keys
        names_dict - labels for plotting
        '''
        self.exps = exps
        self.experiments = experiments_dict
        self.names = names_dict

    def __getitem__(self, q):
        ''' 
        Access experiments with key values directly
        '''
        try:
            return self.experiments[q]
        except:
            print('item not found')
    
    def __add__(self, otherCollection):
        # merge dictionaries and lists
        exps = [*self.exps, *otherCollection.exps]
        experiments_dict = {**self.experiments, **otherCollection.experiments}
        names_dict = {**self.names, **otherCollection.names}

        return CollectionOfExperiments(exps, experiments_dict, names_dict)
    
    def remesh(self, input, target, exp=None, name=None, compute=False, operator=remesh, FGR=None):
        '''
        input  - key of experiment to coarsegrain
        target - key of experiment we want to take coordinates from
        '''

        if exp is None:
            exp = input+'_'+target
        if name is None:
            name = input+' coarsegrained to '+target

        result = self[input].remesh(self[target], exp, compute, operator, FGR) # call experiment method

        print('Experiment '+input+' coarsegrained to '+target+
            ' is created. Its identificator='+exp)
        self.exps.append(exp)
        self.experiments[exp] = result
        self.names[exp] = name

    @classmethod
    def init_folder(cls, common_folder, exps=None, exps_names=None, additional_subfolder='output', prefix=None):
        '''
        Scan folders in common_folder and returns class instance with exps given by these folders
        exps - list of folders can be specified
        exps_names - list of labels can be specified
        additional_subfolder - if results are stored not in common_folder+exps[i],
        but in an additional subfolder 
        '''
        folders = []
        for root, dirs, files in os.walk(common_folder):
            if os.path.isfile(os.path.join(root, additional_subfolder, 'ocean.stats.nc')):
                folder = root[len(common_folder)+1:] # Path w.r.t. common_folder
                folders.append(
                    folder
                    )

        if exps_names is None:
            exps_names = folders

        exps = [folder.replace("/", "-") for folder in folders] # modify folder to be used as a key for caching files
        if prefix:
            exps = [prefix+'-'+exp for exp in exps]
            
        # Construct dictionary of experiments, where keys are given by exps
        experiments_dict = {}
        names_dict = {}
        for i in range(len(exps)):
            folder = os.path.join(common_folder,folders[i],additional_subfolder)
            experiments_dict[exps[i]] = Experiment(folder, exps[i])
            names_dict[exps[i]] = exps_names[i] # convert array to dictionary

        return cls(exps, experiments_dict, names_dict)
    
    def plot_series(self, exps, labels=None, colors=['gray', violet, 'tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:cyan', 'b', 'g', 'y']):
        default_rcParams({'font.size':12})
        plt.figure(figsize=(6,6))
        if labels is None:
            labels=exps
        if colors is None:
            colors = [None] * len(labels)

        for j, (exp, label) in enumerate(zip(exps, labels)):
            ds = self[exp].series
            ds['Time'] = ds['Time'] - ds['Time'][0]
            kw = {'lw':2, 'color':colors[j]}
            
            plt.subplot(2,1,1)
            (ds.KE.sum('Layer')).plot(**kw)
            plt.xlabel('Years')
            plt.xticks(np.arange(6)*365,np.arange(6))
            plt.grid()
            plt.ylabel('Kinetic energy, Joules')
            plt.ylim([0,6e+18])

            plt.subplot(2,1,2)
            (ds.APE.sum('Interface')).plot(label=label, **kw)
            plt.xlabel('Years')
            plt.xticks(np.arange(6)*365,np.arange(6))
            plt.grid()
            plt.ylabel('Available potential energy, Joules')
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.5,1))