import xarray as xr
import os
import numpy as np
import xrft
from functools import cached_property
from helpers.computational_tools import *
from helpers.netcdf_cache import netcdf_property
import math

class main_property(cached_property):
    '''
    https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
    '''
    pass

class Experiment:
    '''
    Imitates xarray. All variables are
    returned as @property. Compared to xarray, allows
    additional computational tools and initialized instantly (within ms)
    '''
    def __init__(self, folder, key=''):
        '''
        Initializes with folder containing all netcdf files corresponding
        to a given experiment.
        Xarray datasets are read only by demand within @property decorator
        @cached_property allows to read each netcdf file only ones

        All fields needed for plotting purposes are suggested to be
        registered with @cached_property decorator (for convenience)
        '''
        self.folder = folder
        self.key = key # for storage of statistics
        self.recompute = False # default value of recomputing of cached on disk properties

        if not os.path.exists(os.path.join(self.folder, 'ocean_geometry.nc')):
            print('Error, cannot find files in folder'+self.folder)

    def remesh(self, target, key, compute=False, operator=remesh, FGR=None):
        '''
        Returns object "experiment", where "Main variables"
        are coarsegrained according to resolution of the target experiment
        operator - filtering/coarsegraining operator 
        '''

        # The coarsegrained experiment is no longer attached to the folder
        result = Experiment(folder=self.folder, key=key)
        result.operator = operator
        result.FGR = FGR

        # Coarsegrain "Main variables" explicitly
        for key in Experiment.get_list_of_main_properties():
            if key in ['u', 'ua']:
                mask = 'wet_u'
            elif key in ['v', 'va']:
                mask = 'wet_v'
            elif key in ['e', 'h', 'ea', 'ha']:
                mask = 'wet'
            else:
                mask = 'wet_c'

            input_field = self.__getattribute__(key)
            output_mask = target.param_extended[mask]

            if operator == gaussian_remesh:
                kw = {'FGR': FGR, 'input_mask': self.param_extended[mask]}
            else:
                kw = {}

            if compute:
                setattr(result, key, operator(input_field,output_mask,**kw).compute())
            else:
                setattr(result, key, operator(input_field,output_mask,**kw))

        result.param = target.param # copy coordinates from target experiment
        result._hires = self # store reference to the original experiment

        return result

    def set_averaging_time(self):
        self.Averaging_time = slice(None,None)

    @classmethod
    def get_list_of_main_properties(cls):
        '''
        Allows to know what properties should be coarsegrained
        '''
        result = []
        for name, value in vars(cls).items():
            if isinstance(value, main_property):
                result.append(name)
        return result
    
    ################### Getters for netcdf files as xarrays #####################
    @cached_property
    def series(self):
        result = xr.open_dataset(os.path.join(self.folder, 'ocean.stats.nc'), decode_times=False)
        return result

    @cached_property
    def param(self):
        result = xr.open_dataset(os.path.join(self.folder, 'ocean_geometry.nc')).rename(
                {'latq': 'yq', 'lonq': 'xq', 'lath': 'yh', 'lonh': 'xh'} # change coordinates notation as in other files
            )
        return rename_coordinates(result)
        return result

    @cached_property
    def ocean_daily(self):
        result = xr.open_mfdataset(os.path.join(self.folder, '*ocean_daily*.nc'), parallel=True)
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_month(self):
        result = xr.open_mfdataset(os.path.join(self.folder, '*ocean_month_0*.nc'), parallel=True)
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_month_z(self):
        result = xr.open_mfdataset(os.path.join(self.folder, '*ocean_month_z*.nc'), parallel=True)
        rename_coordinates(result)
        return result

    ######################## Auxiliary variables #########################
    @main_property
    def ssh(self):
        return self.ocean_faily.zos

    ####################### Grid variables ###########################
    @cached_property
    def param_extended(self):
        grid = create_grid_global(self.param)

        param = self.param
        param['wet_u']=np.floor(grid.interp(param.wet,'X'))
        param['wet_v']=np.floor(grid.interp(param.wet,'Y'))
        param['wet_c']=np.floor(grid.interp(param.wet,['X','Y']))

        return param

    ########################  Statistical tools  #########################

    #-------------------  Mean flow and variability  --------------------#
    @netcdf_property
    def ssh_mean(self):
        return self.ssf.sel(Time=self.Averaging_time).mean(dim='Time')