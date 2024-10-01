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

    @property
    def Averaging_time(self):
        return slice('1979','1981')

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
        result = sort_longitude(xr.open_dataset('../data/ocean_static.nc')).drop_vars('time')
        rename_coordinates(result)
        return result

    @cached_property
    def ocean_daily(self):
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, '*ocean_daily*.nc'), parallel=True))
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_month(self):
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, '*ocean_month_0*.nc'), parallel=True))
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_month_z(self):
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, '*ocean_month_z*.nc'), parallel=True)).rename({'z_l': 'zl'})
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
    # @netcdf_property
    # def ssh_mean(self):
    #     return self.ssf.sel(Time=self.Averaging_time).mean(dim='Time')

    @cached_property
    def woa_temp(self):
        '''
        WOA temperature data on its native horizontal 1x1 grid
        and vertical grid of MOM6 output
        '''
        woa = sort_longitude(xr.open_dataset('../data/woa_1981_2010.nc', decode_times=False).rename({'lat':'yh', 'lon': 'xh'}).t_an.chunk({}))
        woa_interp = woa.interp(depth=self.ocean_month_z.zl)
        woa_interp[{'zl':0}] = woa[{'depth':0}]
        return woa_interp.squeeze().drop_vars(['time', 'depth'])
    
    @cached_property
    def MLD_summer_obs(self):
        obs = np.load('../data/mod_r6_cycle1_MLE1_zgrid_MLD_003_min.npy', allow_pickle='TRUE').item()
        obs = sort_longitude(xr.DataArray(obs['obs'], dims=['xh', 'yh'], coords={'xh':obs['lon'], 'yh':obs['lat']})).T
        return remesh(obs, self.woa_temp)
    
    @cached_property
    def MLD_winter_obs(self):
        obs = np.load('../data/mod_r6_cycle1_MLE1_zgrid_MLD_003_max.npy', allow_pickle='TRUE').item()
        obs = sort_longitude(xr.DataArray(obs['obs'], dims=['xh', 'yh'], coords={'xh':obs['lon'], 'yh':obs['lat']})).T
        return remesh(obs, self.woa_temp)
    
    @cached_property
    def ssh_std_obs(self):
        '''
        Copernicus data
        '''
        obs = sort_longitude(xr.open_dataset('../data/ssh_std_obs.nc', decode_times=False).rename({'lat':'yh', 'lon': 'xh'}).adt.chunk({}))
        return obs
    
    @netcdf_property
    def thetao(self):
        out = self.ocean_month_z.thetao.sel(time=self.Averaging_time).mean('time')
        out = remesh(out, self.woa_temp)
        return xr.where(np.isnan(self.woa_temp), np.nan, out)
    
    @netcdf_property
    def MLD_summer(self):
        MLD_003 = remesh(self.ocean_month.MLD_003, self.woa_temp).sel(time=self.Averaging_time)
        MLD_003_month = MLD_003.groupby('time.month').mean('time')
        return MLD_003_month.min('month')
    
    @netcdf_property
    def MLD_winter(self):
        MLD_003 = remesh(self.ocean_month.MLD_003, self.woa_temp).sel(time=self.Averaging_time)
        MLD_003_month = MLD_003.groupby('time.month').mean('time')
        return MLD_003_month.max('month')
    
    @netcdf_property
    def ssh_std(self):
        ssh = self.ocean_daily.zos.sel(time=self.Averaging_time)
        return remesh(ssh.std('time'), self.woa_temp)
    
    @cached_property
    def geoU(self):
        '''
        u = - g / f * d ssh/dy
        '''
        Omega = 7.2921e-5
        g = 9.8
        deg_to_rad = np.pi / 180 # degrees to radians factor
        fq = 2 * Omega * np.sin(self.param.yq * deg_to_rad)
        grid = create_grid_global(self.param)

        hy = grid.diff(self.ocean_daily.zos, 'Y') / self.param.dyCv
        
        u = grid.interp(- g / fq * hy, 'Y')
        u = xr.where(np.abs(u.yh)<10, np.nan, u)

        return u.chunk({'yh':-1,'xh':-1,'time':1})

    @cached_property
    def geoV(self):
        '''
        v = + g / f * d ssh/dx
        '''
        Omega = 7.2921e-5
        g = 9.8
        deg_to_rad = np.pi / 180 # degrees to radians factor
        fh = 2 * Omega * np.sin(self.param.yh * deg_to_rad)
        grid = create_grid_global(self.param)

        hx = grid.diff(self.ocean_daily.zos, 'X') / self.param.dxCu
        
        v = grid.interp(+ g / fh * hx, 'X')
        v = xr.where(np.abs(v.yh)<10, np.nan, v)

        return v.chunk({'yh':-1,'xh':-1,'time':1})
