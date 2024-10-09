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

    @classmethod
    def get_list_of_netcdf_properties(cls):
        '''
        Allows to know what properties should be cached
        https://stackoverflow.com/questions/27503965/list-property-decorated-methods-in-a-python-class
        '''
        result = []
        for name, value in vars(cls).items():
            if isinstance(value, netcdf_property):
                result.append(name)
        return result

    @property
    def Averaging_time(self):
        return slice('1979','1981')

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
        Copernicus data 1993-1995
        '''
        obs = sort_longitude(xr.open_dataset('../data/ssh_std_obs.nc', decode_times=False).rename({'lat':'yh', 'lon': 'xh'}).adt.chunk({}))
        return obs
    
    @cached_property
    def geoKE_Gulf_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoKE_Gulf.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoKE_Kuroshio_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoKE_Kuroshio.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoKE_Aghulas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoKE_Aghulas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoKE_Malvinas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoKE_Malvinas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_Gulf_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoEKE_Gulf.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_Kuroshio_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoEKE_Kuroshio.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_Aghulas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoEKE_Aghulas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_Malvinas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoEKE_Malvinas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_Gulf_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoMKE_Gulf.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_Kuroshio_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoMKE_Kuroshio.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_Aghulas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoMKE_Aghulas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_Malvinas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoMKE_Malvinas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoKE_map_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoKE_map.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_map_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoEKE_map.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_map_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset('../data/geoMKE_map.nc').__xarray_dataarray_variable__
    
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

        u['time'] = self.ocean_daily.time.copy()

        return u

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

        v['time'] = self.ocean_daily.time.copy()

        return v
    
    @netcdf_property
    def geoKE_map(self):
        return ((self.geoU**2 + self.geoV**2) / 2).sel(time=self.Averaging_time).mean('time').compute()
    
    @netcdf_property
    def geoMKE_map(self):
        geoU = self.geoU.sel(time=self.Averaging_time).mean('time')
        geoV = self.geoV.sel(time=self.Averaging_time).mean('time')
        return ((geoU**2 + geoV**2) / 2).compute()
    
    @netcdf_property
    def geoEKE_map(self):
        return self.geoKE_map - self.geoMKE_map

    @cached_property
    def RV(self):
        param = self.param
        grid = create_grid_global(param)

        dyCv = param.dyCv
        dxCu = param.dxCu
        IareaBu = 1. / param.areacello_bu

        u = self.ocean_daily.ssu
        v = self.ocean_daily.ssv

        dvdx = grid.diff(v * dyCv,'X')
        dudy = grid.diff(u * dxCu,'Y')

        return (dvdx - dudy) * IareaBu
    
    def geoKE_spectrum(self, zos, Lat=(25,45), Lon=(-60,-40)):
        '''
        We estimate KE spectrum from ssh, i.e.
        it is spectrum of geostrophic motions.

        Given the relation u = g/f nabla^perp ssh,
        The KE spectrum is given by:
        KE = g^2 / f^2 * k^2 * E,
        where E is the power spectrum of SSH
        '''
        E = compute_isotropic_PE(zos.chunk({'xh':-1}), self.param.dxt, self.param.dyt, 
                                 Lat=Lat, Lon=Lon)
        Omega = 7.2921e-5
        g = 9.8
        deg_to_rad = np.pi / 180 # degrees to radians factor
        #  Coriolis parameter in the box averaged
        f = 2 * Omega * np.sin(self.param.yh * deg_to_rad).sel(yh=slice(Lat[0],Lat[1])).mean()

        KE = g**2 / f**2 * E.freq_r**2 * E
        return KE
    
    @netcdf_property
    def geoKE_Gulf(self):
        return self.geoKE_spectrum(self.ocean_daily.zos, Lat=(25,45), Lon=(-60,-40)).sel(time=self.Averaging_time).mean('time').compute()
    
    @netcdf_property
    def geoKE_Kuroshio(self):
        return self.geoKE_spectrum(self.ocean_daily.zos, Lat=(25,45), Lon=(150,170)).sel(time=self.Averaging_time).mean('time').compute()
    
    @netcdf_property
    def geoKE_Aghulas(self):
        return self.geoKE_spectrum(self.ocean_daily.zos, Lat=(-50,-30), Lon=(40,60)).sel(time=self.Averaging_time).mean('time').compute()
    
    @netcdf_property
    def geoKE_Malvinas(self):
        return self.geoKE_spectrum(self.ocean_daily.zos, Lat=(-51,-31), Lon=(-49,-29)).sel(time=self.Averaging_time).mean('time').compute()
    
    @netcdf_property
    def geoMKE_Gulf(self):
        return self.geoKE_spectrum(self.ocean_daily.zos.sel(time=self.Averaging_time).mean('time'), Lat=(25,45), Lon=(-60,-40)).compute()
    
    @netcdf_property
    def geoMKE_Kuroshio(self):
        return self.geoKE_spectrum(self.ocean_daily.zos.sel(time=self.Averaging_time).mean('time'), Lat=(25,45), Lon=(150,170)).compute()
    
    @netcdf_property
    def geoMKE_Aghulas(self):
        return self.geoKE_spectrum(self.ocean_daily.zos.sel(time=self.Averaging_time).mean('time'), Lat=(-50,-30), Lon=(40,60)).compute()
    
    @netcdf_property
    def geoMKE_Malvinas(self):
        return self.geoKE_spectrum(self.ocean_daily.zos.sel(time=self.Averaging_time).mean('time'), Lat=(-51,-31), Lon=(-49,-29)).compute()
    
    @netcdf_property
    def geoEKE_Gulf(self):
        return self.geoKE_Gulf - self.geoMKE_Gulf
    
    @netcdf_property
    def geoEKE_Kuroshio(self):
        return self.geoKE_Kuroshio - self.geoMKE_Kuroshio
    
    @netcdf_property
    def geoEKE_Aghulas(self):
        return self.geoKE_Aghulas - self.geoMKE_Aghulas
    
    @netcdf_property
    def geoEKE_Malvinas(self):
        return self.geoKE_Malvinas - self.geoMKE_Malvinas