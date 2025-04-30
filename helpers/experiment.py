import helpers.netcdf_cache
import xarray as xr
import os
import numpy as np
import xrft
from functools import cached_property, cache
import helpers
from helpers.computational_tools import *
from helpers.netcdf_cache import netcdf_property
import math
import xesmf as xe
from xoverturning.compfunc import select_basins
from xoverturning import calcmoc
import gsw
import glob
from cmip_basins import generate_basin_codes

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
    def __init__(self, folder, key='', Averaging_time=slice('1979','1981')):
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
        self.Averaging_time=Averaging_time
        self.data_folder = os.path.join(os.path.dirname(helpers.netcdf_cache.__file__), '../data')  # Where observational data is stored

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

    ################### Getters for netcdf files as xarrays #####################
    @cached_property
    def series(self):
        result = xr.open_dataset(os.path.join(self.folder, 'ocean.stats.nc'), decode_times=False)
        return result

    @cached_property
    def param(self):
        result = sort_longitude(xr.open_dataset(f'{self.data_folder}/ocean_static.nc')).drop_vars('time')
        rename_coordinates(result)
        return result

    @cached_property
    def ocean_daily(self):
        '''
        This return the last year of simulation for fast checks of the simulation
        '''
        year = self.Averaging_time.stop
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, f'{year}*ocean_daily*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_daily_long(self):
        '''
        This returns as much data as we have in Averaging_time
        '''
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, f'*ocean_daily*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_month(self):
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, '*ocean_month_0*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_month_z(self):
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, '*ocean_month_z*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1})).rename({'z_l': 'zl', 'z_i': 'zi'})
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_annual_z(self):
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, '*ocean_annual_z*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1})).rename({'z_l': 'zl', 'z_i': 'zi'})
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean_annual_rho2(self):
        try:
            result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, 'diagnostics/*ocean_annual_rho2*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
        except:
            result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, '*ocean_annual_rho2*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
        rename_coordinates(result)
        return result
    
    @cached_property
    def budget(self):
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, 'diagnostics/*budget*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
        rename_coordinates(result)
        return result
    
    @cached_property
    def ocean3d(self):
        result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, 'diagnostics/*ocean_3d*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
        rename_coordinates(result)
        return result
    
    @cached_property
    def ice(self):
        try:
            result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, 'diagnostics/*ice_month*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
        except:
            result = sort_longitude(xr.open_mfdataset(os.path.join(self.folder, '*ice_month*.nc'), parallel=True, combine='nested', compat='no_conflicts', concat_dim='time', chunks={'time':1}))
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
    
    @property
    def dz(self):
        '''
        Data by Ray computed by differencing zi
        '''
        return xr.DataArray([  5. ,  10. ,  10. ,  15. ,  22.5,  25. ,  25. ,  25. ,  37.5,
                              50. ,  50. ,  75. , 100. , 100. , 100. , 100. , 100. , 100. ,
                             100. , 100. , 100. , 100. , 100. , 175. , 250. , 375. , 500. ,
                             500. , 500. , 500. , 500. , 500. , 500. , 500. , 500. ], dims='zl')

    @property
    def zi(self):
        '''
        Data from ocean_annual_z
        '''
        data = [0.000e+00, 5.000e+00, 1.500e+01, 2.500e+01, 4.000e+01, 6.250e+01,
                             8.750e+01, 1.125e+02, 1.375e+02, 1.750e+02, 2.250e+02, 2.750e+02,
                             3.500e+02, 4.500e+02, 5.500e+02, 6.500e+02, 7.500e+02, 8.500e+02,
                             9.500e+02, 1.050e+03, 1.150e+03, 1.250e+03, 1.350e+03, 1.450e+03,
                             1.625e+03, 1.875e+03, 2.250e+03, 2.750e+03, 3.250e+03, 3.750e+03,
                             4.250e+03, 4.750e+03, 5.250e+03, 5.750e+03, 6.250e+03, 6.750e+03]
        return xr.DataArray(data, dims='zi', coords={'zi': data})

    @property
    def zl(self):
        '''
        Data from ocean_annual_z
        '''
        data = [2.5000e+00, 1.0000e+01, 2.0000e+01, 3.2500e+01, 5.1250e+01, 7.5000e+01,
                             1.0000e+02, 1.2500e+02, 1.5625e+02, 2.0000e+02, 2.5000e+02, 3.1250e+02,
                             4.0000e+02, 5.0000e+02, 6.0000e+02, 7.0000e+02, 8.0000e+02, 9.0000e+02,
                             1.0000e+03, 1.1000e+03, 1.2000e+03, 1.3000e+03, 1.4000e+03, 1.5375e+03,
                             1.7500e+03, 2.0625e+03, 2.5000e+03, 3.0000e+03, 3.5000e+03, 4.0000e+03,
                             4.5000e+03, 5.0000e+03, 5.5000e+03, 6.0000e+03, 6.5000e+03]
        
        return xr.DataArray(data, dims='zl', coords={'zl': data})

    ########################  Statistical tools  #########################

    #-------------------  Mean flow and variability  --------------------#
    @cached_property
    def woa_temp(self):
        '''
        WOA temperature data on its native horizontal 1x1 grid
        and vertical grid of MOM6 output
        '''
        woa = sort_longitude(xr.open_dataset(f'{self.data_folder}/woa18_decav81B0_t00_01.nc', decode_times=False).rename({'lat':'yh', 'lon': 'xh'}).t_an.chunk({}))
        woa_interp = woa.interp(depth=self.zl)
        woa_interp[{'zl':0}] = woa[{'depth':0}]
        return woa_interp.squeeze().drop_vars(['time', 'depth']).compute()
    
    @cached_property
    def woa_salt(self):
        '''
        WOA salinity data on its native horizontal 1x1 grid
        and vertical grid of MOM6 output
        '''
        woa = sort_longitude(xr.open_dataset(f'{self.data_folder}/woa18_decav81B0_s00_01.nc', decode_times=False).rename({'lat':'yh', 'lon': 'xh'}).s_an.chunk({}))
        woa_interp = woa.interp(depth=self.zl)
        woa_interp[{'zl':0}] = woa[{'depth':0}]
        return woa_interp.squeeze().drop_vars(['time', 'depth']).compute()
    
    @cached_property
    def woa_sigma0(self):
        return gsw.sigma0(self.woa_salt, self.woa_temp).compute()
    
    @cached_property
    def woa_sigma2(self):
        return gsw.sigma2(self.woa_salt, self.woa_temp).compute()
    
    @cached_property
    def MLD_summer_obs(self):
        obs = np.load(f'{self.data_folder}/mod_r6_cycle1_MLE1_zgrid_MLD_003_min.npy', allow_pickle='TRUE').item()
        obs = sort_longitude(xr.DataArray(obs['obs'], dims=['xh', 'yh'], coords={'xh':obs['lon'], 'yh':obs['lat']})).T
        return obs
    
    @cached_property
    def MLD_winter_obs(self):
        obs = np.load(f'{self.data_folder}/mod_r6_cycle1_MLE1_zgrid_MLD_003_max.npy', allow_pickle='TRUE').item()
        obs = sort_longitude(xr.DataArray(obs['obs'], dims=['xh', 'yh'], coords={'xh':obs['lon'], 'yh':obs['lat']})).T
        return obs
    
    @cached_property
    def ssh_std_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        obs = sort_longitude(xr.open_dataset(f'{self.data_folder}/ssh_std_obs.nc', decode_times=False).rename({'lat':'yh', 'lon': 'xh'}).adt.chunk({}))
        return obs
    
    @cached_property
    def ssh_mean_obs(self):
        '''
        Copernicus data 1993-2012
        '''
        obs = sort_longitude(xr.open_dataset(f'{self.data_folder}/ssh_mean.nc', decode_times=False).adt.chunk({}))
        return obs
    
    @cached_property
    def ssh_mean_glorys(self):
        '''
        GLORYS12V1 Reanalysis
        1993-2016
        '''
        obs = xr.open_dataset(f'{self.data_folder}/ssh_mean_glorys.nc', decode_times=False).zos.chunk({})
        return obs
    
    @cached_property
    def ssh_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return sort_longitude(xr.open_dataset('/scratch/pp2681/altimetry_Copernicus.nc', chunks={'time':1}).rename(
            {'longitude': 'xh', 'latitude': 'yh'}).adt.sel(time=slice('1993','1995')))
    
    @cached_property
    def geoKE_Gulf_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoKE_Gulf.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoKE_Kuroshio_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoKE_Kuroshio.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoKE_Aghulas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoKE_Aghulas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoKE_Malvinas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoKE_Malvinas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_Gulf_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoEKE_Gulf.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_Kuroshio_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoEKE_Kuroshio.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_Aghulas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoEKE_Aghulas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_Malvinas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoEKE_Malvinas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_Gulf_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoMKE_Gulf.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_Kuroshio_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoMKE_Kuroshio.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_Aghulas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoMKE_Aghulas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_Malvinas_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoMKE_Malvinas.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoKE_map_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoKE_map.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoEKE_map_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoEKE_map.nc').__xarray_dataarray_variable__
    
    @cached_property
    def geoMKE_map_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoMKE_map.nc').__xarray_dataarray_variable__
    
    @cached_property
    def eddy_scale_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/eddy_scale.nc').__xarray_dataarray_variable__
    
    @property
    def rossby_radius_lat(self):
        '''
        CM2.6 data
        '''
        radius = xr.open_dataset('/vast/pp2681/CM26_datasets/ocean3d/subfilter/FGR3/factor-4/train-0.nc').deformation_radius / 1000. # in km
        wet = xr.open_dataset('/vast/pp2681/CM26_datasets/ocean3d/subfilter/FGR3/factor-4/param.nc').wet

        return radius.sum('xh') / wet.isel(zl=0).sum('xh')
    
    @cached_property
    def geovel_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geovel.nc', chunks={'time':1}).__xarray_dataarray_variable__
    
    @cached_property
    def geoRV_obs(self):
        '''
        Copernicus data 1993-1995
        '''
        return xr.open_dataset(f'{self.data_folder}/geoRV.nc').__xarray_dataarray_variable__
    
    @cached_property
    def BT_fraction_obs(self):
        '''
        CM2.6 monthly data coarsened to 0.3^o grid
        Averaged over the last year
        '''
        return self.regrid(sort_longitude(xr.open_dataset(f'{self.data_folder}/BT_fraction_30days_coarsen.nc').__xarray_dataarray_variable__.rename({'xu_ocean':'xh', 'yu_ocean':'yh'})), self.BT_fraction)
    
    @cached_property
    def KE_obs(self):
        '''
        CM2.6 monthly data coarsened to 0.3^o grid
        Averaged over the last year
        Depth-averaged kinetic energy
        '''
        return sort_longitude(xr.open_dataset(f'{self.data_folder}/KE.nc').__xarray_dataarray_variable__.rename({'xu_ocean':'xh', 'yu_ocean':'yh'}))
    
    @cached_property
    def KE_BT_obs(self):
        '''
        CM2.6 monthly data coarsened to 0.3^o grid
        Averaged over the last year
        Barotropic kinetic energy
        '''
        return sort_longitude(xr.open_dataset(f'{self.data_folder}/KE_BT.nc').__xarray_dataarray_variable__.rename({'xu_ocean':'xh', 'yu_ocean':'yh'}))
    
    def regrid(self, input_data, target):
        '''
        Target should be on regular lon-lat grid
        '''
        xin = x_coord(input_data)
        yin = y_coord(input_data)
        xout = x_coord(target)
        yout = y_coord(target)
        if xin.name == 'xh' and yin.name == 'yh':
            lon = self.param.geolon
            lat = self.param.geolat
        elif xin.name == 'xq' and yin.name == 'yh':
            lon = self.param.geolon_u
            lat = self.param.geolat_u
        elif xin.name == 'xh' and yin.name == 'yq':
            lon = self.param.geolon_v
            lat = self.param.geolat_v
        elif xin.name == 'xq' and yin.name == 'yq':
            lon = self.param.geolon_c
            lat = self.param.geolat_c
        else:
            print('Wrong combination of coordinates for interpolation')

        coords = xr.Dataset()
        coords['lon'] = lon
        coords['lat'] = lat
        target_rename = target.rename({xout.name: 'lon', yout.name: 'lat'})
        regridder = xe.Regridder(coords, target.rename({xout.name: 'lon', yout.name: 'lat'}), "nearest_s2d", ignore_degenerate=True, periodic=True, unmapped_to_nan=True)

        out = regridder(input_data.rename({xin.name:'lon', yin.name:'lat'})).rename({'lon':xout.name, 'lat': yout.name})
        if 'time' in input_data.dims:
            out = out.assign_coords({'time': input_data.time})
        return out
    
    @cached_property
    def depth(self):
        depth = self.param.deptho
        return self.regrid(depth, self.woa_temp) # regrid on regular mesh

    @netcdf_property
    def thetao(self):
        try:
            input_data = self.ocean_annual_z.thetao
        except:
            input_data = self.ocean_month_z.thetao#.sel(time=self.Averaging_time).mean('time')

        out = self.regrid(input_data, self.woa_temp)
        return xr.where(np.isnan(self.woa_temp), np.nan, out)
    
    @netcdf_property
    def salto(self):
        try:
            input_data = self.ocean_annual_z.so
        except:
            input_data = self.ocean_month_z.so#.sel(time=self.Averaging_time).mean('time')

        out = self.regrid(input_data, self.woa_temp)
        return xr.where(np.isnan(self.woa_temp), np.nan, out)
    
    @netcdf_property
    def sigma0(self):
        return gsw.sigma0(self.salto, self.thetao)
    
    @netcdf_property
    def sigma2(self):
        return gsw.sigma2(self.salto, self.thetao)
    
    @netcdf_property
    def N_buoyancy_frequency(self):
        rho = self.sigma2 + 1000.
        rho_zi = rho.interp(zl=self.zi).drop_vars('zi')
        rho_zi[{'zi':0}] = rho[{'zl':0}]
        drho_dz = (rho_zi.isel(zi=slice(0,-1)) - rho_zi.isel(zi=slice(1,None))).rename({'zi': 'zl'}) / self.dz
        g = 9.8
        N = np.sqrt(np.maximum(-g * drho_dz / rho,0))
        N['zl'] = self.zl
        return N
    
    @cached_property
    def N_buoyancy_frequency_woa(self):
        rho = self.woa_sigma2 + 1000.
        rho_zi = rho.interp(zl=self.zi).drop_vars('zi')
        rho_zi[{'zi':0}] = rho[{'zl':0}]
        drho_dz = (rho_zi.isel(zi=slice(0,-1)) - rho_zi.isel(zi=slice(1,None))).rename({'zi': 'zl'}) / self.dz
        g = 9.8
        N = np.sqrt(np.maximum(-g * drho_dz / rho,0))
        N['zl'] = self.zl
        return N.compute()
    
    @netcdf_property
    def ubar(self):
        u = self.ocean_month_z.uo
        dt = self.ocean_month_z.average_DT.astype('float64')
        dz = self.dz

        mask = xr.where(np.isnan(u.isel(time=-1)), 0., 1.).compute().chunk({})

        # Depth-average; Nans are skipped by xarray, and so mask
        u_z = (u * dz).sum('zl') / (mask * dz).sum('zl')

        # Time-averaging
        u_zt = (u_z * dt).sel(time=self.Averaging_time).sum('time') / (dt).sel(time=self.Averaging_time).sum('time')

        # restore nans
        u_zt = xr.where(self.param.wet_u==1., u_zt, np.nan)

        return u_zt
    
    @netcdf_property
    def KE(self):
        grid = create_grid_global(self.param)

        u = grid.interp(self.ocean_month_z.uo.sel(time=self.Averaging_time), 'X')
        v = grid.interp(self.ocean_month_z.vo.sel(time=self.Averaging_time), 'Y')
        dz = self.dz
        masku = xr.where(np.isnan(u.isel(time=-1)), np.nan, 1.)
        maskv = xr.where(np.isnan(v.isel(time=-1)), np.nan, 1.)
        mask = masku * maskv
        mask_2d = mask.isel(zl=0)

        def ave_z(x):
            return (((mask * x * dz).sum('zl') / (mask * dz).sum('zl'))) * mask_2d
        
        KE = ave_z(0.5*(u**2 + v**2))

        return sort_longitude(KE.mean('time'))
    
    @netcdf_property
    def KE_BT(self):
        grid = create_grid_global(self.param)

        u = grid.interp(self.ocean_month_z.uo.sel(time=self.Averaging_time), 'X')
        v = grid.interp(self.ocean_month_z.vo.sel(time=self.Averaging_time), 'Y')
        dz = self.dz
        masku = xr.where(np.isnan(u.isel(time=-1)), np.nan, 1.)
        maskv = xr.where(np.isnan(v.isel(time=-1)), np.nan, 1.)
        mask = masku * maskv
        mask_2d = mask.isel(zl=0)

        def ave_z(x):
            return (((mask * x * dz).sum('zl') / (mask * dz).sum('zl'))) * mask_2d
        
        KE_BT = 0.5 * (ave_z(u)**2 + ave_z(v)**2)
        
        return sort_longitude(KE_BT.mean('time'))

    @netcdf_property
    def BT_fraction(self):
        grid = create_grid_global(self.param)

        u = grid.interp(self.ocean_month_z.uo.sel(time=self.Averaging_time), 'X')
        v = grid.interp(self.ocean_month_z.vo.sel(time=self.Averaging_time), 'Y')
        dz = self.dz
        masku = xr.where(np.isnan(u.isel(time=-1)), np.nan, 1.)
        maskv = xr.where(np.isnan(v.isel(time=-1)), np.nan, 1.)
        mask = masku * maskv
        mask_2d = mask.isel(zl=0)

        def ave_z(x):
            return (((mask * x * dz).sum('zl') / (mask * dz).sum('zl'))) * mask_2d
        
        KE = ave_z(0.5*(u**2 + v**2))
        KE_BT = 0.5 * (ave_z(u)**2 + ave_z(v)**2)

        BT_fraction = KE_BT / KE

        return sort_longitude(BT_fraction.mean('time'))
    
    @netcdf_property
    def MLD_summer(self):
        return self.regrid(self.ocean_month.MLD_003.sel(time=self.Averaging_time).groupby('time.month').mean('time').min('month'), 
                            self.MLD_summer_obs)
    
    @netcdf_property
    def MLD_winter(self):
        return self.regrid(self.ocean_month.MLD_003.sel(time=self.Averaging_time).groupby('time.month').mean('time').max('month'),
                            self.MLD_winter_obs)
    
    @netcdf_property
    def ssh_std(self):
        ssh = self.ocean_daily_long.zos.sel(time=self.Averaging_time)
        return self.regrid(ssh.std('time'), self.woa_temp).chunk({'yh':10})
    
    @netcdf_property
    def ssh_mean(self):
        try:
            ssh = self.ocean_month.zos.sel(time=self.Averaging_time)
            return self.regrid(ssh.mean('time'), self.woa_temp)
        except:
            ssh = self.ocean_daily.zos.sel(time=self.Averaging_time)
            return self.regrid(ssh.mean('time'), self.woa_temp)
    
    @netcdf_property
    def u_mean(self):
        try:
            return self.ocean_annual_z.uo.sel(time=self.Averaging_time).mean('time')
        except:
            return self.ocean_month_z.uo.sel(time=self.Averaging_time).mean('time')
    
    @netcdf_property
    def v_mean(self):
        try:
            return self.ocean_annual_z.vo.sel(time=self.Averaging_time).mean('time')
        except:
            return self.ocean_month_z.vo.sel(time=self.Averaging_time).mean('time')
    
    @netcdf_property
    def uabs(self):
        grid = create_grid_global(self.param)
        velocity = np.sqrt(grid.interp(self.u_mean, 'X')**2 + grid.interp(self.v_mean, 'Y')**2)
        return self.regrid(velocity, velocity)
    
    @netcdf_property
    def barotropic_streamfunction(self):
        '''
        We use algorithm of CDFtools, and integrate zonal transport
        from the South Pole in meridional direction through land
        We set streamfunction to zero in America land (for convenience of plotting)
        '''
        umo = self.ocean_annual_z.umo.sel(time=self.Averaging_time).mean('time')
        # 1e+9 is conversion of kg to m^3 (1e+3) and to Sverdrups (another 1e+6)
        psi_global = - (umo.sum('zl').cumsum('yh')).compute() / 1e+9
        # Streamfunction becomes to be deined in a corner after integration
        psi_global = psi_global.rename({'yh': 'yq'})
        psi_global['yq'] = self.param.yq

        # Here we set streamfunction to be zero at the America land
        #psi_global = psi_global - psi_global.sel(xq=-80,yq=40, method='nearest')
        # This interpolation intends to regrid data on regular grid
        return self.regrid(psi_global, psi_global)
    
    @property
    def AMOC_mask(self):
        '''
        This is mask of Atlantic and Arctic oceans as they are defined in xoverturning
        '''
        param = self.param
        param['zi'] = self.ocean_annual_z['zi']
        param['zl'] = self.ocean_annual_z['zl']
        names = dict(
        x_center="xh",
        y_center="yh",
        x_corner="xq",
        y_corner="yq",
        lon_t="geolon",
        lat_t="geolat",
        mask_t="wet",
        lon_v="geolon_v",
        lat_v="geolat_v",
        mask_v="wet_v",
        bathy="deptho",
        interface="zi",
        layer="zl"
        )
        out = select_basins(param[['geolon_v', 'geolat_v', 'wet_v']], names, basin="atl-arc", lon="geolon_v", lat="geolat_v", mask="wet_v", vertical="zl", verbose=True)
        return xr.where(out[0], 1., 0.)
    
    @property
    def AMOC_mask_center(self):
        '''
        This is mask of Atlantic and Arctic oceans as they are defined in xoverturning
        '''
        param = self.param
        param['zi'] = self.ocean_annual_z['zi']
        param['zl'] = self.ocean_annual_z['zl']
        names = dict(
        x_center="xh",
        y_center="yh",
        x_corner="xq",
        y_corner="yq",
        lon_t="geolon",
        lat_t="geolat",
        mask_t="wet",
        lon_v="geolon_v",
        lat_v="geolat_v",
        mask_v="wet_v",
        bathy="deptho",
        interface="zi",
        layer="zl"
        )
        out = select_basins(param[['geolon', 'geolat', 'wet']], names, basin="atl-arc", lon="geolon", lat="geolat", mask="wet", vertical="zl", verbose=True)
        return xr.where(out[0], 1., 0.)
    
    @property
    def AMOC(self):
        '''
        Here we use xoverturning for brevity
        '''
        dsgrid = self.param
        data = xr.Dataset()
        try:
            data['umo'] = self.ocean_annual_z.umo
            data['z_i'] = self.zi.rename({'zi':'z_i'})
        except:
            data['umo'] = self.ocean_month_z.uo * 1e+3 * dsgrid.dxCu * self.dz
            data['z_i'] = self.zi

        try:
            data['vmo'] = self.ocean_annual_z.vmo
        except:
            data['vmo'] = self.ocean_month_z.vo * 1e+3 * dsgrid.dyCv * self.dz
        
        data['umo'] = data['umo']#.sel(time=self.Averaging_time).mean('time') 
        data['vmo'] = data['vmo']#.sel(time=self.Averaging_time).mean('time')
        data = data.rename({'zl': 'z_l'})

        amoc = calcmoc(data, dsgrid=dsgrid, basin='atl-arc')

        return amoc.compute()
    
    @property
    def AMOC_new(self):
        '''
        Here we compute AMOC ourselves
        '''
        basincodes = generate_basin_codes(self.param, lon='geolon_v', lat='geolat_v', mask='wet_v')
        selected_codes = [2, 4, 6, 7, 8, 9]
        mask = xr.where(basincodes.isin(selected_codes),1.,np.nan)

        vmo = (self.ocean_annual_z.vmo * mask).sum('xh') / 1.035e9

        amoc = -vmo.sel(zl=slice(None,None,-1)).cumsum('zl').isel(zl=slice(None,None,-1))
        
        # Above amoc array has vertical coordinate zl, but not zi. Here we extend it with B.C. at the bottom
        amoc = amoc.pad({'zl':(0,1)}, constant_values=0)
        # Change vertical coordinate
        amoc = amoc.drop_vars('zl').rename({'zl':'zi'})
        amoc['zi'] = self.zi

        return amoc.compute()

    @property
    def AMOC_rho2(self):
        dsgrid = self.param
        data = xr.Dataset()

        for key in ['umo', 'vmo']:
            data[key] = self.ocean_annual_rho2[key]

        for key in ['rho2_l', 'rho2_i']:
            dsgrid[key] = self.ocean_annual_rho2[key]

        return calcmoc(data, dsgrid=dsgrid, basin='atl-arc', vertical='rho2')
    
    @property
    def MOC_rho2(self):
        dsgrid = self.param
        data = xr.Dataset()

        for key in ['umo', 'vmo']:
            data[key] = self.ocean_annual_rho2[key]

        for key in ['rho2_l', 'rho2_i']:
            dsgrid[key] = self.ocean_annual_rho2[key]

        return calcmoc(data, dsgrid=dsgrid, basin='global', vertical='rho2')

    @property
    def MOC(self):
        '''
        Here we use xoverturning for brevity
        '''
        dsgrid = self.param
        data = xr.Dataset()
        try:
            data['umo'] = self.ocean_annual_z.umo
        except:
            data['umo'] = self.ocean_month_z.uo * 1e+3 * dsgrid.dxCu * self.dz

        try:
            data['vmo'] = self.ocean_annual_z.vmo
        except:
            data['vmo'] = self.ocean_month_z.vo * 1e+3 * dsgrid.dyCv * self.dz
        
        data['umo'] = data['umo']#.sel(time=self.Averaging_time).mean('time') 
        data['vmo'] = data['vmo']#.sel(time=self.Averaging_time).mean('time')
        data = data.rename({'zl': 'z_l'})

        data['zi'] = self.zi

        moc = calcmoc(data, dsgrid=dsgrid, basin='global')

        return moc.compute()
    
    @property
    def AMOC_map(self):
        '''
        This function shows AMOC without integration in zonal direction.
        Looking in this this function, we can understand, what currents contribute to
        AMOC
        '''
        mask = self.AMOC_mask
        vmo = self.ocean_annual_z.vmo.sel(time=self.Averaging_time).mean('time') * mask

        # Here summation is remove for special
        # To be able to see spatial map
        zonal_integral = vmo#.sum('xh')

        # Integration from the surface to the bottom
        # with zero surface B.C.
        # Result is in the interface points below surface
        amoc = zonal_integral.cumsum('zl')

        # Place explicitly zero B.C. on the surface
        amoc = amoc.pad({'zl':(1,0)}, constant_values=0)

        # Rename the coordinate to the interface
        amoc = amoc.rename({'zl': 'zi'})
        amoc['zi'] = self.zi

        # Now we want to place zero B.C. on the bottom
        # Add some non-zero B.C. on the surface
        #amoc = amoc - amoc.isel(zi=-1)

        # The conversion by 1e+9 is to convert kg to Sv
        return amoc.compute() / 1e+9
    
    @netcdf_property
    def eddy_scale(self):
        '''
        According to Thompson and Young. Also see Yankovsky2022
        '''
        zos = self.ocean_daily.zos.sel(time=self.Averaging_time)
        # Averaging operator is time mean
        mean = lambda x: x.mean(['time'])

        zos_anomaly = zos - mean(zos)

        zos_anomaly_square = mean((zos_anomaly)**2)

        grid = create_grid_global(self.param)
        zos_anomaly_x = grid.interp(grid.diff(zos_anomaly, 'X') / self.param.dxCu,'X')
        zos_anomaly_y = grid.interp(grid.diff(zos_anomaly, 'Y') / self.param.dyCv,'Y')

        zos_grad_anomaly_square = mean( (zos_anomaly_x)**2 + (zos_anomaly_y)**2 )

        # in km
        return (np.sqrt(zos_anomaly_square / zos_grad_anomaly_square) / 1000.)

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
        #u = xr.where(np.abs(u.yh)<10, np.nan, u)

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
        #v = xr.where(np.abs(v.yh)<10, np.nan, v)

        v['time'] = self.ocean_daily.time.copy()

        return v
    
    @netcdf_property
    def geovel(self):
        '''
        Modulus of geostrophic velocity over the last year
        '''
        return np.sqrt(self.geoU**2 + self.geoV**2).isel(time=slice(None,None,3)).astype('float32')
    
    @netcdf_property
    def geoRV(self):
        '''
        Relative vorticity of geostrophic velocity
        '''
        param = self.param
        grid = create_grid_global(param)

        dyCv = param.dyCv
        dxCu = param.dxCu
        IareaBu = 1. / param.areacello_bu

        u = grid.interp(self.geoU, 'X')
        v = grid.interp(self.geoV, 'Y')

        dvdx = grid.diff(v * dyCv,'X')
        dudy = grid.diff(u * dxCu,'Y')

        RV = (dvdx - dudy) * IareaBu

        RV['time'] = self.ocean_daily.time.copy()

        return RV.isel(time=slice(None,None,3)).astype('float32')

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
