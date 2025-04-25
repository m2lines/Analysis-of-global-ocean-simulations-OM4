import xarray as xr

for zl in range(15):
    file_zl = f'/scratch/pp2681/mom6/Neverworld2/simulations/R32/filter_scale_0.75/centers/time_*_zl_{zl}.nc'
    file_zi = f'/scratch/pp2681/mom6/Neverworld2/simulations/R32/filter_scale_0.75/interfaces/time_*_zl_{zl}.nc'

    ds_zl = xr.open_mfdataset(file_zl, combine='nested', concat_dim=['time']).sortby('time')
    ds_zi = xr.open_mfdataset(file_zi, combine='nested', concat_dim=['time']).sortby('time')

    ds_zl.to_netcdf(f'/scratch/pp2681/mom6/Neverworld2/simulations/R32/filter_scale_0.75/centers_{zl}.nc')
    ds_zi.to_netcdf(f'/scratch/pp2681/mom6/Neverworld2/simulations/R32/filter_scale_0.75/interfaces_{zl}.nc')