import numpy as np
import math
import xrft
import numpy.fft as npfft
from scipy import signal
import xarray as xr
import os
import gcm_filters
import xgcm
import cmocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def x_coord(array):
    '''
    Returns horizontal coordinate, 'xq', 'xh' or 'lon'
    as xarray
    '''
    for name in ['xh', 'xq', 'lon']:
        if name in array.dims:
            return array[name]
            
def x_coord_iterator(array):
    '''
    Returns horizontal coordinate, 'xq', 'xh' or 'lon'
    as xarray
    '''
    for name in ['xh', 'xq', 'lon']:
        if name in array.dims:
            yield array[name]
            
def y_coord(array):
    '''
    Returns horizontal coordinate, 'yq' or 'yh'
    as xarray
    '''
    for name in ['yh', 'yq', 'lon']:
        if name in array.dims:
            return array[name]

def sort_longitude(x, lon_min=-180.):
    if lon_min is None:
        return x
    lon_max=lon_min + 360.
    for lon in x_coord_iterator(x):
        if lon.min() < lon_min:
            lon = xr.where(lon<lon_min, lon+360, lon)
            lon = xr.where(lon>lon_max, lon-360, lon)
        else:
            lon = xr.where(lon>lon_max, lon-360, lon)
            lon = xr.where(lon<lon_min, lon+360, lon)   
        x[lon.name] = lon.values
        x = x.sortby(lon.name)
    return x

def rename_coordinates(xr_dataset):
    '''
    in-place change of coordinate names to Longitude and Latitude.
    For convenience of plotting with xarray.plot()
    '''
    for key in ['xq', 'xh']:
        try:
            xr_dataset[key].attrs['long_name'] = 'Longitude'
            xr_dataset[key].attrs['units'] = ''
        except:
            pass

    for key in ['yq', 'yh']:
        try:
            xr_dataset[key].attrs['long_name'] = 'Latitude'
            xr_dataset[key].attrs['units'] = ''
        except:
            pass

def select_LatLon(array, Lat=(35,45), Lon=(5,15)):
    '''
    array is xarray
    Lat, Lon = tuples of floats
    '''
    x = x_coord(array)
    y = y_coord(array)

    return array.sel({x.name: slice(Lon[0],Lon[1]), 
                      y.name: slice(Lat[0],Lat[1])})

def select_NA(array):
    return select_LatLon(array, Lat=(20, 60), Lon=(260-360,330-360))

def select_NA_large(array):
    return select_LatLon(array, Lat=(20, 70), Lon=(-80,20))

def select_Pacific(array):
    return select_LatLon(array, Lat=(10, 65), Lon=(-250+360,-130+360))

def select_Cem(array):
    return select_LatLon(array, Lat=(-10,15), Lon=(-260+360,-230+360))

def select_globe(array):
    return select_LatLon(array, Lat=(None,None), Lon=(None,None))

def select_Equator(array):
    return select_LatLon(array, Lat=(-20,20), Lon=(0,360))

# Juricke regions below:

def select_Gulf(array):
    return select_LatLon(array, Lat=(30, 60), Lon=(-80,-20))

def select_Kuroshio(array):
    return select_LatLon(array, Lat=(20, 50), Lon=(120,180))

def select_SO(array):
    return select_LatLon(array, Lat=(-70,-30), Lon=(0,360))

def select_Aghulas(array):
    return select_LatLon(array, Lat=(-60,-30), Lon=(0,60))

def select_Malvinas(array):
    return select_LatLon(array, Lat=(-60,-30), Lon=(-60,0))

# Sections
def select_Drake(array):
    return select_LatLon(array, Lat=(-70,-55), Lon=(-70,-69)).squeeze()

def select_Atlantic_transect(array):
    return select_LatLon(array, Lat=(-80,90), Lon=(-30,-29)).squeeze()

def select_Pacific_transect(array):
    return select_LatLon(array, Lat=(-80,90), Lon=(-130,-129)).squeeze()

def select_Indian_transect(array):
    return select_LatLon(array, Lat=(-80,30), Lon=(80,81)).squeeze()

def remesh(input, target, fillna=False):
    '''
    Input and target should be xarrays of any type (u-array, v-array, q-array, h-array).
    Datasets are prohibited.
    Horizontal mesh of input changes according to horizontal mesh of target.
    Other dimensions are unchanged!

    If type of arrays is different:
        - Interpolation to correct points occurs
    If input is Hi-res:
        - Coarsening with integer grain and subsequent interpolation to correct mesh if needed
    if input is Lo-res:
        - Interpolation to Hi-res mesh occurs

    Input and output Nan values are treates as zeros (see "fillna")
    '''

    # Define coordinates
    x_input  = x_coord(input)
    y_input  = y_coord(input)
    x_target = x_coord(target)
    y_target = y_coord(target)

    # ratio of mesh steps
    ratiox = np.diff(x_target).max() / np.diff(x_input).max()
    ratiox = round(ratiox)

    ratioy = np.diff(y_target).max() / np.diff(y_input).max()
    ratioy = round(ratioy)
    
    # B.C.
    if fillna:
        result = input.fillna(0)
    else:
        result = input
    
    if (ratiox > 1 or ratioy > 1):
        # Coarsening; x_input.name returns 'xq' or 'xh'
        result = result.coarsen({x_input.name: ratiox, y_input.name: ratioy}, boundary='pad').mean()

    # Coordinate points could change after coarsegraining
    x_result = x_coord(result)
    y_result = y_coord(result)

    # Interpolate if needed
    if not x_result.equals(x_target) or not y_result.equals(y_target):
        result = result.interp({x_result.name: x_target, y_result.name: y_target})
        if fillna:
            result = result.fillna(0)

    # Remove unnecessary coordinates
    if x_target.name != x_input.name:
        result = result.drop_vars(x_input.name)
    if y_target.name != y_input.name:
        result = result.drop_vars(y_input.name)
    
    return result

def create_grid_global(param):
    grid = xgcm.Grid(param, coords={
        'X': {'center': 'xh', 'right': 'xq'},
        'Y': {'center': 'yh', 'right': 'yq'}
    },
    boundary={"X": 'periodic', 'Y': 'fill'},
    fill_value = {'Y':0})

    return grid

def compute_isotropic_KE(u_in, v_in, dx, dy, Lat=(35,45), Lon=(5,15), window='hann', 
        nfactor=2, truncate=True, detrend='linear', window_correction=True, nd_wavenumber=False):
    '''
    u, v - "velocity" arrays defined on corresponding staggered grids
    dx, dy - grid step arrays defined in the center of the cells
    Default options: window correction + linear detrending
    Output:
    mean(u^2+v^2)/2 = int(E(k),dk)
    This equality is expected for detrend=None, window='boxcar'
    freq_r - radial wavenumber, m^-1
    window = 'boxcar' or 'hann'
    '''
    # Interpolate to the center of the cells
    u = remesh(u_in, dx)
    v = remesh(v_in, dy)

    # Select desired Lon-Lat square
    u = select_LatLon(u,Lat,Lon)
    v = select_LatLon(v,Lat,Lon)

    # mean grid spacing in metres
    dx = select_LatLon(dx,Lat,Lon).mean().values
    dy = select_LatLon(dy,Lat,Lon).mean().values

    # define uniform grid
    x = dx*np.arange(len(u.xh))
    y = dy*np.arange(len(u.yh))
    u['xh'] = x
    u['yh'] = y
    v['xh'] = x
    v['yh'] = y

    Eu = xrft.isotropic_power_spectrum(u, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)
    Ev = xrft.isotropic_power_spectrum(v, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = (Eu+Ev) / 2 # because power spectrum is twice the energy
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers

    if nd_wavenumber:
        Lx = x.max() - x.min()
        Ly = y.max() - y.min()
        kmin = 2*np.pi * min(1/Lx, 1/Ly)
        E['freq_r'] = E['freq_r'] / kmin
        E = E * kmin
    
    ############## normalization tester #############
    #print('Energy balance:')
    #print('mean(u^2+v^2)/2=', ((u**2+v**2)/2).mean(dim=('Time', 'xh', 'yh')).values)
    #spacing = np.diff(E.freq_r).mean()
    #print('int(E(k),dk)=', (E.sum(dim='freq_r').mean(dim='Time') * spacing).values)
    #print(f'Max wavenumber={E.freq_r.max().values} [1/m], \n x-grid-scale={np.pi/dx} [1/m], \n y-grid-scale={np.pi/dy} [1/m]')
    
    return E

def compute_isotropic_cospectrum(u_in, v_in, fu_in, fv_in, dx, dy, Lat=(35,45), Lon=(5,15), window='hann', 
        nfactor=2, truncate=False, detrend='linear', window_correction=True, compensated=False):
    # Interpolate to the center of the cells
    u = remesh(u_in, dx)
    v = remesh(v_in, dy)
    fu = remesh(fu_in, dx).transpose(*u.dims)
    fv = remesh(fv_in, dy).transpose(*v.dims)

    # Select desired Lon-Lat square
    u = select_LatLon(u,Lat,Lon)
    v = select_LatLon(v,Lat,Lon)
    fu = select_LatLon(fu,Lat,Lon)
    fv = select_LatLon(fv,Lat,Lon)

    # mean grid spacing in metres
    dx = select_LatLon(dx,Lat,Lon).mean().values
    dy = select_LatLon(dy,Lat,Lon).mean().values

    # define uniform grid
    x = dx*np.arange(len(u.xh))
    y = dy*np.arange(len(u.yh))
    for variable in [u, v, fu, fv]:
        variable['xh'] = x
        variable['yh'] = y

    Eu = xrft.isotropic_cross_spectrum(u, fu, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)
    Ev = xrft.isotropic_cross_spectrum(v, fv, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = (Eu+Ev)
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers
    
    if compensated:
        return np.real(E) * E['freq_r']
    else:
        return np.real(E)

def compute_isotropic_PE(h_int, dx, dy, Lat=(35,45), Lon=(5,15), window='hann', 
        nfactor=2, truncate=True, detrend='linear', window_correction=True):
    '''
    hint - interface displacement in metres
    dx, dy - grid step arrays defined in the center of the cells
    Default options: window correction + linear detrending
    Output:
    mean(h^2)/2 = int(E(k),dk)
    This equality is expected for detrend=None, window='boxcar'
    freq_r - radial wavenumber, m^-1
    window = 'boxcar' or 'hann'
    '''
    # Select desired Lon-Lat square
    hint = select_LatLon(h_int,Lat,Lon)

    # mean grid spacing in metres
    dx = select_LatLon(dx,Lat,Lon).mean().values
    dy = select_LatLon(dy,Lat,Lon).mean().values

    # define uniform grid
    x = dx*np.arange(len(hint.xh))
    y = dy*np.arange(len(hint.yh))
    hint['xh'] = x
    hint['yh'] = y

    E = xrft.isotropic_power_spectrum(hint, dim=('xh','yh'), window=window, nfactor=nfactor, 
        truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = E / 2 # because power spectrum is twice the energy
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers
    
    return E

def compute_KE_time_spectrum(u_in, v_in, Lat=(35,45), Lon=(5,15), Time=slice(0,None), window='hann', 
        nchunks=2, detrend='linear', window_correction=True):
    '''
    Returns KE spectrum with normalization:
    mean(u^2+v^2)/2 = int(E(nu),dnu),
    where nu - time frequency in 1/day (not "angle frequency")
    E(nu) - energy density, i.e. m^2/s^2 * day
    '''

    # Select range of Lat-Lon-time
    u = select_LatLon(u_in,Lat,Lon).sel(Time=Time)
    v = select_LatLon(v_in,Lat,Lon).sel(Time=Time)

    # Let integer division by nchunks
    nTime = len(u.Time)
    chunk_length = math.floor(nTime / nchunks)
    nTime = chunk_length * nchunks

    # Divide time series to time chunks
    u = u.isel(Time=slice(nTime)).chunk({'Time': chunk_length})
    v = v.isel(Time=slice(nTime)).chunk({'Time': chunk_length})

    # compute spatial-average time spectrum
    ps_u = xrft.power_spectrum(u, dim='Time', window=window, window_correction=window_correction, 
        detrend=detrend, chunks_to_segments=True).mean(dim=('xq','yh'))
    ps_v = xrft.power_spectrum(v, dim='Time', window=window, window_correction=window_correction, 
        detrend=detrend, chunks_to_segments=True).mean(dim=('xh','yq'))

    ps = ps_u + ps_v

    # in case of nchunks > 1
    try:
        ps = ps.mean(dim='Time_segment')
    except:
        pass

    # Convert 2-sided power spectrum to one-sided
    ps = ps[ps.freq_Time>=0]
    freq = ps.freq_Time
    ps[freq==0] = ps[freq==0] / 2

    # Drop zero frequency for better plotting
    ps = ps[ps.freq_Time>0]

    ############## normalization tester #############
    #print('Energy balance:')
    #print('mean(u^2+v^2)/2=', ((u**2)/2).mean(dim=('Time', 'xq', 'yh')).values + ((v**2)/2).mean(dim=('Time', 'xh', 'yq')).values)
    #print('int(E(nu),dnu)=', (ps.sum(dim='freq_Time') * ps.freq_Time.spacing).values)
    #spacing = np.diff(u.Time).mean()
    #print(f'Minimum period {2*spacing} [days]')
    #print(f'Max frequency={ps.freq_Time.max().values} [1/day], \n Max inverse period={0.5/spacing} [1/day]')

    return ps

def Lk_error(input, target, normalize=False, k=2):
    '''
    Universal function for computation of NORMALIZED error.
    target - "good simulation", it is used for normalization
    Output is a scalar value.
    error = target-input
    result = mean(abs(error)) / mean(abs(target))
    numerator and denominator could be vectors
    only if variables have layers.
    In this case list of two elements is returned
    '''
    # Check dimensions
    if sorted(input.dims) != sorted(target.dims) or sorted(input.shape) != sorted(target.shape):
        import sys
        sys.exit(f'Dimensions disagree: {sorted(input.dims)} {sorted(target.dims)} {sorted(input.shape)} {sorted(target.shape)}')

    error = target - input

    average_dims = list(input.dims)
    
    # if layer is present, do not average over it at first stage!
    
    if 'zl' in average_dims:
        average_dims.remove('zl')

    if 'zi' in average_dims:
        average_dims.remove('zi')
    
    def lk_norm(x,k):
        '''
        k - order of norm
        k = -1 is the L-infinity norm
        '''
        if k > 0:
            return ((np.abs(x)**k).mean(dim=average_dims))**(1./k)
        elif k==-1:
            return np.abs(x).max(dim=average_dims)
    
    result = lk_norm(error,k)
    if normalize:
        result = result / lk_norm(target,k)

    return list(np.atleast_1d(result))

def compare(tested, control, mask=None, vmax=None, vmin = None, selector=select_NA, cmap=cmocean.cm.balance, time=-1, zl=0,
            label_test = 'Tested field', label_control = 'Control field'):
    if mask is not None:
        mask_nan = mask.data.copy()
        mask_nan[mask_nan==0.] = np.nan
        mask_nan = mask_nan + mask*0
        tested = tested * mask_nan
        control = control * mask_nan
    tested = selector(remesh(tested,control))
    control = selector(control)

    if 'time' in tested.dims:
        tested = tested.isel(time=time)
        control = control.isel(time=time)

    if 'zl' in tested.dims:
        tested = tested.isel(zl=zl)
        control = control.isel(zl=zl)

    tested = tested.compute()
    control = control.compute()
    
    if vmax is None:
        control_mean = control.mean()
        control_std = control.std()
        vmax = control_mean + control_std * 4
        vmin = control_mean - control_std * 4
    else:
        control_mean = 0.
        if vmin is None:
            vmin = - vmax
    
    central_latitude = float(y_coord(control).mean())
    central_longitude = float(x_coord(control).mean())
    fig, axes = plt.subplots(2,2, figsize=(12, 7), subplot_kw={'projection': ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude)})
    cmap.set_bad('gray')
    
    ax = axes[0][0]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    im = tested.plot(ax=ax, vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title(label_test)
    ax = axes[0][1]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    control.plot(ax=ax, vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title(label_control)
    ax = axes[1][0]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    (tested-control).plot(ax=ax, vmax=vmax-control_mean, vmin=vmin-control_mean, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title(f'{label_test}-{label_control}')
    plt.tight_layout()
    plt.colorbar(im, ax=axes, shrink=0.9, aspect=30, extend='both')
    axes[1][1].remove()
    
    ########## Metrics ##############
    error = tested-control
    relative_error = np.abs(error).mean() / np.abs(control).mean()
    R2 = 1 - (error**2).mean() / (control**2).mean()
    optimal_scaling = (tested*control).mean() / (tested**2).mean()
    error = tested * optimal_scaling - control
    R2_max = 1 - (error**2).mean() / (control**2).mean()
    corr = xr.corr(tested, control)
    print('Correlation:', float(corr))
    print('Relative Error:', float(relative_error))
    print('R2 = ', float(R2))
    print('R2 max = ', float(R2_max))
    print('Optinal scaling:', float(optimal_scaling))
    print(f'Nans [test/control]: [{int(np.sum(np.isnan(tested)))}, {int(np.sum(np.isnan(control)))}]')