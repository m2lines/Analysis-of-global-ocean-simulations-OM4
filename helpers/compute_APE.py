import gsw
import numpy as np
import xarray as xr
from cmip_basins import generate_basin_codes

def compute_mask(T, param):
    '''
    Input array is temperature of OM4 at 35 depth levels with land values masked as NaNs
    Note: temperature should have only one time moment

    Compute mask of the domain similar to VERDIÈRE 2018 in three steps:
    * Consider only horizontal points where depth is larger than 1000m
    * Keep only Atlantic, Pacific, Southern and Indian oceans according to cmpi6 classification
    * Remove a little Part of Arctic ocean from the resulting mask
    '''
    basincodes = generate_basin_codes(param, lon='geolon', lat='geolat', mask='wet')

    # Choose Southern Ocean (1), Atlantic Ocean (2), Pacific Ocean (3), and Indian Ocean(5)
    mask = xr.where(basincodes.isin([1,2,3,5]),1.,np.nan)

    # Remove Arctic APE basin
    mask.loc[{'xh':slice(-11,30), 'yh':slice(59,70)}] = np.nan 

    # Consider only points deeper than 1000m
    mask = mask * xr.where(np.isnan(T.isel(zl=18)), np.nan, 1.)

    return mask.values + xr.zeros_like(param.areacello)

def sort_xarray(_x,*_y):
    # Form 1D array
    x = _x.values.ravel()
    # Skip nans
    x = x[~np.isnan(x)]

    # Find sort indices
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    out = [x]
    for y in _y:
        if not(np.array_equal(np.isnan(_x), np.isnan(y))):
            print('Check that NaNs are in the same places')
            return
        y = y.values.ravel()
        y = y[~np.isnan(y)]
        y = y[sort_indices]
        out.append(y)
    return out

def compute_interfaces(rho, edges):
    '''
    rho is 3D array zl, yh, xh
    edges is 1D array of zi

    The algorithm returns depth of each edge as a function of horizontal coordiante:
    interfaces array of size zi, yh, xh

    The algorithm is based on the:
    * Using density rho as a vertical coordinate. 
    * It is possible only if the density is a monotonic function of depth.
    * Density also should be monotonic to make sure that APE is positive (which depends on rho_i - rho_i+1)
    * So we ensure monotonicity by preprocessing density
    * In this case, there is an inverse function
    * The inverse function is interpolated linearly to compute the interfaces
    '''

    zl_depth = rho.zl.values
    rho_monotonic = np.sort(rho.values, axis=0)
    
    z_i, yh, xh = len(edges), len(rho.yh), len(rho.xh)
    interfaces = np.zeros((z_i,yh,xh)) * np.nan

    for j in range(yh):
        for i in range(xh):
            # We compute only internal interfaces
            # edges are densities
            # right and left = nan means that if interface does not fit to the water column, we set it to nan
            interfaces[1:-1,j,i] = np.interp(edges[1:-1], rho_monotonic[:,j,i], zl_depth, right=np.nan, left=np.nan)
    return xr.DataArray(interfaces, dims=['zi','yh','xh'], coords={'z_i':edges.zi, 'yh': rho.yh, 'xh': rho.xh})

def compute_APE(T, S, areacello, dz, zi_coord):
    '''
    T is potential temperature in C^o
    S is salinity [psu]
    areacello is the area of the grid cell [m^2]
    zi_coord>0 [m] depth of interfaces between grid cells
    dz [m] is the vertical grid spacing

    T,S,areacello of size zl,yh,xh
    dz has size of zl
    size of zi is one larger than zl

    Land points must be masked with NaNs in T and S, and consistently

    Algorithms:
    * Use zi as reference heights
    * Compute reference sigma2 density profile by sorting algorithm (Griffies et al 2000)
    * Compute reduced gravities based on sigma2 of reference density profile
    * Find position of interfaces between sigma2 reference levels
    * Compute interface-integrated APE per unit mass as follows (VERDIÈRE et al 2018):
        APE = 0.5 * g_prime (interface - zi)**2 [m^3/s^2]
    * And depth-integrated APE is:
        sum(APE, zi=0..-1) [m^3/s^2]
    * Global integral is given in Joules:
    APE_total = (rho0 * APE.sum('zi') * areacello).sum()
    '''
    # Mask points
    mask_nan = xr.where(np.isnan(T), np.nan, 1.)
    
    # Compute the volume of grid cells
    dV = (areacello * dz * mask_nan).transpose('zl',...)

    # Compute sigma2 potential density
    rho_sigma2 = gsw.density.sigma2(S,T)

    # Sort potential density and rearrange volumes accordingly
    rho_sigma2_sorted, dV_sorted = sort_xarray(rho_sigma2, dV)

    # Sorted fluid particles are distributed over vertical levels 
    # according to volumes of each vertical level. Bin edges are below: 
    dV_bin_edges = np.pad(dV.sum(['xh','yh']).cumsum('zl').values,(1,0)) # Cumsum starting from the surface
    dV_bin_edges[-1] = dV_bin_edges[-1] + 1e+18 # To make sure that all values correspond to some bin

    # Distribute sorted fluid particles over bins corresponding to vertical levels
    bin_indices = []
    dV_sorted_cumsum = np.cumsum(dV_sorted)
    for zi in range(len(dV_bin_edges)-1):
        bin_indices.append(np.logical_and(dV_sorted_cumsum >= dV_bin_edges[zi], dV_sorted_cumsum < dV_bin_edges[zi+1]))

    # Find sigma2 of reference state
    rho_sigma2_ref = xr.zeros_like(T.zl)
    for zl in range(len(T.zl)):
        idx = bin_indices[zl]
        # Volume-averaged density for each sigma-2 layer
        rho_sigma2_ref[zl] = (rho_sigma2_sorted[idx] * dV_sorted[idx]).sum() / (dV_sorted[idx]).sum()

    # Find sigma2-edges of bins of reference state
    rho_sigma2_ref_edges = xr.zeros_like(zi_coord) # Edges of the binning algorithm

    rho_sigma2_ref_edges[0] = rho_sigma2_sorted[0]-1e-1 # -1e-1 so all grid points are for sure here
    rho_sigma2_ref_edges[-1] = rho_sigma2_sorted[-1]+1e-1 # +1e-1 so all grid points are for sure here
    
    for zi in range(1,len(rho_sigma2_ref_edges)-1):
        idx_upper = bin_indices[zi-1]
        idx_lower = bin_indices[zi]
        # Find the middle point between edges of the two sets
        rho_sigma2_ref_edges[zi] = (np.max(rho_sigma2_sorted[idx_upper]) + np.min(rho_sigma2_sorted[idx_lower])) * 0.5

    # Find the position of interfaces corresponding to the sigma2-edges
    interfaces = compute_interfaces(rho_sigma2, rho_sigma2_ref_edges)

    # Compute reduced gravities
    # Reduced gravity for internal interfaces
    rho0 = 1035.
    g = 9.8
    rho_vals = rho_sigma2_ref.values
    g_prime = (rho_vals[1:] - rho_vals[:-1])/rho0 * g
    # add nan values on surface and bottom interfaces, where we do not compute APE
    g_prime = np.pad(g_prime,1,constant_values=np.nan)
    g_prime = xr.DataArray(g_prime, dims='zi', coords={'zi': zi_coord})

    # Note: zi is interfaces height at rest by construction!
    APE = 0.5 * g_prime * (interfaces-zi_coord)**2

    APE_total = (rho0 * APE.sum('zi') * areacello).sum()

    return {'APE': APE, 'APE_total': APE_total,
            'interfaces': interfaces, 'rho_sigma2': rho_sigma2, 
            'rho_sigma2_ref': rho_sigma2_ref, 'rho_sigma2_ref_edges': rho_sigma2_ref_edges,
            'reference_height': zi_coord, 'g_prime': g_prime, 'dV': dV}