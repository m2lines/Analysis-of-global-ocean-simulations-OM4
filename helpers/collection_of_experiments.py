import xarray as xr
import os
from helpers.experiment import Experiment
from helpers.computational_tools import *
from helpers.plot_helpers import *
import cmocean
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import dask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

def init_subplots(exps, labels, ncols=3):
    if labels is None:
            labels=exps
    nfig = len(exps)
    ncol = min(ncols,nfig)
    nrows = nfig / ncols
    if nrows > 1:
        nrows = int(np.ceil(nrows))
    else:
        nrows = 1
    
    return labels, nrows, ncol

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
    
    def compute_statistics(self):
        for exp in self.exps:
            for key in Experiment.get_list_of_netcdf_properties():
                try:
                    self[exp].__getattribute__(key)
                    print('Computed: ', exp, key, ' '*100,end='\r')
                except:
                    print('Error occured: ', exp, key, ' '*100,end='\r')

    @classmethod
    def init_folder(cls, common_folder, exps=None, exps_names=None, additional_subfolder='output', prefix=None, **kw):
        '''
        Scan folders in common_folder and returns class instance with exps given by these folders
        exps - list of folders can be specified
        exps_names - list of labels can be specified
        additional_subfolder - if results are stored not in common_folder+exps[i],
        but in an additional subfolder 
        '''
        dask.config.set(**{'array.slicing.split_large_chunks': True})
        warnings.filterwarnings("ignore")
        folders = []
        for root, _, _ in os.walk(common_folder):
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
            experiments_dict[exps[i]] = Experiment(folder, exps[i], **kw)
            names_dict[exps[i]] = exps_names[i] # convert array to dictionary

        return cls(exps, experiments_dict, names_dict)
    
    def plot_series(self, exps, labels=None, colors=['gray', violet, 'tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:cyan', 'b', 'g', 'y'], lws=2, CFL=False):
        default_rcParams({'font.size':12, 'figure.subplot.hspace': 0.7})
        if CFL:
            nrows=3
        else:
            nrows=2
        plt.figure(figsize=(6,nrows*3), dpi=200)
        if labels is None:
            labels=exps
        if colors is None:
            colors = [None] * len(labels)

        for j, (exp, label) in enumerate(zip(exps, labels)):
            ds = self[exp].series
            ds['Time'] = ds['Time'] - ds['Time'][0]
            kw = {'lw': lws if isinstance(lws,int) else lws[j], 'color':colors[j]}
            
            plt.subplot(nrows,1,1)
            (ds.KE.sum('Layer')).plot(**kw)
            plt.xlabel('Years')
            plt.xticks(np.arange(6)*365,np.arange(6))
            plt.grid()
            plt.ylabel('Kinetic energy [J]')
            plt.ylim([0,6e+18])

            plt.subplot(nrows,1,2)
            (ds.APE.sum('Interface')).plot(label=label, **kw)
            plt.xlabel('Years')
            plt.xticks(np.arange(6)*365,np.arange(6))
            plt.grid()
            plt.ylim([3.75e+20, 3.95e+20])
            plt.ylabel('Available potential energy [J]')

            if CFL:
                plt.subplot(nrows,1,3)
                ds.max_CFL_lin.plot(label=label,**kw)
                plt.grid()
                plt.ylim([0,0.5])
                plt.xlabel('Years')
                plt.xticks(np.arange(6)*365,np.arange(6))
                plt.grid()
                plt.ylabel('CFL number')
        
        #plt.tight_layout()
        plt.legend(bbox_to_anchor=(1,1))

    def plot_map(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', 
                 cmap_bias = cmocean.cm.balance, cmap_field=cmocean.cm.thermal,
                 cilev_bias=None, cilev_field=None,
                 field = lambda x: x.thetao.isel(zl=0), 
                 target = lambda x: x.woa_temp.isel(zl=0),
                 scale = '$^oC$', cmap_label = 'Temperature, $^oC$',
                 range_field=(0,30), range_bias=(-5,5),
                 ncols=2, demean=False,
                 contour_lines=False,
                 overlay_psi=False, isotherm_17=False, bathymetry=False, gridlines=False,
                 overlay_depth=False, contours_SST=False, SST_levels = [9, 18],
                 overlay_ssh=False, GS_NAC=False):
        '''
        Generic function for plotting 2D fields
        '''
        default_rcParams({'font.size': 8})
        labels, nrows, ncol = init_subplots(exps, labels, ncols=ncols)
        
        fig = plt.figure(figsize=(4*ncol, 2*nrows), layout='constrained', dpi=200)
        
        # Some arrays were identified as all bad values but these are not
        #cmap_bias.set_bad('white', alpha=1)
        #cmap_field.set_bad('white', alpha=1)

        data = select(field(self['unparameterized']))
        central_latitude = float(y_coord(data).mean())
        central_longitude = float(x_coord(data).mean())

        norm_bias = None; norm_field = None
        if cilev_bias is not None:
            norm_bias = plt.matplotlib.colors.BoundaryNorm(boundaries=cilev_bias, ncolors=cmap_bias.N)
        if cilev_field is not None:
            norm_field = plt.matplotlib.colors.BoundaryNorm(boundaries=cilev_field, ncolors=cmap_field.N)

        if projection == '3D':
            if select == select_globe:
                transform = ccrs.Robinson()
            else:
                transform = ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude)
        elif projection == '2D':
            transform = ccrs.PlateCarree()
        else:
            print('Specify projection as 2D or 3D')
        
        for ifig, exp in enumerate(exps):
            ax = fig.add_subplot(nrows,ncol,ifig+1,projection=transform)
            if gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=1,alpha=1.0, linestyle='--')
            else:
                gl = ax.gridlines(draw_labels=True, linewidth=1,alpha=0.0, linestyle='-')
            gl.top_labels = False
            gl.right_labels = False
            if isinstance(transform, ccrs.PlateCarree) or isinstance(transform, ccrs.Orthographic):
                ax.coastlines(zorder=101)
            
            label = labels[ifig]

            if demean:
                data = select(target(self['unparameterized']))
                mean_obs = data.mean()

                data = select(field(self['unparameterized']))
                mean_ctr = data.mean()
                
            if exp == 'obs':
                try:
                    data = select(target(self['unparameterized']))
                except:
                    data = select(field(self['unparameterized'])) * np.nan
                if demean:
                    data = data - mean_obs
                vmin, vmax = range_field[0:2]
                cmap = cmap_field; norm=norm_field; cilev=cilev_field
            elif plot_type == 'default':
                data = select(field(self[exp]))
                if demean:
                    data = data - mean_ctr
                vmin, vmax = range_field[0:2]
                cmap = cmap_field; norm=norm_field; cilev=cilev_field

                try:
                    error = data - target(self[exp])
                    rmse = float(np.sqrt(np.nanmean(error**2)))
                    label = label + ' bias' + f'\n RMSE=%.4f{scale}' % rmse
                except:
                    pass
            elif plot_type == 'bias':
                data = select(field(self[exp]) - target(self[exp]))
                if demean:
                    data = data - (mean_ctr - mean_obs)
                rmse = float(np.sqrt(np.nanmean(data**2)))
                label = label + ' bias' + f'\n RMSE=%.4f{scale}' % rmse
                vmin, vmax = range_bias[0:2]
                cmap = cmap_bias; norm=norm_bias; cilev=cilev_bias
            elif plot_type == 'response':
                if exp == 'unparameterized':
                    if target(self[exp]) is not None:
                        data = select(field(self[exp]) - target(self[exp]))
                        if demean:
                            data = data - (mean_ctr - mean_obs)
                        rmse = float(np.sqrt(np.nanmean(data**2)))
                        label = label + ' bias' + f'\n RMSE=%.4f{scale}' % rmse
                        bias = data.copy()
                        vmin, vmax = range_bias[0:2]
                        cmap = cmap_bias; norm=norm_bias; cilev=cilev_bias
                    else:
                        data = select(field(self[exp]))
                        vmin, vmax = range_field[0:2]
                        cmap = cmap_field; norm=norm_field; cilev=cilev_field
                else:
                    data = select(field(self[exp])- field(self['unparameterized']))
                    try:
                        corr = - np.nanmean(bias * data) / np.sqrt(np.nanmean(bias**2) * np.nanmean(data**2))
                        attenuation = - np.nanmean(bias * data) / np.nanmean(data**2)
                        rmse = float(np.sqrt(np.nanmean((data+bias)**2)))
                        #label = label + ' response' + f'\n RMSE=%.4f{scale}, Corr=%.2f\n to atenuate=%.1f' % (rmse, corr, attenuation)
                        label = label + ' response' + f'\n RMSE=%.4f{scale}' % (rmse)
                    except:
                        label = label + ' response'

                    vmin, vmax = range_bias[0:2]
                    cmap = cmap_bias; norm=norm_bias; cilev=cilev_bias

            add_colorbar = not(plot_type == 'default')
            if add_colorbar:
                kw = dict(cbar_kwargs={'label':cmap_label, 'shrink': 0.8})
            else:
                kw = {}

            if norm is not None:
                plot_function = data.plot.contourf
            else:
                plot_function = data.plot.pcolormesh
            im=plot_function(ax=ax, transform=ccrs.PlateCarree(), 
                                    rasterized=True, cmap=cmap, norm=norm, add_colorbar=add_colorbar, vmin=vmin, vmax=vmax, extend='both', **kw)
            if contour_lines:
                contours = data.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='k', levels=cilev_field, linewidths=0.5)
                plt.clabel(contours, inline=True, fontsize=5, fmt="%d")
                #data.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='k', levels=[-40,-30,-20,-10,10,20,30,40], linewidths=0.5)
                #data.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='k', levels=[0], linewidths=1.5)

            if contours_SST:
                therm_true = self['unparameterized'].woa_temp.isel(zl=0)
                if exp == 'obs':
                    contours = select(therm_true).plot.contour(ax=ax, colors='b', transform=ccrs.PlateCarree(), linewidths=1.5, levels=SST_levels)
                    plt.clabel(contours, inline=True, fontsize=5, fmt="%d")
                else:
                    contours = select(therm_true).plot.contour(ax=ax, colors='b', transform=ccrs.PlateCarree(), linewidths=1.5, levels=SST_levels)
                    plt.clabel(contours, inline=True, fontsize=5, fmt="%d")
                    therm_exp = self[exp].thetao.isel(zl=0).sel(time=self[exp].Averaging_time).mean('time')
                    contours = select(therm_exp).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=1.5, levels=SST_levels)
                    plt.clabel(contours, inline=True, fontsize=5, fmt="%d")

            if overlay_psi:
                try:
                    contours = select(self[exp].barotropic_streamfunction).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=0.5, levels=np.arange(-195,200,15))
                    #plt.clabel(contours, inline=True, fontsize=8, fmt="%dSv")
                except:
                    pass

            if overlay_ssh:
                if exp == 'obs':
                    contours = select(self['unparameterized'].ssh_mean_obs+2).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=0.5, levels=np.arange(0,4,0.2))
                else:
                    contours = select(self[exp].ssh_mean+2).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=0.5, levels=np.arange(0,4,0.2))


            if isotherm_17:
                therm_true = self['unparameterized'].woa_temp.isel(zl=8)
                if exp == 'obs':
                    contours = select(therm_true).plot.contour(ax=ax, colors='b', transform=ccrs.PlateCarree(), linewidths=2, levels=[17])
                    plt.clabel(contours, inline=True, fontsize=6, fmt="%d$C^\circ$")
                else:
                    contours = select(therm_true).plot.contour(ax=ax, colors='b', transform=ccrs.PlateCarree(), linewidths=2, levels=[17])
                    #plt.clabel(contours, inline=True, fontsize=6, fmt="%d$C^\circ$")
                    therm_exp = self[exp].thetao.isel(zl=8).sel(time=self[exp].Averaging_time).mean('time')
                    contours = select(therm_exp).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=2, levels=[17])
                    #plt.clabel(contours, inline=True, fontsize=6, fmt="%d$C^\circ$")

            if GS_NAC:
                therm_true = self['unparameterized'].woa_temp
                GS_select = lambda x: x.sel(xh=slice(-80,-30), yh=slice(30,60)).isel(zl=9) # Depth200
                NAC_select = lambda x: x.sel(xh=slice(-50,-30), yh=slice(30,60)).isel(zl=9) # Depth200
                #Azores_select = lambda x: x.sel(xh=slice(-50,-30), yh=slice(30,60)).isel(zl=14) # Depth 600
                contours = GS_select(therm_true).plot.contour(ax=ax, colors='b', transform=ccrs.PlateCarree(), linewidths=0.5, levels=[15])
                contours = NAC_select(therm_true).plot.contour(ax=ax, colors='r', transform=ccrs.PlateCarree(), linewidths=0.5, levels=[10])
                #contours = Azores_select(therm_true).plot.contour(ax=ax, colors='g', transform=ccrs.PlateCarree(), linewidths=2, levels=[10])
                if exp != 'obs':
                    therm_exp = self[exp].thetao.sel(time=self[exp].Averaging_time).mean('time')
                    contours = GS_select(therm_exp).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=0.5, levels=[15])
                    contours = NAC_select(therm_exp).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=0.5, levels=[10])
                    #contours = Azores_select(therm_exp).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=2, levels=[10])
            
            if overlay_depth:
                depth = self['unparameterized'].depth
                contours = select(depth/1000).plot.contour(ax=ax, colors='k', transform=ccrs.PlateCarree(), linewidths=0.5, levels=[0.5, 1, 2, 3, 4, 5])
                #plt.clabel(contours, inline=True, fontsize=6, fmt="%d")
                
            ax.set_title(label)
            ax.add_feature(cfeature.LAND, color='gray', zorder=100)
            if projection=='2D':
                try:
                    ax.set_xlim(data.xh[0], data.xh[-1])
                except:
                    ax.set_xlim(data.xq[0], data.xq[-1])
                try:
                    ax.set_ylim(data.yh[0], data.yh[-1])
                except:
                    ax.set_ylim(data.yq[0], data.yq[-1])
        
        if not(add_colorbar):
            plt.colorbar(im, ax=fig.axes, label=cmap_label, extend='both', shrink=0.8)

    def plot_temp(self, exps, labels=None, zl=0, select=select_globe, projection='2D', plot_type = 'default', ncols=2, 
                  time_range=None, **kw):
        if time_range is None:
            time_range = self['unparameterized'].Averaging_time
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.RdYlBu_r, 
                    #cmap_bias=cmocean.cm.balance,
                    cmap_field=cmocean.cm.balance,
                    cilev_bias=cilev(0.5), cilev_field=np.arange(-2,30),
                    field = lambda x: x.thetao.isel(zl=zl).sel(time=time_range).mean('time'), 
                    target = lambda x: x.woa_temp.isel(zl=zl),
                    scale = '$^oC$', cmap_label = 'Temperature, $^oC$',
                    range_field=(None,None), range_bias=(-2.25,2.25),
                    ncols=ncols, **kw)
        if zl>0:
            plt.suptitle('Depth=%.1f' % self['unparameterized'].thetao.zl[zl])

    def plot_so(self, exps, labels=None, zl=0, select=select_globe, projection='2D', plot_type = 'default', ncols=2, 
                  time_range=None, **kw):
        if time_range is None:
            time_range = self['unparameterized'].Averaging_time
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.RdYlBu_r, 
                    cmap_field=cmocean.cm.balance,
                    cilev_bias=cilev(0.1), cilev_field=np.arange(32,37.25,0.25),
                    field = lambda x: x.salto.isel(zl=zl).sel(time=time_range).mean('time'), 
                    target = lambda x: x.woa_salt.isel(zl=zl),
                    scale = 'psu', cmap_label = 'Salinity, psu',
                    range_field=(None,None), range_bias=(None,None),
                    ncols=ncols, **kw)
        if zl>0:
            plt.suptitle('Depth=%.1f' % self['unparameterized'].thetao.zl[zl])

    def plot_sigma0(self, exps, labels=None, zl=0, select=select_globe, projection='2D', plot_type = 'default', ncols=2, 
                  time_range=None, **kw):
        if time_range is None:
            time_range = self['unparameterized'].Averaging_time
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.RdYlBu_r, 
                    cmap_field=cmocean.cm.balance,
                    cilev_bias=np.arange(-0.09,0.11,0.02)*2, cilev_field=None,
                    field = lambda x: x.sigma0.isel(zl=zl).sel(time=time_range).mean('time'), 
                    target = lambda x: x.woa_sigma0.isel(zl=zl),
                    scale = 'kg/$m^3$', cmap_label = 'Density [$\sigma_0$, kg/m$^3$]',
                    range_field=(None,None), range_bias=(None,None),
                    ncols=ncols, **kw)
        if zl>0:
            plt.suptitle('Depth=%.1f' % self['unparameterized'].thetao.zl[zl])

    def plot_ubar(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', ncols=2, **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.RdYlBu, cmap_field=cmocean.cm.balance,
                    field = lambda x: x.ubar * 100, 
                    target = lambda x: None,
                    scale = 'cm/s', cmap_label = 'Mean zonal \nbarotropic velocity, cm/s',
                    range_field=(-10,10), range_bias=(-10,10),
                    ncols=ncols, **kw)
        
    def plot_MLD_summer(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', ncols=2, **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.RdYlBu, cmap_field=plt.cm.BuPu,
                    field = lambda x: x.MLD_summer, 
                    target = lambda x: x.MLD_summer_obs,
                    scale = 'm', cmap_label = 'Summer MLD, metres',
                    range_field=(0,80), range_bias=(-20,20),
                    ncols=ncols, **kw)
        
    def plot_MLD_winter(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', ncols=2, vmax=1000., **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.RdYlBu, cmap_field=plt.cm.RdYlBu_r,
                    field = lambda x: x.MLD_winter, 
                    target = lambda x: x.MLD_winter_obs,
                    scale = 'm', cmap_label = 'Winter MLD, metres',
                    range_field=(-vmax,vmax), range_bias=(-vmax,vmax),
                    ncols=ncols, **kw)

    def plot_ssh(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', ncols=2, idx=-1, **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.seismic, cmap_field=cmocean.cm.balance,
                    field = lambda x: x.ocean_daily.zos.isel(time=idx), 
                    target = lambda x: x.ssh_obs.isel(time=idx),
                    scale = 'm', cmap_label = 'SSH snapshot, m',
                    range_field=(-1,1), range_bias=(-0.1,0.1),
                    ncols=ncols, demean=True, **kw)

    def plot_BT_fraction(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', ncols=2, idx=-1, **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = cmocean.cm.balance, cmap_field=plt.cm.inferno,
                    field = lambda x: x.BT_fraction, 
                    target = lambda x: x.BT_fraction_obs,
                    scale = '', cmap_label = 'BT KE fraction',
                    range_field=(0,1), range_bias=(-0.2,0.2),
                    ncols=ncols, demean=False, **kw)

    def plot_ssh_std(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', ncols=2, **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.seismic, cmap_field=cmocean.cm.amp,
                    field = lambda x: x.ssh_std, 
                    target = lambda x: x.ssh_std_obs,
                    scale = 'm', cmap_label = 'STD SSH, m',
                    range_field=(0,0.3), range_bias=(-0.1,0.1),
                    ncols=ncols, **kw)
        
    def plot_ssh_mean(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', ncols=2, contour_lines=False, **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.seismic, cmap_field=cmocean.cm.balance,
                    cilev_bias=cilev(0.08), cilev_field=np.arange(-1.5,1.6,0.1),
                    field = lambda x: x.ssh_mean, 
                    target = lambda x: x.ssh_mean_glorys,
                    scale = 'm', cmap_label = 'Mean ZOS [m]',
                    range_field=(None,None), range_bias=(None,None),
                    ncols=ncols, contour_lines=contour_lines, **kw)
        
    def plot_psi(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', ncols=2, contour_lines=False, vmax=45, **kw):
        import matplotlib.colors as mcolors
        #Here we use colorbar similar to https://www.sciencedirect.com/science/article/pii/S0924796314002437
        # Define custom colormap
        colors = [
            #(1, 0.0, 1),  # Purple
            (0.5, 0.0, 1),  # Purple
            #(0.0, 0.0, 1),  # Blue
            (0.0, 0.5, 0),  # Green
            (1, 1, 0.0),  # Yellow
            (1.0, 0, 0.0),  # Red
            #(0.5, 0, 0.0),  # Red
        ]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)
        levels = np.arange(-45, 30, 5)
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = cmocean.cm.balance, cmap_field=custom_cmap,
                    cilev_bias=np.arange(-47.5,50,5), cilev_field=levels,
                    field = lambda x: x.barotropic_streamfunction, 
                    target = lambda x: None,
                    scale = 'Sv', cmap_label = 'Barotropic Streamfunction, Sv',
                    range_field=(None,None), range_bias=(None,None),
                    ncols=ncols, contour_lines=contour_lines, **kw)
        
    def plot_geoRV(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', idx=-1, ncols=2, **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = cmocean.cm.balance, cmap_field=cmocean.cm.balance,
                    field = lambda x: x.geoRV.isel(time=idx), 
                    target = lambda x: x.geoRV_obs.isel(time=idx),
                    scale = '1/s', cmap_label = 'Geostrophic relative vorticity 1/s',
                    range_field=(-3e-5,3e-5), range_bias=(-3e-5,3e-5),
                    ncols=ncols, **kw)
        
    def plot_geovel(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', idx=-1, ncols=2, **kw):
        from matplotlib.colors import LinearSegmentedColormap
        original_cmap = cmocean.cm.balance

        # Extract the red part (positive values, typically > 0.5 in normalized space)
        n_colors = 256  # Number of colors in the colormap
        colors = original_cmap(np.linspace(0.5, 1, n_colors // 2))  # Keep the red part

        # Create a new colormap with only the red part
        red_cmap = LinearSegmentedColormap.from_list("RedBalance", colors)
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.inferno, cmap_field=red_cmap,
                    field = lambda x: np.log10(x.geovel.isel(time=idx)), 
                    target = lambda x: np.log10(x.geovel_obs.isel(time=idx)),
                    scale = 'm', cmap_label = 'Geovelocity, log10',
                    range_field=(-1,0.176), range_bias=(-1.30,0.176),
                    ncols=ncols, **kw)
        
    def plot_uabs(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', idx=-1, ncols=2, zl=0, vmax=1, **kw):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.seismic, cmap_field=plt.cm.viridis,
                    field = lambda x: x.uabs.isel(zl=zl), 
                    target = lambda x: None,
                    scale = 'm/s', cmap_label = 'Modulus of mean velocity [m/s]',
                    range_field=(0,vmax), range_bias=(-vmax,vmax),
                    ncols=ncols, **kw)
        plt.suptitle('Depth = %d' % int(self['unparameterized'].ocean_month_z.zl[zl]))
        
    def plot_temp_section(self, exps, labels=None, select=select_Drake, plot_type = 'default', contour_lines=True):
        default_rcParams({'font.size': 10})
        labels, nrows, ncol = init_subplots(exps, labels, ncols=2)
        
        fig = plt.figure(figsize=(4*ncol, 3*nrows), layout='constrained', dpi=200)
        cmap_bias = cmocean.cm.balance
        cmap_bias.set_bad('gray', alpha=1)

        cmap_temp = cmocean.cm.thermal
        cmap_temp.set_bad('gray', alpha=1)

        select_exp = lambda x, exp: select(x).sel(time=self[exp].Averaging_time).mean('time')
        
        for ifig, exp in enumerate(exps):
            ax = fig.add_subplot(nrows,ncol,ifig+1)
            
            label = labels[ifig]
            data_obs = select(self['unparameterized'].woa_temp)
            if exp == 'obs':
                data = data_obs
                vmin = 0; vmax=30
                cmap = cmap_temp
            elif plot_type == 'default':
                data = select_exp(self[exp].thetao, exp)
                vmin = 0; vmax=30
                cmap = cmap_temp
                rmse = float(np.sqrt(np.nanmean((data-data_obs)**2)))
                label = label + '\n RMSE=%.3f' % rmse + '$^oC$'
            elif plot_type == 'bias':
                data = select_exp(self[exp].thetao, exp) - data_obs
                rmse = float(np.sqrt(np.nanmean(data**2)))
                label = label + '\n RMSE=%.3f' % rmse + '$^oC$'
                vmin = -1; vmax=1
                cmap = cmap_bias
            elif plot_type == 'response':
                if exp == 'unparameterized':
                    data = select_exp(self[exp].thetao, exp) - data_obs
                    rmse = float(np.sqrt(np.nanmean(data**2)))
                    label = label + ' bias' + '\n RMSE=%.3f' % rmse + '$^oC$'
                    bias = data.copy()
                else:
                    data = select_exp(self[exp].thetao, exp) - select_exp(self['unparameterized'].thetao, 'unparameterized')
                    corr = - np.nanmean(bias * data) / np.sqrt(np.nanmean(bias**2) * np.nanmean(data**2))
                    attenuation = - np.nanmean(bias * data) / np.nanmean(data**2)
                    rmse = float(np.sqrt(np.nanmean((data+bias)**2)))
                    label = label + ' response' + f'\n RMSE=%.3f$^oC$, Corr=%.2f\n to atenuate=%.1f' % (rmse, corr, attenuation)
                vmin = -1; vmax=1
                cmap = cmap_bias

            if contour_lines:
                mask = xr.where(np.isnan(data_obs), 1., 0.)
                mask.plot.pcolormesh(cmap='gray_r', vmin=0, vmax=2, add_colorbar=False)
                contours = data_obs.plot.contour(ax=ax, colors='gray', levels=[0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30])
                plt.clabel(contours, inline=True, fontsize=5, fmt="%d")
                data.plot.contour(ax=ax, colors='b', linestyles='dashed', levels=[0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30])
                plt.clabel(contours, inline=True, fontsize=5, fmt="%d")
            else:
                im=data.plot.pcolormesh(ax=ax, rasterized=True, cmap=cmap, add_colorbar=True, vmin=vmin, vmax=vmax,
                                    cbar_kwargs={'label':'Temperature, $^oC$'})
            ax.set_title(label)
            ax.set_ylim([0,6500])
            #ax.set_yscale('log')
            ax.invert_yaxis()
            plt.ylabel('Depth, m')
            plt.xlabel('Latitude')

    def plot_KE_spectrum(self, exps, labels=None, colors=None, type='EKE'):
        default_rcParams({'font.size': 14})
        if labels is None:
            labels=exps
        fig, ax = plt.subplots(2,2,figsize=(12,8))

        if colors is None:
            colors = [None] * len(exps)
            colors[0] = 'tab:gray'
            colors[-1] = 'k'
            
        lw = [2] * len(exps)
        lw[-1] = 1.5

        for j_exp,exp in enumerate(exps):
            for j_region, region in enumerate(['Gulf', 'Kuroshio', 'Aghulas', 'Malvinas']):
                plt.subplot(2,2,1+j_region)
                if exp == 'obs':
                    KE = self['unparameterized'].__getattribute__(f'geo{type}_{region}_obs')
                else:
                    KE = self[exp].__getattribute__(f'geo{type}_{region}')

                KE.plot(label=labels[j_exp], color=colors[j_exp], xscale='log', yscale='log', lw=lw[j_exp])
                plt.title(region)

        for j_region in range(4):
            plt.subplot(2,2,j_region+1)
            plt.xlabel(r'wavenumber $[\mathrm{km}^{-1}]$')
            plt.ylabel(r'Geostrophic %s spectrum $[\mathrm{m}^3/\mathrm{s}^2]$' % type)
            plt.xlim([None,2e-4])
            plt.ylim([1e-2,1e+4])
            plt.xticks([1e-5, 1e-4], ['10$^{-2}$', '10$^{-1}$'])

            k=np.array([2e-5,1e-4])
            plt.plot(k,7e+3*(k/2e-5)**(-3), ls='--', color='k')
            plt.text(6e-5,5e+2,'$k^{-3}$')

        plt.legend(bbox_to_anchor=(1,1))
        plt.tight_layout()

    def plot_lat(self, exps, labels=None, colors=None, type='EKE'):
        default_rcParams({'font.size': 14})
        plt.figure(figsize=(10,8))
        if labels is None:
            labels=exps

        if colors is None:
            colors = [None] * len(exps)
            colors[0] = 'tab:gray'
            colors[-1] = 'k'

        lw = [3] * len(exps)
        lw[-1] = 1.5

        for j_exp,exp in enumerate(exps):
            plt.subplot(2,1,1)
            if exp == 'obs':
                KE = self['unparameterized'].__getattribute__(f'geo{type}_map_obs')
            else:
                KE = self[exp].__getattribute__(f'geo{type}_map')

            KE = (KE).mean('xh')
            KE.plot(label=labels[j_exp], color=colors[j_exp], lw=lw[j_exp])

            plt.subplot(2,1,2)
            if exp == 'obs':
                scale = self['unparameterized'].__getattribute__('eddy_scale_obs')
            else:
                scale = self[exp].__getattribute__('eddy_scale')

            scale = scale.mean('xh')
            scale.plot(label=labels[j_exp], color=colors[j_exp], lw=lw[j_exp])

        plt.subplot(2,1,1)
        plt.xlabel('Latitude')
        plt.xlim([-60,60])
        plt.ylim([0,0.05])
        plt.ylabel('Zonally-averaged \ngeostrophic %s [$\mathrm{m}^2/\mathrm{s}^2$]' % type)
        plt.grid()
        
        plt.subplot(2,1,2)
        dxt = self['unparameterized'].param.dxt
        dyt = self['unparameterized'].param.dyt
        dx = np.sqrt(dxt*dyt) / 1000. # in km
        dx.mean('xh').plot(lw=1.5, color='tab:gray', ls='--', label='Grid spacing OM4')
        self['unparameterized'].rossby_radius_lat.plot(lw=1.5, color='k', ls='-.', label='Rossby radius')
        plt.title('')
        plt.xlabel('Latitude')
        plt.xlim([-60,60])
        plt.ylim([0,400])
        plt.yticks([0,50,100,150,200,250,300,350,400])
        plt.grid()
        plt.ylabel('Energy-containing scale [km]')    
        plt.legend(bbox_to_anchor=(1.2,1))