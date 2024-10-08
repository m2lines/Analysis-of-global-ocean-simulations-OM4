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
        dask.config.set(**{'array.slicing.split_large_chunks': True})
        warnings.filterwarnings("ignore")
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
    
    def plot_series(self, exps, labels=None, colors=['gray', violet, 'tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:cyan', 'b', 'g', 'y'], CFL=False):
        default_rcParams({'font.size':12})
        if CFL:
            nrows=3
        else:
            nrows=2
        plt.figure(figsize=(6,nrows*3))
        if labels is None:
            labels=exps
        if colors is None:
            colors = [None] * len(labels)

        for j, (exp, label) in enumerate(zip(exps, labels)):
            ds = self[exp].series
            ds['Time'] = ds['Time'] - ds['Time'][0]
            kw = {'lw':2, 'color':colors[j]}
            
            plt.subplot(nrows,1,1)
            (ds.KE.sum('Layer')).plot(**kw)
            plt.xlabel('Years')
            plt.xticks(np.arange(6)*365,np.arange(6))
            plt.grid()
            plt.ylabel('Kinetic energy, Joules')
            plt.ylim([0,6e+18])

            plt.subplot(nrows,1,2)
            (ds.APE.sum('Interface')).plot(label=label, **kw)
            plt.xlabel('Years')
            plt.xticks(np.arange(6)*365,np.arange(6))
            plt.grid()
            plt.ylabel('Available potential energy, Joules')

            if CFL:
                plt.subplot(nrows,1,3)
                ds.max_CFL_lin.plot(label=label,**kw)
                plt.grid()
                plt.ylim([0,0.5])
                plt.xlabel('Years')
                plt.xticks(np.arange(6)*365,np.arange(6))
                plt.grid()
                plt.ylabel('CFL number')
        
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1,1))

    def plot_map(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default', 
                 cmap_bias = cmocean.cm.balance, cmap_field=cmocean.cm.thermal,
                 field = lambda x: x.thetao.isel(zl=0), 
                 target = lambda x: x.woa_temp.isel(zl=0),
                 scale = '$^oC$', cmap_label = 'Temperature, $^oC$',
                 range_field=(0,30), range_bias=(-5,5)):
        '''
        Generic function for plotting 2D fields
        '''
        default_rcParams({'font.size': 8})
        labels, nrows, ncol = init_subplots(exps, labels, ncols=2)
        
        fig = plt.figure(figsize=(4*ncol, 2*nrows), layout='constrained', dpi=200)
        
        cmap_bias.set_bad('white', alpha=1)
        cmap_field.set_bad('white', alpha=1)

        if projection == '3D':
            projection = ccrs.Robinson()
        elif projection == '2D':
            projection = ccrs.PlateCarree()
        else:
            print('Specify projection as 2D or 3D')
        
        for ifig, exp in enumerate(exps):
            ax = fig.add_subplot(nrows,ncol,ifig+1,projection=projection)
            gl = ax.gridlines(draw_labels=True, linewidth=0.01,alpha=0.0, linestyle='-')
            gl.top_labels = False
            gl.right_labels = False
            if isinstance(projection, ccrs.PlateCarree):
                ax.coastlines(zorder=101)
            
            label = labels[ifig]
            
            if exp == 'obs':
                data = select(target(self['unparameterized']))
                vmin, vmax = range_field[0:2]
                cmap = cmap_field
            elif plot_type == 'default':
                data = select(field(self[exp]))
                vmin, vmax = range_field[0:2]
                cmap = cmap_field
            elif plot_type == 'bias':
                data = select(field(self[exp]) - target(self[exp]))
                rmse = float(np.sqrt(np.nanmean(data**2)))
                label = label + f'\n RMSE=%.3f{scale}' % rmse
                vmin, vmax = range_bias[0:2]
                cmap = cmap_bias
            elif plot_type == 'response':
                if exp == 'unparameterized':
                    data = select(field(self[exp]) - target(self[exp]))
                    rmse = float(np.sqrt(np.nanmean(data**2)))
                    label = label + ' bias' + f'\n RMSE=%.3f{scale}' % rmse
                    bias = data.copy()
                else:
                    data = select(field(self[exp])- field(self['unparameterized']))
                    corr = - np.nanmean(bias * data) / np.sqrt(np.nanmean(bias**2) * np.nanmean(data**2))
                    attenuation = - np.nanmean(bias * data) / np.nanmean(data**2)
                    rmse = float(np.sqrt(np.nanmean((data+bias)**2)))
                    label = label + ' response' + f'\n RMSE=%.3f{scale}, Corr=%.2f\n to atenuate=%.1f' % (rmse, corr, attenuation)
                vmin, vmax = range_bias[0:2]
                cmap = cmap_bias

            im=data.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), 
                                    rasterized=True, cmap=cmap, add_colorbar=True, vmin=vmin, vmax=vmax,
                                    cbar_kwargs={'label':cmap_label})
            ax.set_title(label)
            ax.add_feature(cfeature.LAND, color='gray', zorder=100)

    def plot_temp(self, exps, labels=None, zl=0, select=select_globe, projection='2D', plot_type = 'default'):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = cmocean.cm.balance, cmap_field=cmocean.cm.thermal,
                    field = lambda x: x.thetao.isel(zl=zl), 
                    target = lambda x: x.woa_temp.isel(zl=zl),
                    scale = '$^oC$', cmap_label = 'Temperature, $^oC$',
                    range_field=(0,30), range_bias=(-5,5))
        
    def plot_MLD_summer(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default'):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.RdYlBu, cmap_field=plt.cm.BuPu,
                    field = lambda x: x.MLD_summer, 
                    target = lambda x: x.MLD_summer_obs,
                    scale = 'm', cmap_label = 'Summer MLD, metres',
                    range_field=(0,80), range_bias=(-20,20))
        
    def plot_MLD_winter(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default'):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.RdYlBu, cmap_field=plt.cm.BuPu,
                    field = lambda x: x.MLD_winter, 
                    target = lambda x: x.MLD_winter_obs,
                    scale = 'm', cmap_label = 'Winter MLD, metres',
                    range_field=(0,300), range_bias=(-100,100))
        
    def plot_ssh_std(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default'):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = plt.cm.seismic, cmap_field=cmocean.cm.amp,
                    field = lambda x: x.ssh_std, 
                    target = lambda x: x.ssh_std_obs,
                    scale = 'm', cmap_label = 'STD SSH, m',
                    range_field=(0,0.3), range_bias=(-0.1,0.1))
        
    def plot_RV(self, exps, labels=None, select=select_globe, projection='2D', plot_type = 'default'):
        self.plot_map(exps, labels=labels, select=select, projection=projection, plot_type = plot_type,
                    cmap_bias = cmocean.cm.balance, cmap_field=cmocean.cm.balance,
                    field = lambda x: x.RV.isel(time=-1), 
                    target = lambda x: None,
                    scale = 'm', cmap_label = 'Relative vorticity 1/s',
                    range_field=(-3e-5,3e-5), range_bias=(-3e-5,3e-5))
        
    def plot_temp_section(self, exps, labels=None, select=select_Drake, plot_type = 'default'):
        default_rcParams({'font.size': 10})
        labels, nrows, ncol = init_subplots(exps, labels, ncols=2)
        
        fig = plt.figure(figsize=(4*ncol, 3*nrows), layout='constrained', dpi=200)
        cmap_bias = cmocean.cm.balance
        cmap_bias.set_bad('gray', alpha=1)

        cmap_temp = cmocean.cm.thermal
        cmap_temp.set_bad('gray', alpha=1)
        
        for ifig, exp in enumerate(exps):
            ax = fig.add_subplot(nrows,ncol,ifig+1)
            
            label = labels[ifig]
            if exp == 'obs':
                data = select(self['unparameterized'].woa_temp)
                vmin = 0; vmax=30
                cmap = cmap_temp
            elif plot_type == 'default':
                data = select(self[exp].thetao)
                vmin = 0; vmax=30
                cmap = cmap_temp
            elif plot_type == 'bias':
                data = select(self[exp].thetao - self[exp].woa_temp)
                rmse = float(np.sqrt(np.nanmean(data**2)))
                label = label + '\n RMSE=%.2f' % rmse + '$^oC$'
                vmin = -5; vmax=5
                cmap = cmap_bias
            elif plot_type == 'response':
                if exp == 'unparameterized':
                    data = select(self[exp].thetao - self[exp].woa_temp)
                    rmse = float(np.sqrt(np.nanmean(data**2)))
                    label = label + ' bias' + '\n RMSE=%.2f' % rmse + '$^oC$'
                    bias = data.copy()
                else:
                    data = select(self[exp].thetao - self['unparameterized'].thetao)
                    corr = - np.nanmean(bias * data) / np.sqrt(np.nanmean(bias**2) * np.nanmean(data**2))
                    attenuation = - np.nanmean(bias * data) / np.nanmean(data**2)
                    rmse = float(np.sqrt(np.nanmean((data+bias)**2)))
                    label = label + ' response' + f'\n RMSE=%.3f$^oC$, Corr=%.2f\n to atenuate=%.1f' % (rmse, corr, attenuation)
                vmin = -5; vmax=5
                cmap = cmap_bias

            im=data.plot.pcolormesh(ax=ax, rasterized=True, cmap=cmap, add_colorbar=True, vmin=vmin, vmax=vmax,
                                    cbar_kwargs={'label':'Temperature, $^oC$'})
            ax.set_title(label)
            ax.set_ylim([1,6500])
            ax.set_yscale('log')
            ax.invert_yaxis()
            plt.ylabel('Depth, m')
            plt.xlabel('Latitude')

    def plot_KE_spectrum(self, exps, labels=None, colors=None):
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
                    KE = self['unparameterized'].__getattribute__(f'geoKE_{region}_obs')
                else:
                    KE = self[exp].__getattribute__(f'geoKE_{region}')

                KE.plot(label=labels[j_exp], color=colors[j_exp], xscale='log', yscale='log', lw=lw[j_exp])
                plt.title(region)

        for j_region in range(4):
            plt.subplot(2,2,j_region+1)
            plt.xlabel(r'wavenumber, $[\mathrm{km}^{-1}]$')
            plt.ylabel(r'Geostrophic KE spectrum $[\mathrm{m}^3/\mathrm{s}^2]$')
            plt.xlim([None,2e-4])
            plt.ylim([1e-2,1e+4])
            plt.xticks([1e-5, 1e-4], ['10$^{-2}$', '10$^{-1}$'])

            k=np.array([2e-5,1e-4])
            plt.plot(k,7e+3*(k/2e-5)**(-3), ls='--', color='k')
            plt.text(6e-5,5e+2,'$k^{-3}$')

        plt.legend(bbox_to_anchor=(1,1))
        plt.tight_layout()