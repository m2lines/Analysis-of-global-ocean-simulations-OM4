import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import imageio

yellow = [0.9290, 0.6940, 0.1250]
violet = [0.4940, 0.1840, 0.5560]
lightblue = [0.3010, 0.7450, 0.9330]
    
def create_animation_ffmpeg(fun, idx, filename='my-video.mp4', dpi=200, FPS=18, resolution=None):
    folder = '.ffmpeg/'+filename.split('.')[0]
    from time import time
    def create_snapshots():
        t0 = time()
        for frame, i in enumerate(idx):
            fun(i)
            plt.savefig(f'{folder}/frame-{frame}.png', dpi=dpi, bbox_inches='tight')
            plt.close()
            nframes = len(idx)
            remaining_frames = nframes - frame
            ETA = (time()-t0) / (frame+1) * remaining_frames
            print(f'Frame {frame}/{nframes} is created, ETA: {ETA}', end='\r')
            
    if os.path.exists(folder):
        if os.path.exists(folder+'/frame-0.png'):
            print(f'Frames already exists in folder {folder}')
            x = input('Do you want to update snapshots?: [y/n]')
            if x=='y':
                create_snapshots()
            elif x=='n':
                print('Frames are not updated\n')
    else:
        os.system(f'mkdir -p {folder}')
        create_snapshots()

    if resolution is None:
        resolution = list(Image.open(f'{folder}/frame-0.png').size)
        for i in [0,1]:
            resolution[i] = (resolution[i]//2)*2
        print(f'Native resolution of snapshots is used: {resolution[0]}x{resolution[1]}\n')
    else:
        for i in [0,1]:
            resolution[i] = (resolution[i]//2)*2
        print(f'Resolution is set to {resolution[0]}x{resolution[1]}\n')

    print(f'Animation {filename} at FPS={FPS} will last for {round(len(idx)/FPS,1)} seconds. The frames are saved to \n{folder}\n')
    ffmpeg_command = f'/home/ctrsp-2024/pp2681/ffmpeg-git-20240629-amd64-static/ffmpeg -y -framerate {FPS} -i {folder}/frame-%d.png -s:v {resolution[0]}x{resolution[1]} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {filename}'
    print('Running the command:')
    print(f'cd {os.getcwd()}; {ffmpeg_command}')
    try:
        os.system(ffmpeg_command)
    except:
        print('Something went wrong. Try to run the following command in the terminal:\n')
        print('Optional: module load ffmpeg/4.2.4')
        print(f'cd {os.getcwd()}; {ffmpeg_command}')
    
def default_rcParams(kw={}):
    '''
    Also matplotlib.rcParamsDefault contains the default values,
    but:
    - backend is changed
    - without plotting something as initialization,
    inline does not work
    '''
    plt.plot()
    plt.close()
    rcParams = matplotlib.rcParamsDefault.copy()
    
    # We do not change backend because it can break
    # inlining; Also, 'backend' key is broken and 
    # we cannot use pop method
    for key, val in rcParams.items():
        if key != 'backend':
            rcParams[key] = val

    matplotlib.rcParams.update({
        #'font.family': 'DejaVuSans_Condensed',
        'mathtext.fontset': 'cm',

        'figure.figsize': (4, 4),

        'figure.subplot.wspace': 0.3,
        
        'font.size': 14,
        #'axes.labelsize': 10,
        #'axes.titlesize': 12,
        #'xtick.labelsize': 10,
        #'ytick.labelsize': 10,
        #'legend.fontsize': 10,

        'axes.formatter.limits': (-2,3),
        'axes.formatter.use_mathtext': True,
        'axes.labelpad': 0,
        'axes.titlelocation' : 'center',
        
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    matplotlib.rcParams.update(**kw)

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str
    
def set_letters(x=-0.2, y=1.05, fontsize=11, letters=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], color='k'):
    fig = plt.gcf()
    axes = fig.axes
    j = 0
    for ax in axes:
        if hasattr(ax, 'collections'):
            if len(ax.collections) > 0:
                collection = ax.collections[0]
            else:
                collection = ax.collections
            if isinstance(collection, matplotlib.collections.LineCollection):
                print('Colorbar-like object skipped')
            else:
                try:
                    ax.text(x,y,f'({letters[j]})', transform = ax.transAxes, fontweight='bold', fontsize=fontsize, color=color)
                except:
                    print('Cannot set letter', letters[j])
                j += 1
        