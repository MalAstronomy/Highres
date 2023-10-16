import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import sys
sys.path.insert(0, '/home/mvasist/Highres/simulations/') #WISEJ1738/sbi' WISEJ1738.sbi.
from spectra_simulator import make_pt, SpectrumMaker
from parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal

from adding_legends import legends

from lampe.plots import LinearAlphaColormap

sim_res = 2e5
dlam = 2.350/sim_res
wavelengths = np.arange(2.320, 2.371, dlam)

# Simulator
simulator = SpectrumMaker(wavelengths, param_set)

def levels_and_creds(creds, alpha):
    creds = np.sort(np.asarray(creds))[::-1]
    creds = np.append(creds, 0)
    levels = (creds - creds.min()) / (creds.max() - creds.min())
    levels = (levels[:-1] + levels[1:]) / 2
    
    return levels, creds

def PT_plot(fig, ax, theta, theta_nom= None, color = 'steelblue', creds= [0.997, 0.955, 0.683], alpha = [0, 0.9], invert= False):
    
    levels, creds = levels_and_creds(creds, alpha)
    cmap= LinearAlphaColormap(color, levels=creds, alpha=alpha)
    
    pressures = simulator.atmosphere.press / 1e6
    temperatures = []
    for th in theta :
        values_actual = deNormVal(th.numpy(), param_list)
        params = param_set.param_dict(values_actual)
        temperatures.append(make_pt(params , pressures))  

    # temperatures = make_pt(params , pressures) 
    # print(temperatures)

    for q, l in zip(creds[:-1], levels):
        left, right = np.quantile(temperatures, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax.fill_betweenx(pressures, left, right, color= cmap(l), linewidth=0)

    # lines = ax.plot([], [], color='black', label='Nominal P-T profile')
    handles, texts = legends(ax, alpha=alpha) #[0.15,0.75]

    if theta_nom != None:
        ax.plot(make_pt(theta_nom, pressures), pressures, color='black', label= 'Synthetic observation')

    # ax.set_xticklabels(np.arange(500,4000,500),fontsize=8)
    # ax.set_yticklabels(np.arange(1e-2, 1e1, np.log10(0.1)),fontsize=8)
    ax.set_xlabel(r'Temperature $(\mathrm{K})$', fontsize= 10)
    ax.set_ylabel(r'Pressure $(\mathrm{bar})$', fontsize= 10)
    ax.set_xlim(0, 2000)
#     ax.set_ylim(1e-2, 1e1)
    ax.set_yscale('log')
    ax.legend(handles, texts, prop={'size': 8})
    if invert :
        ax.invert_yaxis()
#     ax.grid()
    
    return fig 