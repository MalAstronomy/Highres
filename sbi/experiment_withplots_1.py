import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import wandb

from dawgz import job, after, ensure, schedule
from itertools import chain, islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from typing import *
import pandas as pd

from lampe.data import H5Dataset
from zuko.distributions import BoxUniform
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from zuko.flows import NAF, NSF, MAF, NCSF, SOSPF, UNAF, CNF 
from lampe.plots import nice_rc, corner, coverage_plot, mark_point
from lampe.utils import GDStep

import sys
sys.path.insert(0, '/home/mvasist/Highres/simulations/')
from spectra_simulator import make_pt, SpectrumMaker
from DataProcuring import Data 
from ProcessingSpec import ProcessSpec
from parameter import *
from parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal

print('here i am')

# sys.path.insert(0, '/home/mvasist/Highres/sbi/added_scripts/')
from added_scripts.corner_modified import *
from added_scripts.pt_plotting import *


# from ees import Simulator, LOWER, UPPER, LABELS, pt_profile
LABELS, LOWER, UPPER = zip(*[
[                  r'$FeH$',  -1.5, 1.5],   # temp_node_9
[                  r'$CO$',  0.1, 1.6],  # CO_mol_scale
[                  r'$\log g$',   2.5, 5.5],          # log g
[                  r'$Tint$',  300,   3500],   # temp_node_5
[                  r'$T1$',  300,   3500],      # T_bottom
[                  r'$T2$',  300,   3500],   # temp_node_1
[                  r'$T3$',  300,   3500],   # temp_node_2
[                  r'$alpha$',  1.0, 2.0],   # temp_node_4
[                  r'$log_delta$', 3.0, 8.0],   # temp_node_3
[                  r'$log_Pquench$', -6.0, 3.0],   # temp_node_6
[                  r'$logFe$',  -2.3, 1.0], # CH4_mol_scale
[                  r'$fsed$',  0.0, 10.0],   # temp_node_8
[                  r'$logKzz$',  5.0, 13.0], # H2O_mol_scale \_mol\_scale
[                  r'$sigmalnorm$',  1.05, 3.0], # C2O_mol_scale
[                  r'$log\_iso\_rat$',  -11.0, -1.0],   # temp_node_7
[                  r'$R\_P$', 0.8, 2.0],             # R_P / R_Jupyter
[                  r'$rv$',  10.0, 30.0], # NH3_mol_scale 20, 35
[                  r'$vsini$', 0.0, 50 ], # H2S_mol_scale 10.0, 30.0
[                  r'$limb\_dark$',  0.0, 1.0], # PH3_mol_scale
[                  r'$b$',  1, 20.0], # PH3_mol_scale

])

scratch = os.environ['SCRATCH']
datapath = Path(scratch) / 'highres-sbi/data_nic5'
savepath = Path(scratch) / 'highres-sbi/runs/sweep_lognormnoise'

processing = ProcessSpec()
d = Data()
sim = SpectrumMaker(wavelengths=d.model_wavelengths, param_set=param_set, lbl_opacity_sampling=2)


def simulator(theta):
    values_actual = theta[:-4].numpy()
    values_ext_actual = theta[-4:].numpy()
    spectrum = sim(values_actual)
    spec = np.vstack((np.array(spectrum), d.model_wavelengths))
    th, x = processing(torch.Tensor([values_actual]), torch.Tensor(spec), sample= False, \
                       values_ext_actual= torch.Tensor([values_ext_actual]))    
    return x.squeeze()


## Loading from a model to plot
CONFIGS = {
    'embedding': ['shallow'],
    'flow': ['MAF'],  #, 'NCSF', 'SOSPF', 'UNAF', 'CNF'], #'NAF', 
    'transforms': [3], #, 7], #3, 
    # 'signal': [16, 32],  # not important- the autoregression network output , 32
    'hidden_features': [512], # hidden layers of the autoregression network , 256, 
    'hidden_features_no' : [5], 
    'activation': [nn.ELU], #, nn.ReLU],
    'optimizer': ['AdamW'],
    'init_lr':  [1e-3], #[5e-4, 1e-5]
    'weight_decay': [1e-4], #[1e-4], #
    'scheduler': ['ReduceLROnPlateau'], #, 'CosineAnnealingLR'],
    'min_lr': [1e-5], # 1e-6
    'patience': [16], #8
    'epochs': [350],
    'stop_criterion': ['early'], #, 'late'],
    'batch_size':  [256],
    'spectral_length' : [6144], #[1536, 3072, 6144]
    'factor' : [0.3], 
    'noise_scaling' : [2], 
    'noise' : ['lognormaldist']
    # 'SOSF_degree' : [2,3,4],
    # 'SOSF_poly' : [2,4,6],
}


@job(array=1, cpus=2, gpus=1, ram='128GB', time='10-00:00:00')
def experiment(index: int) -> None:
    # Config
    config = {
        key: random.choice(values)
        for key, values in CONFIGS.items()
    }

    def noisy(x, b= None): #50 is 10% of the median of the means of spectra in the training set.
        bs = x.size()[0]
        data_uncertainty = Data().err * Data().flux_scaling
        data_uncertainty = torch.from_numpy(data_uncertainty).cuda()

        if b == None: 
            if config['noise'] == 'uniformdist' :
                b = 1  + torch.rand(bs) * (10-1)
                b = torch.unsqueeze(b,1)
            elif config['noise'] == 'lognormaldist' :
                m = torch.distributions.log_normal.LogNormal(torch.tensor([1.5]), torch.tensor([0.5]))
                b = m.sample([bs])
        else: 
            b = torch.unsqueeze(b, 1)
            print(b.size())

        x = x + torch.mul(data_uncertainty * b.cuda() , torch.randn_like(x))
        return x, b

    class NPEWithEmbedding(nn.Module):
        def __init__(self):
            super().__init__()

            # Estimator
            if config['embedding'] == 'shallow':
                self.embedding = ResMLP(6144, 64, hidden_features=[512] * 2 + [256] * 3 + [128] * 5, activation= nn.ELU)
            else:
                self.embedding = ResMLP(6144, 128, hidden_features=[512] * 3 + [256] * 5 + [128] * 7, activation= nn.ELU)
            
            if config['flow'] == 'MAF':
                self.npe = NPE(
                    20, self.embedding.out_features,
                    # moments=((l + u) / 2, (l - u) / 2),
                    transforms=config['transforms'],
                    build=MAF,
                    # bins=config['signal'],
                    hidden_features=[config['hidden_features']] * config['hidden_features_no'],
                    activation=config['activation'],
                )

        # def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        def forward(self, theta, x): # -> Tensor:
            y = self.embedding(x)
            return self.npe(theta, y)

        # def flow(self, x: Tensor):  # -> Distribution
        def flow(self, x):  # -> Distribution
            out = self.npe.flow(self.embedding(x)) #.to(torch.double)) #
            return out

    if (config['flow'] == 'SOSPF') | (config['flow'] == 'UNAF'):
        estimator = NPEWithEmbedding().cuda()
    
    estimator = NPEWithEmbedding().double().cuda()

    def pipeout(theta , x):
        theta, x = theta.cuda(), x.cuda()
        x , b = noisy(x)
        theta = torch.hstack((theta, b.cuda()))
        return theta, x

    m = 'peachy-feather-81' #'comfy-dawn-59'
    epoch = config['epochs']
    runpath = savepath / m
    runpath.mkdir(parents=True, exist_ok=True)
    
    estimator = NPEWithEmbedding().double()
    states = torch.load(runpath / ('states_' + str(epoch) + '.pth'), map_location='cpu')
    estimator.load_state_dict(states['estimator'])
    estimator.cuda().eval()

############################################################
        
    savepath_plots = runpath  / ('plots_sim_b_' + str(epoch))
    savepath_plots.mkdir(parents=True, exist_ok=True)

    def thetascalebackup(theta):
        theta[:-1] =  torch.Tensor(LOWER[:-1]) + theta[:-1] * (torch.Tensor(UPPER[:-1]) - torch.Tensor(LOWER[:-1]))
        return theta

# #     ## Corner    
    obs = torch.Tensor(np.loadtxt('/home/mvasist/Highres/observation/simulated_obs/x_sim_b.npy'))
    theta_star = torch.Tensor(np.loadtxt('/home/mvasist/Highres/observation/simulated_obs/theta_sim_b.npy'))
    obs = torch.unsqueeze(obs[0], 0)
    theta_star = torch.unsqueeze(theta_star, 0)
    theta_star, x_star = pipeout(theta_star, obs)  #[1,6144] dimensions [1,20]
    theta_star, x_star = theta_star[0], x_star[0]
    
    #Then, to reload:
    df_theta = pd.read_csv( savepath_plots / 'theta.csv')
    theta = df_theta.values
    theta = torch.from_numpy(theta)

#     ## NumPy
    # def filter_limbdark_mask(theta):
    #     mask = theta[:,-2]<0
    #     mask += theta[:,-2]>1
    #     return mask 
    # def filter_logdelta_mask(theta):
    #     mask = theta[:,8]>8
    #     return mask 

    # mask1 = filter_limbdark_mask(theta)
    # theta_filterLD = theta[~mask1]
    # mask2 = filter_logdelta_mask(theta_filterLD)
    # theta_filterLD = theta_filterLD[~mask2]

    ### PT profile
    # fig, ax = plt.subplots(figsize=(4,4))

    ##sim PT
    # pressures = sim.atmosphere.press / 1e6
    # val_act = deNormVal(theta_star.cpu().numpy(), param_list)
    # params = param_set.param_dict(val_act)
    # temp= make_pt(params , pressures)
    # ax.plot(temp, pressures, color = 'black')  ##sim
    ##sim PT

    # print(theta_filterLD[:2**8, :-1].size(), theta_filterLD)
    # fig_pt = PT_plot(fig, ax, theta_filterLD[:2**8, :-1], invert = True) #, self.theta_star)
    # fig_pt = PT_plot(fig_pt, ax, self.theta_paul[:2**8], invert = True, color = 'orange') #, theta_star)
    # fig_pt.savefig(self.savepath_plots / 'pt_profile_Paul_unregPTwithb_24Apr2023.pdf')
    # fig_pt.savefig(savepath_plots / 'pt_profile.pdf')



if __name__ == '__main__':
    schedule(
        experiment,
        name='EES sweep',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate HighResear',
            'export WANDB_SILENT=true',
        ]
    )
