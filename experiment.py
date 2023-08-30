#!/usr/bin/env python

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
from zuko.flows import NAF, NSF
from lampe.plots import nice_rc, corner, coverage_plot
from lampe.utils import GDStep

from DataProcuring import Data 
# from spectra_simulator import Simulator, LOWER, UPPER
# from AverageEstimator import avgestimator
# from corner_modified import *
# from pt_plotting import *


# from ees import Simulator, LOWER, UPPER, LABELS, pt_profile
LABELS, LOWER, UPPER = zip(*[
[                  r'$T1$',  300,   3500],      # T_bottom
[                  r'$T2$',  300,   3500],   # temp_node_1
[                  r'$T3$',  300,   3500],   # temp_node_2
[                  r'$log_delta$', 3.0, 8.0],   # temp_node_3
[                  r'$alpha$',  1.0, 2.0],   # temp_node_4
[                  r'$Tint$',  300,   3500],   # temp_node_5
[                  r'$FeH$',  -1.5, 1.5],   # temp_node_9
[                  r'$CO$',  0.1, 1.6],  # CO_mol_scale
[                  r'$\log g$',   2.5, 5.5],          # log g
[                  r'$log_Pquench$', -6.0, 3.0],   # temp_node_6
[                  r'$log_iso_rat$',  -11.0, -1.0],   # temp_node_7
[                  r'$fsed$',  0.0, 10.0],   # temp_node_8
[                  r'$logKzz$',  5.0, 13.0], # H2O_mol_scale \_mol\_scale
[                  r'$sigmalnorm$',  1.05, 3.0], # C2O_mol_scale
[                  r'$logFe$',  -2.3, 1.0], # CH4_mol_scale
[                  r'$R_P$', 0.8, 2.0],             # R_P / R_Jupyter
[                  r'$rv$',  20.0, 35.0], # NH3_mol_scale
[                  r'$limb_dark$',  0.0, 1.0], # PH3_mol_scale
[                  r'$vsini$',  10.0, 30.0], # H2S_mol_scale
])

scratch = os.environ['SCRATCH']
datapath = Path(scratch) / 'highres-sbi/data_fulltheta'
savepath = Path(scratch) / 'highres-sbi/runs/sweep_more'


CONFIGS = {
    'embedding': ['shallow', 'deep'],
    'flow': ['NSF'], #'NAF', 
    'transforms': [5], #3, 
    'signal': [16],  # not important- the autoregression network output , 32
    'hidden_features': [512], # hidden layers of the autoregression network , 256, 
    'activation': [nn.ELU],
    'optimizer': ['AdamW'],
    'init_lr': [5e-4, 1e-5] , # [1e-3, 5e-4, 1e-4, 1e-5],
    'weight_decay': [1e-4], #[0, 1e-4, 1e-3, 1e-2],
    'scheduler': ['ReduceLROnPlateau'], #, 'CosineAnnealingLR'],
    'min_lr': [1e-5], # 1e-6
    'patience': [16, 32], #8
    'epochs': [1024],
    'stop_criterion': ['early'], #, 'late'],
    'batch_size':  [512, 1024], #[256, 512, 1024, 2048],
    'spectral_length' : [6144], #[1536, 3072, 6144]
}


@job(array=2**4, cpus=2, gpus=1, ram='64GB', time='10-00:00:00')
def experiment(index: int) -> None:
    # Config
    config = {
        key: random.choice(values)
        for key, values in CONFIGS.items()
    }
    
    run = wandb.init(project='highres--sweep-moree', config=config)
    
    # Simulator
    # simulator = Simulator(noisy=False)
    
    def noisy(x: Tensor) -> Tensor:
        # return x + simulator.sigma * torch.randn_like(x)
        data_uncertainty = Data().err * Data().flux_scaling*50 #this is 10% of the median of the means of spectra in the training set.
        # print(x.is_cuda, (torch.from_numpy(data_uncertainty)).is_cuda, torch.randn_like(x).is_cuda)
        x = x + torch.from_numpy(data_uncertainty).cuda() * torch.randn_like(x)
        return x

    l, u = torch.tensor(LOWER), torch.tensor(UPPER)
    
    # Estimator
    if config['embedding'] == 'shallow':
        embedding = ResMLP(6144, 64, hidden_features=[512] * 2 + [256] * 3 + [128] * 5, activation= nn.ELU)
    else:
        embedding = ResMLP(6144, 128, hidden_features=[512] * 3 + [256] * 5 + [128] * 7, activation= nn.ELU)

    if config['flow'] == 'NSF':
        estimator = NPE(
            19, embedding.out_features,
            # moments=((l + u) / 2, (l - u) / 2),
            transforms=config['transforms'],
            build=NSF,
            bins=config['signal'],
            hidden_features=[config['hidden_features']] * 5,
            activation=config['activation'],
        )
    else:
        estimator = NPE(
            19, embedding.out_features,
            # moments=((l + u) / 2, (l - u) / 2),
            transforms=config['transforms'],
            build=NAF,
            signal=config['signal'],
            hidden_features=[config['hidden_features']] * 5,
            activation=config['activation'],
        )
    
    # print(embedding, flush=True)
    # print(estimator, flush=True)

    embedding.double().cuda(), estimator.double().cuda()

    # Optimizer
    loss = NPELoss(estimator)
    optimizer = optim.AdamW(chain(embedding.parameters(), estimator.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])
    scheduler = sched.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=config['min_lr'], patience=config['patience'], threshold=1e-2, threshold_mode='abs')
    step = GDStep(optimizer, clip=1)

    # Data
    trainset = H5Dataset(datapath / 'train.h5', batch_size=config['batch_size'], shuffle=True)
    validset = H5Dataset(datapath / 'valid.h5', batch_size=config['batch_size'], shuffle=True)

    # Training
    def pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        x = noisy(x)
        x = embedding(x)
        return loss(theta, x)

    for epoch in tqdm(range(config['epochs']), unit='epoch'):
        embedding.train(), estimator.train()
        
        start = time.time()
        losses = torch.stack([
            step(pipe(theta.float(), x[:,0].float()))
            for theta, x in islice(trainset, 2**8)
        ]).cpu().numpy()
        end = time.time()
        
        embedding.eval(), estimator.eval()
        
        with torch.no_grad():            
            losses_val = torch.stack([
                pipe(theta.float(), x[:,0].float())
                for theta, x in islice(validset, 2**5)
            ]).cpu().numpy()

        run.log({
            'lr': optimizer.param_groups[0]['lr'],
            'loss': np.nanmean(losses),
            'loss_val': np.nanmean(losses_val),
            'nans': np.isnan(losses).mean(),
            'nans_val': np.isnan(losses_val).mean(),
            'speed': len(losses) / (end - start),
        })

        scheduler.step(np.nanmean(losses_val))

        runpath = savepath / f'{run.name}' #_{run.id}'
        runpath.mkdir(parents=True, exist_ok=True)

        if epoch % 50 ==0 : 
                torch.save({
                        'embedding': embedding.state_dict(),
                        'estimator': estimator.state_dict(),
                        'optimizer': optimizer.state_dict(),
            },  runpath / f'states_{epoch}.pth')

        if config['stop_criterion'] == 'early' and optimizer.param_groups[0]['lr'] <= config['min_lr']:
            break


    # Evaluation
    plt.rcParams.update(nice_rc(latex=True))

    ## Coverage
    testset = H5Dataset(datapath / 'test.h5', batch_size=2**4)
    d = Data()
    ranks = []

    with torch.no_grad():
        for theta, x in tqdm(islice(testset, 2**8)):
            theta, x = theta.cuda(), x.cuda()
            x = x[:,0]
            x = noisy(x)
            x = embedding(x)

            posterior = estimator.flow(x)
            samples = posterior.sample((2**10,))
            log_p = posterior.log_prob(theta)
            log_p_samples = posterior.log_prob(samples)

            ranks.append((log_p_samples < log_p).float().mean(dim=0).cpu())

    ranks = torch.cat(ranks)
    ecdf_fig = coverage_plot(ranks, coverages = np.linspace(0, 1, 256))
    ecdf_fig.savefig(runpath / 'coverage.pdf')

    ## Corner
    # dataset = H5Dataset(datapath / 'event.h5')
    # theta_star, x_star = dataset[1]
    x_star = d.flux*d.flux_scaling

    with torch.no_grad():
        x = embedding(x_star.cuda())
        theta = torch.cat([
            estimator.sample(x, (2**14,)).cpu()
            for _ in range(2**6)
        ])
    
    theta_numpy = theta.double().numpy() #convert to Numpy array
    df_theta = pd.DataFrame(theta_numpy) #convert to a dataframe
    df_theta.to_csv(runpath/ 'theta.csv' ,index=False) #save to file

    corner_fig = corner(
        theta,
        smooth=2,
        bounds=(LOWER, UPPER),
        labels=LABELS,
        legend=r'$p_{\phi}(\theta | x^*)$',
        # markers=[theta_star],
        figsize=(12, 12),
    )
    corner_fig.savefig(runpath / 'corner.pdf')
    
    # ## NumPy
    # theta_star, x_star = theta_star.double().numpy(), x_star.double().numpy()
    # theta = theta[:2**8].double().numpy()

    # ## PT profile
    # pt_fig, ax = plt.subplots(figsize=(4.8, 4.8))

    # pressures = simulator.atmosphere.press / 1e6
    # temperatures = pt_profile(theta, pressures)

    # for q in [0.997, 0.95, 0.68]:
    #     left, right = np.quantile(temperatures, [0.5 - q / 2, 0.5 + q / 2], axis=0)
    #     ax.fill_betweenx(pressures, left, right, color='C0', alpha=0.25, linewidth=0)

    # ax.plot(pt_profile(theta_star, pressures), pressures, color='k', linestyle='--')

    # ax.set_xlabel(r'Temperature $[\mathrm{K}]$')
    # ax.set_xlim(0, 4000)
    # ax.set_ylabel(r'Pressure $[\mathrm{bar}]$')
    # ax.set_ylim(1e-2, 1e1)
    # ax.set_yscale('log')
    # ax.invert_yaxis()
    # ax.grid()

    # pt_fig.savefig(runpath / 'pt_profile.pdf')

    # ## Residuals
    # res_fig, ax = plt.subplots(figsize=(4.8, 4.8))

    # x = np.stack([simulator(t) for t in tqdm(theta)])
    # mask = ~np.isnan(x).any(axis=-1)
    # theta, x = theta[mask], x[mask]

    # wlength = np.linspace(0.95, 2.45, x.shape[-1])
    
    # for q in [0.997, 0.95, 0.68]:
    #     lower, upper = np.quantile(x, [0.5 - q / 2, 0.5 + q / 2], axis=0)
    #     ax.fill_between(wlength, lower, upper, color='C0', alpha=0.25, linewidth=0)

    # ax.plot(wlength, x_star, color='k', linestyle=':')
    
    # ax.set_xlabel(r'Wavelength $[\mu\mathrm{m}]$')
    # ax.set_ylabel(r'Flux $[\mathrm{W} \, \mathrm{m}^{-2} \, \mu\mathrm{m}^{-1}]$')
    # ax.grid()

    # res_fig.savefig(runpath / 'residuals.pdf')

    run.log({
        'coverage': wandb.Image(ecdf_fig),
        'corner': wandb.Image(corner_fig),
        # 'pt_profile': wandb.Image(pt_fig),
        # 'res_fig': wandb.Image(res_fig),
    })
    run.finish()


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
