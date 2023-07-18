#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import astropy.units as u
import wandb
#from specutil import *
import astropy.units as u
import astropy.constants as const
from PyAstronomy.pyasl import fastRotBroad

from dawgz import job, schedule
from itertools import islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm

from lampe.data import H5Dataset
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from lampe.utils import GDStep

from zuko.flows import NAF, MAF
from zuko.distributions import BoxUniform
from generate import param_set
from parameter import *

from DataProcuring import Data 
from ProcessingSpec import ProcessSpec
# from Embedding.CNNwithAttention import CNNwithAttention
from Embedding.MHA import MultiHeadAttentionwithMLP, GPTLanguageModel
# from Embedding.MLP import MLP 
from Embedding.CausalConv1D import CausalConv1d, CausalConvLayers

import GISIC
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from ees import Simulator, LOWER, UPPER
# from Embedding.SelfAttention import SelfAttention
# import Embedding.CNN as CNN

scratch = os.environ.get('SCRATCH', '')
# scratch = '/users/ricolandman/Research_data/npe_crires/'
datapath = Path(scratch) / 'highres-sbi/data_fulltheta'
savepath = Path(scratch) / 'highres-sbi/runs'

d = Data()

class SoftClip(nn.Module):
    def __init__(self, bound: float = 1.0):
        super().__init__()

        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))

class stacking(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.hstack((x[:, 0, :], x[:, 1, :]))

class NPEWithEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Sequential(
            SoftClip(100.0),
            # CNNwithAttention(2, 128),

            # MultiHeadAttentionwithMLP(128, 4, 8, 377),
            
            # stacking()

            # GPTLanguageModel(128, 4, 8, 377), #n_embedding, n_head, n_blocks, block_size

            # ResMLP(
            #     6144 , 64, hidden_features=[512] * 2 + [256] * 3 + [128] * 5, 
            #     activation=nn.ELU,
            # ),

            CausalConvLayers(1, 4, 32, 2, 32),  #in_channels, out_channels, MM, stride, kernel_size
            nn.Flatten(),
            )
        # self.flatten()

        self.npe = NPE(
            19, 200, 
            #moments=((l + u) / 2, (u - l) / 2),43q  r7890q q=-09875            transforms=3,
            build=NAF,
            hidden_features=[512] * 5,
            activation=nn.ELU,
            # dropout = 0.2,
        ).to(torch.float64)
        

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        print(x.is_cuda,theta.is_cuda)
        y = self.embedding(x)
        print(y.is_cuda)
        # print(y.size())
        if torch.isnan(y).sum()>0:
             print('NaNs in embedding')
        return self.npe(theta.to(torch.double), y)

    def flow(self, x: Tensor):  # -> Distribution
        # print(x.size())
        print(x.is_cuda)
        out = self.npe.flow(self.embedding(x).to(torch.double)) 
        print(out.is_cuda)
#         print(type(out))
#         if np.any(np.isnan(out.detach().cpu().numpy())):
#              print('NaNs in flow')
        return out
    

class BNPELoss(nn.Module):
    def __init__(self, estimator, prior, lmbda=100.0):
        super().__init__()
        self.estimator = estimator
        self.prior = prior
        self.lmbda = lmbda
    def forward(self, theta, x):
        theta_prime = torch.roll(theta, 1, dims=0)
        log_p, log_p_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )
        l0 = -log_p.mean()
        lb = (torch.sigmoid(log_p - self.prior.log_prob(theta)) + torch.sigmoid(log_p_prime - self.prior.log_prob(theta_prime)) - 1).mean().square()
        return l0 + self.lmbda * lb
                    
                    
def noisy(theta, x ):
    data_uncertainty = Data().err * Data().flux_scaling*10
    # data_uncertainty = Data().err /250
    # x[:,0, :] = x[:,0, :] + torch.from_numpy(data_uncertainty) * torch.randn(x[:,0,:].size())
    x = x + torch.from_numpy(data_uncertainty) * torch.randn_like(x)
    # theta = theta.numpy()
    # print(theta.size(), x.size())
    return theta, x
   

@job(array=1, cpus=2, gpus=1, ram='32GB', time='10-00:00:00')
def train(i: int):

    config_dict = {
        
                'embedding': 'CausalConv(1, 4, 32, 2, 32)', #'MAH_nopositional-512, 8, 8, 512',  #shallow = [2,3,5], deep = [3,5,7] ResMLP[2,3,5]
                'embedding features' : 'in_channels, out_channels, MM, stride, kernel_size', 
                'embedding_output_len' : '200', 
                'NPE_input_len': '200' ,
                'flow': 'NAF',
                'transforms': 3, 
                'hidden_features': 512, # hidden layers of the autoregression network
                'activation': 'ELU',
                'optimizer': 'AdamW',
                'init_lr': 1e-3,
                'weight_decay': 1e-2,
                'scheduler': 'ReduceLROnPlateau',
                'min_lr': 1e-6,
                'patience': 32,
                'epochs': 2000,
                'stop_criterion': 'early', 
                'batch_size': 16,
                'gradient_steps_train': 1024, 
                'gradient_steps_valid': 256
             } 

    # Run
    run = wandb.init(project='highres-CausalConv1D',  config = config_dict)

    # Data
    trainset = H5Dataset(datapath / 'train.h5', batch_size=16, shuffle=True)
    validset = H5Dataset(datapath / 'valid.h5', batch_size=16, shuffle=True)

    # Training
#     process = Processing()
    estimator = NPEWithEmbedding().double().cuda()
    prior = BoxUniform(torch.tensor(param_set.lower).cuda(), torch.tensor(param_set.upper).cuda())
    loss = NPELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=1e-4, weight_decay=1e-2)
    step = GDStep(optimizer, clip=1.0)
    scheduler = sched.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        min_lr=1e-7,
        patience=32,
        threshold=1e-2,
        threshold_mode='abs',
    )

    def pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = noisy(theta,x)
        # v = torch.stack([torch.from_numpy(np.asarray(GISIC.normalize(d.data_wavelengths_norm, x[i].numpy(), sigma=30))) for i in range(len(x))]) #B, 3, 6144 , wavelengths, flux, continuum
        # theta, x = theta.cuda(), v[:,1,:].cuda()

        # theta, x = torch.from_numpy(theta).cuda(), torch.from_numpy(x).cuda()
        # x = torch.hstack((x[:, 0, :], x[:,1,:]))
        # print(x.size())
        # x = torch.permute(x, (0, 2, 1))
        
        theta, x = theta.cuda(), x.cuda()
        return loss(theta, x)

    for epoch in tqdm(range(2001), unit='epoch'):
        estimator.train()
        start = time.time()

        losses = torch.stack([
            step(pipe(theta, x[:,0])) #16,6144
            for theta, x in islice(trainset, 1024)
        ]).cpu().numpy()
        

        end = time.time()
        estimator.eval()

        with torch.no_grad():
            losses_val = torch.stack([
                pipe(theta, x[:,0])
                for theta, x in islice(validset, 256)
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

        runpath = savepath / run.name
        runpath.mkdir(parents=True, exist_ok=True)

        if epoch % 50 ==0 : 
                torch.save({
                'estimator': estimator.state_dict(),
                'optimizer': optimizer.state_dict(),
            },  runpath / f'states_{epoch}.pth')
                
        if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

    run.finish()


# train()

if __name__ == '__main__':
    schedule(
        train, #coverageplot, cornerplot,
        name='Training',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate HighResear',
            'export WANDB_SILENT=true',
        ]
    )