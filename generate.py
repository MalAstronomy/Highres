#!/usr/bin/env python

import numpy as np
import os
import torch
from multiprocessing import Pool

from dawgz import job, after, ensure, schedule
from itertools import starmap
from pathlib import Path
from typing import *

from lampe.data import JointLoader, H5Dataset
#from lampe.distributions import BoxUniform

#from ees import Simulator, LOWER, UPPER
from parameter import *
# from spectra_simulator import *

from ProcessingSpec import ProcessSpec
from DataProcuring import Data

from torch.utils.data import DataLoader, Dataset, IterableDataset

import GISIC
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

process = ProcessSpec()

with_isotope = True
include_clouds = True
#Define parameters
FeH = Parameter('FEH', uniform_prior(-1.5, 1.5))
CO = Parameter('C_O', uniform_prior(0.1, 1.6))
#log_g = Parameter('log_g', gaussian_prior(log_g_mu, log_g_sigma))
log_g = Parameter('log_g', uniform_prior(2.5, 5.5))
T_int = Parameter('T_int', uniform_prior(300, 3500))

# Change parameter definition to temperature ratios!!
T1 = Parameter('T1', uniform_prior(300, 3500))
T2 = Parameter('T2', uniform_prior(300, 3500))
T3 = Parameter('T3', uniform_prior(300, 3500))
#T1 = Parameter('T1', lambda x : (x*0.5 + 0.5)*T_int.value)
#T2 = Parameter('T2', lambda x : (x*0.5 + 0.5)*T1.value)
#T3 = Parameter('T3', lambda x : (x*0.5 + 0.5)*T2.value)
alpha = Parameter('alpha', uniform_prior(1, 2))
log_delta = Parameter('log_delta', uniform_prior(3, 8))
#P_phot = Parameter('P_phot', uniform_prior(-3, 2))
log_Pquench = Parameter('log_Pquench', uniform_prior(-6, 3))

#Non-pRT parameters that we skip here
limb_dark = Parameter('limb_dark', uniform_prior(0,1))
vsini = Parameter('vsini', uniform_prior(10, 30))
rv = Parameter('rv', uniform_prior(20, 35))
radius = Parameter('radius', uniform_prior(0.8, 2.0))

param_set = ParameterSet([FeH, CO, log_g, T_int, T1, T2, T3, alpha, log_delta, log_Pquench])


if include_clouds:
    MgSiO3 = Parameter('log_MgSiO3', uniform_prior(-2.3, 1))
    Fe = Parameter('log_Fe', uniform_prior(-2.3, 1))
    fsed = Parameter('fsed', uniform_prior(0,10))
    Kzz = Parameter('log_Kzz', uniform_prior(5,13))
    sigma_lnorm= Parameter('sigma_lnorm', uniform_prior(1.05, 3))
    param_set.add_params([Fe, fsed, Kzz, sigma_lnorm])

iso_rat = Parameter('log_iso_rat', uniform_prior(-11, -1))
if with_isotope:
    param_set.add_params(iso_rat)

ndim = param_set.N_params
species = ['CO_main_iso', 'H2O_main_iso']
if with_isotope:
    species += ['CO_36']


scratch = os.environ.get('SCRATCH', '')
path = Path(scratch) / 'highres-sbi/data'
path_full = Path(scratch) / 'highres-sbi/data_fulltheta'
path_norm = Path(scratch) / 'highres-sbi/data_fulltheta_norm'
path_norm.mkdir(parents=True, exist_ok=True)

# @ensure(lambda i: (path_full / f'samples_{i:06d}.h5').exists())
# @job(array=3, cpus=1, ram='64GB', time='10-00:00:00')
@job(array=200, cpus=1, ram='64GB', time='10-00:00:00')
def simulate(i: int):
    #prior = BoxUniform(torch.tensor(LOWER), torch.tensor(UPPER))
    #simulator = Simulator(noisy=False)
    # sim_res = 2e5
    # dlam = 2.350/sim_res
    # wavelengths = np.arange(2.320, 2.371, dlam)
    # simulator = SpectrumMaker(wavelengths=wavelengths, param_set=param_set, lbl_opacity_sampling=2)
    # loader = JointLoader(param_set, simulator, batch_size=16, numpy=False)

    # def filter_nan(theta, x):
    #     mask = torch.any(torch.isnan(x), dim=-1)
    #     mask += torch.any(~torch.isfinite(x), dim=-1)
    #     return theta[~mask], x[~mask]

    # H5Dataset.store(
    #     starmap(filter_nan, loader),
    #     path / f'samples_{i:06d}.h5',
    #     size=32,
    # )

    ######################################################################################################
    #  generating dataset with full theta (this code is not tested)

    # loader = H5Dataset(path/ f'samples_{i:06d}.h5', batch_size=32)

    # def filter_nan(theta, x):
    #     mask = torch.any(torch.isnan(x), dim=-1)
    #     mask += torch.any(~torch.isfinite(x), dim=-1)
    #     return theta[~mask], x[~mask]

    # def Processing_fulltheta(theta,x):
    #     theta_new, x_new = process(theta, x)
    #     # theta_new, x_new = filter_nan(theta_new, x_new)
    #     return theta_new, x_new[:,0,:]
        
    # H5Dataset.store(
    #     starmap(Processing_fulltheta, loader),
    #     path_full / f'samples_{i:06d}.h5',
    #     size=32,
    # )

    # ######################################################################################################

    #   generating dataset with normalized fluxes using GISIC
    # @job(array=3, cpus=1, ram='64GB', time='10-00:00:00')

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # file = 'train.h5'
    loader = H5Dataset(path_full/ f'samples_{i:06d}.h5', batch_size=32)
    # loader = H5Dataset(path_full/ file, batch_size=32)

    def filter_nan(theta, x):
        mask = torch.any(torch.isnan(x), dim=-1)
        mask1 = torch.any(~torch.isfinite(x[mask]), dim=-1)
        return theta[mask][mask1], x[mask][mask1]
    
    
    def noisy(theta, x ):
        data_uncertainty = Data().err * Data().flux_scaling * 10 
        x = x + torch.from_numpy(data_uncertainty) * torch.randn_like(x)
        return theta, x

    def normalizing(theta,x):
        # theta, x = noisy(theta,x[:,0,:])
        # v = torch.stack([torch.from_numpy(np.asarray(GISIC.normalize(Data().data_wavelengths_norm, x[i].numpy(), sigma=30))) for i in range(len(x))]) #B, 3, 6144 , wavelengths, flux, continuum
        v = torch.stack([torch.from_numpy(np.array(GISIC.normalize(Data().data_wavelengths_norm, x[i, 0, :].numpy(), sigma=20))) for i in range(len(x))])
        wave, norm_flux, continuum =  v[:, 0, :], v[:, 1, :], v[:, 2, :]
        theta_new, x_new = theta, v
        # theta_new, x_new = filter_nan(theta, x_new)
        # theta_new, x_new = filter_nan(theta_new, x_new)
        return theta_new, x_new

        
    H5Dataset.store(
        starmap(normalizing, loader),
        path_norm / f'samples_{i:06d}.h5',
        # path_norm / file[i],
        size= len(loader),
    )


    # ######################################################################################################



#@after(simulate)
@job(array=26111, cpus=1, ram='32GB', time='01:00:00')
def revaggregate(i: int):
    file = 'train.h5'
    torch.manual_seed(0)
    trainset = H5Dataset(path_full/ file, batch_size=32)

    for k, (theta, x) in enumerate(trainset):
        if k == i:
            loader = DataLoader(tuple(zip(theta,x)), batch_size=32)
            H5Dataset.store(
                # starmap(filter_nan, loader),
                loader,
                path_full / f'samples_{i:06d}.h5',
                size=32,
            )

@job(cpus=1, ram='4GB', time='01:00:00')
def aggregate():
    files = list(path.glob('samples_*.h5'))
    length = len(files)
    print('Length:', length)

    i = int(0.8 * length)
    j = int(0.9 * length)
    splits = {
        'train': files[:i],
        'valid': files[i:j],
        'test': files[j:],
    }

    for name, files in splits.items():
        dataset = H5Dataset(*files, batch_size=32)
        print(dataset)
        print(len(dataset))

        H5Dataset.store(
            dataset,
            path / f'{name}.h5',
            size=len(dataset),
        )


#@ensure(lambda: (path / 'event.h5').exists())
#@job(cpus=1, ram='4GB', time='05:00')
def event():
    simulator = Simulator(noisy=False)

    theta_star = np.array([0.55, 0., -5., -0.86, -0.65, 3., 8.5, 2., 3.75, 1., 1063.6, 0.26, 0.29, 0.32, 1.39, 0.48])
    x_star = simulator(theta_star)

    theta = theta_star[None].repeat(256, axis=0)
    x = x_star[None].repeat(256, axis=0)

    noise = simulator.sigma * np.random.standard_normal(x.shape)
    noise[0] = 0

    H5Dataset.store(
        [(theta, x + noise)],
        path / 'event.h5',
        size=256,
    )

if __name__ == '__main__':
    # N_workers = 8
    # N_datasets = 100
    # #f = lambda x: simulate(simulator, param_set, x)
    # print('Testing...')
    # simulate(1)
    # print('Done testing, simulating for real...')
    # with Pool(N_workers) as p:
    #     p.map(simulate, np.arange(2, N_datasets+1))
    # simulate()

    #for i in range(10):
    #    simulate(simulator, param_set, i)
    #aggregate()

    schedule(
        simulate, #simulate, # event, revaggregate
        name='Data generation',
        backend='slurm',
        prune=True,
        env=[
            'source ~/.bashrc',
            'conda activate HighResear',
        ]
    )
