from parameter import *


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

param_list = [FeH, CO, log_g, T_int, T1, T2, T3, alpha, log_delta, log_Pquench, Fe, fsed, Kzz, sigma_lnorm, iso_rat]

## Non pRT params included here
radius = Parameter('radius', uniform_prior(0.8, 2.0))
rv = Parameter('rv', uniform_prior(10, 30))
limb_dark = Parameter('limb_dark', uniform_prior(0,1))
vsini = Parameter('vsini', uniform_prior(0, 50))
param_set_ext = ParameterSet([radius, rv, vsini, limb_dark])

param_list_ext = [radius, rv, vsini, limb_dark]

def deNormVal(values, param_list):
    values_actual = []
    for i, param in enumerate(param_list):
        param.prior(values[i])
        values_actual.append(param.value)
    return values_actual