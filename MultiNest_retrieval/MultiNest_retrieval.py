import os
os.environ["OMP_NUM_THREADS"] = "1"

# TODO: change to correct path on cluster
# os.environ["pRT_input_data_path"] = "/u/nnas/packages/petitRADTRANS/petitRADTRANS/input_data"
#os.environ['pRT_input_data_path'] = '/u/pmolliere/packages/input_data'

import numpy as np

from petitRADTRANS.retrieval import RetrievalConfig, Retrieval
from petitRADTRANS.retrieval.util import gaussian_prior

run_mode = 'retrieve'
from emission_model import emission_model_diseq

from petitRADTRANS import nat_cst as nc

# retrieve.py expects this object to be called RunDefinition.
run_mode_use = run_mode
if run_mode_use == 'Teff':
    run_mode_use = 'retrieve'
    AMR_use = False
else:
    AMR_use = True

RunDefinition = RetrievalConfig(retrieval_name = "Highres_retrieval", # Give a useful name for your retrieval
                                run_mode = run_mode_use, # 'retrieval' to run, or 'evaluate' to make plots
                                AMR = AMR_use,             # Adaptive mesh refinement, slower if True
                                scattering = False)      # Add scattering for emission spectra clouds
                                # Todo: want scattering for specific datasets (i.e., keep for Cushing)!

if run_mode == 'Teff':
    #wlen_range_micron = [0.11, 250]
    wlen_range_micron = [0.9, 250] # was 0.11 but no alkalis...
    data_resolution = 100
    model_resolution = 10
    pRT_grid = False
else:
    wlen_range_micron = [4.80, 18.22]
    data_resolution = None
    model_resolution = None
    pRT_grid = True

RunDefinition.add_data('MIRI1',
                    #    '../obs_polychronis/separated_WISE1828_spectrum_200223_pRT_grid/1A.dat',
                        '/home/mvasist/Paul_MultiNest_1828/separated_WISE1828_spectrum_200223_pRT_grid/1A.dat', 
                       data_resolution = data_resolution,
                       model_resolution = model_resolution,
                       model_generating_function = emission_model_diseq,
                       scale_err = False,
                       wlen_range_micron = wlen_range_micron,
                       pRT_grid = pRT_grid)

RunDefinition.data['MIRI1'].flux_error = 1. * RunDefinition.data['MIRI1'].flux_error
if run_mode == 'evaluate':
    RunDefinition.data['MIRI1'].flux_error = np.sqrt(RunDefinition.data['MIRI1'].flux_error**2. + \
                                                     10**(-9.267))

err_min = np.min(RunDefinition.data['MIRI1'].flux_error)
err_max = np.max(RunDefinition.data['MIRI1'].flux_error)

import glob
paths = glob.glob('/home/mvasist/Paul_MultiNest_1828/separated_WISE1828_spectrum_200223_pRT_grid/*.dat')
i = 2
for path in paths:
    if '1A.dat' not in path:
        if True: #'13.44' not in path:
            if True: #'15.58' not in path:
                #bins = np.genfromtxt(path)[:, -1]
                RunDefinition.add_data('MIRI'+str(i),
                                           path,
                                           data_resolution=data_resolution,
                                           model_resolution=model_resolution,
                                           model_generating_function=emission_model_diseq,
                                           scale_err=False,
                                           wlen_range_micron=wlen_range_micron,
                                           external_pRT_reference = 'MIRI1',
                                           pRT_grid = pRT_grid)

                if run_mode == 'evaluate':
                    RunDefinition.data['MIRI' + str(i)].flux_error = np.sqrt(RunDefinition.data['MIRI' + str(i)].flux_error ** 2. + \
                                                                     10 ** (-9.267))
                err_min = min(np.min(RunDefinition.data['MIRI' + str(i)].flux_error), err_min)
                err_max = max(np.max(RunDefinition.data['MIRI' + str(i)].flux_error), err_max)
                i += 1

RunDefinition.add_data('Cushing',
                    #    '../obs_polychronis/WISE1828.fl.txt',
                    '/home/mvasist/Paul_MultiNest_1828/separated_WISE1828_spectrum_200223_pRT_grid/1A.dat',
                       data_resolution=320,
                       model_resolution=640,
                       model_generating_function=emission_model_diseq,
                       scale_err=False)

RunDefinition.data['Cushing'].flux = RunDefinition.data['Cushing'].flux * 1e7 * 1e23 * \
                                     (RunDefinition.data['Cushing'].wlen*1e-4)**2. / nc.c
RunDefinition.data['Cushing'].flux_error = RunDefinition.data['Cushing'].flux_error * 1e7 * 1e23 * \
                                            (RunDefinition.data['Cushing'].wlen*1e-4)**2. / nc.c

if run_mode == 'evaluate':
    RunDefinition.data['Cushing'].flux_error = np.sqrt(RunDefinition.data['Cushing'].flux_error ** 2. + \
                                                                 10 ** (-12.804))



#################################################
# Add parameters, and priors for free parameters.
#################################################

RunDefinition.add_parameter('D_pl',
                            False,
                            #  Calculated from Gaia parallax,
                            #  Consistent with https://arxiv.org/abs/2004.05180
                            value = 9.9 * nc.pc
                            )

RunDefinition.add_parameter('R_pl',
                            True,
                            transform_prior_cube_coordinate = \
                                lambda x: (0.5 + 2.5 * x) * nc.r_jup_mean
                            )

# This run uses the model of Molliere (2020) for HR8799e
# Check out models.py for a description of the parameters.

RunDefinition.add_parameter('log_g',True,
                            transform_prior_cube_coordinate = \
                            lambda x : 2.5+3.5*x)

RunDefinition.add_parameter('T_bottom',
                            True,
                            transform_prior_cube_coordinate = lambda x : 100.+8900.*x)

RunDefinition.add_parameter('N_nodes',
                            False,
                            value = 10)

for i in range(RunDefinition.parameters['N_nodes'].value-1):
    RunDefinition.add_parameter('temp_node_'+str(i+1),
                                True,
                                transform_prior_cube_coordinate=lambda x: 0.2 + 0.8 * x)

'''
RunDefinition.add_parameter('sigma_lnorm', True,
                            transform_prior_cube_coordinate = \
                            lambda x : 1.0 + 2*1e1**(-2+2*x))
RunDefinition.add_parameter('log_pquench',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : -6.0+9.0*x)
'''

RunDefinition.add_parameter('H2O_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
RunDefinition.add_parameter('CO2_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
RunDefinition.add_parameter('CO_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
RunDefinition.add_parameter('CH4_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
RunDefinition.add_parameter('NH3_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
RunDefinition.add_parameter('PH3_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
RunDefinition.add_parameter('H2S_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
#RunDefinition.add_parameter('alkali_mol_scale',
#                            True,
#                            transform_prior_cube_coordinate= \
#                                lambda x: -10+10*x)
RunDefinition.add_parameter('15NH3_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
'''
RunDefinition.add_parameter('SO2_mol_scale',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: -10+10*x)
'''
RunDefinition.add_parameter('Cushing_scale_factor',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : 0.5+ 1.*x)

RunDefinition.add_parameter('N_data_sets',
                            False,
                            value = len(RunDefinition.data))

RunDefinition.add_parameter('Fe/H', False, value = 0.)
RunDefinition.add_parameter('C/O', False, value = 0.55)

def gaussian_prior_safe(x, mu, sig, bord_left, bord_right):
    retVal = gaussian_prior(x, mu, sig)
    retVal = max(retVal, bord_left)
    retVal = min(retVal, bord_right)
    return retVal

'''
RunDefinition.add_parameter('gamma',
                            True,
                            transform_prior_cube_coordinate= \
                                lambda x: gaussian_prior_safe(x, 1, 1, 0.001, 5000.)**2 * 100.)
                            # this corresponds to delta (delta T)**2 between to layers centered at (1 K)**2.,
                            # Over a width of (delta log P)^2 = 0.1^2 = 1/100^2
'''

logbmin = np.log10(0.01*err_min**2.)
logbmax = np.log10(100*err_max**2.)
dellogbs = logbmax - logbmin
RunDefinition.add_parameter('Mike_Line_b_MIRI',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : logbmin + dellogbs*x)

logbminC = np.log10(0.01*np.min(RunDefinition.data['Cushing'].flux_error)**2.)
logbmaxC = np.log10(100*np.max(RunDefinition.data['Cushing'].flux_error)**2.)
dellogbsC = logbmaxC - logbminC
RunDefinition.add_parameter('Mike_Line_b_Cushing',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : logbminC + dellogbsC*x)

'''
RunDefinition.add_parameter('log_kzz',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : 5.0+8.*x)
RunDefinition.add_parameter('fsed_Fe(c)',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : 0.0 + 10.0*x)
RunDefinition.add_parameter('fsed_MgSiO3(c)',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : 0.0 + 10.0*x)
'''

#######################################################
# Define species to be included as absorbers
#######################################################
RunDefinition.set_rayleigh_species(['H2', 'He'])
RunDefinition.set_continuum_opacities(['H2-H2', 'H2-He'])

# TODO: Add FeH, VO, TiO, CrH
# TODO: Compare standard w ExoMol

RunDefinition.set_line_species(['CH4_hargreaves',
                                'H2O_Exomol',
                                'CO2',
                                'CO_all_iso_HITEMP',
                                'H2S',
                                'NH3',
                                'PH3',
                                '15NH3'], eq=True)

# Origin run
#RunDefinition.add_cloud_species("Fe(c)_cd",eq = True,abund_lim = (-3.5,4.5))
#RunDefinition.add_cloud_species("MgSiO3(c)_cd",eq = True,abund_lim = (-3.5,4.5))

#RunDefinition.add_cloud_species("Fe(c)_ad",eq = False,abund_lim = (-10,0.), PBase_lim = (-5.0,2.0))
#RunDefinition.add_cloud_species("MgSiO3(c)_ad",eq = False,abund_lim = (-10,0.), PBase_lim = (-5.0,2.0))

##################################################################
# Define what to put into corner plot if run_mode == 'evaluate'
##################################################################
'''
RunDefinition.parameters['log_g'].plot_in_corner = True
RunDefinition.parameters['log_g'].corner_ranges = [2., 5.]
RunDefinition.parameters['log_g'].corner_label = "log g"
#RunDefinition.parameters['fsed_Fe(c)'].plot_in_corner = True
#RunDefinition.parameters['fsed_MgSiO3(c)'].plot_in_corner = True
#RunDefinition.parameters['log_kzz'].plot_in_corner = True
#RunDefinition.parameters['log_kzz'].corner_label = "log Kzz"
RunDefinition.parameters['C/O'].plot_in_corner = True
RunDefinition.parameters['Fe/H'].plot_in_corner = True
#RunDefinition.parameters['log_pquench'].plot_in_corner = True
#RunDefinition.parameters['log_pquench'].corner_label = "log pquench"
'''
'''
for spec in RunDefinition.cloud_species:
    cname = spec.split('_')[0]
    RunDefinition.parameters['log_X_cb_'+cname].plot_in_corner = True
    RunDefinition.parameters['log_X_cb_'+cname].corner_label = cname
'''

for param in RunDefinition.parameters.keys():
    if RunDefinition.parameters[param].is_free_parameter:
        if 'temp_node' in param:
            continue
        RunDefinition.parameters[param].plot_in_corner = True



##################################################################
# Define axis properties of spectral plot if run_mode == 'evaluate'
##################################################################
RunDefinition.plot_kwargs["spec_xlabel"] = 'Wavelength [micron]'

RunDefinition.plot_kwargs["spec_ylabel"] = "Flux [~normalized]"
RunDefinition.plot_kwargs["y_axis_scaling"] = 1.0
RunDefinition.plot_kwargs["xscale"] = 'log'
RunDefinition.plot_kwargs["wavelength_lim"] = [0.8, 18]
RunDefinition.plot_kwargs["yscale"] = 'log'
RunDefinition.plot_kwargs["resolution"] = None
RunDefinition.plot_kwargs["nsample"] = 10.
RunDefinition.plot_kwargs["flux_lim"] = [1e-7,1e-3]

##################################################################
# Define from which observation object to take P-T
# in evaluation mode (if run_mode == 'evaluate'),
# add PT-envelope plotting options
##################################################################
RunDefinition.plot_kwargs["take_PTs_from"] = 'MIRI1'
RunDefinition.plot_kwargs["temp_limits"] = [0, 2000]
RunDefinition.plot_kwargs["press_limits"] = [1e3, 1e-5]


##################################################################
# Run the Retrieval
##################################################################

retrieval = Retrieval(RunDefinition,
                      sample_spec = False,
		              ultranest = False,
                      test_plotting = False,
                      pRT_plot_style = False)

retrieval.run(n_live_points = 2000,
              const_efficiency_mode=True,
              sampling_efficiency = 0.05,
              resume = True)

import pylab as plt
sample_dict, parameter_dict = retrieval.get_samples()
# Pick the current retrieval to look at.
samples_use = sample_dict[retrieval.retrieval_name]
parameters_read = parameter_dict[retrieval.retrieval_name]

if run_mode == 'evaluate':
    fig,ax,ax_r = retrieval.plot_spectra(samples_use,parameters_read)
    plt.show()
    #retrieval.plot_corner(sample_dict,parameter_dict,parameters_read,title_kwargs = {"fontsize" : 10})
    #plt.show()
    fig,ax =retrieval.plot_PT(sample_dict,parameters_read, contribution = True, pRT_reference = 'MIRI1')
    plt.show()
elif run_mode == 'Teff':

    samples_use = samples_use[:,:-1]

    import glob
    path = glob.glob('evaluate*/*full.npy')
    best_fit = np.load(path[0])

    from matplotlib import pyplot as plt
    from petitRADTRANS.retrieval import parameter as pm

    len_sample = len(samples_use)

    file = open(path[0].split('/')[0]+'/Teffs.dat', 'w')
    file2 = open(path[0].split('/')[0]+'/MMWs.dat', 'w')

    for i_s, sample in enumerate(samples_use):

        sample_parameters = {}
        i = 0
        for param in parameters_read:
            sample_parameters[param] = pm.Parameter(param,
                                                      is_free_parameter=False,
                                                      value=sample[i])
            i += 1

        sample_parameters['N_data_sets'] = pm.Parameter('N_data_sets',
                                    is_free_parameter=False,
                                    value=len(RunDefinition.data))

        sample_parameters['Fe/H'] = pm.Parameter('Fe/H', is_free_parameter=False, value=0.)
        sample_parameters['C/O'] = pm.Parameter('C/O', is_free_parameter=False, value=0.55)

        sample_parameters['N_nodes'] = pm.Parameter('N_nodes',
                                    is_free_parameter=False,
                                    value=10)
        sample_parameters['D_pl'] = pm.Parameter('D_pl',
                                    is_free_parameter=False,
                                    #  Calculated from Gaia parallax,
                                    #  Consistent with https://arxiv.org/abs/2004.05180
                                    value=9.9 * nc.pc
                                    )

        wlen, model, __, = emission_model_diseq(retrieval.data['MIRI1'].pRT_object,
                                                sample_parameters,
                                                AMR = False,
                                                outside = True,
                                                MMW_file = file2)


        if False:
            plt.semilogx(best_fit[:,0], best_fit[:,1])
            plt.semilogx(wlen, model)
            plt.show()

        nu = retrieval.data['MIRI1'].pRT_object.freq
        Teff = (-np.sum((retrieval.data['MIRI1'].pRT_object.flux[1:]+retrieval.data['MIRI1'].pRT_object.flux[:-1])*np.diff(nu))/nc.sigma/2.)**0.25
        file.write(str(Teff)+'\n')
        print(i_s+1, len_sample, Teff)


    file.close()
    file2.close()
