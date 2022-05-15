from learning_functions import *
import random
import dill
import pickle
from pathlib import Path
import numpy as np
import os
from scipy import interpolate
from scipy import optimize
from sklearn.decomposition import IncrementalPCA

#Loading in all of the data and parameters
with open('parameters.pkl', 'rb') as handle:
    params = dill.load(handle)

preParams = params['preParams']
normParams = params['normParams']
learnParams = params['learnParams']
physParams = params['physParams']
loading = params['loadParams']
optParams = params['optParams']
nullParams = params['nullParams']
emitted_wavelengths = params['emitted_wavelengths']
observed_wavelengths = params['observed_wavelengths']
kms_to_z = params['kms_to_z']

# optimization parameters
initial_c     = 0.1                          # initial guess for c
initial_tau_0 = 0.0023                       # initial guess for τ₀
initial_beta  = 3.65                         # initial guess for β

training_release  = 'dr12q'
dla_catalog_name = 'dr9q_concordance'
training_set_name = 'dr9q_minus_concordance'

random.seed()
print(random.random())

p = Path(os.getcwd())
parent_dir = str(p.parent)
release = "{}/processed/catalog".format(training_release)
filename = os.path.join(parent_dir, release)
#getting back pickled data for catalog
with open(release,'rb') as f:
    catalog = pickle.load(f)

in_dr9 = catalog['in_dr9']
filter_flags = catalog['new_filter_flags']
filtered_flags = (filter_flags == 0)
los_inds = catalog['los_inds']['dr9q_concordance']
dla_inds = catalog['dla_inds']['dr9q_concordance']
dla_inds = np.invert(dla_inds)

train_ind = in_dr9 & filtered_flags & los_inds & dla_inds
z_qsos             =        catalog['z_qsos'][train_ind]

#getting back preprocessed qso data
release = "{}/processed/preloaded_qsos".format(training_release)
filename = os.path.join(parent_dir, release)
with open(release, 'rb') as f:
    preqsos = pickle.load(f)


all_wavelengths = preqsos['all_wavelengths']
all_wavelengths = all_wavelengths[train_ind]
all_flux           =                preqsos['all_flux']
all_flux           =           all_flux[train_ind]
all_noise_variance =       preqsos['all_noise_variance']
all_noise_variance = all_noise_variance[train_ind]
all_pixel_mask     =           preqsos['all_pixel_mask']
all_pixel_mask     =     all_pixel_mask[train_ind]
#deallocating memory i think
preqsos = 0

num_quasars = len(z_qsos)
print("num_quasars", num_quasars)

rest_wavelengths = np.arange(nullParams.min_lambda,nullParams.max_lambda+nullParams.dlambda,nullParams.dlambda)
num_rest_pixels  = rest_wavelengths.size

lya_1pzs             = np.empty([num_quasars, num_rest_pixels])
lya_1pzs[:] = np.NaN
all_lyman_1pzs       = np.empty([learnParams.num_forest_lines, num_quasars, num_rest_pixels])
all_lyman_1pzs[:] = np.NaN
rest_fluxes          = np.empty([num_quasars, num_rest_pixels])
rest_fluxes[:] = np.NaN
rest_noise_variances = np.empty([num_quasars, num_rest_pixels])
rest_noise_variances[:] = np.NaN

# the preload_qsos should fliter out empty spectra;
# this line is to prevent there is any empty spectra
# in preloaded_qsos.mat for some reason
is_empty             = np.zeros((num_quasars, 1), dtype=int)
is_empty = np.logical_not(is_empty)

# interpolate quasars onto chosen rest wavelength grid

bluewards_flux = []
bluewards_nv = []
redwards_flux = []
redwards_nv = []

#Creation of the base model
for i in range(num_quasars):
    z_qso = z_qsos[i]

    this_wavelengths    =    all_wavelengths[i]
    this_flux           =           all_flux[i]
    this_noise_variance = all_noise_variance[i]
    this_pixel_mask     =     all_pixel_mask[i]

    this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso)
    this_pixel_mask = np.array(this_pixel_mask, dtype=bool)

    this_flux[this_pixel_mask]           = np.NaN
    this_noise_variance[this_pixel_mask] = np.NaN

    print("processing quasar {num} with lambda size = {size} ...\n".format(num=i+1, size=this_wavelengths.shape[0]))

    this_pixel_mask = np.logical_not(this_pixel_mask)
    
    if this_wavelengths.shape == np.shape([[0, 0]]):
        is_empty[i, 1] = 1
        continue

    z_interp = interpolate.interp1d(this_rest_wavelengths, 1 + (this_wavelengths - physParams.lya_wavelength) / physParams.lya_wavelength,bounds_error=False)
    lya_1pzs[i] = z_interp(rest_wavelengths)
 
    # this_wavelength is raw wavelength (w/t ind)
    # so we need an indicator here to comfine lya_1pzs
    # below Lyman alpha (do we need to make the indicator
    # has a lower bound at Lyman limit here?)
    # indicator = lya_1pzs(i, :) <= (1 + z_qso);
    # lya_1pzs(i, :) = lya_1pzs(i, :) .* indicator;

    # include all members in Lyman series to the forest
    for j in range(learnParams.num_forest_lines):
        this_transition_wavelength = learnParams.all_transition_wavelengths[j]

        trans_interp = interpolate.interp1d(this_rest_wavelengths, 1 + (this_wavelengths - this_transition_wavelength) / this_transition_wavelength,bounds_error=False)
        all_lyman_1pzs[j, i] = trans_interp(rest_wavelengths)

        # indicator function: z absorbers <= z_qso
        indicator = all_lyman_1pzs[j, i] <= (1 + z_qso)

        all_lyman_1pzs[j, i] = np.multiply(all_lyman_1pzs[j, i], indicator)
    
    flux_interp = interpolate.interp1d(this_rest_wavelengths, this_flux, bounds_error=False)
    rest_fluxes[i, :] = flux_interp(rest_wavelengths)

    #normalizing here
    ind = (this_rest_wavelengths >= normParams.normalization_min_lambda) & (this_rest_wavelengths <= normParams.normalization_max_lambda) & (this_pixel_mask)

    this_median = np.nanmedian(this_flux[ind])
    rest_fluxes[i, :] = rest_fluxes[i, :] / this_median

    rest_interp = interpolate.interp1d(this_rest_wavelengths, this_noise_variance,bounds_error=False)
    rest_noise_variances[i, :] = rest_interp(rest_wavelengths)
    rest_noise_variances[i, :] = rest_noise_variances[i, :] / np.power(this_median, 2)  #setting up bluward/redwards of restframe txt files

    # normalise the data we put into end model fitting
    this_norm_flux           = this_flux / this_median
    this_norm_noise_variance = this_noise_variance / np.power(this_median, 2)
    
    less = (this_rest_wavelengths < nullParams.min_lambda) & this_pixel_mask
    more = (this_rest_wavelengths > nullParams.max_lambda) & this_pixel_mask

    bluewards_flux.append(this_norm_flux[less])
    bluewards_nv.append(this_norm_noise_variance[less])
    redwards_flux.append(this_norm_flux[more])
    redwards_nv.append(this_norm_noise_variance[more])

bluewards_flux = np.concatenate(bluewards_flux).astype(bluewards_flux[0].dtype)
bluewards_nv = np.concatenate(bluewards_nv).astype(bluewards_nv[0].dtype)
redwards_flux = np.concatenate(redwards_flux).astype(redwards_flux[0].dtype)
redwards_nv = np.concatenate(redwards_nv).astype(redwards_nv[0].dtype)

[bluewards_mu, bluewards_sigma] = fitendmodel(bluewards_flux, bluewards_nv)
[redwards_mu, redwards_sigma] = fitendmodel(redwards_flux, redwards_nv)
    
all_wavelengths = None
all_flux = None
all_noise_variance = None
all_pixel_mask = None
bluewards_flux= None
bluewards_nv = None
redwards_flux = None
redwards_nv = None

space_save = [all_wavelengths, all_flux, all_noise_variance, all_pixel_mask, bluewards_flux, bluewards_nv, redwards_flux, redwards_nv]

# filter out empty spectra
# note: if you've done this in preload_qsos then skip these lines
z_qsos               = z_qsos[is_empty[:,0]]
lya_1pzs             = lya_1pzs[is_empty[:,0], :]
rest_fluxes          = rest_fluxes[is_empty[:,0], :]
rest_noise_variances = rest_noise_variances[is_empty[:,0], :]
all_lyman_1pzs       = all_lyman_1pzs[:, is_empty[:,0], :]

# update num_quasars in consideration
#should update with z_qsos
num_quasars = len(z_qsos)

print('Get rid of empty spectra, num_quasars = {num}\n'.format(num=num_quasars))

# mask noisy pixels
ind = (rest_noise_variances > nullParams.max_noise_variance)
print("Masking {val} of pixels\n".format(val=np.count_nonzero(ind) * (1.0 / ind.size)))
lya_1pzs[ind]             = np.NaN
rest_fluxes[ind]          = np.NaN
rest_noise_variances[ind] = np.NaN


for i in range(num_quasars):
    for j in range(learnParams.num_forest_lines):
        all_lyman_1pzs[j, i, ind[i, :]]  = np.NaN

# reverse the rest_fluxes back to the fluxes before encountering Lyα forest
prev_tau_0 = 0.0023 # Kim et al. (2007) priors
prev_beta  = 3.65

rest_fluxes_div_exp1pz      = np.empty((num_quasars, num_rest_pixels))
rest_noise_variances_exp1pz = np.empty((num_quasars, num_rest_pixels))
rest_fluxes_div_exp1pz[:]      = np.NaN
rest_noise_variances_exp1pz[:] = np.NaN

for i in range(num_quasars):
    # compute the total optical depth from all Lyman series members
    # Apr 8: not using NaN here anymore due to range beyond Lya will all be NaNs
    total_optical_depth = np.zeros((learnParams.num_forest_lines, num_rest_pixels))

    for j in range(learnParams.num_forest_lines):
         #calculate the oscillator strengths for Lyman series
        this_tau_0 = prev_tau_0 * learnParams.all_oscillator_strengths[j] / learnParams.lya_oscillator_strength * learnParams.all_transition_wavelengths[j] / physParams.lya_wavelength
    
        # remove the leading dimension
        this_lyman_1pzs = np.squeeze(all_lyman_1pzs[j, i, :])#'; % (1, num_rest_pixels)

        total_optical_depth[j, :] = np.multiply(this_tau_0, np.power(this_lyman_1pzs,prev_beta))

    # Apr 8: using zeros instead so not nansum here anymore
    # beyond lya, absorption fcn shoud be unity
    lya_absorption = np.exp(- np.sum(total_optical_depth, axis=0) )

    # We have to reverse the effect of Lyα for both mean-flux and observational noise
    rest_fluxes_div_exp1pz[i, :]      = rest_fluxes[i, :] / lya_absorption
    rest_noise_variances_exp1pz[i, :] = rest_noise_variances[i, :] / lya_absorption**2

all_lyman_1pzs = None

# Filter out spectra which have too many NaN pixels
#Filtering section
ind = (np.sum(np.isnan(rest_fluxes_div_exp1pz), axis=1) < (num_rest_pixels-preParams.min_num_pixels))

print("Filtering {width} quasars for NaN\n".format(width=rest_fluxes_div_exp1pz.shape[1] - np.count_nonzero(ind)))

z_qsos                      = z_qsos[ind]
rest_fluxes_div_exp1pz      = rest_fluxes_div_exp1pz[ind, :]
rest_noise_variances_exp1pz = rest_noise_variances_exp1pz[ind, :]
lya_1pzs                    = lya_1pzs[ind, :]

# Check for columns which contain only NaN on either end.
nancolfrac = np.sum(np.isnan(rest_fluxes_div_exp1pz), axis=0) / float(np.count_nonzero(ind))
print("Columns with nan > 0.9: ")

#print(np.max(np.nonzero(nancolfrac > 0.9)))
#print(np.max(nancolfrac[nancolfrac>0.9]))

# find empirical mean vector and center data
mu = np.nanmean(rest_fluxes_div_exp1pz, axis=0)
centered_rest_fluxes = rest_fluxes_div_exp1pz[...,:] - mu
rest_fluxes = None

# small fix to the data fit into the pca:
# make the NaNs to the medians of a given row
# rememeber not to inject this into the actual
# joint likelihood maximisation
pca_centered_rest_flux = centered_rest_fluxes

num_quasars = len(pca_centered_rest_flux)

for i in range(num_quasars):
    this_pca_centered_rest_flux = pca_centered_rest_flux[i, :]

    # assign median value for each row to nan
    ind = np.isnan(this_pca_centered_rest_flux)

    pca_centered_rest_flux[i, ind] = np.nanmedian(this_pca_centered_rest_flux, axis=0)

# get top-k PCA vectors to initialize M
ipca = IncrementalPCA(n_components=nullParams.k)
ipca.fit(pca_centered_rest_flux)
coefficients = ipca.components_.T
latent = ipca.explained_variance_

objective_function = lambda x : objective(x, centered_rest_fluxes, lya_1pzs, rest_noise_variances_exp1pz, learnParams.num_forest_lines, learnParams.all_transition_wavelengths, learnParams.all_oscillator_strengths, z_qsos)

# initialize A to top-k PCA components of non-DLA-containing spectra
initial_M = coefficients * np.sqrt(latent)

# initialize log omega to log of elementwise sample standard deviation
centered_rest_fluxes = rest_fluxes_div_exp1pz[...,:] - mu
initial_log_omega = np.log(np.nanstd(centered_rest_fluxes, axis=0))

initial_log_c_0   = np.log(optParams.initial_c_0)
initial_log_tau_0 = np.log(optParams.initial_tau_0)
initial_log_beta  = np.log(optParams.initial_beta)

init_M = np.reshape(initial_M, initial_M.shape[0]*initial_M.shape[1], order='F')

#initial_M[:] is actually changing it to a gigantic column vector, id say the same for log omeaga but it already is???
initial_x = np.concatenate((init_M, initial_log_omega))
initial_x = np.append(initial_x, [initial_log_c_0, initial_log_tau_0, initial_log_beta])

#minimization of negative log likelihood
# maximize likelihood via L-BFGS
maxes = {'maxfun':8000, 'maxiter':4000}
#method = trust- or CG ones that I would try first: CG, BFGS, Newton-CG, trust-ncg, SLSQP
result = optimize.minimize(objective_function, initial_x, method='L-BFGS-B', jac=True, options=maxes, callback=callbackF)
#try method Nelder-Mead
#result = optimize.minimize(objective_function, initial_x, method='CG', jac=True, options={'maxiter':3000})
x = result.x
log_likelihood = result.fun
message = result.message
success = result.success
ind = list(range(num_rest_pixels * nullParams.k))
ind = np.array(ind)
M = np.reshape(x[ind], [num_rest_pixels, nullParams.k], order='F')

ind = list(range((num_rest_pixels * nullParams.k), (num_rest_pixels * (nullParams.k + 1))))
ind = np.array(ind)
print("ind", ind, ind.shape)
log_omega = x[ind].T

log_c_0   = x[-3]
log_tau_0 = x[-2]
log_beta  = x[-1]

variables_to_save = {'training_release':training_release, 'train_ind':train_ind, 'max_noise_variance':nullParams.max_noise_variance,
                     'rest_wavelengths':rest_wavelengths, 'mu':mu, 'initial_M':initial_M, 'initial_log_omega':initial_log_omega,
                     'initial_log_c_0':initial_log_c_0, 'initial_tau_0':initial_tau_0, 'initial_beta':initial_beta, 'opt':opt,
                     'M':M, 'log_omega':log_omega, 'log_c_0':log_c_0, 'log_tau_0':log_tau_0, 'log_beta':log_beta, 'result':result,
                     'log_likelihood':log_likelihood, 'message':message, 'success':success, 'bluewards_mu':bluewards_mu,
                     'bluewards_sigma':bluewards_sigma, 'redwards_mu':redwards_mu, 'redwards_sigma':redwards_sigma}

direct = 'dr12q/processed'
#directory = os.path.join(parent_dir, direct)
                   
place = '{}/learned_model_outdata_{}_norm_{}-{}'.format(direct, training_set_name, normParams.normalization_min_lambda, normParams.normalization_max_lambda)
             
# Open a file for writing data
file_handler = open(place, 'wb')

# Dump the data of the object into the file
dill.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()
