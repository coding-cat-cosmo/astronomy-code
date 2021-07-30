import os
import dill
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool


dill.load_session('parameters.pkl')

# log_mvnpdf_iid: computes mutlivariate normal dist with
#    each dim is iid, so no covariance. 
#   log N(y; mu, diag(d))

def log_mvnpdf_iid(y, mu, d):

    log_2pi = 1.83787706640934534
  
    n = d.shape[0]
   
    y = y - (mu)

    d_inv = 1 / d
    D_inv_y = d_inv * y
  
    K_inv_y = D_inv_y
  
    log_det_K = np.sum(np.log(d))
  
    log_p = -0.5 * (np.dot(y, K_inv_y) + log_det_K + n * log_2pi)
    
    #print("n",n)
    #print("\ny", y, y.shape)
    #print("\nd_inv", d_inv, d_inv.shape)
    #print("\nD_inv_y", D_inv_y, D_inv_y.shape)
    #print("\nK_inv_y", K_inv_y, K_inv_y.shape)
    #print("\nlog_det_K", log_det_K, log_det_K.shape)
    #print("\nlog_p", log_p, log_p.shape)
                    
    return log_p

# log_mvnpdf_low_rank: efficiently computes
#
#   log N(y; mu, MM' + diag(d))

def log_mvnpdf_low_rank(y, mu, M, d):

    #disp(mean(y));
    #disp(mean(mu));
    #disp(mean(M(:)));
    #disp(mean(d));
    log_2pi = 1.83787706640934534

    [n, k] = M.shape
    #print("\nthen", n)
    #print("\nk", k)
 
    y = y - (mu)
    #print("\ny", y, y.shape)
    #d = np.ones(len(d)) * .001 #here
    d_inv = 1 / d
    #print("\nd_inv", d_inv, d_inv.shape)
    D_inv_y = d_inv * y
    #print("\nD_inv_y", D_inv_y, D_inv_y.shape)

    D_inv_M = d_inv[:, None] * M
    #print("D_inv_M", D_inv_M, D_inv_M.shape)

    # use Woodbury identity, define
    #   B = (I + M' D^-1 M),
    # then
    #   K^-1 = D^-1 - D^-1 M B^-1 M' D^-1

    B = np.dot(M.T, D_inv_M)
    #print("\nB before", B, B.shape)
    B = np.reshape(B, B.shape[0]*B.shape[1], order='F')
    B[0::k+1] = B[0::k+1] + 1
    B = np.reshape(B, [k, k], order='F')
    #print("\nB after", B, B.shape)
    L = np.linalg.cholesky(B)
    L= L.T
    #print("\nL", L, L.shape)
    # C = B^-1 M' D^-1
    ld = np.linalg.solve(L.T, D_inv_M.T)
    #print("\nld", ld, ld.shape)
    C= np.linalg.solve(L, ld)
    #C = np.linalg.solve(L.T, np.linalg.solve(L, D_inv_M.T))
    #print("\nC", C, C.shape)

    mid = np.dot(C, y)
    #print("\nmid", mid, mid.shape)
    after = np.dot(D_inv_M, mid)
    #print("\nafter", after, after.shape)
    K_inv_y = D_inv_y - np.dot(D_inv_M, np.dot(C, y))
    #print("\nK_inv_y", K_inv_y, K_inv_y.shape)

    log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))
    #print("\nlog_det_K", log_det_K, log_det_K.shape)

    log_p = -0.5 * (np.dot(y, K_inv_y) + log_det_K + n * log_2pi)
    #print("\nlog_p", log_p)

    return log_p


 '''mex voigt.c -lcerf
% specify the learned quasar model to use
training_release  = 'dr12q';
training_set_name = 'dr9q_minus_concordance';
% specify the spectra to use for computing the DLA existence prior
dla_catalog_name  = 'dr9q_concordance';
prior_ind = ...
    [' prior_catalog.in_dr9 & ' ...
     ' prior_catalog.los_inds(dla_catalog_name) & ' ...
     '(prior_catalog.filter_flags == 0)'];
     % specify the spectra to process
release = 'dr12q';
test_set_name = 'dr12q';
test_ind = '(catalog.filter_flags == 0)';
% model prior parameters
prior_z_qso_increase = kms_to_z(30000);       % use QSOs with z < (z_QSO + x) for prior

% instrumental broadening parameters
width = 3;                                    % width of Gaussian broadening (# pixels)
pixel_spacing = 1e-4;                         % wavelength spacing of pixels in dex

% DLA model parameters: absorber range and model
num_lines = 3;                                % number of members of the Lyman series to use

max_z_cut = kms_to_z(3000);                   % max z_DLA = z_QSO - max_z_cut
max_z_dla = @(wavelengths, z_qso) ...         % determines maximum z_DLA to search
    (max(wavelengths) / lya_wavelength - 1) - max_z_cut;

min_z_cut = kms_to_z(3000);                   % min z_DLA = z_Ly∞ + min_z_cut
min_z_dla = @(wavelengths, z_qso) ...         % determines minimum z_DLA to search
max(min(wavelengths) / lya_wavelength - 1,                          ...
    observed_wavelengths(lyman_limit, z_qso) / lya_wavelength - 1 + ...
    min_z_cut);
% produce catalog searching [Lyoo + 3000 km/s, Lya - 3000 km/s]
set_parameters;

% specify the learned quasar model to use
training_release  = 'dr12q';
training_set_name = 'dr9q_minus_concordance';

% specify the spectra to use for computing the DLA existence prior
dla_catalog_name  = 'dr9q_concordance';
prior_ind = ...
    [' prior_catalog.in_dr9 & '             ...
     '(prior_catalog.filter_flags == 0) & ' ...
     ' prior_catalog.los_inds(dla_catalog_name)'];

% specify the spectra to process
release = 'dr12q';
test_set_name = 'dr12q';
test_ind = '(catalog.filter_flags == 0)';

% process the spectra
process_qsos;
'''
  
# process_qsos: run DLA detection algorithm on specified objects
# 
# Apr 8, 2020: add all Lyman series to the effective optical depth
#   effective_optical_depth := ∑ τ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β where 
#   1 + z_i1 =  λobs / λ_i1 = λ_lya / λ_i1 *  (1 + z_a)
# Dec 25, 2019: add Lyman series to the noise variance training
#   s(z)     = 1 - exp(-effective_optical_depth) + c_0 
# the mean values of Kim's effective optical depth
# Apr 28: add occams razor for penalising the missing pixels,
#   this factor is tuned to affect log likelihood in a range +- 500,
#   this value could be effective to penalise every likelihoods for zQSO > zCIV
#   the current implemetation is:
#     likelihood - occams_factor * (1 - lambda_observed / (max_lambda - min_lambda) )
#   and occams_factor is a tunable hyperparameter
# 
# May 11: out-of-range data penalty,
#   adding additional log-likelihoods to the null model log likelihood,
#   this additional log-likelihoods are:
#     log N(y_bluewards; bluewards_mu, diag(V_bluewards) + bluewards_sigma^2 )
#     log N(y_redwards;  redwards_mu,  diag(V_redwards)  + redwards_sigma^2 )
import os
import dill
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool

# specify the learned quasar model to use
training_release  = 'dr12q'
training_set_name = 'dr9q_minus_concordance'

# specify the spectra to use for computing the DLA existence prior
dla_catalog_name  = 'dr9q_concordance'
    
# specify the spectra to process
release = 'dr12q'
test_set_name = 'dr12q'

modelParams = model_params()
instrumentParams = instrument_params()
moreParams = more_dla_params()
nullParams = null_params()
dlaParams = dla_params()
normParams = normalization_params()
learnParams = learning_params()

prev_tau_0 = 0.0023
prev_beta  = 3.65

occams_factor = 0 # turn this to zero if you don't want the incomplete data penalty

# load redshifts/DLA flags from training release
#prior_catalog = load(sprintf('%s/catalog', processed_directory(training_release)));
p = Path(os.getcwd())
parent_dir = str(p.parent)
rel = "dr12q/processed/catalog"
filename = os.path.join(parent_dir, rel)
#getting back pickled data for catalog
#try:
with open(filename,'rb') as f:
    prior_catalog = pickle.load(f)
#except:
#print('\ncatalog')
#print(prior_catalog)

prior_ind = prior_catalog['in_dr9'] & prior_catalog['los_inds'][dla_catalog_name] & (prior_catalog['new_filter_flags'] == 0)

orig_z_qsos  = prior_catalog['z_qsos'][prior_ind]
dla_ind = prior_catalog['dla_inds'][dla_catalog_name]
dla_ind = dla_ind[prior_ind]
print("\nz_qsos")
print(orig_z_qsos)
print(orig_z_qsos.shape)

# filter out DLAs from prior catalog corresponding to region of spectrum below Ly∞ QSO rest
z_dlas = prior_catalog['z_dlas'][dla_catalog_name]
#z_dlas = [x[0] for x in z_dlas if x!=0]
#print("orig", prior_catalog['z_dlas'], len(prior_catalog['z_dlas']))
f#or num, name in enumerate(presidents, start=1):
for i, z in enumerate(z_dlas):
    if (z!=0):
        z_dlas[i] = z[0]
        
#print("z_dlas", z_dlas, len(z_dlas))
#print("prior_ind", prior_ind, len(prior_ind), np.count_nonzero(prior_ind))
z_dlas = np.array(z_dlas)
z_dlas = z_dlas[prior_ind]

thing = np.where(dla_ind > 0)
thing = thing[0]
#print("thing", thing)
for i in thing:
    if (observed_wavelengths(physConst.lya_wavelength, z_dlas[i]) < observed_wavelengths(physConst.lyman_limit, orig_z_qsos[i])):
            dla_ind[i] = False

#prior = rmfield(prior, 'z_dlas');
#z_dlas = 0

# load QSO model from training release
#variables_to_load = {'rest_wavelengths', 'mu', 'M', 'log_omega', ...
#    'log_c_0', 'log_tau_0', 'log_beta', ...
#    'bluewards_mu', 'bluewards_sigma', ...
#    'redwards_mu', 'redwards_sigma'};
#load(sprintf('%s/learned_model_outdata_%s_norm_%d-%d',processed_directory(training_release),training_set_name, ...
#     normalization_min_lambda, normalization_max_lambda),  ...
#     variables_to_load{:});
rel = "dr12q/processed/"
directory = os.path.join(parent_dir, rel)
training_set_name = 'dr9q_minus_concordance'
normParams = normalization_params()

place = '{}/learned_model_outdata_{}_norm_{}-{}'.format(directory, training_set_name, normParams.normalization_min_lambda, normParams.normalization_max_lambda)
#getting back pickled data for model
#try:
with open(place,'rb') as f:
    model = pickle.load(f)
#except:
#print('\nmodel')
#print(model)
rest_wavelengths = model['rest_wavelengths']
mu = model['mu']
#M = model['M']
#log_omega = model['log_omega']
#log_c_0 = model['log_c_0']
#log_tau_0 = model['log_tau_0']
#log_beta = model['log_beta']
bluewards_mu = model['bluewards_mu']
bluewards_sigma = model['bluewards_sigma']
redwards_mu = model['redwards_mu']
redwards_sigma = model['redwards_sigma']

# load DLA samples from training release
#variables_to_load = {'offset_samples', 'offset_samples_qso', 'log_nhi_samples', 'nhi_samples'};
#load(sprintf('%s/dla_samples', processed_directory(training_release)), ...
#    variables_to_load{:});

rel = "dr12q/processed/"
direc = os.path.join(parent_dir, rel)
lease = "dla_samples"
filename = os.path.join(direc, lease)
with open(filename, 'rb') as f:
    samples = pickle.load(f)
    
offset_samples = samples['offset_samples']
offset_samples_qso = samples['offset_samples_qso']
log_nhi_samples = samples['log_nhi_samples']
nhi_samples = samples['nhi_samples']

# load preprocessed QSOs
#variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
#    'all_pixel_mask'};
#load(sprintf('%s/preloaded_qsos', processed_directory(release)), ...
#    variables_to_load{:});

# load redshifts from catalog to process
#catalog = load(sprintf('%s/catalog', processed_directory(release)));
rel = "dr12q/processed/catalog"
filename = os.path.join(parent_dir, rel)
with open(filename, 'rb') as f:
    catalog = pickle.load(f)

#getting back preprocessed qso data
rel = "dr12q/processed/preloaded_qsos"
#release = "dr12q/processed/preloaded_zqso_only_qsos.mat"
filename = os.path.join(parent_dir, rel)
with open(filename, 'rb') as f:
    preqsos = pickle.load(f)
    
all_wavelengths = preqsos['all_wavelengths']
all_flux = preqsos['all_flux']
all_noise_variance = preqsos['all_noise_variance']
all_pixel_mask = preqsos['all_pixel_mask']

# enable processing specific QSOs via setting to_test_ind
test_ind = (catalog['new_filter_flags'] == 0)
#print("before", test_ind, len(test_ind))
#i only have 5000 from preqsos stuff so artificially reducing test_ind...
test_ind = test_ind[:5000]
#print("after", test_ind, len(test_ind))

all_wavelengths    =    all_wavelengths[test_ind]
all_flux           =           all_flux[test_ind]
all_noise_variance = all_noise_variance[test_ind]
all_pixel_mask     =     all_pixel_mask[test_ind]
#all_thing_ids      =   catalog['thing_ids'][test_ind]
#more fixing done here since catalog has the full amount but test_ind is only 5000
all_thing_ids = catalog['thing_ids'][:5000][test_ind]
#all_thing_ids = all_thing_ids[:5000]
#all_thing_ids[test_ind]

z_qsos = catalog['z_qsos'][:5000][test_ind]
dla_inds = catalog['dla_inds']['dr12q_visual']
dla_inds = dla_inds[:5000][test_ind]

num_quasars = len(z_qsos)
try:
    qso_ind and var
except NameError:
    qso_ind = [x for x in range(int(np.floor(num_quasars/100)))]

num_quasars = len(qso_ind)

#load('./test/M.mat');
# preprocess model interpolants
#my_interpolating_function = RegularGridInterpolator((x, y, z), data)
print("lengths", rest_wavelengths.shape, mu.shape)
mu_interpolator = RegularGridInterpolator((rest_wavelengths,), mu)
#temp = (rest_wavelengths, [x for x in range(k)])
M_interpolator = RegularGridInterpolator((rest_wavelengths, [x for x in range(nullParams.k)]), M)
log_omega_interpolator = RegularGridInterpolator((rest_wavelengths,), log_omega)

# initialize results
# prevent parfor error, should use nan(num_quasars, num_dla_samples); or not save these variables;
min_z_dlas = np.empty((num_quasars, dlaParams.num_dla_samples))
min_z_dlas[:] = np.NaN
max_z_dlas = np.empty((num_quasars, dlaParams.num_dla_samples))
max_z_dlas[:] = np.NaN
# sample_log_priors_no_dla = np.empty((num_quasars, dlaParams.num_dla_samples)) # comment out these to save memory
#sample_log_priors_no_dla[:] = np.NaN
# sample_log_priors_dla = np.empty((num_quasars, dlaParams.num_dla_samples))
#sample_log_priors_dla[:] = np.NaN
# sample_log_likelihoods_no_dla = np.empty((num_quasars, dlaParams.num_dla_samples))
#sample_log_likelihoods_no_dl[:] = np.NaN
# sample_log_likelihoods_dla = np.empty((num_quasars, dlaParams.num_dla_samples))
#sample_log_likelihoods_dla[:] = np.NaN
sample_log_posteriors_no_dla  = np.empty((num_quasars, dlaParams.num_dla_samples))
sample_log_posteriors_no_dla[:] = np.NaN
sample_log_posteriors_dla = np.empty((num_quasars, dlaParams.num_dla_samples))
sample_log_posteriors_dla[:] = np.NaN
log_posteriors_no_dla = np.empty((num_quasars))
log_posteriors_no_dla[:] = np.NaN
log_posteriors_dla_sub = np.empty((num_quasars))
log_posteriors_dla_sub[:] = np.NaN
log_posteriors_dla_sup = np.empty((num_quasars))
log_posteriors_dla_sup[:] = np.NaN
log_posteriors_dla = np.empty((num_quasars))
log_posteriors_dla[:] = np.NaN
z_true = np.empty((num_quasars))
z_true[:] = np.NaN
dla_true = np.empty((num_quasars))
dla_true[:] = np.NaN
z_map = np.empty((num_quasars))
z_map[:] = np.NaN
z_dla_map = np.empty((num_quasars))
z_dla_map[:] = np.NaN
n_hi_map = np.empty((num_quasars))
n_hi_map[:] = np.NaN
log_nhi_map = np.empty((num_quasars))
log_nhi_map[:] = np.NaN
signal_to_noise = np.empty((num_quasars))
signal_to_noise[:] = np.NaN

c_0   = np.exp(log_c_0)
tau_0 = np.exp(log_tau_0)
beta  = np.exp(log_beta)

z_list = [x for x in range(len(offset_samples_qso))]
z_list = np.array(z_list)
#Debug output
#all_mus =[]

fluxes = []
rest_wavelengths = []
this_p_dlas = np.zeros((len(z_list)))

# this is just an array allow you to select a range of quasars to run
quasar_ind = 0
                            
#try
#    load(['./checkpointing/curDLA_', optTag, '.mat']); %checkmarking code
#catch ME
#    0;

q_ind_start = quasar_ind

# catch the exceptions
all_exceptions = np.zeros((num_quasars))

all_exceptions = [False for x in all_exceptions]
#for i in range(all_exceptions):
 #   all_exceptions[i] = False
    
all_posdeferrors = np.zeros((num_quasars))

#print("\nprior_ind")
#print(prior_ind)
#print(len(prior_ind), np.count_nonzero(prior_ind))
#print("\ndla_ind")
#print(dla_ind)
#print(dla_ind.shape, np.count_nonzero(dla_ind))
#print("\nrest_wavelengths")
#print(rest_wavelengths)
#print(rest_wavelengths.shape)
#print("\nmu")
#print(mu)
#print(mu.shape)
#print("\nM")
#print(M)
#print(M.shape)
#print("\nlog_omega")
#print(log_omega)
#print(len(log_omega))
#print("\nlog_c_0")
#print(log_c_0)
#print("\nlog_tau_0")
#print(log_tau_0)
#print("\nlog_beta")
#print(log_beta)
#print("\nbluewards_mu")
#print(bluewards_mu)
#print("\nbluewards_sigma")
#print(bluewards_sigma)
#print("\nredwards_mu")
#print(redwards_mu)
#print("\nredwards_sigma")
#print(redwards_sigma)
#print("\noffset_samples")
#print(offset_samples)
#print(offset_samples.shape)
#print("\noffset_samples_qso")
#print(offset_samples_qso)
#print(offset_samples_qso.shape)
#print("\nlog_nhi_samples")
##print(log_nhi_samples)
#print(log_nhi_samples.shape)
#print("\nhi_samples")
#print(nhi_samples)
#print(nhi_samples.shape)
#print("\nall_wavelengths")
#print(all_wavelengths)
#print(all_wavelengths.shape)
#print("\nall_flux")
#print(all_flux)
#print(all_flux.shape)
#print("\nall_noise_variance")
#print(all_noise_variance)
#print(all_noise_variance.shape)
#print("\nall_pixel_mask")
#print(all_pixel_mask)
#print(all_pixel_mask.shape)
#print("\ntest_ind")
#print(test_ind)
#print(len(test_ind))
#print("\nall_thing_ids")
#print(all_thing_ids)
#print(all_thing_ids.shape)
#print("\nz_qsos")
#print(z_qsos)
#print(len(z_qsos))
#print("\ndla_inds")
#print(dla_inds)
#print(dla_inds.shape)
#print("\nnum_quasars")
#print(num_quasars)
#print("\nmin_z_dlas")
#print(min_z_dlas)
#print(min_z_dlas.shape)
#print("\nmax_z_dlas")
#print(max_z_dlas)
#print(max_z_dlas.shape)
#print("\nsample_log_posteriors_no_dla")
#print(sample_log_posteriors_no_dla)
#print(sample_log_posteriors_no_dla.shape)
#print("\nsample_log_posteriors_dla")
#print(sample_log_posteriors_dla)
#print(sample_log_posteriors_dla.shape)
#print("\nlog_posteriors_no_dla")
#print(log_posteriors_no_dla)
#print(log_posteriors_no_dla.shape)
#print("\nlog_posteriors_dla_sub")
#print(log_posteriors_dla_sub)
#print(log_posteriors_dla_sub.shape)
#print("\nlog_posteriors_dla_sup")
#print(log_posteriors_dla_sup)
#print(log_posteriors_dla_sup.shape)
#print("\nlog_posteriors_dla")
#print(log_posteriors_dla)
#print(log_posteriors_dla.shape)
#print("\nz_true")
#print(z_true)
#print(z_true.shape)
#print("\ndla_true")
#print(dla_true)
#print(dla_true.shape)
#print("\nz_map")
#print(z_map)
#print(z_map.shape)
#print("\nz_dla_map")
#print(z_dla_map)
#print(z_dla_map.shape)
#print("\nwn_hi_map")
#print(n_hi_map)
#print(n_hi_map.shape)
#print("\nlog_nhi_map")
#print(log_nhi_map)
#print(log_nhi_map.shape)
#print("\nsignal_to_noise")
#print(signal_to_noise)
#print(signal_to_noise.shape)
#print("\nc_0")
#print(c_0)
#print("\ntau_0")
#print(tau_0)
#print("\nbeta")
#print(beta)
#print("\nz_list")
#print(z_list)
#print(len(z_list))
#print("\nfluxes")
#print(fluxes)
#print(fluxes.shape)
#print("\nrest_wavelengths")
#print(rest_wavelengths)
#print(rest_wavelengths.shape)
#print("\nthis_p_dlas")
#print(this_p_dlas)
#print(this_p_dlas.shape)
#print("\nq_ind_start")
#print(q_ind_start)
#print("\nall_exceptions")
#print(all_exceptions)
#print(len(all_exceptions))
#print("\nall_posdeferrors")
#print(all_posdeferrors)
#print(all_posdeferrors.shape)
print("\n\n\n")

for quasar_ind in range(q_ind_start, num_quasars): #quasar list
    t = time.time()
    quasar_num = qso_ind[quasar_ind]
    
    z_true[quasar_ind]   = z_qsos[quasar_num]
    dla_true[quasar_ind] = dla_inds[quasar_num]
    print('processing quasar {qua}/{num} (z_true = {zt:0.4f}) ...'.format(qua=quasar_ind+1, num=num_quasars, zt=z_true[quasar_ind]))

    #computing signal-to-noise ratio
    this_wavelengths    =    all_wavelengths[quasar_num]
    this_flux           =           all_flux[quasar_num]
    this_noise_variance = all_noise_variance[quasar_num]
    this_pixel_mask     =     all_pixel_mask[quasar_num]

    this_rest_wavelengths = emitted_wavelengths(this_wavelengths, 4.4088) 
    #roughly highest redshift possible (S2N for everything that may be in restframe)
    
    ind  = this_rest_wavelengths <= nullParams.max_lambda

    #print("\nquasar_num")
    #print(quasar_num)
    #print("\nz_true")
    #print(z_true)
    #print(z_true.shape)
    #print("\ndla_true")
    #print(dla_true)
    #print(dla_true.shape)
    #print("\nthis_wavelengths")
    #print(this_wavelengths)
    #print(this_wavelengths.shape)
    #print("\nthis_flux")
    #print(this_flux)
    #print(this_flux.shape)
    #print("\nthis_noise_variance")
    #print(this_noise_variance)
    #print(this_noise_variance.shape)
    #print("\nthis_rest_wavelengths")
    #print(this_rest_wavelengths)
    #print(this_rest_wavelengths.shape)
    #print("\nind")
    #print(ind)
    #print(len(ind), np.count_nonzero(ind))
    #print("\n\n\n")
    
    this_rest_wavelengths = this_rest_wavelengths[ind]
    this_flux             =             this_flux[ind]
    this_noise_variance   =   this_noise_variance[ind]

    this_noise_variance[np.isinf(this_noise_variance)] = .01 #kludge to fix bad data
    
    this_pixel_signal_to_noise  = np.sqrt(this_noise_variance) / abs(this_flux)

    # this is before pixel masking; nanmean to avoid possible NaN values
    signal_to_noise[quasar_ind] = np.nanmean(this_pixel_signal_to_noise)

    # this is saved for the MAP esitmate of z_QSO
    used_z_dla = np.empty((dlaParams.num_dla_samples))
    used_z_dla[:] = np.NaN

    # initialise some dummy arrays to reduce memory consumption 
    this_sample_log_priors_no_dla      = np.empty((dlaParams.num_dla_samples))
    this_sample_log_priors_no_dla[:] = np.NaN
    this_sample_log_priors_dla         = np.empty((dlaParams.num_dla_samples))
    this_sample_log_priors_dla[:] = np.NaN
    this_sample_log_likelihoods_no_dla = np.empty((dlaParams.num_dla_samples))
    this_sample_log_likelihoods_no_dla[:] = np.NaN
    this_sample_log_likelihoods_dla    = np.empty((dlaParams.num_dla_samples))
    this_sample_log_likelihoods_dla[:] = np.NaN


    # move these outside the parfor to avoid constantly querying these large arrays
    this_out_wavelengths    =    all_wavelengths[quasar_num]
    this_out_flux           =           all_flux[quasar_num]
    this_out_noise_variance = all_noise_variance[quasar_num]
    this_out_pixel_mask     =     all_pixel_mask[quasar_num]

    # Test: see if this spec is empty; this error handling line be outside parfor
    # would avoid running lots of empty spec in parallel workers
    if np.all(len(this_out_wavelengths) == 0):
        all_exceptions[quasar_ind] = 1
        continue;

    # record posdef error;
    # if it only happens for some samples not all of the samples, I would prefer
    # to think it is due to the noise_variace of the incomplete data combine with
    # the K causing the Covariance behaving weirdly.
    this_posdeferror = np.zeros((dlaParams.num_dla_samples))
    this_posdeferror = [False for x in this_posdeferror]
    this_posdeferror = np.array(this_posdeferror)
    
    #print("\nthis_noise_variance")
    #print(this_noise_variance)
    #print(this_noise_variance.shape)
    #print("\nthis_pixel_signal_to_noise")
    #print(this_pixel_signal_to_noise)
    #print(this_pixel_signal_to_noise.shape)
    #print("\nsignal_to_noise")
    #print(signal_to_noise)
    #print(signal_to_noise.shape)
    print("\nthis_out_wavelengths")
    print(this_out_wavelengths)
    print(this_out_wavelengths.shape)
    #print("\nthis_out_flux")
    #print(this_out_flux)
    #print(this_out_flux.shape)
    #print("\nthis_out_noise_variance")
    #print(this_out_noise_variance)
    #print(this_out_noise_variance.shape)
    #print("\nthis_out_pixel_mask")
    #print(this_out_pixel_mask)
    #print(this_out_pixel_mask.shape)
    #print("\nall_exceptions")
    #print(all_exceptions)
    #print(len(all_exceptions))
    #print("\nthis_posdeferror")
    #print(this_posdeferror)
    #print(len(this_posdeferror))
    
    #parfor i = 1:num_dla_samples       #variant redshift in quasars 
    #    z_qso = offset_samples_qso(i);
    for i in range(dlaParams.num_dla_samples):  #variant redshift in quasars 
        z_qso = offset_samples_qso[i]
        print("\nz_qso")
        print(z_qso)
        
        # keep a copy inside the parfor since we are modifying them
        this_wavelengths    = this_out_wavelengths
        this_flux           = this_out_flux
        this_noise_variance = this_out_noise_variance
        this_pixel_mask     = this_out_pixel_mask
        
        #print("\nthis_wavelengths")
        #print(this_wavelengths)
        #print(this_wavelengths.shape)

        #Cut off observations
        max_pos_lambda = observed_wavelengths(nullParams.max_lambda, z_qso)
        min_pos_lambda = observed_wavelengths(nullParams.min_lambda, z_qso)
        #print("\nmax_pos_lambda")
        #print(max_pos_lambda)
        #print("\nmin_pos_lambda")
        #print(min_pos_lambda)
        max_observed_lambda = min(max_pos_lambda, np.max(this_wavelengths))

        min_observed_lambda = max(min_pos_lambda, np.min(this_wavelengths))
        lambda_observed = (max_observed_lambda - min_observed_lambda)

        ind = (this_wavelengths > min_observed_lambda) & (this_wavelengths < max_observed_lambda)
        this_flux           = this_flux[ind]
        this_noise_variance = this_noise_variance[ind]
        this_wavelengths    = this_wavelengths[ind]
        this_pixel_mask     = this_pixel_mask[ind]

        # convert to QSO rest frame
        this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso)
        
        #print("\nz_qso")
        #print(z_qso)
        #print("\norig ind")
        #print(ind)
        #print(len(ind), np.count_nonzero(ind))
        #print("\nthis_wavelengths")
        #print(this_wavelengths)
        #print(this_wavelengths.shape)
        #print("\nthis_flux")
        #print(this_flux)
        #print(this_flux.shape)
        #print("\nthis_noise_variance")
        #print(this_noise_variance)
        #print(this_noise_variance.shape)
        #print("\nthis_pixel_mask")
        #print(this_pixel_mask)
        #print(this_pixel_mask.shape)
        #print("\nmax_observed_lambda")
        #print(max_observed_lambda)
        #print("\nmin_observed_lambda")
        #print(min_observed_lambda)
        #print("\nlambda_observed")
        #print(lambda_observed)
        #print("\nthis_rest_wavelenths")
        #print(this_rest_wavelengths)
        #print(this_rest_wavelengths.shape)
        #print("\n\n\n")
        
         #normalizing here
        ind = (this_rest_wavelengths >= normParams.normalization_min_lambda) & (this_rest_wavelengths <= normParams.normalization_max_lambda)
        #print("\nind")
        #print(ind)
        #print(len(ind), np.count_nonzero(ind))
        #print("this_flux before")
        #print(this_flux)
        #print(this_flux.shape)
        #print("this_flux after")
        #print(this_flux[ind])
        #print(this_flux[ind].shape)

        this_median         = np.nanmedian(this_flux[ind])
        this_flux           = this_flux / this_median
        this_noise_variance = this_noise_variance / pow(this_median, 2)
        
        #print("\nthis_median")
        #print(this_median)
        #print("\nthis_flux")
        #print(this_flux)
        #print(this_flux.shape)
        #print("\nthis_noise_variance")
        #print(this_noise_variance)
        #print(this_noise_variance.shape)

        ind = (this_rest_wavelengths >= nullParams.min_lambda) & (this_rest_wavelengths <= nullParams.max_lambda)
        #print("\nind 2")
        #print(ind)
        #print(len(ind), np.count_nonzero(ind))

        # keep complete copy of equally spaced wavelengths for absorption computation
        this_unmasked_wavelengths = this_wavelengths[ind]

        
        # Normalise the observed flux for out-of-range model since the
        # redward- and blueward- models were trained with normalisation.
        #Find probability for out-of-range model
        this_normalized_flux = this_out_flux / this_median  # since we've modified this_flux, we thus need to
                                                            # use this_out_flux which is outside the parfor loop
        this_normalized_noise_variance = this_out_noise_variance / pow(this_median, 2)

        # select blueward region
        ind_bw = (this_out_wavelengths < min_observed_lambda) & np.logical_not(this_out_pixel_mask)
        this_normalized_flux_bw           = this_normalized_flux[ind_bw]
        this_normalized_noise_variance_bw = this_normalized_noise_variance[ind_bw]
        n_bw = this_normalized_flux_bw.shape[0]
        
        # select redward region
        ind_rw = (this_out_wavelengths > max_observed_lambda)  & np.logical_not(this_out_pixel_mask)
        this_normalized_flux_rw           = this_normalized_flux[ind_rw]
        this_normalized_noise_variance_rw = this_normalized_noise_variance[ind_rw]
        n_rw = this_normalized_flux_rw.shape[0]

        # calculate log likelihood of iid multivariate normal with log N(y; mu, diag(V) + sigma^2 )
        bw_log_likelihood = log_mvnpdf_iid(this_normalized_flux_bw, bluewards_mu * np.ones((n_bw)), bluewards_sigma**2 * np.ones((n_bw)) + this_normalized_noise_variance_bw)
        rw_log_likelihood = log_mvnpdf_iid(this_normalized_flux_rw, redwards_mu * np.ones((n_rw)), redwards_sigma**2 * np.ones((n_rw)) + this_normalized_noise_variance_rw )
        
        ind = ind & np.logical_not(this_pixel_mask)
        
        this_wavelengths      =      this_wavelengths[ind]
        this_rest_wavelengths = this_rest_wavelengths[ind]
        this_flux             =             this_flux[ind]
        this_noise_variance   =   this_noise_variance[ind]

        this_noise_variance[np.isinf(this_noise_variance)] = np.nanmean(this_noise_variance) #rare kludge to fix bad data
        
        if (np.count_nonzero(this_rest_wavelengths) < 1):
            print(' took {tc:0.3f}s.\n'.format(tc=time.time() - t))
            fluxes.append(0)
            rest_wavelengths.append(0)
            
            continue;
            
        fluxes.append(this_flux)
        rest_wavelengths.append(this_rest_wavelengths)
        
        this_lya_zs = (this_wavelengths - physConst.lya_wavelength) / physConst.lya_wavelength
        
        # To count the effect of Lyman series from higher z,
        # we compute the absorbers' redshifts for all members of the series
        this_lyseries_zs = np.empty((len(this_wavelengths), learnParams.num_forest_lines))
        this_lyseries_zs[:] = np.NaN
        
        #print("\nthis_unmasked_wavelengths")
        #print(this_unmasked_wavelengths)
        #print(this_unmasked_wavelengths.shape)
        #print("\nthis_normalized_flux")
        #print(this_normalized_flux)
        #print(this_normalized_flux.shape)
        #print("\nthis_normalized_noise_variance")
        #print(this_normalized_noise_variance)
        #print(this_normalized_noise_variance.shape)
        #print("\nind_bw")
        #print(ind_bw)
        #print(len(ind_bw), np.count_nonzero(ind_bw))
        #print("\nthis_normalized_flux_bw")
        #print(this_normalized_flux_bw)
        #print(this_normalized_flux_bw.shape)
        #print("\nthis_normalized_noise_variance_bw")
        #print(this_normalized_noise_variance_bw)
        #print(this_normalized_noise_variance_bw.shape)
        #print("\nthen_bw")
        #print(n_bw)
        #print("\nind_rw")
        #print(ind_rw)
        #print(len(ind_rw), np.count_nonzero(ind_rw))
        #print("\nthis_normalized_flux_rw")
        #print(this_normalized_flux_rw)
        #print(this_normalized_flux_rw.shape)
        #print("\nthis_normalized_noise_variance_rw")
        #print(this_normalized_noise_variance_rw)
        #print(this_normalized_noise_variance_rw.shape)
        #print("\nthen_rw")
        #print(n_rw)
        #print("\nbw_log_likelihood")
        #print(bw_log_likelihood)
        #print(bw_log_likelihood.shape)
        #print("\nrw_log_likelihood")
        #print(rw_log_likelihood)
        #print(rw_log_likelihood.shape)
        #print("\nind 3")
        #print(ind)
        #print(len(ind), np.count_nonzero(ind))
        #print("\nthis_wavelengths")
        #print(this_wavelengths)
        #print(this_wavelengths.shape)
        #print("\nthis_rest_wavelengths")
        #print(this_rest_wavelengths)
        #print(this_rest_wavelengths.shape)
        #print("\nthis_flux")
        #print(this_flux)
        #print(this_flux.shape)
        #print("\nthis_noise_variance")
        #print(this_noise_variance)
        #print(this_noise_variance.shape)
        #print("\nfluxes")
        #print(fluxes)
        #print(len(fluxes))
        #print("\nrest_wavelengths")
        #print(rest_wavelengths)
        #print(len(rest_wavelengths))
        #print("\nthis_lya_zs")
        #print(this_lya_zs)
        #print(this_lya_zs.shape)
        #print("\nthis_lyseries_zs")
        #print(this_lyseries_zs)
        #print(this_lyseries_zs.shape)
        print("\n\n\n")
        
        for l in range(learnParams.num_forest_lines):
            this_lyseries_zs[:, l] = (this_wavelengths - learnParams.all_transition_wavelengths[l]) / learnParams.all_transition_wavelengths[l]

        #print("\nthis_lyseries_zs")
        #print(this_lyseries_zs)
        #print(this_lyseries_zs.shape)
        # DLA existence prior
        less_ind = (orig_z_qsos < (z_qso + modelParams.prior_z_qso_increase))
        #print("\nless_ind")
        #print(less_ind)
        #print(len(less_ind), np.count_nonzero(less_ind))
        #print("\ndla_ind")
        #print(dla_ind)
        #print(dla_ind.shape)

        this_num_dlas    = np.count_nonzero(dla_ind[less_ind])
        this_num_quasars = np.count_nonzero(less_ind)
        this_p_dla       = this_num_dlas / float(this_num_quasars)
        this_p_dlas[i]   = this_p_dla
        #print("\nthis_num_dlas")
        #print(this_num_dlas)
        #print("\nthis_num_quasars")
        #print(this_num_quasars)
        #print("\nthis_p_dla")
        #print(this_p_dla)
        #print("\nthis_p_dlas")
        #print(this_p_dlas)
        #print(this_p_dlas.shape)

        #minimal plausible prior to prevent NaN on low z_qso;
        if this_num_dlas == 0:
            this_num_dlas = 1
            this_num_quasars = len(less_ind)
        
        #print("\nafter this_num_dlas")
        #print(this_num_dlas)
        #print("\nafter this_num_quasars")
        #print(this_num_quasars)
        this_sample_log_priors_dla[i] = np.log(this_num_dlas) - np.log(this_num_quasars)
        this_sample_log_priors_no_dla[i] = np.log(this_num_quasars - this_num_dlas) - np.log(this_num_quasars)
        #print("\nthis_sample_log_priors_dla")
        #print(this_sample_log_priors_dla)
        #print(this_sample_log_priors_dla.shape)
        #print("\nthis_sample_log_priors_no_dla")
        #print(this_sample_log_priors_no_dla)
        #print(this_sample_log_priors_no_dla.shape)

        #sample_log_priors_dla(quasar_ind, z_list_ind) = log(.5);
        #sample_log_priors_no_dla(quasar_ind, z_list_ind) = log(.5);

        # fprintf_debug('\n');
        print(' ...     p(   DLA | z_QSO)        : {th:0.3f}\n'.format(th=this_p_dla))
        print(' ...     p(no DLA | z_QSO)        : {th:0.3f}\n'.format(th=1 - this_p_dla))

        # interpolate model onto given wavelengths
        this_mu = mu_interpolator(this_rest_wavelengths)
        #print("\nthis_mu")
        #print(this_mu)
        #print(this_mu.shape)
        this_M  =  M_interpolator((this_rest_wavelengths[:, None], np.arange(nullParams.k)[None, :]))
        #print("\nthis_M")
        #print(this_M)
        #print(this_M.shape)
        #Debug output
        #all_mus[z_list_ind] = this_mu
        #all_Ms[z_list_ind] = this_M

        this_log_omega = log_omega_interpolator(this_rest_wavelengths)
        #print("\nthis_log_omega")
        #print(this_log_omega)
        #print(this_log_omega.shape)
        this_omega2 = np.exp(2 * this_log_omega)
        #print("\nthis_omega2")
        #print(this_omega2)
        #print(this_omega2.shape)
        
        # Lyman series absorption effect for the noise variance
        # note: this noise variance must be trained on the same number of members of Lyman series
        lya_optical_depth = tau_0 * (1 + this_lya_zs)**beta
        #print("\nlya_optical_depth")
        #print(lya_optical_depth)
        #print(lya_optical_depth.shape)

        # Note: this_wavelengths is within (min_lambda, max_lambda)
        # so it may beyond lya_wavelength, so need an indicator;
        # Note: 1 - exp( -0 ) + c_0 = c_0
        indicator         = this_lya_zs <= z_qso
        lya_optical_depth = lya_optical_depth * indicator
        #print("\nindicator")
        #print(indicator)
        #print(len(indicator))
        #print("\nlya_optical_depth")
        #print(lya_optical_depth)
        #print(lya_optical_depth.shape)

        for l in range(1,learnParams.num_forest_lines):
            lyman_1pz = learnParams.all_transition_wavelengths[0] * (1 + this_lya_zs) / learnParams.all_transition_wavelengths[l]

            # only include the Lyman series with absorber redshifts lower than z_qso
            indicator = lyman_1pz <= (1 + z_qso)
            lyman_1pz = lyman_1pz * indicator

            tau = tau_0 * learnParams.all_transition_wavelengths[l] * learnParams.all_oscillator_strengths[l] / (learnParams.all_transition_wavelengths[0] * learnParams.all_oscillator_strengths[0])

            lya_optical_depth = lya_optical_depth + tau * lyman_1pz**beta

        this_scaling_factor = 1 - np.exp( -lya_optical_depth ) + c_0
        
        this_omega2 = this_omega2 * this_scaling_factor**2

        # Lyman series absorption effect on the mean-flux
        # apply the lya_absorption after the interpolation because NaN will appear in this_mu
        total_optical_depth = np.empty((len(this_wavelengths), learnParams.num_forest_lines))
        total_optical_depth[:] = np.NaN
        #print("\nlyman_1pz")
        #print(lyman_1pz)
        #print(lyman_1pz.shape, np.count_nonzero(lyman_1pz))
        #print("\nindicator")
        #print(indicator)
        #print(len(indicator), np.count_nonzero(indicator))
        #print("\ntau")
        #print(tau)
        #print(tau.shape)
        #print("\nlya_optical_depth")
        #print(lya_optical_depth)
        #print(lya_optical_depth.shape)
        #print("\nthis_scaling_factor")
        #print(this_scaling_factor)
        #print("\nthis_omega2")
        #print(this_omega2)
        #print("\ntotal_optical_depth")
        #print(total_optical_depth)
        #print(total_optical_depth.shape)

        for l in range(learnParams.num_forest_lines):
            # calculate the oscillator strength for this lyman series member
            this_tau_0 = prev_tau_0 * learnParams.all_oscillator_strengths[l] / learnParams.lya_oscillator_strength * learnParams.all_transition_wavelengths[l] / physConst.lya_wavelength

            total_optical_depth[:, l] = this_tau_0 * ( (1 + this_lyseries_zs[:, l])**prev_beta )

            # indicator function: z absorbers <= z_qso
            # here is different from multi-dla processing script
            # I choose to use zero instead or nan to indicate
            # values outside of the Lyman forest
            indicator = this_lyseries_zs[:, l] <= z_qso
            total_optical_depth[:, l] = total_optical_depth[:, l] * indicator

        # change from nansum to simply sum; shouldn't be different
        # because we also change indicator from nan to zero,
        # but if this script is glitchy then inspect this line
        lya_absorption = np.exp(- np.sum(total_optical_depth, 1) )
        #print("\nthis_tau_0")
        #print(this_tau_0)
        #print("\ntotal_optical_depth")
        #print(total_optical_depth)
        #print(total_optical_depth.shape)
        #print("\nindicator")
        #print(indicator)
        #print(len(indicator), np.count_nonzero(indicator))
        #print("\nlya_absorption")
        #print(lya_absorption)

        this_mu = this_mu * lya_absorption
        this_M  = this_M  * lya_absorption[:, None]

        # re-adjust (K + Ω) to the level of μ .* exp( -optical_depth ) = μ .* a_lya
        # now the null model likelihood is:
        # p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
        this_omega2 = this_omega2 * lya_absorption**2

        occams = occams_factor * (1 - lambda_observed / (nullParams.max_lambda - nullParams.min_lambda) )
        #print("\nthis_mu")
        #print(this_mu)
        #print(this_mu.shape)
        #print("\nthis_M")
        #print(this_M)
        #print(this_M.shape)
        #print("\nthis_omega2")
        #print(this_omega2)
        #print(this_omega2.shape)
        #print("\noccams")
        #print(occams)
        #print(occams.shape)
        #broked

        # baseline: probability of no DLA model
        # The error handler to deal with Postive definite errors sometime happen
        #try:
        this_sample_log_likelihoods_no_dla[i] = log_mvnpdf_low_rank(this_flux, this_mu, this_M, this_omega2 + this_noise_variance) + bw_log_likelihood + rw_log_likelihood - occams
        #this_sample_log_likelihoods_no_dla[i] = this_sample_log_likelihoods_no_dla[i] + bw_log_likelihood + rw_log_likelihood - occams
        #except:
            #if (strcmp(ME.identifier, 'MATLAB:posdef')):
        #    this_posdeferror[i] = True
        #    print('(QSO {qua}, Sample {it}): Matrix must be positive definite. We skip this sample but you need to be careful about this spectrum'.format(qua=quasar_num, it=i))
        #    continue;
                
        #    raise

        print(' ... log p(D | z_QSO, no DLA)     : {ts:0.2f}\n'.format(ts=this_sample_log_likelihoods_no_dla[i]))

        # Add
        if this_wavelengths.shape[0] == 0:
            print("dunded")
            continue;
            #break

        # use a temp variable to avoid the possible parfor issue
        # should be fine after change size of min_z_dlas to (num_quasar, num_dla_samples)
        this_min_z_dlas = moreParams.min_z_dla(this_wavelengths, z_qso)
        this_max_z_dlas = moreParams.max_z_dla(this_wavelengths, z_qso)

        min_z_dlas[quasar_ind, i] = this_min_z_dlas
        max_z_dlas[quasar_ind, i] = this_max_z_dlas

        sample_z_dlas = this_min_z_dlas + (this_max_z_dlas - this_min_z_dlas) * offset_samples

        used_z_dla[i] = sample_z_dlas[i]
        print("\nthis_min_z_dlas")
        print(this_min_z_dlas)
        print(this_min_z_dlas.shape)
        print("\nthis_max_z_dlas")
        print(this_max_z_dlas)
        print(this_max_z_dlas.shape)
        print("\nmin_z_dlas")
        print(min_z_dlas)
        print(min_z_dlas.shape)
        print("\nmax_z_dlas")
        print(max_z_dlas)
        print(max_z_dlas.shape)
        print("\nsample_z_dlas")
        print(sample_z_dlas)
        print(sample_z_dlas.shape)
        print("\nused_z_dla")
        print(used_z_dla)
        print(used_z_dla.shape)

        # ensure enough pixels are on either side for convolving with instrument profile
        padded_wavelengths = [np.logspace(np.log10(np.min(this_unmasked_wavelengths)) - instrumentParams.width * instrumentParams.pixel_spacing,
                            np.log10(np.min(this_unmasked_wavelengths)) - instrumentParams.pixel_spacing, instrumentParams.width).T,
                            this_unmasked_wavelengths,
                            np.logspace(np.log10(np.max(this_unmasked_wavelengths)) + instrumentParams.pixel_spacing,
                            np.log10(np.max(this_unmasked_wavelengths)) + instrumentParams.width * instrumentParams.pixel_spacing, instrumentParams.width).T]

        padded_wavelengths = np.array([x for ls in padded_wavelengths for x in ls])
        print("\npadded_wavelengths")
        print(padded_wavelengths)
        print(padded_wavelengths.shape)
        # to retain only unmasked pixels from computed absorption profile
        ind = np.logical_not(this_pixel_mask[ind])
        print("\nind")
        print(ind)
        print(len(ind), np.count_nonzero(ind))

        # compute probabilities under DLA model for each of the sampled
        # (normalized offset, log(N HI)) pairs absorption corresponding to this sample
        absorption = voigt(padded_wavelengths, sample_z_dlas[i], nhi_samples[i], num_lines)
        print("\nabsorption")
        print(absorption)
        print(absorption.shape)
        break
    
    elapsed = time.time() - t
    print("elapsed")
    print(elapsed)
    break