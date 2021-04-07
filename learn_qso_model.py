'''training_release  = 'dr12q';
dla_catalog_name = 'dr9q_concordance';
train_ind = ...
    [' catalog.in_dr9                     & ' ...
     '(catalog.filter_flags == 0)         & ' ...
     ' catalog.los_inds(dla_catalog_name) & ' ...
     '~catalog.dla_inds(dla_catalog_name)']; 

% null model parameters
min_lambda         =     910;                 % range of rest wavelengths to       Å
max_lambda         =    3000;                 %   model
dlambda            =    0.25;                 % separation of wavelength grid      Å
k                  = 20;                      % rank of non-diagonal contribution
max_noise_variance = 4^2;                     % maximum pixel noise allowed during model training

% optimization parameters
initial_c     = 0.1;                          % initial guess for c
initial_tau_0 = 0.0023;                       % initial guess for τ₀
initial_beta  = 3.65;                         % initial guess for β
minFunc_options =               ...           % optimization options for model fitting
    struct('MaxIter',     4000, ...
           'MaxFunEvals', 8000);
       
training_set_name = 'dr9q_minus_concordance';'''

# learn_qso_model: fits GP to training catalog via maximum likelihood

#rng('default');

# load catalog
#catalog = load(sprintf('%s/catalog', processed_directory(training_release)));

# load preprocessed QSOs
#variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
#                     'all_pixel_mask'};
#preqsos = matfile(sprintf('%s/preloaded_qsos.mat', processed_directory(training_release)));

# determine which spectra to use for training; allow string value for
# train_ind
#if (ischar(train_ind))
#  train_ind = eval(train_ind);
#end

# select training vectors
#all_wavelengths    =          preqsos.all_wavelengths;
#all_wavelengths    =    all_wavelengths(train_ind, :);
#all_flux           =                 preqsos.all_flux;
#all_flux           =           all_flux(train_ind, :);
#all_noise_variance =       preqsos.all_noise_variance;
#all_noise_variance = all_noise_variance(train_ind, :);
#all_pixel_mask     =           preqsos.all_pixel_mask;
#all_pixel_mask     =     all_pixel_mask(train_ind, :);
#z_qsos             =        catalog.z_qsos(train_ind);
#clear preqsos

#num_quasars = numel(z_qsos);

#rest_wavelengths = (min_lambda:dlambda:max_lambda);
#num_rest_pixels  = numel(rest_wavelengths);

#lya_1pzs             = nan(num_quasars, num_rest_pixels);
#all_lyman_1pzs       = nan(num_forest_lines, num_quasars, num_rest_pixels);
#rest_fluxes          = nan(num_quasars, num_rest_pixels);
#rest_noise_variances = nan(num_quasars, num_rest_pixels);

# the preload_qsos should fliter out empty spectra;
# this line is to prevent there is any empty spectra
# in preloaded_qsos.mat for some reason
#is_empty             = false(num_quasars, 1);

# interpolate quasars onto chosen rest wavelength grid
#for i = 1:num_quasars
#  z_qso = z_qsos(i);

#  this_wavelengths    =    all_wavelengths{i}';
#  this_flux           =           all_flux{i}';
#  this_noise_variance = all_noise_variance{i}';
#  this_pixel_mask     =     all_pixel_mask{i}';

#  this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso);

#  this_flux(this_pixel_mask)           = nan;
#  this_noise_variance(this_pixel_mask) = nan;

#  fprintf('processing quasar %i with lambda_size = %i %i ...\n', i, size(this_wavelengths));

#  if all(size(this_wavelengths) == [0 0])
#    is_empty(i, 1) = 1;
#    continue;
#  end

#  lya_1pzs(i, :) = ...
#      interp1(this_rest_wavelengths, ...
#              1 + (this_wavelengths - lya_wavelength) / lya_wavelength, ...
#              rest_wavelengths);
  
  # this_wavelength is raw wavelength (w/t ind)
  # so we need an indicator here to comfine lya_1pzs
  # below Lyman alpha (do we need to make the indicator
  # has a lower bound at Lyman limit here?)
  # indicator = lya_1pzs(i, :) <= (1 + z_qso);
  # lya_1pzs(i, :) = lya_1pzs(i, :) .* indicator;

  # include all members in Lyman series to the forest
  #for j = 1:num_forest_lines
  #  this_transition_wavelength = all_transition_wavelengths(j);

  #  all_lyman_1pzs(j, i, :) = ...
  #    interp1(this_rest_wavelengths, ...
  #            1 + (this_wavelengths - this_transition_wavelength) / this_transition_wavelength, ... 
  #            rest_wavelengths);

    # indicator function: z absorbers <= z_qso
    #indicator = all_lyman_1pzs(j, i, :) <= (1 + z_qso);

    #all_lyman_1pzs(j, i, :) = all_lyman_1pzs(j, i, :) .* indicator;
  #end

  #rest_fluxes(i, :) = ...
  #    interp1(this_rest_wavelengths, this_flux,           rest_wavelengths);

  #normalizing here
  #ind = (this_rest_wavelengths >= normalization_min_lambda) & ...
  #      (this_rest_wavelengths <= normalization_max_lambda) & ...
  #      (~this_pixel_mask);

  #this_median = nanmedian(this_flux(ind));
  #rest_fluxes(i, :) = rest_fluxes(i, :) / this_median;

  #rest_noise_variances(i, :) = ...
  #    interp1(this_rest_wavelengths, this_noise_variance, rest_wavelengths);
  #rest_noise_variances(i, :) = rest_noise_variances(i, :) / this_median .^ 2;  %setting up bluward/redwards of restframe txt files

  # normalise the data we put into end model fitting
  #this_norm_flux           = this_flux / this_median;
  #this_norm_noise_variance = this_noise_variance / this_median .^ 2;

  #bluewards_flux{i} = this_norm_flux(this_rest_wavelengths < min_lambda & ~this_pixel_mask);
  #bluewards_nv{i}   = this_norm_noise_variance(this_rest_wavelengths < min_lambda & ~this_pixel_mask);
  #redwards_flux{i}  = this_norm_flux(this_rest_wavelengths > max_lambda & ~this_pixel_mask);
  #redwards_nv{i}    = this_norm_noise_variance(this_rest_wavelengths > max_lambda & ~this_pixel_mask);
#end
#bluewards_flux = cell2mat(bluewards_flux);
#bluewards_nv = cell2mat(bluewards_nv);
#redwards_flux = cell2mat(redwards_flux);
#redwards_nv = cell2mat(redwards_nv);

#addpath('./offrestfit');

#[bluewards_mu, bluewards_sigma] = fitendmodel(bluewards_flux, bluewards_nv);
#[redwards_mu, redwards_sigma] = fitendmodel(redwards_flux, redwards_nv);

#clear('all_wavelengths', 'all_flux', 'all_noise_variance', 'all_pixel_mask');
#clear('bluewards_flux', 'bluewards_nv', 'redwards_flux', 'redwards_nv');

# filter out empty spectra
# note: if you've done this in preload_qsos then skip these lines
#z_qsos               = z_qsos(~is_empty);
#lya_1pzs             = lya_1pzs(~is_empty, :);
#rest_fluxes          = rest_fluxes(~is_empty, :);
#rest_noise_variances = rest_noise_variances(~is_empty, :);
#all_lyman_1pzs       = all_lyman_1pzs(:, ~is_empty, :);

# update num_quasars in consideration
#num_quasars = numel(z_qsos);

#fprintf('Get rid of empty spectra, num_quasars = %i\n', num_quasars);

# mask noisy pixels
#ind = (rest_noise_variances > max_noise_variance);
#fprintf("Masking %g of pixels\n", nnz(ind) * 1 ./ numel(ind));
#lya_1pzs(ind)             = nan;
#rest_fluxes(ind)          = nan;
#rest_noise_variances(ind) = nan;
#for i = 1:num_quasars
#  for j = 1:num_forest_lines
#    all_lyman_1pzs(j, i, ind(i, :))  = nan;
#  end
#end

# reverse the rest_fluxes back to the fluxes before encountering Lyα forest
#prev_tau_0 = 0.0023; % Kim et al. (2007) priors
#prev_beta  = 3.65;

#rest_fluxes_div_exp1pz      = nan(num_quasars, num_rest_pixels);
#rest_noise_variances_exp1pz = nan(num_quasars, num_rest_pixels);

#for i = 1:num_quasars
  # compute the total optical depth from all Lyman series members
  # Apr 8: not using NaN here anymore due to range beyond Lya will all be NaNs
  #total_optical_depth = zeros(num_forest_lines, num_rest_pixels);

  #for j = 1:num_forest_lines
  #  % calculate the oscillator strengths for Lyman series
# this_tau_0 = prev_tau_0 * ...
#     all_oscillator_strengths(j)   / lya_oscillator_strength * ...
#     all_transition_wavelengths(j) / lya_wavelength;
    
    # remove the leading dimension
    #this_lyman_1pzs = squeeze(all_lyman_1pzs(j, i, :))'; % (1, num_rest_pixels)

    #total_optical_depth(j, :) = this_tau_0 .* (this_lyman_1pzs.^prev_beta);
  #end

  # Apr 8: using zeros instead so not nansum here anymore
  # beyond lya, absorption fcn shoud be unity
  #lya_absorption = exp(- sum(total_optical_depth, 1) );

  # We have to reverse the effect of Lyα for both mean-flux and observational noise
  #rest_fluxes_div_exp1pz(i, :)      = rest_fluxes(i, :) ./ lya_absorption;
  #rest_noise_variances_exp1pz(i, :) = rest_noise_variances(i, :) ./ (lya_absorption.^2);
#end

#clear('all_lyman_1pzs');

# Filter out spectra which have too many NaN pixels
#ind = sum(isnan(rest_fluxes_div_exp1pz),2) < num_rest_pixels-min_num_pixels;

#fprintf("Filtering %g quasars\n", length(rest_fluxes_div_exp1pz) - nnz(ind));

#z_qsos                      = z_qsos(ind);
#rest_fluxes_div_exp1pz      = rest_fluxes_div_exp1pz(ind, :);
#rest_noise_variances_exp1pz = rest_noise_variances_exp1pz(ind, :);
#lya_1pzs                    = lya_1pzs(ind, :);

# Check for columns which contain only NaN on either end.
#nancolfrac = sum(isnan(rest_fluxes_div_exp1pz), 1) / nnz(ind);
#fprintf("Columns with nan > 0.9: ");

#max(find(nancolfrac > 0.9))

# find empirical mean vector and center data
#mu = nanmean(rest_fluxes_div_exp1pz);
#centered_rest_fluxes = bsxfun(@minus, rest_fluxes_div_exp1pz, mu);
#clear('rest_fluxes', 'rest_fluxes_div_exp1pz');

# small fix to the data fit into the pca:
# make the NaNs to the medians of a given row
# rememeber not to inject this into the actual
# joint likelihood maximisation
#pca_centered_rest_flux = centered_rest_fluxes;

#[num_quasars, ~] = size(pca_centered_rest_flux);

#for i = 1:num_quasars
#  this_pca_centered_rest_flux = pca_centered_rest_flux(i, :);

  # assign median value for each row to nan
  #ind = isnan(this_pca_centered_rest_flux);

  #pca_centered_rest_flux(i, ind) = nanmedian(this_pca_centered_rest_flux);
#end

# get top-k PCA vectors to initialize M
#[coefficients, ~, latent] = ...
#  pca(pca_centered_rest_flux, ...
#        'numcomponents', k, ...
#        'rows',          'complete');

#objective_function = @(x) objective(x, centered_rest_fluxes, lya_1pzs, ...
#        rest_noise_variances_exp1pz, num_forest_lines, all_transition_wavelengths, ...
#        all_oscillator_strengths, z_qsos);

# initialize A to top-k PCA components of non-DLA-containing spectra
#initial_M = bsxfun(@times, coefficients(:, 1:k), sqrt(latent(1:k))');

# initialize log omega to log of elementwise sample standard deviation
#initial_log_omega = log(nanstd(centered_rest_fluxes));

#initial_log_c_0   = log(initial_c_0);
#initial_log_tau_0 = log(initial_tau_0);
#initial_log_beta  = log(initial_beta);

#initial_x = [initial_M(:);         ...
#             initial_log_omega(:); ...
#             initial_log_c_0;      ...
#             initial_log_tau_0;    ...
#             initial_log_beta];

# maximize likelihood via L-BFGS
#[x, log_likelihood, ~, minFunc_output] = ...
#    minFunc(objective_function, initial_x, minFunc_options);

#ind = (1:(num_rest_pixels * k));
#M = reshape(x(ind), [num_rest_pixels, k]);

#ind = ((num_rest_pixels * k + 1):(num_rest_pixels * (k + 1)));
#log_omega = x(ind)';

#log_c_0   = x(end - 2);
#log_tau_0 = x(end - 1);
#log_beta  = x(end);

#variables_to_save = {'training_release', 'train_ind', 'max_noise_variance', ...
#                     'minFunc_options', 'rest_wavelengths', 'mu', ...
#                     'initial_M', 'initial_log_omega', 'initial_log_c_0', ...
#                     'initial_tau_0', 'initial_beta',  'M', 'log_omega', ...
#                     'log_c_0', 'log_tau_0', 'log_beta', 'log_likelihood', ...
#                     'minFunc_output', 'bluewards_mu', 'bluewards_sigma', ...
#                     'redwards_mu', 'redwards_sigma'};

#save(sprintf('%s/learned_model_outdata_%s_norm_%d-%d',             ...
#             processed_directory(training_release), ...
#             training_set_name, ...
#	           normalization_min_lambda, normalization_max_lambda), ...
#variables_to_save{:}, '-v7.3');
import random
import dill
import pickle
from pathlib import Path
import numpy as np
import os

dill.load_session("parameters.pkl")
preParams = preproccesing_params()
normParams = normalization_params()
learnParams = learning_params()
loading = file_loading()
nullParams = null_params()
nullParams.min_lambda = 910             # range of rest wavelengths to       Å
nullParams.max_lambda = 3000            #   model
nullParams.dlambda = 0.25               # separation of wavelength grid      Å
nullParams.k = 20                       # rank of non-diagonal contribution
nullParams.max_noise_variance = 4^2     # maximum pixel noise allowed during model training

# optimization parameters
initial_c     = 0.1                          # initial guess for c
initial_tau_0 = 0.0023                       # initial guess for τ₀
initial_beta  = 3.65                         # initial guess for β

train_ind = [' catalog.in_dr9                     & ', '(catalog.filter_flags == 0)         & ', ' catalog.los_inds(dla_catalog_name) & ','~catalog.dla_inds(dla_catalog_name)']

training_release  = 'dr12q'
dla_catalog_name = 'dr9q_concordance'
training_set_name = 'dr9q_minus_concordance'

random.seed()
print(random.random())

p = Path(os.getcwd())
parent_dir = str(p.parent)
release = "dr12q/processed/catalog"
filename = os.path.join(parent_dir, release)
#getting back pickled data for catalog
with open(filename,'rb') as f:
     catalog = pickle.load(f)

print('\ncatalog')
print(catalog)

#getting back preprocessed qso data
release = "dr12q/processed/preloaded_qsos"
filename = os.path.join(parent_dir, release)
with open(filename, 'rb') as f:
    preqsos = pickle.load(f)

#print('\npreqsos')
#print(preqsos)
#if (train_ind.isalpha()):
#    train_ind = eval(train_ind)
#part1 = catalog['in_dr9']
#part2 = catalog['filter_flags']==0
#part3 =  catalog['los_inds']['dr9q_concordance']
#part4 = ~catalog['dla_inds']['dr9q_concordance']
train_ind = np.bitwise_and(np.bitwise_and(np.bitwise_and(catalog['in_dr9'], (catalog['filter_flags']==0)), catalog['los_inds']['dr9q_concordance']), ~catalog['dla_inds']['dr9q_concordance'])
#truncation for debugging but before does something
z_qsos             =        catalog['z_qsos'][train_ind]
train_ind = train_ind[:500]
#print('\npart1')
#print(part1)
#print('\npart2')
#print(part2)
#print('\npart3')
#print(part3)
#print('\npart4')
#print(part4)
print('\ntrain_ind')
#print(train_ind)
#print(type(train_ind))
print(len(train_ind))
#print('in_dr9')
#print(catalog['in_dr9'])
#print('los_inds')
#print(catalog['los_inds'])
#print(catalog['los_inds']['dr9q_concordance'])

# select training vectors
#all_wavelengths    =          preqsos['all_wavelengths']
#print('\nall_wavelengths')
#print(len(all_wavelengths))
#print(all_wavelengths)
#num_quasars = 500
#for ind in range(num_quasars):
#    numb = all_wavelengths[ind]
#    all_wavelengths[ind] = numb[train_ind]
#all_wavelengths    =    all_wavelengths[wl for wl in all_wavelengths wl=wl[train_ind]]
#output = 0
#for i in train_ind:
#    if i == True:
#        output+=1
#print("output", output)
#filtered_list = [i for (i, v) in zip(all_wavelengths, train_ind) if v]
#print('\nfiltered_list')
#print(len(filtered_list))
#print(filtered_list)
#print('\nall_wavelengths')
#print(len(all_wavelengths))
#print(all_wavelengths)
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

#num_quasars = len(z_qsos)
num_quasars = 500

print("\nlength of values")
print(len(all_wavelengths), len(all_flux), len(all_noise_variance), len(all_pixel_mask), len(z_qsos))

rest_wavelengths = np.arange(nullParams.min_lambda,nullParams.max_lambda+nullParams.dlambda,nullParams.dlambda)
num_rest_pixels  = rest_wavelengths.size

print("\nrest_wavelengths")
print(len(rest_wavelengths))
print(rest_wavelengths)
print('\nisnum_rest_pixels')
print(num_rest_pixels)

lya_1pzs             = np.empty((num_quasars, num_rest_pixels))
lya_1pzs[:] = np.NaN
all_lyman_1pzs       = np.empty((learnParams.num_forest_lines, num_quasars, num_rest_pixels))
all_lyman_1pzs[:] = np.NaN
rest_fluxes          = np.empty((num_quasars, num_rest_pixels))
rest_fluxes[:] = np.NaN
rest_noise_variances = np.empty((num_quasars, num_rest_pixels))
rest_noise_variances[:] = np.NaN

#print("\nlya_1pzs")
#print(len(lya_1pzs))
#print(lya_1pzs)
#print("\nall_lyman_1pzs")
#print(len(all_lyman_1pzs))
#print(all_lyman_1pzs)
#print("\nrest_fluxes")
#print(len(rest_fluxes))
#print(rest_fluxes)
#print("\nrest_noise_variances")
#print(len(rest_noise_variances))
#print(rest_noise_variances)

# the preload_qsos should fliter out empty spectra;
# this line is to prevent there is any empty spectra
# in preloaded_qsos.mat for some reason
is_empty             = np.zeros((num_quasars, 1), dtype=int)
#print('\nis_empty')
#print(len(is_empty))
#print(is_empty)

# interpolate quasars onto chosen rest wavelength grid
for i in range(num_quasars):
    z_qso = z_qsos[i]

    this_wavelengths    =    all_wavelengths[i].transpose()
    this_flux           =           all_flux[i].transpose()
    this_noise_variance = all_noise_variance[i].transpose()
    this_pixel_mask     =     all_pixel_mask[i].transpose()

    this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso)

    this_flux[this_pixel_mask]           = np.NaN
    this_noise_variance[this_pixel_mask] = np.NaN

    #fprintf('processing quasar %i with lambda_size = %i %i ...\n', i, size(this_wavelengths));
    print("processing quasar {num} with lambda size = {size} ...\n".format(num=i, size=this_wavelengths.shape[0]))

    print('\nthis_wavelengths')
    print(len(this_wavelengths), type(this_wavelengths))
    print(this_wavelengths)
    
    if all(this_wavelengths.shape == np.shape([[0, 0]])):
        #np.all(x==0)
        is_empty[i, 1] = 1
        continue

    lya_1pzs[i] = interp1(this_rest_wavelengths, 1 + (this_wavelengths - lya_wavelength) / lya_wavelength, rest_wavelengths)
 
    # this_wavelength is raw wavelength (w/t ind)
    # so we need an indicator here to comfine lya_1pzs
    # below Lyman alpha (do we need to make the indicator
    # has a lower bound at Lyman limit here?)
    # indicator = lya_1pzs(i, :) <= (1 + z_qso);
    # lya_1pzs(i, :) = lya_1pzs(i, :) .* indicator;

    # include all members in Lyman series to the forest