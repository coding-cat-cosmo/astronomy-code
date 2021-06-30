'''% DLA model parameters: parameter samples
num_dla_samples     = 100000;                 % number of parameter samples
alpha               = 0.9;                    % weight of KDE component in mixture
uniform_min_log_nhi = 20.0;                   % range of column density samples    [cm⁻²]
uniform_max_log_nhi = 23.0;                   % from uniform distribution
fit_min_log_nhi     = 20.0;                   % range of column density samples    [cm⁻²]
fit_max_log_nhi     = 22.0;                   % from fit to log PDF
training_release  = 'dr12q';
'''
#% generate_dla_samples: generates DLA parameter samples from training
#% catalog

#% load training catalog
#catalog = load(sprintf('%s/catalog', processed_directory(training_release)));

#% generate quasirandom samples from p(normalized offset, log₁₀(N_HI))
#rng('default');
#sequence = scramble(haltonset(3), 'rr2');

#% the first dimension can be used directly for the uniform prior over
#% offsets
#offset_samples  = sequence(1:num_dla_samples, 1)';

#% ADDING: second dimension for z_qso
#offset_samples_qso  = sequence(1:num_dla_samples, 2)';
#z_qsos = catalog.z_qsos;
#bins = 150;
#[z_freq, z_bin] = histcounts(z_qsos, [z_qso_cut : ((max(z_qsos) - z_qso_cut) / bins) : max(z_qsos)]);
#for i=length(z_freq):-1:1 z_freq(i) = sum(z_freq(1:i)); end
#z_freq = [0 z_freq]; z_freq = z_freq / max(z_freq);
#[z_freq, I] = unique(z_freq); z_bin = z_bin(I);
#offset_samples_qso = interp1(z_freq, z_bin, offset_samples_qso);

#% we must transform the second dimension to have the correct marginal
#% distribution for our chosen prior over column density, which is a
#% mixture of a uniform distribution on log₁₀ N_HI and a distribution
#% we fit to observed data

#% uniform component of column density prior
#u = makedist('uniform', ...
#             'lower', uniform_min_log_nhi, ...
#             'upper', uniform_max_log_nhi);

#% extract observed log₁₀ N_HI samples from catalog
#all_log_nhis = catalog.log_nhis(dla_catalog_name);
#ind = cellfun(@(x) (~isempty(x)), all_log_nhis);
#log_nhis = cat(1, all_log_nhis{ind});

#% make a quadratic fit to the estimated log p(log₁₀ N_HI) over the
#% specified range
#x = linspace(fit_min_log_nhi, fit_max_log_nhi, 1e3);
#kde_pdf = ksdensity(log_nhis, x);
#f = polyfit(x, log(kde_pdf), 2);


#extrapolate_min_log_nhi = 19.0; % normalization range for the extrapolated region
#% convert this to a PDF and normalize
#if ~extrapolate_subdla
#    unnormalized_pdf = @(nhi) (exp(polyval(f, nhi)));
#    Z = integral(unnormalized_pdf, fit_min_log_nhi, 25.0);
#else
#    unnormalized_pdf = ...
#        @(nhi) ( exp(polyval(f,  nhi))       .*      heaviside( nhi - 20.03269 ) ...
#        +   exp(polyval(f,  20.03269))  .* (1 - heaviside( nhi - 20.03269 )) );
#    Z = integral(unnormalized_pdf, extrapolate_min_log_nhi, 25.0);
#end

#% create the PDF of the mixture between the uniform distribution and
#% the distribution fit to the data
#normalized_pdf = @(nhi) ...
#          alpha  * (unnormalized_pdf(nhi) / Z) + ...
#     (1 - alpha) * (pdf(u, nhi));

# if ~extrapolate_subdla
#     cdf = @(nhi) (integral(normalized_pdf, fit_min_log_nhi, nhi));
# else
#     cdf = @(nhi) (integral(normalized_pdf, extrapolate_min_log_nhi, nhi));
# end


#% use inverse transform sampling to convert the quasirandom samples on
#% [0, 1] to appropriate values
#log_nhi_samples = zeros(1, num_dla_samples);
#for i = 1:num_dla_samples
#  log_nhi_samples(i) = ...
#      fzero(@(nhi) (cdf(nhi) - sequence(i, 3)), 20.5);
#end

#% precompute N_HI samples for convenience
#nhi_samples = 10.^log_nhi_samples;

#variables_to_save = {'uniform_min_log_nhi', 'uniform_max_log_nhi', ...
#                     'fit_min_log_nhi', 'fit_max_log_nhi', 'alpha', ...
#                     'offset_samples', 'log_nhi_samples', 'nhi_samples', ...
#                     'offset_samples_qso'};
#save(sprintf('%s/dla_samples', processed_directory(training_release)), ...
#     variables_to_save{:}, '-v7.3');

# generate_dla_samples: generates DLA parameter samples from training
# catalog

import random
import dill
import pickle
from pathlib import Path
import numpy as np
import os
from skopt.sampler import Halton
from scipy import interpolate
from scipy.stats import uniform
from scipy import stats
import scipy.integrate as integrate
from scipy.stats import norm
from scipy.optimize import root_scalar

#dill.load_session("parameters.pkl")

# DLA model parameters: parameter samples
#num_dla_samples     = 100000                 # number of parameter samples
#alpha               = 0.9                    # weight of KDE component in mixture
#uniform_min_log_nhi = 20.0                   # range of column density samples    [cm⁻²]
#uniform_max_log_nhi = 23.0                   # from uniform distribution
#fit_min_log_nhi     = 20.0                   # range of column density samples    [cm⁻²]
#fit_max_log_nhi     = 22.0                   # from fit to log PDF

dlaParams = dla_params()
preParams = preproccesing_params()
flag = flags()

training_release  = 'dr12q'

random.seed()
print(random.random())
dla_catalog_name  = 'dr9q_concordance'

p = Path(os.getcwd())
parent_dir = str(p.parent)
release = "dr12q/processed/catalog"
filename = os.path.join(parent_dir, release)
#getting back pickled data for catalog
#try:
with open(filename,'rb') as f:
    catalog = pickle.load(f)
#except:
#print('\ncatalog')
#print(catalog)

# generate quasirandom samples from p(normalized offset, log₁₀(N_HI))
#rng('default');
#sequence = scramble(haltonset(3), 'rr2');
#offset_samples = skopt.sampler.generate(3, num_dla_samples)
halton = Halton()
ls = [(0.,1.), (0.,1.), (0.,1.)]
print(type(ls))
samples = halton.generate(ls, dlaParams.num_dla_samples)
samples = np.array(samples)
print("\nsamples", samples.shape)
# the first dimension can be used directly for the uniform prior over offsets
offset_samples = samples[:,0]
# ADDING: second dimension for z_qso
offset_samples_qso = samples[:,1]
# ADDING: third dimension for z_qso
other_offset_samples = samples[:,2]
z_qsos = catalog["z_qsos"]
bins = 150

#rest_wavelengths = (min_lambda:dlambda:max_lambda);
#vect = np.arange(nullParams.min_lambda,nullParams.max_lambda+nullParams.dlambda,nullParams.dlambda)
vect = np.arange(preParams.z_qso_cut, np.max(z_qsos) + (np.max(z_qsos) - preParams.z_qso_cut) / bins, (np.max(z_qsos) - preParams.z_qso_cut) / bins)
#[z_freq, z_bin] = histcounts(z_qsos, [z_qso_cut : ((max(z_qsos) - z_qso_cut) / bins) : max(z_qsos)]);
z_freq, z_bin = np.histogram(z_qsos, vect)
#vect = list(z_qso_cut : np.max(z_qsos) : ((np.max(z_qsos) - z_qso_cut) / bins))
print("length", len(z_freq))
for i in range(len(z_freq)-1, -1, -1): #=length(z_freq):-1:1
    z_freq[i] = np.sum(z_freq[:i+1])

z_freq = np.insert(z_freq, 0, 0)
print(np.max(z_freq))
#print("before div", z_freq, z_freq.shape)
z_freq = z_freq / float(np.max(z_freq))
#print("before unique", z_freq, z_freq.shape)
z_freq, I = np.unique(z_freq, return_index=True)
z_bin = z_bin[I]
#rest_fluxes(i, :) = interp1(this_rest_wavelengths, this_flux,           rest_wavelengths);
#offset_samples_qso = interp1(z_freq, z_bin, offset_samples_qso);
offset_interp = interpolate.interp1d(z_freq, z_bin, bounds_error=False)
offset_samples_qso = offset_interp(offset_samples_qso)

# we must transform the second dimension to have the correct marginal
# distribution for our chosen prior over column density, which is a
# mixture of a uniform distribution on log₁₀ N_HI and a distribution
# we fit to observed data

# uniform component of column density prior
#u = makedist('uniform', 'lower', uniform_min_log_nhi, 'upper', uniform_max_log_nhi);
loc = dlaParams.uniform_min_log_nhi
scale = dlaParams.uniform_max_log_nhi - dlaParams.uniform_min_log_nhi
u = uniform(loc, scale)

# extract observed log₁₀ N_HI samples from catalog
log_nhis = catalog['log_nhis']
#print("log_nhis", log_nhis)
all_log_nhis = log_nhis[dla_catalog_name]
#ind = cellfun(@(x) (~isempty(x)), all_log_nhis);
all_log_nhis = np.array(all_log_nhis)
log_nhis = all_log_nhis[all_log_nhis!=0.0]
print("check", np.count_nonzero(all_log_nhis))
log_nhis = [num[0] for num in log_nhis]
#print("temp", temp, len(temp))
#print(ind, all_log_nhis[0])
#print(all_log_nhis)
#log_nhis = all_log_nhis[ind]
#log_nhis = cat(1, all_log_nhis{ind});

# make a quadratic fit to the estimated log p(log₁₀ N_HI) over the specified range
x = np.linspace(dlaParams.fit_min_log_nhi, dlaParams.fit_max_log_nhi, num=1e3)
#kde_pdf = ksdensity(log_nhis, x);
kde = stats.gaussian_kde(log_nhis)
kde_pdf = kde(x)
f = np.polyfit(x, np.log(kde_pdf), 2);


extrapolate_min_log_nhi = 19.0 # normalization range for the extrapolated region convert this to a PDF and normalize
if not flag.extrapolate_subdla:
    unnormalized_pdf = lambda nhi : np.exp(np.polyval(f, nhi))
    Z = integrate.quad(unnormalized_pdf, dlaParams.fit_min_log_nhi, 25.0)
    Z = Z[0]
else:
    unnormalized_pdf = lambda nhi : np.exp(np.polyval(f, nhi))*np.heaviside(nhi-20.03269,0)+exp(np.polyval(f,20.03269))*(1-np.heaviside(nhi-20.03269,0))
    Z = integrate.quad(unnormalized_pdf, dlaParams.extrapolate_min_log_nhi, 25.0)
    
# create the PDF of the mixture between the uniform distribution and the distribution fit to the data
normalized_pdf = lambda nhi : dlaParams.alpha * (unnormalized_pdf(nhi) / Z) + (1 - dlaParams.alpha) * (u.pdf(nhi))

if not flag.extrapolate_subdla:
    #print("min_log", dlaParams.fit_min_log_nhi)
    cdf = lambda nhi : integrate.quad(normalized_pdf, dlaParams.fit_min_log_nhi, nhi)[0]
else:
    cdf = lambda nhi : integrate.quad(normalized_pdf, dlaParams.extrapolate_min_log_nhi, nhi)[0]
    
# use inverse transform sampling to convert the quasirandom samples on [0, 1] to appropriate values
log_nhi_samples = np.zeros(dlaParams.num_dla_samples)
for i in range(dlaParams.num_dla_samples):
    #sol = root_scalar(func, args=(5, 6, 100), method='toms748', bracket=[1e-3, 1])
    #args=(), method='toms748', x0=None
    log_nhi_samples[i] = root_scalar(lambda nhi : (cdf(nhi) - other_offset_samples[i]), method='brentq', bracket=[19.0, 25.0], x0=20.5).root

# precompute N_HI samples for convenience
nhi_samples = pow(10, log_nhi_samples)

variables_to_save = {'uniform_min_log_nhi':dlaParams.uniform_min_log_nhi, 'uniform_max_log_nhi':dlaParams.uniform_max_log_nhi,
                     'fit_min_log_nhi':dlaParams.fit_min_log_nhi, 'fit_max_log_nhi':dlaParams.fit_max_log_nhi, 'alpha':dlaParams.alpha,
                     'offset_samples':offset_samples, 'log_nhi_samples':log_nhi_samples, 'nhi_samples':nhi_samples,
                     'offset_samples_qso':offset_samples_qso}

#save(sprintf('%s/dla_samples', processed_directory(training_release)),variables_to_save{:}, '-v7.3');
direct = 'dr12q/processed'
pathName = os.path.join(parent_dir, direct)
fileName = "{direct}/dla_samples".format(direct = pathName)

# Open a file for writing data
file_handler = open(fileName, 'wb')

# Dump the data of the object into the file
pickle.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()

print("\nsamples", samples)
print("\noffset_samples")
print(offset_samples)
print(offset_samples.shape)
print("\noffset_samples_qso")
print(offset_samples_qso)
print(offset_samples_qso.shape)
print("\nother_offset_samples")
print(other_offset_samples)
print(other_offset_samples.shape)
print("\nz_qsos")
print(z_qsos)
print(z_qsos.shape)
print("\nbins")
print(bins)
print("\nvect")
print(vect)
print(vect.shape)
print("\nz_freq")
print(z_freq)
print(z_freq.shape)
print("\nz_bin")
print(z_bin)
print(z_bin.shape)
#print("\nall_log_nhis")
#print(all_log_nhis)
#print(len(all_log_nhis))
#print(len(log_nhis))
#print("\nlog_nhis")
#print(log_nhis)
#print(len(log_nhis))
print(log_nhis[0], log_nhis[3], log_nhis[5000])
print(log_nhis[0] == 20.8786)
print("\nx")
print(x)
print(x.shape)
print("\nkde_pdf")
print(kde_pdf)
print(len(kde_pdf))
print("\nf")
print(f)
print(len(f))
print("\nZ")
print(Z)
print("\ncdf")
print(cdf)
print("\nlog_nhi_samples")
print(log_nhi_samples)
print(log_nhi_samples.shape)
print("\nhi_samples")
print(nhi_samples)
print(nhi_samples.shape)
print("\nfileName")
print(fileName)