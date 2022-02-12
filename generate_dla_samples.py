import random
import dill
import pickle
from pathlib import Path
import numpy as np
import os
from scipy.stats import qmc
from scipy import interpolate
from scipy.stats import uniform
from scipy import stats
import scipy.integrate as integrate
from scipy.optimize import root_scalar

with open('parameters.pkl', 'rb') as handle:
    params = dill.load(handle)

dlaParams = params['dlaParams']
preParams = params['preParams']
flag = params['flag']

training_release  = 'dr12q'

random.seed()
print(random.random())
dla_catalog_name  = 'dr9q_concordance'

p = Path(os.getcwd())
parent_dir = str(p.parent)
release = "dr12q/processed/catalog"
filename = os.path.join(parent_dir, release)
#getting back pickled data for catalog
with open(filename,'rb') as f:
    catalog = pickle.load(f)

# generate quasirandom samples from p(normalized offset, log₁₀(N_HI))
halton = qmc.Halton(d=3, scramble=False)
samples = halton.random(dlaParams.num_dla_samples)
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

vect = np.arange(preParams.z_qso_cut, np.max(z_qsos) + (np.max(z_qsos) - preParams.z_qso_cut) / bins, (np.max(z_qsos) - preParams.z_qso_cut) / bins)
z_freq, z_bin = np.histogram(z_qsos, vect)
for i in range(len(z_freq)-1, -1, -1): #=length(z_freq):-1:1
    z_freq[i] = np.sum(z_freq[:i+1])

z_freq = np.insert(z_freq, 0, 0)
print(np.max(z_freq))
z_freq = z_freq / float(np.max(z_freq))
z_freq, I = np.unique(z_freq, return_index=True)
z_bin = z_bin[I]
offset_interp = interpolate.interp1d(z_freq, z_bin, bounds_error=False)
offset_samples_qso = offset_interp(offset_samples_qso)

# we must transform the second dimension to have the correct marginal
# distribution for our chosen prior over column density, which is a
# mixture of a uniform distribution on log₁₀ N_HI and a distribution
# we fit to observed data

# uniform component of column density prior
loc = dlaParams.uniform_min_log_nhi
scale = dlaParams.uniform_max_log_nhi - dlaParams.uniform_min_log_nhi
u = uniform(loc, scale)

# extract observed log₁₀ N_HI samples from catalog
log_nhis = catalog['log_nhis']
all_log_nhis = log_nhis[dla_catalog_name]
all_log_nhis = np.array(all_log_nhis)
log_nhis = all_log_nhis[all_log_nhis!=0.0]
log_nhis = [num[0] for num in log_nhis]

# make a quadratic fit to the estimated log p(log₁₀ N_HI) over the specified range
x = np.linspace(dlaParams.fit_min_log_nhi, dlaParams.fit_max_log_nhi, num=1000)
kde = stats.gaussian_kde(log_nhis)
kde_pdf = kde(x)
f = np.polyfit(x, np.log(kde_pdf), 2)


extrapolate_min_log_nhi = 19.0 # normalization range for the extrapolated region convert this to a PDF and normalize
if not flag.extrapolate_subdla:
    unnormalized_pdf = lambda nhi : np.exp(np.polyval(f, nhi))
    Z = integrate.quad(unnormalized_pdf, dlaParams.fit_min_log_nhi, 25.0)
    Z = Z[0]
else:
    unnormalized_pdf = lambda nhi : np.exp(np.polyval(f, nhi))*np.heaviside(nhi-20.03269,0)+np.exp(np.polyval(f,20.03269))*(1-np.heaviside(nhi-20.03269,0))
    Z = integrate.quad(unnormalized_pdf, dlaParams.extrapolate_min_log_nhi, 25.0)
    
# create the PDF of the mixture between the uniform distribution and the distribution fit to the data
normalized_pdf = lambda nhi : dlaParams.alpha * (unnormalized_pdf(nhi) / Z) + (1 - dlaParams.alpha) * (u.pdf(nhi))

if not flag.extrapolate_subdla:
    cdf = lambda nhi : integrate.quad(normalized_pdf, dlaParams.fit_min_log_nhi, nhi)[0]
else:
    cdf = lambda nhi : integrate.quad(normalized_pdf, dlaParams.extrapolate_min_log_nhi, nhi)[0]
    
# use inverse transform sampling to convert the quasirandom samples on [0, 1] to appropriate values
log_nhi_samples = np.zeros(dlaParams.num_dla_samples)
for i in range(dlaParams.num_dla_samples):
    log_nhi_samples[i] = root_scalar(lambda nhi : (cdf(nhi) - other_offset_samples[i]), method='brentq', bracket=[19.0, 25.0], x0=20.5).root

# precompute N_HI samples for convenience
nhi_samples = pow(10, log_nhi_samples)

variables_to_save = {'uniform_min_log_nhi':dlaParams.uniform_min_log_nhi, 'uniform_max_log_nhi':dlaParams.uniform_max_log_nhi,
                     'fit_min_log_nhi':dlaParams.fit_min_log_nhi, 'fit_max_log_nhi':dlaParams.fit_max_log_nhi, 'alpha':dlaParams.alpha,
                     'offset_samples':offset_samples, 'log_nhi_samples':log_nhi_samples, 'nhi_samples':nhi_samples,
                     'offset_samples_qso':offset_samples_qso}

direct = 'dr12q/processed'
#pathName = os.path.join(parent_dir, direct)
fileName = "{dir}/dla_samples".format(dir = direct)

# Open a file for writing data
file_handler = open(fileName, 'wb')

# Dump the data of the object into the file
dill.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()
