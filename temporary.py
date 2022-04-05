import scipy.optimize as opt

def fitendmu(endsigma2,obs,obsnoisevars,precompdens=None):

    if precompdens is None:
        precompdens = obsnoisevars + endsigma2

    return np.sum(np.divide(obs, precompdens))/np.sum(np.divide(1.0, precompdens))

def endnllh(obs,obsnoisevars,endsigma2,endmu=None):

    dens = obsnoisevars + endsigma2

    if endmu is None:
        endmu = fitendmu(endsigma2,obs,obsnoisevars,dens)

    diffs = obs-endmu

    return obs.shape[0]*np.log(np.sqrt(2*np.pi)) + np.sum(np.log(dens) + np.divide(np.multiply(diffs,diffs),dens))/2


def fitendmodel(obs,obsnoisevars):

    touse = np.isfinite(obsnoisevars)
    obs = obs[touse]
    obsnoisevars = obsnoisevars[touse]

    naivemu = np.sum(np.divide(obs, obsnoisevars))/np.sum(np.divide(1.0, obsnoisevars))
    diffs = obs-naivemu
    naivesigma2 = np.sum(np.divide(np.multiply(diffs,diffs), obsnoisevars))/np.sum(np.divide(1.0, obsnoisevars))

    spread = 100

    optsol = opt.minimize_scalar(lambda s2 : endnllh(obs,obsnoisevars,s2), bounds=(naivesigma2/spread,naivesigma2*spread),method='bounded')

    if not optsol.success:
        print("end model failed to converge")
        endsigma2 = naivesigma2
    
    else:
        endsigma2 = optsol.x

    return [fitendmu(endsigma2,obs,obsnoisevars), endsigma2]

def tstfitendmodel(mu,s2,npts=1000,obsvarstddev=25):

    vs = np.random.randn(npts,1)*obsvarstddev
    vals = np.random.randn(npts,1)*np.sqrt(vs+s2)+mu
    [muhat,s2hat] = fitendmodel(vals,vs)
    return [muhat, s2hat]

import numpy as np
from numba import njit
# spectrum_loss: computes the negative log likelihood for centered
# flux y:
#
#     -log p(y | Lyα z, σ², M, ω², c₀, τₒ, β)
#   = -log N(y; 0, MM' + diag(σ² + (ω ∘ (c₀ + a(1 + Lyα z)))²)),
#
# where a(Lyα z) is the approximate absorption due to Lyman α at
# redshift z:
#
#   a(z) = 1 - exp(-τ₀(1 + z)ᵝ)
#
# and its derivatives wrt M, log ω, log c₉, log τ₀, and log β
@njit
def spectrum_loss(y,lya_1pz,noise_variance,M,omega2,c_0,tau_0,beta,num_forest_lines,all_transition_wavelengths,all_oscillator_strengths,zqso_1pz):

    log_2pi = 1.83787706640934534

    [n, k] = np.shape(M)

    # compute approximate Lyα optical depth/absorption
    lya_optical_depth = tau_0 * lya_1pz**beta
    #print("lya_optical_depth before", lya_optical_depth, lya_optical_depth.shape)
    #print("Area 1\n")
    #fileObject = open("data_dump.txt", 'a')
    #fileObject.write("lya_optical_depth\n" + lya_optical_depth)
    #with open('data_dump.txt', 'a') as f:
    #    print >> f, "lya_optical_depth\n"
    #    for item in lya_optical_depth:
    #        print >> f, item

    # compute approximate Lyman series optical depth/absorption
    # using the scaling relationship
    indicator         = lya_1pz <= zqso_1pz
    #print("indicator", indicator, indicator.shape)
    lya_optical_depth = lya_optical_depth * indicator
    #print("lya_optical_depth after", lya_optical_depth, lya_optical_depth.shape)

    for i in range (1, num_forest_lines):
        lyman_1pz = all_transition_wavelengths[0] * lya_1pz / all_transition_wavelengths[i]
        #print("lyman_1pz before", lyman_1pz, lyman_1pz.shape)

        indicator = lyman_1pz <= zqso_1pz
        #print("indicator", indicator, indicator.shape)
        lyman_1pz = lyman_1pz * indicator
        #print("lyman_1pz after", lyman_1pz, lyman_1pz.shape)

        tau = tau_0 * all_transition_wavelengths[i] * all_oscillator_strengths[i] / (all_transition_wavelengths[0] * all_oscillator_strengths[0])
        #print("tau", tau, tau.shape)

        lya_optical_depth = lya_optical_depth + tau * lyman_1pz**beta
        #print("lya_optical_depth", lya_optical_depth, lya_optical_depth.shape)
        #print("Area 2\n")
        #with open('data_dump.txt', 'a') as f:
        #    print >> f, "lya_optical_depth 2\n"
        #    for item in lya_optical_depth:
        #        print >> f, item

    lya_absorption = np.exp(-lya_optical_depth)
    #print("lya_absorption", lya_absorption, lya_absorption.shape)

    # compute "absorption noise" contribution
    scaling_factor = 1 - lya_absorption + c_0
    #print("scaling_factor", scaling_factor, scaling_factor.shape)
    absorption_noise = omega2 * scaling_factor**2
    #print("absorption_noise", absorption_noise, absorption_noise.shape)
    #print("Area 3\n")
    #with open('data_dump.txt', 'a') as f:
    #    print >> f, "absorption_noise\n"
    #    for item in absorption_noise:
    #        print >> f, item

    d = noise_variance + absorption_noise
    #print("d", d, d.shape)

    d_inv = 1.0 / d
    #print("d_inv", d_inv, d_inv.shape)
    D_inv_y = d_inv * y
    #print("D_inv_y", D_inv_y, D_inv_y.shape)
    #D_inv_M = bsxfun(@times, d_inv, M);
    #print("M before", M, M.shape)
    D_inv_M = np.zeros((M.shape[0], M.shape[1]))
    for i in range(M.shape[1]):
        D_inv_M[:,i] = d_inv * M[:,i]
    #D_inv_M = d_inv[:,None] * M

    # use Woodbury identity, define
    #   B = (I + MᵀD⁻¹M),
    # then
    #   K⁻¹ = D⁻¹ - D⁻¹MB⁻¹MᵀD⁻¹
    #print("M", M, M.shape)
    #print("D_inv_M", D_inv_M, D_inv_M.shape)

    B = np.dot(M.T, D_inv_M)
    #print("B before", B, B.shape)
    B = np.reshape(B, B.shape[0]*B.shape[1])
    B[0::k+1] = B[0::k+1] + 1
    B = np.reshape(B, (k, k))
    #print("B after", B, B.shape)
    L = np.linalg.cholesky(B)
    #print("L", L, L.shape)
    # C = B⁻¹MᵀD⁻¹
    ld = np.linalg.solve(L, D_inv_M.T)
    #print("\nld", ld, ld.shape)
    C= np.linalg.solve(L.T, ld)
    #C = np.linalg.solve(L.T, np.linalg.solve(L, D_inv_M.T))
    #print("C", C, C.shape)
    #inbtw = np.dot(C, y)
    #print("inbtw", inbtw, inbtw.shape)
    #doted = np.dot(D_inv_M, inbtw)
    #print("doted", doted, doted.shape)
    #subed = D_inv_y - doted
    #print("subed", subed, subed.shape)
    
    K_inv_y = D_inv_y - np.dot(D_inv_M, np.dot(C, y))
    #print("K_inv_y", K_inv_y, K_inv_y.shape)

    log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))
    #print("log_det_K", log_det_K, log_det_K.shape)

    # negative log likelihood:
    #   ½ yᵀ (K + V + A)⁻¹ y + log det (K + V + A) + n log 2π
    nlog_p = 0.5 * (np.dot(y.T, K_inv_y) + log_det_K + n * log_2pi)
    #print("nlog_p", nlog_p, nlog_p.shape)

    # gradient wrt M
    K_inv_M = D_inv_M - np.dot(D_inv_M, np.dot(C, M))
    #print("M after", M, M.shape)
    #print("C", C, C.shape)
    #temp = np.matmul(C, M)
    #print("C * M", temp, temp.shape)
    #tempty = np.dot(D_inv_M, temp)
    #print("D_inv_M * C * M", tempty, tempty.shape)
    #tempted = D_inv_M - tempty
    #print("K_inv_M", K_inv_M, K_inv_M.shape)
    #K_inv_M = tempted
    
    temp = np.dot(K_inv_y, M)
    temp = np.reshape(temp, (temp.shape[0], 1))
    tempted = np.dot(np.reshape(K_inv_y, (K_inv_y.shape[0], 1)), temp.T)
    dM = -1 * (tempted - K_inv_M)
    #dM = -(np.dot(K_inv_y, np.dot(K_inv_y, M).T) - K_inv_M)
    #print("dM", dM, dM.shape)

    # compute diag K⁻¹ without computing full product
    diag_K_inv = d_inv - np.sum(C * D_inv_M.T, axis=0).T
    #print("d_inv", d_inv, d_inv.shape)
    #print("C", C, C.shape)
    #print("D_inv_M", D_inv_M, D_inv_M.shape)
    #tmp = C * D_inv_M.T
    #print("C * D_inv_M", tmp, tmp.shape)
    #temp = np.sum(C * D_inv_M.T, axis=0).T
    #print("sum of C * D_inv_M.T", temp, temp.shape)
    #print("diag_K_inv", diag_K_inv, diag_K_inv.shape)

    # gradient wrt log ω
    dlog_omega = -(absorption_noise * ((K_inv_y**2) - diag_K_inv))
    #print("dlog_omega", dlog_omega, dlog_omega.shape)
    #print("Area 4\n")
    #with open('data_dump.txt', 'a') as f:
    #    print >> f, "dlog_omega\n"
    #    for item in dlog_omega:
    #        print >> f, item

    # gradient wrt log c₀
    da = c_0 * omega2 * scaling_factor
    #print("da", da, da.shape)
    #print("K_inv_y", K_inv_y, K_inv_y.shape)
    #print("diag_K_inv", diag_K_inv, diag_K_inv.shape)
    da = np.reshape(da, (da.shape[0], 1))
    K_inv_y = np.reshape(K_inv_y, (K_inv_y.shape[0], 1))
    diag_K_inv = np.reshape(diag_K_inv, (diag_K_inv.shape[0], 1))
    dlog_c_0 = np.dot(-(K_inv_y * da).T, K_inv_y) + np.dot(diag_K_inv.T, da)
    #temp = np.dot(diag_K_inv, da.T)
    #print("diag_K_inv * da", temp, temp.shape)
    #other = np.dot(-(K_inv_y * da).T, K_inv_y)
    #print("other", other, other.shape)
    #print("added part \n")
    #print(np.dot(diag_K_inv.T, da))
    #print("\n")
    #print("first part \n")
    #print(np.dot(-(K_inv_y * da).T, K_inv_y))
    #print("\n")
    #dlog_c_0 = np.squeeze(dlog_c_0)
    dlog_c_0 = dlog_c_0[0,0]
    #print("dlog_c_0", dlog_c_0)

    # gradient wrt log τ₀
    da = omega2 * scaling_factor * lya_optical_depth * lya_absorption
    #print("da 2", da, da.shape)
    da = np.reshape(da, (da.shape[0], 1))
    dlog_tau_0 = np.dot(-(K_inv_y * da).T, K_inv_y) + np.dot(diag_K_inv.T, da)
    #dlog_tau_0 = np.squeeze(dlog_tau_0)
    dlog_tau_0 = dlog_tau_0[0,0]
    #print("dlog_tau_0", dlog_tau_0)

    # gradient wrt log β
    # Apr 12: inject indicator in the gradient but outside the log 
    lya_1pz = np.reshape(lya_1pz, (lya_1pz.shape[0], 1))
    indicator = lya_1pz <= zqso_1pz
    #print("first da",da.shape)
    #print("indicator", indicator, indicator.shape)
    #print("log_lya", np.log(lya_1pz).shape)
    #print("beta", beta.shape)
    #one = np.log(lya_1pz)
    #print("shapes", one.shape, beta.shape)
    #two = one * beta
    #three = two * indicator
    #da = da * three
    da = da * (np.log(lya_1pz) * beta) * indicator
    #print("da 3", da, da.shape)
    #da = np.reshape(da, (da.shape[0], 1))
    dlog_beta = np.dot(-(K_inv_y * da).T, K_inv_y) + np.dot(diag_K_inv.T, da)
    #dlog_beta = np.squeeze(dlog_beta)
    dlog_beta = dlog_beta[0,0]
    #print("dlog_beta", dlog_beta)

    return nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta

#objective: computes negative log likelihood of entire training
# dataset as a function of the model parameters, x, a vector defined
# as
#
#   x = [vec M; log ω; log c₉; log τ₀; log β]
#
# as well as its gradient:
#
#   f(x) = -∑ᵢ log(yᵢ | Lyα z, σ², M, ω, c₀, τ₉, β)
#   g(x) = ∂f/∂x

def objective(x, centered_rest_fluxes, lya_1pzs, rest_noise_variances, num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, z_qsos):

    [num_quasars, num_pixels] = np.shape(centered_rest_fluxes)

    k = (len(x) - 3) / num_pixels - 1
    k = int(k)

    M = np.reshape(x[:(num_pixels*k)], [num_pixels, k], order='F')

    log_omega = x[(num_pixels*k):(num_pixels*(k+1))]

    log_c_0   = x[-3]
    log_tau_0 = x[-2]
    log_beta  = x[-1]

    omega2 = np.exp(2 * log_omega)
    c_0    = np.exp(log_c_0)
    tau_0  = np.exp(log_tau_0)
    beta   = np.exp(log_beta)

    f          = 0
    dM         = np.zeros([M.shape[0], M.shape[1]])
    dlog_omega = np.zeros([log_omega.shape[0]])
    dlog_c_0   = 0
    dlog_tau_0 = 0
    dlog_beta  = 0

    for i in range(num_quasars):
        ind = (~np.isnan(centered_rest_fluxes[i, :]))

        # Apr 12: directly pass z_qsos in the argument since we don't want
        # zeros in lya_1pzs to mess up the gradients in spectrum_loss
        zqso_1pz = z_qsos[i] + 1

        [this_f, this_dM, this_dlog_omega, this_dlog_c_0, this_dlog_tau_0, this_dlog_beta] = spectrum_loss(centered_rest_fluxes[i, ind].T, lya_1pzs[i, ind].T, rest_noise_variances[i, ind].T, M[ind, :], omega2[ind], c_0, tau_0, beta, num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, zqso_1pz);

        f               = f               + this_f
        dM[ind, :]      = dM[ind, :]      + this_dM
        dlog_omega[ind] = dlog_omega[ind] + this_dlog_omega
        dlog_c_0        = dlog_c_0        + this_dlog_c_0
        dlog_tau_0      = dlog_tau_0      + this_dlog_tau_0
        dlog_beta       = dlog_beta       + this_dlog_beta

    # apply prior for τ₀ (Kim, et al. 2007)
    tau_0_mu    = 0.0023
    tau_0_sigma = 0.0007

    dlog_tau_0 = dlog_tau_0 + tau_0 * (tau_0 - tau_0_mu) / tau_0_sigma**2

    # apply prior for β (Kim, et al. 2007)
    beta_mu    = 3.65
    beta_sigma = 0.21

    dlog_beta = dlog_beta + beta * (beta - beta_mu) / beta_sigma**2
    
    d_M = np.reshape(dM, dM.shape[0]*dM.shape[1], order='F')

    g = np.concatenate((d_M, dlog_omega))
    g = np.append(g, [dlog_c_0, dlog_tau_0, dlog_beta])

    return f,g


import random
import dill
import pickle
from pathlib import Path
import numpy as np
import os
from scipy import interpolate
from scipy import optimize
from sklearn.decomposition import IncrementalPCA

with open('parameters.pkl', 'rb') as handle:
    params = dill.load(handle)

preParams = params['preParams']
normParams = params['normParams']
learnParams = params['learnParams']
physParams = params['physParams']
loading = params['loadParams']
optParams = params['optParams']
nullParams = params['nullParams']
nullParams.min_lambda = 910             # range of rest wavelengths to       Å
nullParams.max_lambda = 3000            #   model
nullParams.dlambda = 0.25               # separation of wavelength grid      Å
nullParams.k = 20                       # rank of non-diagonal contribution
nullParams.max_noise_variance = 4**2     # maximum pixel noise allowed during model training
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
release = "dr12q/processed/catalog"
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
release = "dr12q/processed/preloaded_qsos"
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

# maximize likelihood via L-BFGS
maxes = {'maxfun':8000, 'maxiter':4000}

variables_to_save = {'training_release':training_release, 'train_ind':train_ind, 'max_noise_variance':nullParams.max_noise_variance,
                     'rest_wavelengths':rest_wavelengths, 'mu':mu, 'initial_M':initial_M, 'initial_log_omega':initial_log_omega,
                     'initial_log_c_0':initial_log_c_0, 'initial_tau_0':initial_tau_0, 'initial_beta':initial_beta, 'maxes':maxes,
                     'initial_x':initial_x,  'bluewards_mu':bluewards_mu, "rest_noise_variances_exp1pz":rest_noise_variances_exp1pz,
                     'bluewards_sigma':bluewards_sigma, 'redwards_mu':redwards_mu, 'redwards_sigma':redwards_sigma,
                     'k':nullParams.k, 'num_rest_pixels':num_rest_pixels, 'training_set_name':training_set_name,
                     'normalization_min_lambda':normParams.normalization_min_lambda, 'normalization_max_lambda':normParams.normalization_max_lambda,
                     'centered_rest_fluxes':centered_rest_fluxes, 'lya_1pzs':lya_1pzs, 'num_forest_lines':learnParams.num_forest_lines,
                     'all_transition_wavelengths':learnParams.all_transition_wavelengths, 'all_oscillator_strengths':learnParams.all_oscillator_strengths,
                     'z_qsos':z_qsos}

direct = 'dr12q/processed'
#directory = os.path.join(parent_dir, direct)
                   
#place = '{}/learned_model_outdata_{}_norm_{}-{}'.format(direct, training_set_name, normParams.normalization_min_lambda, normParams.normalization_max_lambda)
place = os.path.join(direct, "unoptimized_model")
             
# Open a file for writing data
file_handler = open(place, 'wb')

# Dump the data of the object into the file
dill.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()

#method = trust- or CG ones that I would try first: CG, BFGS, Newton-CG, trust-ncg, SLSQP
#result = optimize.minimize(objective_function, initial_x, method='L-BFGS-B', jac=True, options={'maxfun':8000, 'maxiter':4000})
#try method Nelder-Mead
#result = optimize.minimize(objective_function, initial_x, method='CG', jac=True, options={'maxiter':3000})
#result.x, result.fun, result.message. result.success
#x = result.x
#log_likelihood = result.fun
#message = result.message
#success = result.success
#ind = list(range(num_rest_pixels * nullParams.k))
#ind = np.array(ind)
#M = np.reshape(x[ind], [num_rest_pixels, nullParams.k], order='F')

#ind = list(range((num_rest_pixels * nullParams.k), (num_rest_pixels * (nullParams.k + 1))))
#ind = np.array(ind)
#print("ind", ind, ind.shape)
#log_omega = x[ind].T

#log_c_0   = x[-3]
#log_tau_0 = x[-2]
#log_beta  = x[-1]

#variables_to_save = {'training_release':training_release, 'train_ind':train_ind, 'max_noise_variance':nullParams.max_noise_variance,
#                     'rest_wavelengths':rest_wavelengths, 'mu':mu, 'initial_M':initial_M, 'initial_log_omega':initial_log_omega,
#                     'initial_log_c_0':initial_log_c_0, 'initial_tau_0':initial_tau_0, 'initial_beta':initial_beta, 'opt':opt,
#                     'M':M, 'log_omega':log_omega, 'log_c_0':log_c_0, 'log_tau_0':log_tau_0, 'log_beta':log_beta, 'result':result,
#                     'log_likelihood':log_likelihood, 'message':message, 'success':success, 'bluewards_mu':bluewards_mu,
#                     'bluewards_sigma':bluewards_sigma, 'redwards_mu':redwards_mu, 'redwards_sigma':redwards_sigma}

#direct = 'dr12q/processed'
#directory = os.path.join(parent_dir, direct)
                   
#place = '{}/learned_model_outdata_{}_norm_{}-{}'.format(direct, training_set_name, normParams.normalization_min_lambda, normParams.normalization_max_lambda)
             
# Open a file for writing data
#file_handler = open(place, 'wb')

# Dump the data of the object into the file
#dill.dump(variables_to_save, file_handler)

# close the file handler to release the resources
#file_handler.close()