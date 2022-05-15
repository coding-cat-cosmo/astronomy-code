import numpy as np
from numba import njit
import dill
import pickle
from pathlib import Path
from scipy import optimize
import os
import time
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

    # compute approximate Lyman series optical depth/absorption
    # using the scaling relationship
    indicator         = lya_1pz <= zqso_1pz
    lya_optical_depth = lya_optical_depth * indicator

    for i in range (1, num_forest_lines):
        lyman_1pz = all_transition_wavelengths[0] * lya_1pz / all_transition_wavelengths[i]

        indicator = lyman_1pz <= zqso_1pz
        lyman_1pz = lyman_1pz * indicator

        tau = tau_0 * all_transition_wavelengths[i] * all_oscillator_strengths[i] / (all_transition_wavelengths[0] * all_oscillator_strengths[0])

        lya_optical_depth = lya_optical_depth + tau * lyman_1pz**beta

    lya_absorption = np.exp(-lya_optical_depth)

    # compute "absorption noise" contribution
    scaling_factor = 1 - lya_absorption + c_0
    absorption_noise = omega2 * scaling_factor**2

    d = noise_variance + absorption_noise

    d_inv = 1.0 / d
    D_inv_y = d_inv * y
    D_inv_M = np.zeros((M.shape[0], M.shape[1]))
    for i in range(M.shape[1]):
        D_inv_M[:,i] = d_inv * M[:,i]
    #D_inv_M = d_inv[:,None] * M

    # use Woodbury identity, define
    #   B = (I + MᵀD⁻¹M),
    # then
    #   K⁻¹ = D⁻¹ - D⁻¹MB⁻¹MᵀD⁻¹

    B = np.dot(M.T, D_inv_M)
    B = np.reshape(B, B.shape[0]*B.shape[1])
    B[0::k+1] = B[0::k+1] + 1
    B = np.reshape(B, (k, k))
    #L = np.linalg.cholesky(B)
    # C = B⁻¹MᵀD⁻¹
    C = np.linalg.solve(B, D_inv_M.T)
    #C = np.linalg.solve(L.T, np.linalg.solve(L, D_inv_M.T))
    
    K_inv_y = D_inv_y - np.dot(D_inv_M, np.dot(C, y))

    #log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))
    log_det_K = np.sum(np.log(d)) + np.linalg.slogdet(B)[1]

    # negative log likelihood:
    #   ½ yᵀ (K + V + A)⁻¹ y + log det (K + V + A) + n log 2π
    nlog_p = 0.5 * (np.dot(y.T, K_inv_y) + log_det_K + n * log_2pi)

    # gradient wrt M
    K_inv_M = D_inv_M - np.dot(D_inv_M, np.dot(C, M))
    
    temp = np.dot(K_inv_y, M)
    temp = np.reshape(temp, (temp.shape[0], 1))
    tempted = np.dot(np.reshape(K_inv_y, (K_inv_y.shape[0], 1)), temp.T)
    dM = -1 * (tempted - K_inv_M)

    # compute diag K⁻¹ without computing full product
    diag_K_inv = d_inv - np.sum(C * D_inv_M.T, axis=0).T

    # gradient wrt log ω
    dlog_omega = -(absorption_noise * ((K_inv_y**2) - diag_K_inv))

    # gradient wrt log c₀
    da = c_0 * omega2 * scaling_factor
    da = np.reshape(da, (da.shape[0], 1))
    K_inv_y = np.reshape(K_inv_y, (K_inv_y.shape[0], 1))
    diag_K_inv = np.reshape(diag_K_inv, (diag_K_inv.shape[0], 1))
    dlog_c_0 = np.dot(-(K_inv_y * da).T, K_inv_y) + np.dot(diag_K_inv.T, da)
    dlog_c_0 = dlog_c_0[0,0]

    # gradient wrt log τ₀
    da = omega2 * scaling_factor * lya_optical_depth * lya_absorption
    da = np.reshape(da, (da.shape[0], 1))
    dlog_tau_0 = np.dot(-(K_inv_y * da).T, K_inv_y) + np.dot(diag_K_inv.T, da)
    dlog_tau_0 = dlog_tau_0[0,0]

    # gradient wrt log β
    # Apr 12: inject indicator in the gradient but outside the log 
    lya_1pz = np.reshape(lya_1pz, (lya_1pz.shape[0], 1))
    indicator = lya_1pz <= zqso_1pz
    da = da * (np.log(lya_1pz) * beta) * indicator
    dlog_beta = np.dot(-(K_inv_y * da).T, K_inv_y) + np.dot(diag_K_inv.T, da)
    dlog_beta = dlog_beta[0,0]

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

    try:
        yu = open("temp", 'wb')
        dill.dump(x, yu)
    finally:
        yu.close()

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

Nfeval = 1
other = 0
start = 0
end = 0
arr = []
def callbackF(Xi):
    global Nfeval
    global other
    global start
    global end
    global arr
    end = time.time()
    temp = '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}    {5: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], objective_function(Xi)[0], end-start)
    print(temp)
    arr.append(objective_function(Xi)[0])
    if other % 10 == 0:
        try:
            ot = open("iter", 'wb')
            dn = open("change", 'wb')
            dill.dump(Nfeval, ot)
            dill.dump(arr, dn)
        finally:
            ot.close()
            dn.close()
    Nfeval += 1
    other += 1
    start = time.time()

direct = "dr12q/processed"
place = os.path.join(direct, "unoptimized_model")

with open(place, 'rb') as handle:
    unop = dill.load(handle)

training_release = unop['training_release']
train_ind = unop['train_ind']
max_noise_variance = unop['max_noise_variance']
rest_wavelengths = unop['rest_wavelengths']
mu = unop['mu']
initial_M = unop['initial_M']
initial_log_omega = unop['initial_log_omega']
initial_log_c_0 = unop['initial_log_c_0']
initial_tau_0 = unop['initial_tau_0']
initial_beta = unop['initial_beta']
maxes = unop['maxes']
initial_x = unop['initial_x']
bluewards_mu = unop['bluewards_mu']
bluewards_sigma = unop['bluewards_sigma']
redwards_mu = unop['redwards_mu']
redwards_sigma = unop['redwards_sigma']
num_rest_pixels = unop['num_rest_pixels']
k = unop['k']
training_set_name = unop['training_set_name']
normalization_min_lambda = unop['normalization_min_lambda']
normalization_max_lambda = unop['normalization_max_lambda']
centered_rest_fluxes = unop['centered_rest_fluxes']
lya_1pzs = unop['lya_1pzs']
rest_noise_variances_exp1pz = unop['rest_noise_variances_exp1pz']
num_forest_lines = unop['num_forest_lines']
all_transition_wavelengths = unop['all_transition_wavelengths']
all_oscillator_strengths = unop['all_oscillator_strengths']
z_qsos = unop['z_qsos']

objective_function = lambda x : objective(x, centered_rest_fluxes, lya_1pzs, rest_noise_variances_exp1pz, num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, z_qsos)

with open("temp", "rb") as fing:
    initial_x = dill.load(fing)
#method = trust- or CG ones that I would try first: CG, BFGS, Newton-CG, trust-ncg, SLSQP
result = optimize.minimize(objective_function, initial_x, method='L-BFGS-B', jac=True, options={'maxfun':8000, 'maxiter':1000}, callback=callbackF)
#try method Nelder-Mead
#result = optimize.minimize(objective_function, initial_x, method='CG', jac=True, options={'maxiter':3000})
x = result.x
log_likelihood = result.fun
message = result.message
success = result.success
ind = list(range(num_rest_pixels * k))
ind = np.array(ind)
M = np.reshape(x[ind], [num_rest_pixels, k], order='F')

ind = list(range((num_rest_pixels * k), (num_rest_pixels * (k + 1))))
ind = np.array(ind)
#print("ind", ind, ind.shape)
log_omega = x[ind].T

log_c_0   = x[-3]
log_tau_0 = x[-2]
log_beta  = x[-1]

variables_to_save = {'training_release':training_release, 'train_ind':train_ind, 'max_noise_variance':max_noise_variance,
                     'rest_wavelengths':rest_wavelengths, 'mu':mu, 'initial_M':initial_M, 'initial_log_omega':initial_log_omega,
                     'initial_log_c_0':initial_log_c_0, 'initial_tau_0':initial_tau_0, 'initial_beta':initial_beta, 'maxes':maxes,
                     'M':M, 'log_omega':log_omega, 'log_c_0':log_c_0, 'log_tau_0':log_tau_0, 'log_beta':log_beta, 'result':result,
                     'log_likelihood':log_likelihood, 'message':message, 'success':success, 'bluewards_mu':bluewards_mu,
                     'bluewards_sigma':bluewards_sigma, 'redwards_mu':redwards_mu, 'redwards_sigma':redwards_sigma}

direct = 'dr12q/processed'
                   
place = '{}/learned_model_outdata_{}_norm_{}-{}'.format(direct, training_set_name, normalization_min_lambda, normalization_max_lambda)
             
# Open a file for writing data
file_handler = open(place, 'wb')

# Dump the data of the object into the file
dill.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()