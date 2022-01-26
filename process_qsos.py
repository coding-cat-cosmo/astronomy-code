# voigt.c: computes exact Voigt profiles in terms of the complex error function (requires libcerf)

#include "mex.h"
#include <cerf.h>
#include <math.h>

#define LAMBDAS_ARG    prhs[0]
#define Z_ARG          prhs[1]
#define N_ARG          prhs[2]
#define NUM_LINES_ARG  prhs[3]

#define PROFILE_ARG    plhs[0]
from scipy.special import voigt_profile
import numpy as np
from numba import prange, njit, jit
#from scipy.ndimage import convolve1d

#number of lines in the Lyman series to consider 
#NUM_LINES = 31

#/* note: all units are CGS */

#/* physical constants */

c   = 2.99792458e+10             #/* speed of light          cm s⁻¹        */
#/* static const double k   = 1.38064852e-16; */        /* Boltzmann constant      erg K⁻¹       */
#/* static const double m_p = 1.672621898e-24;*/        /* proton mass             g             */
#/* static const double m_e = 9.10938356e-28; */        /* electron mass           g             */
#/* e = 1.6021766208e-19 * c / 10; */
#/* static const double e   = 4.803204672997660e-10; */ /* elementary charge       statC         */

#/* Lyman series */

transition_wavelengths = np.array([1.2156701e-05, 1.0257223e-05, 9.725368e-06, 9.497431e-06, 9.378035e-06, 9.307483e-06,
9.262257e-06, 9.231504e-06, 9.209631e-06, 9.193514e-06, 9.181294e-06, 9.171806e-06,
9.16429e-06, 9.15824e-06, 9.15329e-06, 9.14919e-06, 9.14576e-06, 9.14286e-06, 9.14039e-06,
9.13826e-06, 9.13641e-06, 9.13480e-06, 9.13339e-06, 9.13215e-06, 9.13104e-06, 9.13006e-06,
9.12918e-06, 9.12839e-06, 9.12768e-06, 9.12703e-06, 9.12645e-06]) #/* transition wavelengths  cm 

oscillator_strengths = np.array([0.416400, 0.079120, 0.029000, 0.013940, 0.007799, 0.004814, 0.003183, 0.002216, 0.001605,
0.00120, 0.000921, 0.0007226, 0.000577, 0.000469, 0.000386, 0.000321, 0.000270, 0.000230,
0.000197, 0.000170, 0.000148, 0.000129, 0.000114, 0.000101, 0.000089, 0.000080, 0.000071,
0.000064, 0.000058, 0.000053, 0.000048]) #/* oscillator strengths    dimensionless */

Gammas = np.array([6.265e+08, 1.897e+08, 8.127e+07, 4.204e+07, 2.450e+07, 1.236e+07, 8.255e+06, 5.785e+06,
4.210e+06, 3.160e+06, 2.432e+06, 1.911e+06, 1.529e+06, 1.243e+06, 1.024e+06, 8.533e+05,
7.186e+05, 6.109e+05, 5.237e+05, 4.523e+05, 3.933e+05, 3.443e+05, 3.030e+05, 2.679e+05,
2.382e+05, 2.127e+05, 1.907e+05, 1.716e+05, 1.550e+05, 1.405e+05, 1.277e+05])#/* transition rates        s^-1

# assumed constant */
# static const double T = 1e+04; */                   /* gas temperature         K             */

# derived constants */

# b = sqrt(2 * k * T / m_p); */
# static const double b = 1.28486551932562422e+06; */                       /* Doppler parameter       cm s⁻¹       

# sigma = b / M_SQRT2; */
sigma = 9.08537121627923800e+05   # Gaussian width          cm s⁻¹        */

# leading_constants[i] =  M_PI * e * e * oscillator_strengths[i] * transition_wavelengths[i] / (m_e * c);

leading_constants = np.array([1.34347262962625339e-07, 2.15386482180851912e-08, 7.48525170087141461e-09, 3.51375347286007472e-09,
1.94112336271172934e-09, 1.18916112899713152e-09, 7.82448627128742997e-10, 5.42930932279390593e-10,
3.92301197282493829e-10, 2.92796010451409027e-10, 2.24422239410389782e-10, 1.75895684469038289e-10,
1.40338556137474778e-10, 1.13995374637743197e-10, 9.37706429662300083e-11, 7.79453203101192392e-11,
6.55369055970184901e-11, 5.58100321584169051e-11, 4.77895916635794548e-11, 4.12301389852588843e-11,
3.58872072638707592e-11, 3.12745536798214080e-11, 2.76337116167110415e-11, 2.44791750078032772e-11,
2.15681362798480253e-11, 1.93850080479346101e-11, 1.72025364178111889e-11, 1.55051698336865945e-11,
1.40504672409331934e-11, 1.28383057589411395e-11, 1.16264059622218997e-11]) # leading constants       cm²      

# gammas[i] = Gammas[i] * transition_wavelengths[i] / (4 * M_PI); */
gammas = np.array([6.06075804241938613e+02, 1.54841462408931704e+02, 6.28964942715328164e+01, 3.17730561586147395e+01,
1.82838676775503330e+01, 9.15463131005758157e+00, 6.08448802613156925e+00, 4.24977523573725779e+00,
3.08542121666345803e+00, 2.31184525202557767e+00, 1.77687796208123139e+00, 1.39477990932179852e+00,
1.11505539984541979e+00, 9.05885451682623022e-01, 7.45877170715450677e-01, 6.21261624902197052e-01,
5.22994533400935269e-01, 4.44469874827484512e-01, 3.80923210837841919e-01, 3.28912390446060132e-01,
2.85949711597237033e-01, 2.50280032040928802e-01, 2.20224061101442048e-01, 1.94686521675913549e-01,
1.73082093051965591e-01, 1.54536566013816490e-01, 1.38539175663870029e-01, 1.24652675945279762e-01,
1.12585442799479921e-01, 1.02045988802423507e-01, 9.27433783998286437e-02]) #/* Lorentzian widths       cm s⁻¹        

#/* BOSS spectrograph instrumental broadening */
#/* R = 2000; */                                  /* resolving power          dimensionless */

#/* width of instrument broadening in pixels */
#/* pixel_spacing = 1e-4; */
#/* pixel_sigma = 1 / (R * 2 * sqrt(2 * M_LN2) * (pow(10, pixel_spacing) - 1)); */

width = 3                      # width of convolution     dimensionless */

#  total = 0;
#  for (i = -width, j = 0; i <= width; i++, j++) {
#    instrument_profile[j] = exp(-0.5 * i * i / (pixel_sigma * pixel_sigma));
#    total += instrument_profile[j];
#  }
#  for (i = 0; i < 2 * width + 1; i++)
#    instrument_profile[i] /= total;

instrument_profile = np.array([2.17460992138080811e-03, 4.11623059580451742e-02, 2.40309364651846963e-01,
4.32707438937454059e-01, # center pixel */
2.40309364651846963e-01, 4.11623059580451742e-02, 2.17460992138080811e-03])

#    double *lambdas, *profile, *multipliers, *raw_profile;
#double z, N, velocity, total;
#int num_lines, i, j, k;
#mwSize num_points;
#@jit
def genvoigtapprox(minvel=-9000000000,maxvel=+90000000000,npts=300000):
    from scipy.interpolate import interp1d

    sigma = 9.08537121627923800e+05 # Gaussian width          cm s⁻¹        */
    #/* Lorentzian widths       cm s⁻¹
    gammas = np.array([6.06075804241938613e+02, 1.54841462408931704e+02, 6.28964942715328164e+01, 3.17730561586147395e+01,
                       1.82838676775503330e+01, 9.15463131005758157e+00, 6.08448802613156925e+00, 4.24977523573725779e+00,
                       3.08542121666345803e+00, 2.31184525202557767e+00, 1.77687796208123139e+00, 1.39477990932179852e+00,
                       1.11505539984541979e+00, 9.05885451682623022e-01, 7.45877170715450677e-01, 6.21261624902197052e-01,
                       5.22994533400935269e-01, 4.44469874827484512e-01, 3.80923210837841919e-01, 3.28912390446060132e-01,
                       2.85949711597237033e-01, 2.50280032040928802e-01, 2.20224061101442048e-01, 1.94686521675913549e-01,
                       1.73082093051965591e-01, 1.54536566013816490e-01, 1.38539175663870029e-01, 1.24652675945279762e-01,
                       1.12585442799479921e-01, 1.02045988802423507e-01, 9.27433783998286437e-02])
    vels =   np.linspace(minvel,maxvel,npts)
    #moreParams.num_lines = 3
    voigts = voigt_profile(vels[:,None],sigma,gammas[None,:3])

    myapprox = [interp1d(vels,voigts[:,i]) for i in range(3)]

    def voigtapprox(vel):
        # vel is m-by-n (where n = len(gammas))
        m = vel.shape[0]
        n = 3

        res = np.zeros((m,n))
        for i in range(n):
            res[:,i] = myapprox[i](vel[:,i])
        # just temporarily
        #rescheck = voigt_profile(vel,sigma,gammas[None,:])
        #if ((np.abs(rescheck-res)/rescheck).max() > 1e-6):
        #    print("res\n")
        #    print(res)
        #    print(res.shape)
        #    print("rescheck\n")
        #    print(rescheck)
        #    print(rescheck.shape)
        return res

    return voigtapprox


vapprox = genvoigtapprox()

def voigt(LAMBDAS_ARG, Z_ARG, N_ARG, NUM_LINES_ARG=31):



    #/* get input */
    lambdas = LAMBDAS_ARG                #/* wavelengths             Å             */
    z       = Z_ARG                  #/* redshift                dimensionless */
    N       = N_ARG                  #/* column density          cm⁻²          */

    num_lines = NUM_LINES_ARG

    #num_points = len(LAMBDAS_ARG)

    #/* initialize output */
    #PROFILE_ARG = mxCreateDoubleMatrix(num_points - 2 * width, 1, mxREAL);
    #profile = mxGetPr(PROFILE_ARG);                /* absorption profile      dimensionless */
    #profile = np.zeros(num_points - 2 * width)
    #profile[:] = 0

    #/* to hold the profile before instrumental broadening */
    #raw_profile = mxMalloc(num_points * sizeof(double));
    #raw_profile = []

    #multipliers = mxMalloc(num_lines * sizeof(double));
    #multipliers = []
    #for i in range(num_lines):
        #multipliers.append(c / (transition_wavelengths[i] * (1 + z)) / 100000000)

    multipliers = c / (transition_wavelengths[:num_lines] * (1+z)) / 100000000

    #/* compute raw Voigt profile */
    #for i in range(num_points):
        #/* apply each absorption line */
       # total = 0
       # for j in range(num_lines):
          #/* velocity relative to transition wavelength */
       #     velocity = lambdas[i] * multipliers[j] - c
       #     total += -leading_constants[j] * voigt_profile(velocity, sigma, gammas[j])

       # raw_profile.append(np.exp(N * total))
        
    velocity = lambdas[:,None] * multipliers[None,:num_lines] - c
    #subtotal = -leading_constants[None,:num_lines] * voigt_profile(velocity,sigma,gammas[None,:num_lines])
    subtotal = -leading_constants[None,:num_lines] * vapprox(velocity)
    total = subtotal.sum(axis=1)
    raw_profile = np.exp(N*total)

    #print("\nsum_points before", num_points)
    #print("\nsum_points after", num_points - 2 * width)
    #num_points = num_points - 2 * width
    #print("raw_profile", raw_profile)

    #/* instrumental broadening */
   # for i in range(num_points):
   #     k = 0
   #     for j in range(i, i+2*width+1):
   #         profile[i] += raw_profile[j] * instrument_profile[k]
   #         k+=1
            
    profile = np.convolve(raw_profile, instrument_profile, mode='valid')
    #profile = convolve1d(raw_profile, instrument_profile)

    return profile

class outputs:
    def __init__(self, fluxes, rest_wavelengths, min_z_dlas, max_z_dlas, this_p_dlas, this_sample_log_priors_no_dla,
                 this_sample_log_priors_dla, used_z_dla, this_sample_log_likelihoods_no_dla,
                 this_sample_log_likelihoods_dla, i):
        self.fluxes = fluxes
        self.rest_wavelengths = rest_wavelengths
        self.min_z_dlas = min_z_dlas
        self.max_z_dlas = max_z_dlas
        self.this_p_dlas = this_p_dlas
        self.this_sample_log_priors_no_dla = this_sample_log_priors_no_dla
        self.this_sample_log_priors_dla = this_sample_log_priors_dla
        self.used_z_dla = used_z_dla
        self.this_sample_log_likelihoods_no_dla = this_sample_log_likelihoods_no_dla
        self.this_sample_log_likelihoods_dla = this_sample_log_likelihoods_dla
        self.i = i
    
class inputs:
    def __init__(self, offset_samples_qso, this_out_wavelengths, this_out_flux, this_out_noise_variance,
                    this_out_pixel_mask, bluewards_mu, bluewards_sigma, redwards_mu, redwards_sigma,
                    fluxes, rest_wavelengths, orig_z_qsos, dla_ind, this_p_dlas, c_0, tau_0, beta,
                    prev_tau_0, prev_beta, occams_factor, quasar_ind, offset_samples, nhi_samples,
                    min_z_dlas, max_z_dlas, this_sample_log_priors_no_dla, this_sample_log_priors_dla,
                    this_sample_log_likelihoods_no_dla, this_sample_log_likelihoods_dla, used_z_dla):
        self.offset_samples_qso = offset_samples_qso
        self.this_out_wavelengths = this_out_wavelengths
        self.this_out_flux = this_out_flux
        self.this_out_noise_variance = this_out_noise_variance
        self.this_out_pixel_mask = this_out_pixel_mask
        self.bluewards_mu = bluewards_mu
        self.bluewards_sigma = bluewards_sigma
        self.redwards_mu = redwards_mu
        self.redwards_sigma = redwards_sigma
        self.fluxes = fluxes
        self.rest_wavelengths = rest_wavelengths
        self.orig_z_qsos = orig_z_qsos
        self.dla_ind = dla_ind
        self.this_p_dlas = this_p_dlas
        self.c_0 = c_0
        self.tau_0 = tau_0
        self.beta = beta
        self.prev_tau_0 = prev_tau_0
        self.prev_beta = prev_beta
        self.occams_factor = occams_factor
        self.quasar_ind = quasar_ind
        self.offset_samples = offset_samples
        self.nhi_samples = nhi_samples
        self.min_z_dlas = min_z_dlas
        self.max_z_dlas = max_z_dlas
        self.this_sample_log_priors_no_dla = this_sample_log_priors_no_dla
        self.this_sample_log_priors_dla = this_sample_log_priors_dla
        self.this_sample_log_likelihoods_no_dla = this_sample_log_likelihoods_no_dla
        self.this_sample_log_likelihoods_dla = this_sample_log_likelihoods_dla
        self.used_z_dla = used_z_dla

    def __call__(self, i):

        z_qso = self.offset_samples_qso[i]
        #print("\nz_qso")
        #print(z_qso)

        # keep a copy inside the parfor since we are modifying them
        this_wavelengths    = self.this_out_wavelengths
        this_flux           = self.this_out_flux
        this_noise_variance = self.this_out_noise_variance
        this_pixel_mask     = self.this_out_pixel_mask

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
        bw_log_likelihood = log_mvnpdf_iid(this_normalized_flux_bw, self.bluewards_mu * np.ones((n_bw)), self.bluewards_sigma**2 * np.ones((n_bw)) + this_normalized_noise_variance_bw)
        rw_log_likelihood = log_mvnpdf_iid(this_normalized_flux_rw, self.redwards_mu * np.ones((n_rw)), self.redwards_sigma**2 * np.ones((n_rw)) + this_normalized_noise_variance_rw)

        ind = ind & np.logical_not(this_pixel_mask)

        this_wavelengths      =      this_wavelengths[ind]
        this_rest_wavelengths = this_rest_wavelengths[ind]
        this_flux             =             this_flux[ind]
        this_noise_variance   =   this_noise_variance[ind]

        this_noise_variance[np.isinf(this_noise_variance)] = np.nanmean(this_noise_variance) #rare kludge to fix bad data

        if (np.count_nonzero(this_rest_wavelengths) < 1):
            print(' took {tc:0.3f}s.\n'.format(tc=time.time() - t))
            self.fluxes.append(0)
            self.rest_wavelengths.append(0)
            #self.fluxes = 0
            #self.rest_wavelengths = 0

            values = outputs(self.fluxes, self.rest_wavelengths, self.min_z_dlas, self.max_z_dlas, self.this_p_dlas,
                             self.this_sample_log_priors_no_dla, self.this_sample_log_priors_dla, self.used_z_dla,
                             self.this_sample_log_likelihoods_no_dla, self.this_sample_log_likelihoods_dla, i)

            return values

        self.fluxes.append(this_flux)
        self.rest_wavelengths.append(this_rest_wavelengths)
        #self.fluxes = this_flux
        #self.rest_wavelengths = this_rest_wavelengths

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
        #print("\n\n\n")

        #for l in range(learnParams.num_forest_lines):
        #    this_lyseries_zs[:, l] = (this_wavelengths - learnParams.all_transition_wavelengths[l]) / learnParams.all_transition_wavelengths[l]
        this_lyseries_zs = (this_wavelengths[None, :] - np.array(learnParams.all_transition_wavelengths[:learnParams.num_forest_lines])[:, None]) / np.array(learnParams.all_transition_wavelengths[:learnParams.num_forest_lines])[:, None]
        this_lyseries_zs = this_lyseries_zs.T


        #print("\nthis_lyseries_zs")
        #print(this_lyseries_zs)
        #print(this_lyseries_zs.shape)
        # DLA existence prior
        less_ind = (self.orig_z_qsos < (z_qso + modelParams.prior_z_qso_increase))
        #print("\nless_ind")
        #print(less_ind)
        #print(len(less_ind), np.count_nonzero(less_ind))
        #print("\ndla_ind")
        #print(dla_ind)
        #print(dla_ind.shape)

        this_num_dlas    = np.count_nonzero(self.dla_ind[less_ind])
        this_num_quasars = np.count_nonzero(less_ind)
        this_p_dla       = this_num_dlas / float(this_num_quasars)
        self.this_p_dlas[i]   = this_p_dla
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
        self.this_sample_log_priors_dla[i] = np.log(this_num_dlas) - np.log(this_num_quasars)
        self.this_sample_log_priors_no_dla[i] = np.log(this_num_quasars - this_num_dlas) - np.log(this_num_quasars)
        #print("\nthis_sample_log_priors_dla")
        #print(this_sample_log_priors_dla)
        #print(this_sample_log_priors_dla.shape)
        #print("\nthis_sample_log_priors_no_dla")
        #print(this_sample_log_priors_no_dla)
        #print(this_sample_log_priors_no_dla.shape)

        #sample_log_priors_dla(quasar_ind, z_list_ind) = log(.5);
        #sample_log_priors_no_dla(quasar_ind, z_list_ind) = log(.5);

        # fprintf_debug('\n');
        #print(' ...     p(   DLA | z_QSO)        : {th:0.3f}\n'.format(th=this_p_dla))
        #print(' ...     p(no DLA | z_QSO)        : {th:0.3f}\n'.format(th=1 - this_p_dla))

        # interpolate model onto given wavelengths
        this_mu = mu_interpolator(this_rest_wavelengths)
        #print("\nthis_mu")
        #print(this_mu)
        #print(this_mu.shape)
        #this_M  =  M_interpolator((this_rest_wavelengths[:, None], np.arange(nullParams.k)[None, :]))
        this_M  =  M_interpolator(this_rest_wavelengths)
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
        lya_optical_depth = self.tau_0 * (1 + this_lya_zs)**self.beta
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

        #for l in range(1,learnParams.num_forest_lines):
        #    lyman_1pz = learnParams.all_transition_wavelengths[0] * (1 + this_lya_zs) / learnParams.all_transition_wavelengths[l]

            # only include the Lyman series with absorber redshifts lower than z_qso
        #    indicator = lyman_1pz <= (1 + z_qso)
        #    lyman_1pz = lyman_1pz * indicator

        #    tau = self.tau_0 * learnParams.all_transition_wavelengths[l] * learnParams.all_oscillator_strengths[l] / (learnParams.all_transition_wavelengths[0] * learnParams.all_oscillator_strengths[0])

         #   lya_optical_depth = lya_optical_depth + tau * lyman_1pz**self.beta
            
        lyman_1pz = learnParams.all_transition_wavelengths[0] * (1+this_lya_zs) / learnParams.all_transition_wavelengths[1:learnParams.num_forest_lines][:, None]
        lyman_1pz = lyman_1pz * (lyman_1pz <= (1+z_qso))
        tau = self.tau_0 * learnParams.all_transition_wavelengths[1:learnParams.num_forest_lines][:, None] * learnParams.all_oscillator_strengths[1:learnParams.num_forest_lines][:, None] / (learnParams.all_transition_wavelengths[0] * learnParams.all_oscillator_strengths[0])
        other = tau * (lyman_1pz**self.beta)
        other = other.sum(axis=0)
        lya_optical_depth = lya_optical_depth + other


        this_scaling_factor = 1 - np.exp( -lya_optical_depth ) + self.c_0

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

        #for l in range(learnParams.num_forest_lines):
            # calculate the oscillator strength for this lyman series member
         #   this_tau_0 = self.prev_tau_0 * learnParams.all_oscillator_strengths[l] / learnParams.lya_oscillator_strength * learnParams.all_transition_wavelengths[l] / physConst.lya_wavelength

          #  total_optical_depth[:, l] = this_tau_0 * ( (1 + this_lyseries_zs[:, l])**self.prev_beta )

            # indicator function: z absorbers <= z_qso
            # here is different from multi-dla processing script
            # I choose to use zero instead or nan to indicate
            # values outside of the Lyman forest
          #  indicator = this_lyseries_zs[:, l] <= z_qso
          #  total_optical_depth[:, l] = total_optical_depth[:, l] * indicator
            
        this_tau_0 = self.prev_tau_0 * learnParams.all_oscillator_strengths[:learnParams.num_forest_lines] / learnParams.lya_oscillator_strength * learnParams.all_transition_wavelengths[:learnParams.num_forest_lines] / physConst.lya_wavelength
        total_optical_depth = this_tau_0 * ((1+this_lyseries_zs[None, :])**self.prev_beta)
        total_optical_depth = total_optical_depth * (this_lyseries_zs <= z_qso)
        total_optical_depth = np.squeeze(total_optical_depth)

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

        occams = self.occams_factor * (1 - lambda_observed / (nullParams.max_lambda - nullParams.min_lambda) )
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
        self.this_sample_log_likelihoods_no_dla[i] = log_mvnpdf_low_rank(this_flux, this_mu, this_M, this_omega2 + this_noise_variance) + bw_log_likelihood + rw_log_likelihood - occams
        #this_sample_log_likelihoods_no_dla[i] = this_sample_log_likelihoods_no_dla[i] + bw_log_likelihood + rw_log_likelihood - occams
        #except:
            #if (strcmp(ME.identifier, 'MATLAB:posdef')):
            #    this_posdeferror[i] = True
            #    print('(QSO {qua}, Sample {it}): Matrix must be positive definite. We skip this sample but you need to be careful about this spectrum'.format(qua=quasar_num, it=i))
            #    continue;

        #    raise

        #print(' ... log p(D | z_QSO, no DLA)     : {ts:0.2f}\n'.format(ts=this_sample_log_likelihoods_no_dla[i]))

        # Add
        if this_wavelengths.shape[0] == 0:
            print("dunded")
            values = outputs(self.fluxes, self.rest_wavelengths, self.min_z_dlas, self.max_z_dlas, self.this_p_dlas,
                             self.this_sample_log_priors_no_dla, self.this_sample_log_priors_dla, self.used_z_dla,
                             self.this_sample_log_likelihoods_no_dla, self.this_sample_log_likelihoods_dla, i)

            return values
            #break

        # use a temp variable to avoid the possible parfor issue
        # should be fine after change size of min_z_dlas to (num_quasar, num_dla_samples)
        this_min_z_dlas = moreParams.min_z_dla(this_wavelengths, z_qso)
        this_max_z_dlas = moreParams.max_z_dla(this_wavelengths, z_qso)

        self.min_z_dlas[self.quasar_ind, i] = this_min_z_dlas
        self.max_z_dlas[self.quasar_ind, i] = this_max_z_dlas

        sample_z_dlas = this_min_z_dlas + (this_max_z_dlas - this_min_z_dlas) * self.offset_samples

        self.used_z_dla[i] = sample_z_dlas[i]
        #print("\nthis_min_z_dlas")
        #print(this_min_z_dlas)
        #print(this_min_z_dlas.shape)
        #print("\nthis_max_z_dlas")
        #print(this_max_z_dlas)
        #print(this_max_z_dlas.shape)
        #print("\nmin_z_dlas")
        #print(min_z_dlas)
        #print(min_z_dlas.shape)
        #print("\nmax_z_dlas")
        #print(max_z_dlas)
        #print(max_z_dlas.shape)
        #print("\nsample_z_dlas")
        #print(sample_z_dlas)
        #print(sample_z_dlas.shape)
        #print("\nused_z_dla")
        #print(used_z_dla)
        #print(used_z_dla.shape)

        # ensure enough pixels are on either side for convolving with instrument profile
        padded_wavelengths = [np.logspace(np.log10(np.min(this_unmasked_wavelengths)) - instrumentParams.width * instrumentParams.pixel_spacing,
                              np.log10(np.min(this_unmasked_wavelengths)) - instrumentParams.pixel_spacing, instrumentParams.width).T,
                              this_unmasked_wavelengths,
                              np.logspace(np.log10(np.max(this_unmasked_wavelengths)) + instrumentParams.pixel_spacing,
                              np.log10(np.max(this_unmasked_wavelengths)) + instrumentParams.width * instrumentParams.pixel_spacing, instrumentParams.width).T]

        padded_wavelengths = np.array([x for ls in padded_wavelengths for x in ls])
        #print("\npadded_wavelengths")
        #print(padded_wavelengths)
        #print(padded_wavelengths.shape)
        #print(len(padded_wavelengths))
        # to retain only unmasked pixels from computed absorption profile
        ind = np.logical_not(this_pixel_mask[ind])
        #print("\nind")
        #print(ind)
        #print(len(ind), np.count_nonzero(ind))

        # compute probabilities under DLA model for each of the sampled
        # (normalized offset, log(N HI)) pairs absorption corresponding to this sample
        #scipy.special.voigt_profile(x, sigma, gamma, out=None)
        #print("sample_z_dlas[i]")
        #print(sample_z_dlas[i])
        #print("nhi_samples[i]")
        #print(nhi_samples[i])
        #print("num_lines", moreParams.num_lines)
        absorption = voigt(padded_wavelengths, sample_z_dlas[i], self.nhi_samples[i], moreParams.num_lines)
        #print("\nabsorption")
        #print(absorption)
        #print(absorption.shape)
        # add this line back for implementing pixel masking
        temp = np.zeros(len(absorption)-len(ind))
        ind = np.append(ind, temp)
        ind = np.array(ind, dtype=bool)
        absorption = absorption[ind]
        #print("\nabsorption")
        #print(absorption)
        #print(absorption.shape)

        # delta z = v / c = H(z) d / c = 70 (km/s/Mpc) * sqrt(0.3 * (1+z)^3 + 0.7) * (5 Mpc) / (3x10^5 km/s) ~ 0.005 at z=3
        if flag.add_proximity_zone:
            print("\nreached this")
            delta_z = (70 * np.sqrt(.3 * (1+z_qso)**3 + .7) * 5) / (3 * 10**5)
            #print("\ndelta_z")
            #print(delta_z)


        dla_mu = this_mu * absorption
        dla_M = this_M * absorption[:, None]
        dla_omega2 = this_omega2 * absorption**2
        #print("\ndla_mu")
        #print(dla_mu)
        #print(dla_mu.shape)
        #print("\ndla_omega2")
        #print(dla_omega2)
        #print(dla_omega2.shape)

        # Add a penalty for short spectra: the expected reduced chi^2 of each spectral pixel that would have been observed.
        # Also add an error handler for DLA model likelihood function
        #try:
        self.this_sample_log_likelihoods_dla[i] = log_mvnpdf_low_rank(this_flux, dla_mu, dla_M, dla_omega2 + this_noise_variance) + bw_log_likelihood + rw_log_likelihood - occams
        #except:
            #if (strcmp(ME.identifier, 'MATLAB:posdef')):
            #    this_posdeferror[i] = True
            #    print('(QSO {qua}, Sample {it}): Matrix must be positive definite. We skip this sample but you need to be careful about this spectrum'.format(qua=quasar_num, it=i))
            #    continue;

            #    raise
        #print("\nthis_sample_log_likelihoods_dla")
        #print(this_sample_log_likelihoods_dla)
        #print(this_sample_log_likelihoods_dla.shape)
        #DEBUGGING
        values = outputs(self.fluxes, self.rest_wavelengths, self.min_z_dlas, self.max_z_dlas, self.this_p_dlas,
                             self.this_sample_log_priors_no_dla, self.this_sample_log_priors_dla, self.used_z_dla,
                             self.this_sample_log_likelihoods_no_dla, self.this_sample_log_likelihoods_dla, i)
        return values

        # log_mvnpdf_iid: computes mutlivariate normal dist with
#    each dim is iid, so no covariance. 
#   log N(y; mu, diag(d))
@njit
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

#from scipy.linalg import cho_factor, cho_solve
#from scipy.linalg import lu_factor, lu_solve
@njit
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
    #print("\nd", d, d.shape)
    d_inv = 1 / d
    #print("\nd_inv", d_inv, d_inv.shape)
    D_inv_y = d_inv * y
    #print("\nD_inv_y", D_inv_y, D_inv_y.shape)

    #D_inv_M = d_inv[:, None] * M
    d_inv = np.reshape(d_inv, (d_inv.shape[0], 1))
    D_inv_M = d_inv * M
    #print("D_inv_M", D_inv_M, D_inv_M.shape)

    # use Woodbury identity, define
    #   B = (I + M' D^-1 M),
    # then
    #   K^-1 = D^-1 - D^-1 M B^-1 M' D^-1

    B = np.dot(M.T, D_inv_M)
    #print("\nB before", B, B.shape)
    B = np.reshape(B, B.shape[0]*B.shape[1])
    B[0::k+1] = B[0::k+1] + 1
    B = np.reshape(B, (k, k))
    #B.flat[0::k+1] = B.flat[0::k+1] + 1
    #print("\nB after", B, B.shape)
    #L = np.linalg.cholesky(B)
    #L= L.T
    #print("\nL", L, L.shape)
    # C = B^-1 M' D^-1
    #ld = np.linalg.solve(L.T, D_inv_M.T)
    #print("\nld", ld, ld.shape)
    #C= np.linalg.solve(L, ld)
    C = np.linalg.solve(B, D_inv_M.T)
    #lu, piv = lu_factor(B)
    #C = lu_solve((lu, piv), D_inv_M.T, check_finite=False)
    #C = np.linalg.solve(L.T, np.linalg.solve(L, D_inv_M.T))
    #print("\nC", C, C.shape)
    #L, low = cho_factor(B)
    #ld = cho_solve((L.T, low), D_inv_M.T)
    #C = cho_solve((L, low),  D_inv_M.T)

    #mid = np.dot(C, y)
    #print("\nmid", mid, mid.shape)
    #after = np.dot(D_inv_M, mid)
    #print("\nafter", after, after.shape)
    K_inv_y = D_inv_y - np.dot(D_inv_M, np.dot(C, y))
    #print("\nK_inv_y", K_inv_y, K_inv_y.shape)

    #log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))
    log_det_K = np.sum(np.log(d)) + np.linalg.slogdet(B)[1]
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
#from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from multiprocessing import Pool
from scipy.special import voigt_profile
#import cProfile
#from voigt import voigt

#dill.load_session('parameters.pkl')
#flags for changes
#extrapolate_subdla = 0 #0 = off, 1 = on
#add_proximity_zone = 0
#integrate          = 1
#optTag = (str(integrate), str(extrapolate_subdla), str(add_proximity_zone))

# physical constants
#lya_wavelength = 1215.6701                   # Lyman alpha transition wavelength  Å
#lyb_wavelength = 1025.7223                   # Lyman beta  transition wavelength  Å
#lyman_limit    =  911.7633                   # Lyman limit wavelength             Å
#speed_of_light = 299792458                   # speed of light                     m s⁻¹

# converts relative velocity in km s^-1 to redshift difference
#kms_to_z = lambda kms : (kms * 1000) / speed_of_light

# utility functions for redshifting
#emitted_wavelengths = lambda observed_wavelengths, z : (observed_wavelengths / (1 + z))

#observed_wavelengths = lambda emitted_wavelengths, z : (emitted_wavelengths * (1 + z))

# preprocessing parameters
#z_qso_cut      = 2.15                        # filter out QSOs with z less than this threshold
#z_qso_training_max_cut = 5                   # roughly 95% of training data occurs before this redshift; 
                                             # assuming for normalization purposes (move to set_parameters when pleased)
#min_num_pixels = 400                         # minimum number of non-masked pixels

# normalization parameters
# I use 1216 is basically because I want integer in my saved filenames
#normalization_min_lambda = 1216 - 40              # range of rest wavelengths to use   Å
#normalization_max_lambda = 1216 + 40              #   for flux normalization

# file loading parameters: this is no longer used.
#loading_min_lambda = 700                   # range of rest wavelengths to load  Å
#loading_max_lambda = 5000                  # This maximum is set so we include CIV.
# The maximum allowed is set so that even if the peak is redshifted off the end, the
# quasar still has data in the range

class flags: #flags for changes
    def __init__(self): #0 = off, 1 = on
        self.extrapolate_subdla = 0
        self.add_proximity_zone = 0
        self.integrate = 1
        self.optTag = (str(self.integrate), str(self.extrapolate_subdla), str(self.add_proximity_zone))

   # def __getstate__(self):
   #     attributes = self.__dict__.copy()
   #     return attributes

#saving_flag = flags()
#my_pickle_string = pickle.dumps(my_foobar_instance)
#my_new_instance = pickle.loads(my_pickle_string)
#print(my_new_instance.__dict__)

class physical_constants:
    def __init__(self):
        self.lya_wavelength = 1215.6701                   # Lyman alpha transition wavelength  Å
        self.lyb_wavelength = 1025.7223                   # Lyman beta  transition wavelength  Å
        self.lyman_limit    =  911.7633                   # Lyman limit wavelength             Å
        self.speed_of_light = 299792458                   # speed of light                     m s⁻¹

physConst = physical_constants()        
# converts relative velocity in km s^-1 to redshift difference
kms_to_z = lambda kms : (kms * 1000) / physConst.speed_of_light

# utility functions for redshifting
emitted_wavelengths = lambda observed_wavelengths, z : (observed_wavelengths / float((1 + z)))

observed_wavelengths = lambda emitted_wavelengths, z : (emitted_wavelengths * (1 + z))

class preproccesing_params:
    def __init__(self):
        self.z_qso_cut      = 2.15                        # filter out QSOs with z less than this threshold
        self.z_qso_training_max_cut = 5                   # roughly 95% of training data occurs before this redshift; 
                                             # assuming for normalization purposes (move to set_parameters when pleased)
        self.min_num_pixels = 400                         # minimum number of non-masked pixels

class normalization_params:
    def __init__(self):
        self.normalization_min_lambda = 1216 - 40              # range of rest wavelengths to use   Å
        self.normalization_max_lambda = 1216 + 40              #   for flux normalization
        
class file_loading:
    def __init__(self):
        self.loading_min_lambda = 700                   # range of rest wavelengths to load  Å
        self.loading_max_lambda = 5000                  # This maximum is set so we include CIV.
# The maximum allowed is set so that even if the peak is redshifted off the end, the
# quasar still has data in the range

# null model parameters
class null_params:
    def __init__(self):
        self.min_lambda = 910              # range of rest wavelengths to       Å
        self.max_lambda = 3000             #   model
        self.dlambda = 0.25                # separation of wavelength grid      Å
        self.k = 20                        # rank of non-diagonal contribution
        self.max_noise_variance = 4**2     # maximum pixel noise allowed during model training


# Lyman-series array: for modelling the forests of Lyman series
class learning_params:
    def __init__(self):
        self.num_forest_lines = 6
        self.all_transition_wavelengths = np.array([1215.6701, 1025.7223, 972.5368, 949.7431, 937.8035,
                                           930.7483, 926.2257, 923.1504, 920.9631, 919.3514,
                                           918.1294, 917.1806, 916.429, 915.824, 915.329, 914.919,
                                           914.576, 914.286, 914.039,913.826, 913.641, 913.480,
                                           913.339, 913.215, 913.104, 913.006, 912.918, 912.839,
                                           912.768, 912.703, 912.645]) # transition wavelengths, Å
        self.all_oscillator_strengths = np.array([0.416400, 0.079120, 0.029000, 0.013940, 0.007799, 0.004814, 0.003183, 0.002216, 
                                                  0.001605, 0.00120, 0.000921, 0.0007226, 0.000577, 0.000469, 0.000386, 0.000321, 0.000270, 
                                           0.000230, 0.000197, 0.000170, 0.000148, 0.000129, 0.000114, 0.000101, 0.000089, 0.000080,
                                           0.000071, 0.000064, 0.000058, 0.000053, 0.000048])
        # oscillator strengths
        self.lya_oscillator_strength = 0.416400
        self.lyb_oscillator_strength = 0.079120

# optimization parameters
class optimization_params:
    def __init__(self):
        self.initial_c_0 = 0.1   # initial guess for c₀
        self.initial_tau_0 = 0.0023   # initial guess for τ₀
        self.initial_beta = 3.65  # initial guess for β
        
# DLA model parameters: parameter samples
class dla_params:
    def __init__(self):
        self.num_dla_samples     = 100000                 # number of parameter samples
        self.alpha               = 0.9                    # weight of KDE component in mixture
        self.uniform_min_log_nhi = 20.0                   # range of column density samples    [cm⁻²]
        self.uniform_max_log_nhi = 23.0                   # from uniform distribution
        self.fit_min_log_nhi     = 20.0                   # range of column density samples    [cm⁻²]
        self.fit_max_log_nhi     = 22.0                   # from fit to log PDF
        
# model prior parameters
class model_params:
    def __init__(self):
        self.prior_z_qso_increase = kms_to_z(30000.0)       # use QSOs with z < (z_QSO + x) for prior

# instrumental broadening parameters
class instrument_params:
    def __init__(self):
        self.width = 3                                    # width of Gaussian broadening (# pixels)
        self.pixel_spacing = .0001                        # wavelength spacing of pixels in dex

# DLA model parameters: absorber range and model
class more_dla_params:
    def __init__(self):
        self.num_lines = 3                                # number of members of the Lyman series to use
        self.max_z_cut = kms_to_z(3000.0)                   # max z_DLA = z_QSO - max_z_cut
 # determines maximum z_DLA to search
        self.max_z_dla = lambda wavelengths, z_qso : min((np.max(wavelengths) / physConst.lya_wavelength - 1) - kms_to_z(3000.0), z_qso - kms_to_z(3000.0))
        self.min_z_cut = kms_to_z(3000.0)                   # min z_DLA = z_Ly∞ + min_z_cut
# determines minimum z_DLA to search
        self.min_z_dla = lambda wavelengths, z_qso : max(np.min(wavelengths) / physConst.lya_wavelength - 1,observed_wavelengths(physConst.lyman_limit, z_qso) / physConst.lya_wavelength - 1 + kms_to_z(3000.0))


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
flag = flags()

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
M = model['M']
log_omega = model['log_omega']
log_c_0 = model['log_c_0']
log_tau_0 = model['log_tau_0']
log_beta = model['log_beta']
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
#lease = "dla_samples"
lease = "dlaSamples"
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
#print("lengths", rest_wavelengths.shape, mu.shape)
mu_interpolator = interp1d(rest_wavelengths, mu, assume_sorted=True)

#temp = (rest_wavelengths, [x for x in range(k)])
M_interpolator = interp1d(rest_wavelengths , M, axis=0, assume_sorted=True)

log_omega_interpolator = interp1d(rest_wavelengths, log_omega, assume_sorted=True)

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
#DLA_cut = 20.3
#sub20pt3_ind = (log_nhi_samples < DLA_cut)
#size = np.count_nonzero(sub20pt3_ind)
#sample_log_posteriors_dla_sub = np.empty((size))
#sample_log_posteriors_dla_sub[:] = np.NaN
#sample_log_posteriors_dla_sup = np.empty((size))
#sample_log_posteriors_dla_sup[:] = np.NaN
sample_log_posteriors_dla_sub = []
sample_log_posteriors_dla_sup = []
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
quasar_ind = 0 #instead of 0
                            
#try
#    load(['./checkpointing/curDLA_', optTag, '.mat']); %checkmarking code
#catch ME
#    0;

q_ind_start = quasar_ind

# catch the exceptions
all_exceptions = np.zeros((num_quasars))

all_exceptions = [False for x in all_exceptions]
all_exceptions = np.array(all_exceptions)
#for i in range(all_exceptions):
 #   all_exceptions[i] = False
    
all_posdeferrors = np.zeros((num_quasars))

qi = 0
arr = []

#print("\nprior_ind")
#print(prior_ind)
#print(len(prior_ind), np.count_nonzero(prior_ind))
#print("\ndla_ind")
#print(dla_ind)
#print(dla_ind.shape, np.count_nonzero(dla_ind))
#print("\nrest_wavelengths")
#print(rest_wavelengths)
#print("\nmu")
#print(mu)
#print(mu.shape)
#print("\nM")
#print(M)
#print(M.shape)
print("\nlog_omega")
print(log_omega)
print(len(log_omega))
print("\nlog_c_0")
print(log_c_0)
print("\nlog_tau_0")
print(log_tau_0)
print("\nlog_beta")
print(log_beta)
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
print("\nlog_nhi_samples")
print(log_nhi_samples)
print(log_nhi_samples.shape)
print("\nhi_samples")
print(nhi_samples)
print(nhi_samples.shape)
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

#debugging for now
num_quasars = 2
for quasar_ind in range(q_ind_start, num_quasars): #quasar list
    arr.append(quasar_ind)
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
        continue

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
    #print("\nthis_out_wavelengths")
    #print(this_out_wavelengths)
    #print(this_out_wavelengths.shape)
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
    #should be dlaParams.num_dla_samples
    args = inputs(offset_samples_qso, this_out_wavelengths, this_out_flux,
                      this_out_noise_variance, this_out_pixel_mask,
                      bluewards_mu, bluewards_sigma, redwards_mu, redwards_sigma,
                      fluxes, rest_wavelengths, orig_z_qsos, dla_ind, this_p_dlas,
                      c_0, tau_0, beta, prev_tau_0, prev_beta, occams_factor,
                      quasar_ind, offset_samples, nhi_samples, min_z_dlas,
                      max_z_dlas, this_sample_log_priors_no_dla, this_sample_log_priors_dla,
                      this_sample_log_likelihoods_no_dla, this_sample_log_likelihoods_dla, used_z_dla)
    #pr = cProfile.Profile()
    #pr.enable()
    #with Pool(5) as p: # with index
    #    values = p.map(args, range(dlaParams.num_dla_samples))
    for i in range(dlaParams.num_dla_samples):  #variant redshift in quasars 
        #[fluxes, rest_wavelengths, min_z_dlas, max_z_dlas, this_p_dlas, used_z_dla,
        #this_sample_log_priors_no_dla, this_sample_log_priors_dla, this_sample_log_likelihoods_no_dla,
         #this_sample_log_likelihoods_dla, i]
        values = args(i)
    #pr.disable()
    
    #for val in values:
    #    fluxes.append(val.fluxes)
    #    rest_wavelengths.append(val.rest_wavelengths)
    #    min_z_dlas[quasar_ind, val.i] = val.min_z_dlas[quasar_ind, val.i]
    #    max_z_dlas[quasar_ind, val.i] = val.max_z_dlas[quasar_ind, val.i]
    #    this_p_dlas[val.i] = val.this_p_dlas[val.i]
    #    used_z_dla[val.i] = val.used_z_dla[val.i]
    #    this_sample_log_priors_no_dla[val.i] = val.this_sample_log_priors_no_dla[val.i]
    #    this_sample_log_priors_dla[val.i] = val.this_sample_log_priors_dla[val.i]
    #    this_sample_log_likelihoods_no_dla[val.i] = val.this_sample_log_likelihoods_no_dla[val.i]
    #    this_sample_log_likelihoods_dla[val.i] = val.this_sample_log_likelihoods_dla[val.i]
        
    #
    fluxes = values.fluxes
    rest_wavelengths = values.rest_wavelengths
    min_z_dlas = values.min_z_dlas
    max_z_dlas = values.max_z_dlas
    this_p_dlas = values.this_p_dlas
    used_z_dla = values.used_z_dla
    this_sample_log_priors_no_dla = values.this_sample_log_priors_no_dla
    this_sample_log_priors_dla = values.this_sample_log_priors_dla
    this_sample_log_likelihoods_no_dla = values.this_sample_log_likelihoods_no_dla
    this_sample_log_likelihoods_dla = values.this_sample_log_likelihoods_dla

    # to re-evaluate the model posterior for P(DLA| logNHI > 20.3)
    # we need to select the samples with > DLA_cut and re-calculate the Bayesian model selection
    #print("\nfluxes")
    #print(fluxes)
    #print(len(fluxes))
    #print("\nrest_wavelengths")
    #print(rest_wavelengths)
    #print(len(rest_wavelengths))
    #print("\nmin_z_dlas")
    #print(min_z_dlas)
    #print(min_z_dlas.shape)
    #print("\nmax_z_dlas")
    #print(max_z_dlas)
    #print(max_z_dlas.shape)
    #print("\nthis_p_dlas")
    #print(this_p_dlas)
    #print(this_p_dlas.shape)
    #print("\nused_z_dla")
    #print(used_z_dla)
    #print(used_z_dla.shape)
    #print("\nthis_sample_log_priors_no_dla")
    #print(this_sample_log_priors_no_dla)
    #print(this_sample_log_priors_no_dla.shape)
    #print("\nthis_sample_log_priors_dla")
    #print(this_sample_log_priors_dla)
    #print(this_sample_log_priors_dla.shape)
    #print("\nthis_sample_log_likelihoods_no_dla")
    #print(this_sample_log_likelihoods_no_dla)
    #print(this_sample_log_likelihoods_no_dla.shape)
    #print("\nthis_sample_log_likelihoods_dla")
    #print(this_sample_log_likelihoods_dla)
    #print(this_sample_log_likelihoods_no_dla.shape)
    DLA_cut = 20.3
    sub20pt3_ind = (log_nhi_samples < DLA_cut)
    #print("\nsub20pt3_ind")
    #print(sub20pt3_ind)
    #print(len(sub20pt3_ind), np.count_nonzero(sub20pt3_ind))

    # indicing sample_log_posteriors instead of assignments to avoid create a new array
    sample_log_posteriors_no_dla[quasar_ind] = this_sample_log_priors_no_dla + this_sample_log_likelihoods_no_dla
    sample_log_posteriors_dla[quasar_ind] = this_sample_log_priors_dla + this_sample_log_likelihoods_dla
    #print("\nsample_log_posteriors_no_dla")
    #print(sample_log_posteriors_no_dla)
    #print(sample_log_posteriors_no_dla.shape)
    #print("\nsample_log_posteriors_dla")
    #print(sample_log_posteriors_dla)
    #print(sample_log_posteriors_dla.shape)
    #print("\nshape", sample_log_posteriors_dla_sub.shape, this_sample_log_priors_dla[sub20pt3_ind].shape, this_sample_log_likelihoods_dla[sub20pt3_ind].shape)
    sample_log_posteriors_dla_sub.append(this_sample_log_priors_dla[sub20pt3_ind] + this_sample_log_likelihoods_dla[sub20pt3_ind])
    sub20pt3_ind = np.logical_not(sub20pt3_ind)
    sample_log_posteriors_dla_sup.append(this_sample_log_priors_dla[sub20pt3_ind] + this_sample_log_likelihoods_dla[sub20pt3_ind])
    #print("\nsample_log_posteriors_dla_sub")
    #print(sample_log_posteriors_dla_sub)
    #print(len(sample_log_posteriors_dla_sub))
    #print("\nsample_log_posteriors_dla_sup")
    #print(sample_log_posteriors_dla_sup)
    #print(len(sample_log_posteriors_dla_sup))

    # use nanmax to avoid NaN potentially in the samples
    # not sure whether the z code has many NaNs in array; the multi-dla codes have many NaNs
    max_log_likelihood_no_dla = np.nanmax(sample_log_posteriors_no_dla[quasar_ind])
    max_log_likelihood_dla = np.nanmax(sample_log_posteriors_dla[quasar_ind])
    max_log_likelihood_dla_sub = np.nanmax(sample_log_posteriors_dla_sub[qi])
    max_log_likelihood_dla_sup = np.nanmax(sample_log_posteriors_dla_sup[qi])
    #print("\nmax_log_likelihood_no_dla")
    #print(max_log_likelihood_no_dla)
    #print("\nmax_log_likelihood_dla")
    #print(max_log_likelihood_dla)
    #print("\nmax_log_likelihood_dla_sub")
    #print(max_log_likelihood_dla_sub)
    #print("\nmax_log_likelihood_dla_sup")
    #print(max_log_likelihood_dla_sup)

    probabilities_no_dla = np.exp(sample_log_posteriors_no_dla[quasar_ind] - max_log_likelihood_no_dla)
    probabilities_dla    = np.exp(sample_log_posteriors_dla[quasar_ind] - max_log_likelihood_dla)
    probabilities_dla_sub = np.exp(sample_log_posteriors_dla_sub[qi] - max_log_likelihood_dla_sub)
    probabilities_dla_sup = np.exp(sample_log_posteriors_dla_sup[qi] - max_log_likelihood_dla_sup)
    #print("\nprobabilities_no_dla")
    #print(probabilities_no_dla)
    #print("\nprobabilities_dla")
    #print(probabilities_dla)
    #print("\nprobabilites_dla_sub")
    #print(probabilities_dla_sub)
    #print("\nprobabilites_dla_sup")
    #print(probabilities_dla_sup)

    I = np.nanargmax(probabilities_no_dla + probabilities_dla)
    #print("\nI")
    #print(I)

    z_map[quasar_ind] = offset_samples_qso[I]                                  #MAP estimate
    #print("\nz_map")
    #print(z_map)
    #print(z_map.shape)

    I = np.nanargmax(probabilities_dla)
    z_dla_map[quasar_ind] = used_z_dla[I]
    n_hi_map[quasar_ind] = nhi_samples[I]
    log_nhi_map[quasar_ind] = log_nhi_samples[I]
    #print("\nI")
    #print(I)
    #print("\nz_dla_map")
    #print(z_dla_map)
    #print(z_dla_map.shape)
    #print("\nn_hi_map")
    #print(n_hi_map)
    #print(n_hi_map.shape)
    #print("\nlog_nhi_map")
    #print(log_nhi_map)
    #print(log_nhi_map.shape)

    log_posteriors_no_dla[quasar_ind] = np.log(np.mean(probabilities_no_dla)) + max_log_likelihood_no_dla   #Expected
    log_posteriors_dla[quasar_ind] = np.log(np.mean(probabilities_dla)) + max_log_likelihood_dla            #Expected
    log_posteriors_dla_sub[quasar_ind] = np.log(np.mean(probabilities_dla_sub)) + max_log_likelihood_dla_sub #Expected
    log_posteriors_dla_sup[quasar_ind] = np.log(np.mean(probabilities_dla_sup)) + max_log_likelihood_dla_sup #Expected
    #print("\nlog_posteriors_no_dla")
    #print(log_posteriors_no_dla)
    #print(log_posteriors_no_dla.shape)
    #print("\nlog_posteriors_dla")
    #print(log_posteriors_dla)
    #print(log_posteriors_dla.shape)
    #print("\nlog_posteriors_dla_sub")
    #print(log_posteriors_dla_sub)
    #print(log_posteriors_dla_sub.shape)
    #print("\nlog_posteriors_dla_sup")
    #print(log_posteriors_dla_sup)
    #print(log_posteriors_dla_sup.shape)

    elapsed = time.time() - t
    print(' took {tm:0.3f}s.\n'.format(tm=elapsed))
    
    # z-estimation checking printing at runtime
    zdiff = z_map[quasar_ind] - z_qsos[quasar_num]
    #print("\nzdiff")
    #print(zdiff)
    #print(zdiff.shape)
    if quasar_ind % 1 == 0:
        elapsed = time.time() - t
        print('Done QSO {first} of {tot} in {tm:0.3f} s. True z_QSO = {tru:0.4f}, I={isi} map={mp:0.4f} dif = {di:.04f}\n'.format(first=quasar_ind+1, tot=num_quasars, tm=elapsed, tru=z_qsos[quasar_num], isi=I, mp=z_map[quasar_ind], di=zdiff))  

    #print(' ... log p(DLA | D, z_QSO)        : {fn:0.2f}\n'.format(fn=log_posteriors_dla[quasar_ind]))

    elapsed = time.time() - t
    print(' took {tm:0.3f}s. (z_map = {mp:0.4f})\n'.format(tm=elapsed, mp=z_map[quasar_ind]))
    if quasar_ind % 5 == 0:
        var = {'log_posteriors_dla_sub':log_posteriors_dla_sub, 'log_posteriors_dla_sup':log_posteriors_dla_sup, 
               'log_posteriors_dla':log_posteriors_dla, 'log_posteriors_no_dla':log_posteriors_no_dla, 'z_true':z_true,
               'dla_true':dla_true, 'quasar_ind':quasar_ind, 'quasar_num':quasar_num, 'used_z_dla':used_z_dla,
               'nhi_samples':nhi_samples, 'offset_samples_qso':offset_samples_qso, 'offset_samples':offset_samples,
               'z_map':z_map, 'signal_to_noise':signal_to_noise, 'z_dla_map':z_dla_map, 'n_hi_map':n_hi_map}
        check = "checkpoints/checkpoint-{inte}-{nu}".format(inte=quasar_ind, nu=num_quasars)
        fname = os.path.join(parent_dir, check)
        file_hand = open(fname, 'wb')
        pickle.dump(var, file_hand)
        file_hand.close()
        #dill.dump_session('checkpoint.pkl')
    #    save(['./checkpointing/curDLA_', optTag, '.mat'], 'log_posteriors_dla_sub', 'log_posteriors_dla_sup', 
    #'log_posteriors_dla', 'log_posteriors_no_dla', 'z_true', 'dla_true', 'quasar_ind', 'quasar_num',...
#'used_z_dla', 'nhi_samples', 'offset_samples_qso', 'offset_samples', 'z_map', 'signal_to_noise', 'z_dla_map', 'n_hi_map')

    # record posdef error;
    # count number of posdef errors; if it is == num_dla_samples, then we have troubles.
    all_posdeferrors[quasar_ind] = sum(this_posdeferror)
    qi = qi + 1
    #print("\nall_posdeferrors")
    #print(all_posdeferrors)
    #print(all_posdeferrors.shape)
    #DEBUGGING
    break
#end

    
    #elapsed = time.time() - t
    #print("elapsed")
    #print(elapsed)
    #break
    
# compute model posteriors in numerically safe manner
combined = np.array(([log_posteriors_no_dla, log_posteriors_dla_sub, log_posteriors_dla_sup])).T
print("\ncombined")
print(combined)
print(combined.shape)
max_log_posteriors = np.nanmax(combined, axis=1)
print("\nmax_log_posteriors")
print(max_log_posteriors)
print(max_log_posteriors.shape)

model_posteriors = np.exp(combined - max_log_posteriors[:,None])
print("\nmodel_posteriors")
print(model_posteriors)
print(model_posteriors.shape)

model_posteriors = model_posteriors / np.nansum(model_posteriors, axis=1)[:, None]
print("\nmodel_posteriors after")
print(model_posteriors)
print(model_posteriors.shape)

p_no_dlas = model_posteriors[:, 0]
p_dlas    = 1 - p_no_dlas
print("\np_no_dlas")
print(p_no_dlas)
print("\np_dlas")
print(p_dlas)
print("\nz_map")
print(z_map)
print("\nz_true")
print(z_true)

# save results
variables_to_save = {'training_release':training_release, 'training_set_name':training_set_name,
                    'dla_catalog_name':dla_catalog_name, 'release':release, 'test_set_name':test_set_name,
                    'test_ind':test_ind, 'prior_z_qso_increase':modelParams.prior_z_qso_increase, 'max_z_cut':moreParams.max_z_cut,
                    'num_lines':moreParams.num_lines, 'min_z_dlas':min_z_dlas, 'max_z_dlas':max_z_dlas, # you need to save DLA search length to compute CDDF
                    'sample_log_posteriors_no_dla':sample_log_posteriors_no_dla, 'sample_log_posteriors_dla':sample_log_posteriors_dla,
                    'sample_log_posteriors_dla_sub':sample_log_posteriors_dla_sub, 'sample_log_posteriors_dla_sup':sample_log_posteriors_dla_sup,
                    'log_posteriors_no_dla':log_posteriors_no_dla, 'log_posteriors_dla':log_posteriors_dla,
                    'log_posteriors_dla_sub':log_posteriors_dla_sub, 'log_posteriors_dla_sup':log_posteriors_dla_sup,
                    'model_posteriors':model_posteriors, 'p_no_dlas':p_no_dlas, 'p_dlas':p_dlas, 'z_map':z_map,
                    'z_true':z_true, 'dla_true':dla_true, 'z_dla_map':z_dla_map, 'n_hi_map':n_hi_map, 'log_nhi_map':log_nhi_map,
                    'signal_to_noise':signal_to_noise, 'all_thing_ids':all_thing_ids, 'all_posdeferrors':all_posdeferrors,
                    'all_exceptions':all_exceptions, 'qso_ind':qso_ind, 'arr':arr}

    # 'sample_log_priors_no_dla':sample_log_priors_no_dla, 'sample_log_priors_dla':sample_log_priors_dla,
    # 'sample_log_likelihoods_no_dla', 'sample_log_likelihoods_dla', ...
#print("\nvariables_to_save")
#print(variables_to_save)
    
direct = 'dr12q/processed'
filename = '{dirt}/processed_qsos_{tst}-{opt}_{beg}-{en}_norm_{minl}-{maxl}'.format(dirt=direct, tst=test_set_name, opt=100, beg=qso_ind[0], en=qso_ind[0] + len(qso_ind), minl=normParams.normalization_min_lambda, maxl=normParams.normalization_max_lambda)
#filename = 'second_part'

print("\nfilename")
print(filename)
fileName = os.path.join(parent_dir, filename)
#save(filename, variables_to_save{:}, '-v7.3');
# Open a file for writing data
#file_handler = open(fileName, 'wb')

# Dump the data of the object into the file
#pickle.dump(variables_to_save, file_handler)

# close the file handler to release the resources
#file_handler.close()
#pr.print_stats(sort='tottime')