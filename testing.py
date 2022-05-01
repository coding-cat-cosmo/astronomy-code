# voigt.c: computes exact Voigt profiles in terms of the complex error function (requires libcerf)
from scipy.special import voigt_profile
import numpy as np
from numba import prange, njit, jit

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

width = 3                      # width of convolution     dimensionless */

instrument_profile = np.array([2.17460992138080811e-03, 4.11623059580451742e-02, 2.40309364651846963e-01,
4.32707438937454059e-01, # center pixel */
2.40309364651846963e-01, 4.11623059580451742e-02, 2.17460992138080811e-03])

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
    voigts = voigt_profile(vels[:,None],sigma,gammas[None,:3])

    myapprox = [interp1d(vels,voigts[:,i]) for i in range(3)]

    def voigtapprox(vel):
        # vel is m-by-n (where n = len(gammas))
        m = vel.shape[0]
        n = 3

        res = np.zeros((m,n))
        for i in range(n):
            res[:,i] = myapprox[i](vel[:,i])
        return res

    return voigtapprox


vapprox = genvoigtapprox()

def voigt(LAMBDAS_ARG, Z_ARG, N_ARG, NUM_LINES_ARG=31):



    #/* get input */
    lambdas = LAMBDAS_ARG                #/* wavelengths             Å             */
    z       = Z_ARG                  #/* redshift                dimensionless */
    N       = N_ARG                  #/* column density          cm⁻²          */

    num_lines = NUM_LINES_ARG

    multipliers = c / (transition_wavelengths[:num_lines] * (1+z)) / 100000000

    #/* compute raw Voigt profile */
    velocity = lambdas[:,None] * multipliers[None,:num_lines] - c
    #subtotal = -leading_constants[None,:num_lines] * voigt_profile(velocity,sigma,gammas[None,:num_lines])
    subtotal = -leading_constants[None,:num_lines] * vapprox(velocity)
    total = subtotal.sum(axis=1)
    raw_profile = np.exp(N*total)

    #/* instrumental broadening */
    profile = np.convolve(raw_profile, instrument_profile, mode='valid')

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

        # keep a copy inside the parfor since we are modifying them
        this_wavelengths    = self.this_out_wavelengths
        this_flux           = self.this_out_flux
        this_noise_variance = self.this_out_noise_variance
        this_pixel_mask     = self.this_out_pixel_mask

        #Cut off observations
        max_pos_lambda = observed_wavelengths(nullParams.max_lambda, z_qso)
        min_pos_lambda = observed_wavelengths(nullParams.min_lambda, z_qso)
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

        #normalizing here
        ind = (this_rest_wavelengths >= normParams.normalization_min_lambda) & (this_rest_wavelengths <= normParams.normalization_max_lambda)

        this_median         = np.nanmedian(this_flux[ind])
        this_flux           = this_flux / this_median
        this_noise_variance = this_noise_variance / pow(this_median, 2)

        ind = (this_rest_wavelengths >= nullParams.min_lambda) & (this_rest_wavelengths <= nullParams.max_lambda)

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

        this_lyseries_zs = (this_wavelengths[None, :] - np.array(learnParams.all_transition_wavelengths[:learnParams.num_forest_lines])[:, None]) / np.array(learnParams.all_transition_wavelengths[:learnParams.num_forest_lines])[:, None]
        this_lyseries_zs = this_lyseries_zs.T

        # DLA existence prior
        less_ind = (self.orig_z_qsos < (z_qso + modelParams.prior_z_qso_increase))

        this_num_dlas    = np.count_nonzero(self.dla_ind[less_ind])
        this_num_quasars = np.count_nonzero(less_ind)
        this_p_dla       = this_num_dlas / float(this_num_quasars)
        self.this_p_dlas[i]   = this_p_dla

        #minimal plausible prior to prevent NaN on low z_qso;
        if this_num_dlas == 0:
            this_num_dlas = 1
            this_num_quasars = len(less_ind)

        self.this_sample_log_priors_dla[i] = np.log(this_num_dlas) - np.log(this_num_quasars)
        self.this_sample_log_priors_no_dla[i] = np.log(this_num_quasars - this_num_dlas) - np.log(this_num_quasars)

        # fprintf_debug('\n');
        #print(' ...     p(   DLA | z_QSO)        : {th:0.3f}\n'.format(th=this_p_dla))
        #print(' ...     p(no DLA | z_QSO)        : {th:0.3f}\n'.format(th=1 - this_p_dla))

        # interpolate model onto given wavelengths
        this_mu = mu_interpolator(this_rest_wavelengths)
        this_M  =  M_interpolator(this_rest_wavelengths)

        this_log_omega = log_omega_interpolator(this_rest_wavelengths)
        this_omega2 = np.exp(2 * this_log_omega)

        # Lyman series absorption effect for the noise variance
        # note: this noise variance must be trained on the same number of members of Lyman series
        lya_optical_depth = self.tau_0 * (1 + this_lya_zs)**self.beta

        # Note: this_wavelengths is within (min_lambda, max_lambda)
        # so it may beyond lya_wavelength, so need an indicator;
        # Note: 1 - exp( -0 ) + c_0 = c_0
        indicator         = this_lya_zs <= z_qso
        lya_optical_depth = lya_optical_depth * indicator
            
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
            
        this_tau_0 = self.prev_tau_0 * learnParams.all_oscillator_strengths[:learnParams.num_forest_lines] / learnParams.lya_oscillator_strength * learnParams.all_transition_wavelengths[:learnParams.num_forest_lines] / physConst.lya_wavelength
        total_optical_depth = this_tau_0 * ((1+this_lyseries_zs[None, :])**self.prev_beta)
        total_optical_depth = total_optical_depth * (this_lyseries_zs <= z_qso)
        total_optical_depth = np.squeeze(total_optical_depth)

        # change from nansum to simply sum; shouldn't be different
        # because we also change indicator from nan to zero,
        # but if this script is glitchy then inspect this line
        lya_absorption = np.exp(- np.sum(total_optical_depth, 1) )

        this_mu = this_mu * lya_absorption
        this_M  = this_M  * lya_absorption[:, None]

        # re-adjust (K + Ω) to the level of μ .* exp( -optical_depth ) = μ .* a_lya
        # now the null model likelihood is:
        # p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
        this_omega2 = this_omega2 * lya_absorption**2

        occams = self.occams_factor * (1 - lambda_observed / (nullParams.max_lambda - nullParams.min_lambda) )

        # baseline: probability of no DLA model
        self.this_sample_log_likelihoods_no_dla[i] = log_mvnpdf_low_rank(this_flux, this_mu, this_M, this_omega2 + this_noise_variance) + bw_log_likelihood + rw_log_likelihood - occams

        #print(' ... log p(D | z_QSO, no DLA)     : {ts:0.2f}\n'.format(ts=this_sample_log_likelihoods_no_dla[i]))

        # Add
        if this_wavelengths.shape[0] == 0:
            print("dunded")
            values = outputs(self.fluxes, self.rest_wavelengths, self.min_z_dlas, self.max_z_dlas, self.this_p_dlas,
                             self.this_sample_log_priors_no_dla, self.this_sample_log_priors_dla, self.used_z_dla,
                             self.this_sample_log_likelihoods_no_dla, self.this_sample_log_likelihoods_dla, i)

            return values

        # use a temp variable to avoid the possible parfor issue
        # should be fine after change size of min_z_dlas to (num_quasar, num_dla_samples)
        this_min_z_dlas = moreParams.min_z_dla(this_wavelengths, z_qso)
        this_max_z_dlas = moreParams.max_z_dla(this_wavelengths, z_qso)

        self.min_z_dlas[self.quasar_ind, i] = this_min_z_dlas
        self.max_z_dlas[self.quasar_ind, i] = this_max_z_dlas

        sample_z_dlas = this_min_z_dlas + (this_max_z_dlas - this_min_z_dlas) * self.offset_samples

        self.used_z_dla[i] = sample_z_dlas[i]

        # ensure enough pixels are on either side for convolving with instrument profile
        padded_wavelengths = [np.logspace(np.log10(np.min(this_unmasked_wavelengths)) - instrumentParams.width * instrumentParams.pixel_spacing,
                              np.log10(np.min(this_unmasked_wavelengths)) - instrumentParams.pixel_spacing, instrumentParams.width).T,
                              this_unmasked_wavelengths,
                              np.logspace(np.log10(np.max(this_unmasked_wavelengths)) + instrumentParams.pixel_spacing,
                              np.log10(np.max(this_unmasked_wavelengths)) + instrumentParams.width * instrumentParams.pixel_spacing, instrumentParams.width).T]

        padded_wavelengths = np.array([x for ls in padded_wavelengths for x in ls])
        # to retain only unmasked pixels from computed absorption profile
        ind = np.logical_not(this_pixel_mask[ind])

        # compute probabilities under DLA model for each of the sampled
        # (normalized offset, log(N HI)) pairs absorption corresponding to this sample
        absorption = voigt(padded_wavelengths, sample_z_dlas[i], self.nhi_samples[i], moreParams.num_lines)
        # add this line back for implementing pixel masking
        temp = np.zeros(len(absorption)-len(ind))
        ind = np.append(ind, temp)
        ind = np.array(ind, dtype=bool)
        absorption = absorption[ind]

        # delta z = v / c = H(z) d / c = 70 (km/s/Mpc) * sqrt(0.3 * (1+z)^3 + 0.7) * (5 Mpc) / (3x10^5 km/s) ~ 0.005 at z=3
        if flag.add_proximity_zone:
            print("\nreached this")
            delta_z = (70 * np.sqrt(.3 * (1+z_qso)**3 + .7) * 5) / (3 * 10**5)


        dla_mu = this_mu * absorption
        dla_M = this_M * absorption[:, None]
        dla_omega2 = this_omega2 * absorption**2

        # Add a penalty for short spectra: the expected reduced chi^2 of each spectral pixel that would have been observed.
        # Also add an error handler for DLA model likelihood function
        self.this_sample_log_likelihoods_dla[i] = log_mvnpdf_low_rank(this_flux, dla_mu, dla_M, dla_omega2 + this_noise_variance) + bw_log_likelihood + rw_log_likelihood - occams

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
                    
    return log_p

# log_mvnpdf_low_rank: efficiently computes
#
#   log N(y; mu, MM' + diag(d))

@njit
def log_mvnpdf_low_rank(y, mu, M, d):

    log_2pi = 1.83787706640934534

    [n, k] = M.shape
 
    y = y - (mu)
    d_inv = 1 / d
    D_inv_y = d_inv * y
    d_inv = np.reshape(d_inv, (d_inv.shape[0], 1))
    D_inv_M = d_inv * M
    #print("D_inv_M", D_inv_M, D_inv_M.shape)

    # use Woodbury identity, define
    #   B = (I + M' D^-1 M),
    # then
    #   K^-1 = D^-1 - D^-1 M B^-1 M' D^-1

    B = np.dot(M.T, D_inv_M)
    B = np.reshape(B, B.shape[0]*B.shape[1])
    B[0::k+1] = B[0::k+1] + 1
    B = np.reshape(B, (k, k))
    #L = np.linalg.cholesky(B)
    #L= L.T
    # C = B^-1 M' D^-1
    C = np.linalg.solve(B, D_inv_M.T)
    #C = np.linalg.solve(L.T, np.linalg.solve(L, D_inv_M.T))

    K_inv_y = D_inv_y - np.dot(D_inv_M, np.dot(C, y))

    #log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))
    log_det_K = np.sum(np.log(d)) + np.linalg.slogdet(B)[1]

    log_p = -0.5 * (np.dot(y, K_inv_y) + log_det_K + n * log_2pi)

    return log_p

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
from scipy.interpolate import interp1d
from multiprocessing import Pool
from scipy.special import voigt_profile
import h5py
with open('parameters.pkl', 'rb') as handle:
    params = dill.load(handle)

# specify the learned quasar model to use
training_release  = 'dr12q'
training_set_name = 'dr9q_minus_concordance'

# specify the spectra to use for computing the DLA existence prior
dla_catalog_name  = 'dr9q_concordance'
    
# specify the spectra to process
release = 'dr12q'
test_set_name = 'dr12q'

modelParams = params['modelParams']
instrumentParams = params['instrumentParams']
moreParams = params['moreParams']
nullParams = params['nullParams']
dlaParams = params['dlaParams']
normParams = params['normParams']
learnParams = params['learnParams']
flag = params['flag']
emitted_wavelengths = params['emitted_wavelengths']
observed_wavelengths = params['observed_wavelengths']
physConst = params['physParams']
kms_to_z = params['kms_to_z']

prev_tau_0 = 0.0023
prev_beta  = 3.65

occams_factor = 0 # turn this to zero if you don't want the incomplete data penalty

# load redshifts/DLA flags from training release
p = Path(os.getcwd())
parent_dir = str(p.parent)
filename = "dr12q/processed/catalog"
#getting back pickled data for catalog
with open(filename,'rb') as f:
    prior_catalog = pickle.load(f)


prior_ind = prior_catalog['in_dr9'] & prior_catalog['los_inds'][dla_catalog_name] & (prior_catalog['new_filter_flags'] == 0)

orig_z_qsos  = prior_catalog['z_qsos'][prior_ind]
dla_ind = prior_catalog['dla_inds'][dla_catalog_name]
dla_ind = dla_ind[prior_ind]
print("\nz_qsos")
print(orig_z_qsos)
print(orig_z_qsos.shape)

# filter out DLAs from prior catalog corresponding to region of spectrum below Ly∞ QSO rest
z_dlas = prior_catalog['z_dlas'][dla_catalog_name]
for i, z in enumerate(z_dlas):
    if (z!=0):
        z_dlas[i] = z[0]

z_dlas = np.array(z_dlas)
z_dlas = z_dlas[prior_ind]

thing = np.where(dla_ind > 0)
thing = thing[0]
for i in thing:
    if (observed_wavelengths(physConst.lya_wavelength, z_dlas[i]) < observed_wavelengths(physConst.lyman_limit, orig_z_qsos[i])):
            dla_ind[i] = False

# load QSO model from training release
rel = "dr12q/processed/"
directory = os.path.join(parent_dir, rel)
training_set_name = 'dr9q_minus_concordance'

place = '{}/learned_model_outdata_{}_norm_{}-{}'.format(rel, training_set_name, normParams.normalization_min_lambda, normParams.normalization_max_lambda)
#getting back pickled data for model
with open(place,'rb') as f:
    model = pickle.load(f)

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

###ALTERNATIVE version with learned model from matlab
#place = '{}/learned_model_outdata_dr9q_minus_concordance_norm_1176-1256.mat'.format(rel)
#place = "dr12q/processed/learned_zqso_only_model_outdata_normout_dr9q_minus_concordance_norm_1176-1256.mat"
#f = h5py.File(place,'r')
#print(f.keys())
#rest_wavelengths = f.get("rest_wavelengths")
#rest_wavelengths = np.array(rest_wavelengths)
#rest_wavelengths = np.squeeze(rest_wavelengths)
#print("rest_wavelengths and size", rest_wavelengths, rest_wavelengths.shape)
#mu = f.get("mu")
#mu = np.array(mu)
#mu = np.squeeze(mu)
#print("mu and size", mu, mu.shape)
#M = f.get("M")
#M = np.array(M)
#M = M.T
#print("M and size", M, M.shape)
#log_omega = f.get("log_omega")
#log_omega = np.array(log_omega)
#log_omega = np.squeeze(log_omega)
#print("log_omega and size", log_omega, log_omega.shape)
#log_c_0 = f.get("log_c_0")
#log_c_0 = np.array(log_c_0)
#log_c_0 = np.squeeze(log_c_0)
#print("log_c_0 and size", log_c_0, log_c_0.shape)
#log_tau_0 = f.get("log_tau_0")
#log_tau_0 = np.array(log_tau_0)
#log_tau_0 = np.squeeze(log_tau_0)
#print("log_tau_0 and size", log_tau_0, log_tau_0.shape)
#log_beta = f.get("log_beta")
#log_beta = np.array(log_beta)
#log_beta = np.squeeze(log_beta)
#print("log_beta and size", log_beta, log_beta.shape)
#bluewards_mu = f.get("bluewards_mu")
#bluewards_mu = np.array(bluewards_mu)
#bluewards_mu = np.squeeze(bluewards_mu)
#print("bluewards_mu and size", bluewards_mu, bluewards_mu.shape)
#bluewards_sigma = f.get("bluewards_sigma")
#bluewards_sigma = np.array(bluewards_sigma)
#bluewards_sigma = np.squeeze(bluewards_sigma)
#print("bluewards_sigma and size", bluewards_sigma, bluewards_sigma.shape)
#redwards_mu = f.get("redwards_mu")
#redwards_mu = np.array(redwards_mu)
#redwards_mu = np.squeeze(redwards_mu)
#print("rewards_mu and size", redwards_mu, redwards_mu.shape)
#redwards_sigma = f.get("redwards_sigma")
#redwards_sigma = np.array(redwards_sigma)
#redwards_sigma = np.squeeze(redwards_sigma)
#print("redwards_sigma and size", redwards_sigma, redwards_sigma.shape)
#f.close()

# load DLA samples from training release

rel = "dr12q/processed/"
direc = os.path.join(parent_dir, rel)
lease = "dla_samples"
filename = os.path.join(rel, lease)
with open(filename, 'rb') as f:
    samples = pickle.load(f)
    
offset_samples = samples['offset_samples']
offset_samples_qso = samples['offset_samples_qso']
log_nhi_samples = samples['log_nhi_samples']
nhi_samples = samples['nhi_samples']

#alternative version that takes Halton Sequence from matlab version
#fileName = 'dr12q/processed/dla_samples.mat'
#f = h5py.File(fileName, 'r')
    
#offset_samples = f.get('offset_samples')
#offset_samples = np.array(offset_samples)
#offset_samples = np.squeeze(offset_samples)
#print("offset_samples", offset_samples, offset_samples.shape)
#offset_samples_qso = f.get('offset_samples_qso')
#offset_samples_qso = np.array(offset_samples_qso)
#offset_samples_qso = np.squeeze(offset_samples_qso)
#print("offset_samples_qso", offset_samples_qso, offset_samples_qso.shape)
#log_nhi_samples = f.get('log_nhi_samples')
#log_nhi_samples = np.array(log_nhi_samples)
#log_nhi_samples = np.squeeze(log_nhi_samples)
#print("log_nhi_samples", log_nhi_samples, log_nhi_samples.shape)
#nhi_samples = f.get('nhi_samples')
#nhi_samples = np.array(nhi_samples)
#nhi_samples = np.squeeze(nhi_samples)
#print("nhi_samples", nhi_samples, nhi_samples.shape)
#f.close()

# load preprocessed QSOs
# load redshifts from catalog to process
rel = "dr12q/processed/catalog"
filename = os.path.join(parent_dir, rel)
with open(rel, 'rb') as f:
    catalog = pickle.load(f)

#getting back preprocessed qso data
rel = "dr12q/processed/preloaded_qsos"
filename = os.path.join(parent_dir, rel)
with open(rel, 'rb') as f:
    preqsos = pickle.load(f)
    
all_wavelengths = preqsos['all_wavelengths']
all_flux = preqsos['all_flux']
all_noise_variance = preqsos['all_noise_variance']
all_pixel_mask = preqsos['all_pixel_mask']

# enable processing specific QSOs via setting to_test_ind
test_ind = (catalog['new_filter_flags'] == 0)

all_wavelengths    =    all_wavelengths[test_ind]
all_flux           =           all_flux[test_ind]
all_noise_variance = all_noise_variance[test_ind]
all_pixel_mask     =     all_pixel_mask[test_ind]
#more fixing done here since catalog has the full amount but test_ind is only 5000
all_thing_ids = catalog['thing_ids'][test_ind]

z_qsos = catalog['z_qsos'][test_ind]
dla_inds = catalog['dla_inds']['dr12q_visual']
dla_inds = dla_inds[test_ind]

num_quasars = len(z_qsos)
try:
    qso_ind and var
except NameError:
    qso_ind = [x for x in range(int(np.floor(num_quasars/100)))]

num_quasars = len(qso_ind)

# preprocess model interpolants
mu_interpolator = interp1d(rest_wavelengths, mu, assume_sorted=True)

M_interpolator = interp1d(rest_wavelengths , M, axis=0, assume_sorted=True)

log_omega_interpolator = interp1d(rest_wavelengths, log_omega, assume_sorted=True)

# initialize results
# prevent parfor error, should use nan(num_quasars, num_dla_samples); or not save these variables;
min_z_dlas = np.empty((num_quasars, dlaParams.num_dla_samples))
min_z_dlas[:] = np.NaN
max_z_dlas = np.empty((num_quasars, dlaParams.num_dla_samples))
max_z_dlas[:] = np.NaN

#sample_log_posteriors_dla_sub = []
#sample_log_posteriors_dla_sup = []
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

#need to be external to test but also adds 4 GB per quasar
#fluxes = []
#rest_wavelengths = []
this_p_dlas = np.zeros((len(z_list)))

# this is just an array allow you to select a range of quasars to run
quasar_ind = 0 #instead of 0

q_ind_start = quasar_ind

# catch the exceptions
all_exceptions = np.zeros((num_quasars))

all_exceptions = [False for x in all_exceptions]
all_exceptions = np.array(all_exceptions)
    
all_posdeferrors = np.zeros((num_quasars))

qi = 0
arr = []

#num_quasars = 1500
for quasar_ind in range(q_ind_start, num_quasars): #quasar list
    arr.append(quasar_ind)
    t = time.time()
    quasar_num = qso_ind[quasar_ind]
    fluxes = []
    rest_wavelengths = []
    
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
    
    args = inputs(offset_samples_qso, this_out_wavelengths, this_out_flux,
                      this_out_noise_variance, this_out_pixel_mask,
                      bluewards_mu, bluewards_sigma, redwards_mu, redwards_sigma,
                      fluxes, rest_wavelengths, orig_z_qsos, dla_ind, this_p_dlas,
                      c_0, tau_0, beta, prev_tau_0, prev_beta, occams_factor,
                      quasar_ind, offset_samples, nhi_samples, min_z_dlas,
                      max_z_dlas, this_sample_log_priors_no_dla, this_sample_log_priors_dla,
                      this_sample_log_likelihoods_no_dla, this_sample_log_likelihoods_dla, used_z_dla)

    #with Pool(5) as p: # with index
    #    values = p.map(args, range(dlaParams.num_dla_samples))
    for i in range(dlaParams.num_dla_samples):  #variant redshift in quasars 
        values = args(i)
    
    #for val in values:
        #fluxes.append(val.fluxes)
        #rest_wavelengths.append(val.rest_wavelengths)
        #min_z_dlas[quasar_ind, val.i] = val.min_z_dlas[quasar_ind, val.i]
        #max_z_dlas[quasar_ind, val.i] = val.max_z_dlas[quasar_ind, val.i]
        #this_p_dlas[val.i] = val.this_p_dlas[val.i]
        #used_z_dla[val.i] = val.used_z_dla[val.i]
        #this_sample_log_priors_no_dla[val.i] = val.this_sample_log_priors_no_dla[val.i]
        #this_sample_log_priors_dla[val.i] = val.this_sample_log_priors_dla[val.i]
        #this_sample_log_likelihoods_no_dla[val.i] = val.this_sample_log_likelihoods_no_dla[val.i]
        #this_sample_log_likelihoods_dla[val.i] = val.this_sample_log_likelihoods_dla[val.i]
        
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
    DLA_cut = 20.3
    sub20pt3_ind = (log_nhi_samples < DLA_cut)

    # indicing sample_log_posteriors instead of assignments to avoid create a new array
    #sample_log_posteriors_no_dla[quasar_ind] = this_sample_log_priors_no_dla + this_sample_log_likelihoods_no_dla
    #sample_log_posteriors_dla[quasar_ind] = this_sample_log_priors_dla + this_sample_log_likelihoods_dla
    sample_log_posteriors_no_dla = this_sample_log_priors_no_dla + this_sample_log_likelihoods_no_dla
    sample_log_posteriors_dla = this_sample_log_priors_dla + this_sample_log_likelihoods_dla
    #sample_log_posteriors_dla_sub.append(this_sample_log_priors_dla[sub20pt3_ind] + this_sample_log_likelihoods_dla[sub20pt3_ind])
    #sub20pt3_ind = np.logical_not(sub20pt3_ind)
    #sample_log_posteriors_dla_sup.append(this_sample_log_priors_dla[sub20pt3_ind] + this_sample_log_likelihoods_dla[sub20pt3_ind])
    sample_log_posteriors_dla_sub = this_sample_log_priors_dla[sub20pt3_ind] + this_sample_log_likelihoods_dla[sub20pt3_ind]
    sub20pt3_ind = np.logical_not(sub20pt3_ind)
    sample_log_posteriors_dla_sup = this_sample_log_priors_dla[sub20pt3_ind] + this_sample_log_likelihoods_dla[sub20pt3_ind]

    # use nanmax to avoid NaN potentially in the samples
    # not sure whether the z code has many NaNs in array; the multi-dla codes have many NaNs
    #max_log_likelihood_no_dla = np.nanmax(sample_log_posteriors_no_dla[quasar_ind])
    #max_log_likelihood_dla = np.nanmax(sample_log_posteriors_dla[quasar_ind])
    #max_log_likelihood_dla_sub = np.nanmax(sample_log_posteriors_dla_sub[qi])
    #max_log_likelihood_dla_sup = np.nanmax(sample_log_posteriors_dla_sup[qi])
    max_log_likelihood_no_dla = np.nanmax(sample_log_posteriors_no_dla)
    max_log_likelihood_dla = np.nanmax(sample_log_posteriors_dla)
    max_log_likelihood_dla_sub = np.nanmax(sample_log_posteriors_dla_sub)
    max_log_likelihood_dla_sup = np.nanmax(sample_log_posteriors_dla_sup)

    #probabilities_no_dla = np.exp(sample_log_posteriors_no_dla[quasar_ind] - max_log_likelihood_no_dla)
    #probabilities_dla    = np.exp(sample_log_posteriors_dla[quasar_ind] - max_log_likelihood_dla)
    #probabilities_dla_sub = np.exp(sample_log_posteriors_dla_sub[qi] - max_log_likelihood_dla_sub)
    #probabilities_dla_sup = np.exp(sample_log_posteriors_dla_sup[qi] - max_log_likelihood_dla_sup)
    probabilities_no_dla = np.exp(sample_log_posteriors_no_dla - max_log_likelihood_no_dla)
    probabilities_dla    = np.exp(sample_log_posteriors_dla - max_log_likelihood_dla)
    probabilities_dla_sub = np.exp(sample_log_posteriors_dla_sub - max_log_likelihood_dla_sub)
    probabilities_dla_sup = np.exp(sample_log_posteriors_dla_sup - max_log_likelihood_dla_sup)

    I = np.nanargmax(probabilities_no_dla + probabilities_dla)

    z_map[quasar_ind] = offset_samples_qso[I]                                  #MAP estimate

    I = np.nanargmax(probabilities_dla)
    z_dla_map[quasar_ind] = used_z_dla[I]
    n_hi_map[quasar_ind] = nhi_samples[I]
    log_nhi_map[quasar_ind] = log_nhi_samples[I]

    log_posteriors_no_dla[quasar_ind] = np.log(np.mean(probabilities_no_dla)) + max_log_likelihood_no_dla   #Expected
    log_posteriors_dla[quasar_ind] = np.log(np.mean(probabilities_dla)) + max_log_likelihood_dla            #Expected
    log_posteriors_dla_sub[quasar_ind] = np.log(np.mean(probabilities_dla_sub)) + max_log_likelihood_dla_sub #Expected
    log_posteriors_dla_sup[quasar_ind] = np.log(np.mean(probabilities_dla_sup)) + max_log_likelihood_dla_sup #Expected

    elapsed = time.time() - t
    print(' took {tm:0.3f}s.\n'.format(tm=elapsed))
    
    # z-estimation checking printing at runtime
    zdiff = z_map[quasar_ind] - z_qsos[quasar_num]

    if quasar_ind % 1 == 0:
        elapsed = time.time() - t
        print('Done QSO {first} of {tot} in {tm:0.3f} s. True z_QSO = {tru:0.4f}, I={isi} map={mp:0.4f} dif = {di:.04f}\n'.format(first=quasar_ind+1, tot=num_quasars, tm=elapsed, tru=z_qsos[quasar_num], isi=I, mp=z_map[quasar_ind], di=zdiff))  

    #print(' ... log p(DLA | D, z_QSO)        : {fn:0.2f}\n'.format(fn=log_posteriors_dla[quasar_ind]))

    elapsed = time.time() - t
    print(' took {tm:0.3f}s. (z_map = {mp:0.4f})\n'.format(tm=elapsed, mp=z_map[quasar_ind]))
    if quasar_ind % 10 == 0:
        var = {'log_posteriors_dla_sub':log_posteriors_dla_sub, 'log_posteriors_dla_sup':log_posteriors_dla_sup, 
               'log_posteriors_dla':log_posteriors_dla, 'log_posteriors_no_dla':log_posteriors_no_dla, 'z_true':z_true,
               'dla_true':dla_true, 'quasar_ind':quasar_ind, 'quasar_num':quasar_num, 'used_z_dla':used_z_dla,
               'nhi_samples':nhi_samples, 'offset_samples_qso':offset_samples_qso, 'offset_samples':offset_samples,
               'z_map':z_map, 'signal_to_noise':signal_to_noise, 'z_dla_map':z_dla_map, 'n_hi_map':n_hi_map,
	       'qso_ind':qso_ind, 'arr':arr}
        check = "checkpoints/checkpoint-{inte}-{nu}".format(inte=quasar_ind, nu=num_quasars)
        file_hand = open(check, 'wb')
        pickle.dump(var, file_hand)
        file_hand.close()

    # record posdef error;
    # count number of posdef errors; if it is == num_dla_samples, then we have troubles.
    all_posdeferrors[quasar_ind] = sum(this_posdeferror)
    qi = qi + 1

# compute model posteriors in numerically safe manner
combined = np.array(([log_posteriors_no_dla, log_posteriors_dla_sub, log_posteriors_dla_sup])).T
max_log_posteriors = np.nanmax(combined, axis=1)
model_posteriors = np.exp(combined - max_log_posteriors[:,None])
model_posteriors = model_posteriors / np.nansum(model_posteriors, axis=1)[:, None]

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
                    'log_posteriors_no_dla':log_posteriors_no_dla, 'log_posteriors_dla':log_posteriors_dla,
                    'log_posteriors_dla_sub':log_posteriors_dla_sub, 'log_posteriors_dla_sup':log_posteriors_dla_sup,
                    'model_posteriors':model_posteriors, 'p_no_dlas':p_no_dlas, 'p_dlas':p_dlas, 'z_map':z_map,
                    'z_true':z_true, 'dla_true':dla_true, 'z_dla_map':z_dla_map, 'n_hi_map':n_hi_map, 'log_nhi_map':log_nhi_map,
                    'signal_to_noise':signal_to_noise, 'all_thing_ids':all_thing_ids, 'all_posdeferrors':all_posdeferrors,
                    'all_exceptions':all_exceptions, 'qso_ind':qso_ind, 'arr':arr}
    
direct = 'dr12q/processed'
filename = '{dirt}/processed_qsos_{tst}-{opt}_{beg}-{en}_norm_{minl}-{maxl}'.format(dirt=direct, tst=test_set_name, opt=100, beg=qso_ind[0], en=qso_ind[0] + len(qso_ind), minl=normParams.normalization_min_lambda, maxl=normParams.normalization_max_lambda)
filename = 'sixteenth_part'

print("\nfilename")
print(filename)

# Open a file for writing data
file_handler = open(filename, 'wb')

# Dump the data of the object into the file
pickle.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()
