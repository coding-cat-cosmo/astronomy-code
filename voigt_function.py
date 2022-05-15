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
