import pickle
import dill

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
        self.optTag = (str(integrate), str(extrapolate_subdla), str(add_proximity_zone))

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
emitted_wavelengths = lambda observed_wavelengths, z : (observed_wavelengths / (1 + z))

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


dill.dump_session('parameters.pkl')
