from pathlib import Path
import os
import ssl
import wget
import tarfile
#import sys
#import urllib
#import subprocess

p = Path(os.getcwd())
#gets parent path
base_directory = p.parent
# DR9Q
directory="dr9q/distfiles"
 
# Parent Directory path
parent_dir = str(p.parent)
 
# Path
add = os.path.join(parent_dir, directory)
 
# Create the directory
# 'distfiles' 
try:
    os.makedirs(add)
    print("Directory '%s' created successfully" %directory)
except OSError as error:
    print("Directory '%s' can not be created")
    
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
url = "http://data.sdss3.org/sas/dr9/env/BOSS_QSO/DR9Q/DR9Q.fits"
wget.download(url, add)
#filename = wget.download(url)
#f_name = 'DR9Q.fits'
#subprocess.Popen(['wget', '-O', add, filename])
#urllib.urlretrieve(url, add)
 
# By setting exist_ok as True
# error caused due already
# existing directory can be suppressed
# but other OSError may be raised
# due to other error like
# invalid path name
#DR10Q
directory='dr10q/distfiles'
add = os.path.join(parent_dir, directory)

try:
    os.makedirs(add)
    print("\nDirectory '%s' created successfully\n" %directory)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")
    
url = "http://data.sdss3.org/sas/dr10/boss/qso/DR10Q/DR10Q_v2.fits"
wget.download(url, add)

#DR12Q
directory='dr12q/distfiles'
add = os.path.join(parent_dir, directory)

try:
    os.makedirs(add)
    print("\nDirectory '%s' created successfully\n" %directory)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")
    
url = "http://data.sdss3.org/sas/dr12/boss/qso/DR12Q/DR12Q.fits"
wget.download(url, add)

directory = "dla_catalogs"
catalog = os.path.join(parent_dir, directory)
# concordance catalog from BOSS DR9 Lyman-alpha forest catalog
name = "dr9q_concordance"
cat = os.path.join(catalog, name)
direct = "dr9q_concordance/distfiles"
add = os.path.join(catalog, direct)

try:
    os.makedirs(add)
    print("\nDirectory '%s' created successfully\n" %direct)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")
    
url = "http://data.sdss3.org/sas/dr9/boss/lya/cat/BOSSLyaDR9_cat.txt"
wget.download(url, add)

direct = "processed"
add = os.path.join(cat, direct)

try:
    os.makedirs(add)
    print("\nDirectory '%s' created successfully\n" %direct)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")
    
#COMMAND = "tail -n+2 ./*/*.tsv|cat|awk 'BEGIN{FS=\"\t\"};{split($10,arr,\"-\")}{print arr[1]}'|sort|uniq -c"  

#subprocess.call(COMMAND, shell=True)
#gawking begins here
direct = "distfiles/BOSSLyaDR9_cat.txt"
base = os.path.join(cat, direct)
other = "processed/dla_catalog"
exp = os.path.join(cat, other)
outf = open(exp, 'w')
with open(base) as fpin:
    for nr, line in enumerate(fpin):
        parts = line.rstrip('\n').split()
        if (nr > 1 and not parts[14].startswith('-')):
            outf.write(parts[3])
            outf.write(' ')
            outf.write(parts[14])
            outf.write(' ')
            outf.write(parts[15])
            outf.write("\n")
            
outf.close()

other = "dr9q_concordance/processed/los_catalog"
exp = os.path.join(catalog, other)
outf = open(exp, 'w')
with open(base) as fpin:
    for nr, line in enumerate(fpin):
        parts = line.rstrip('\n').split()
        if (nr > 0):
            outf.write(parts[3])
            outf.write("\n")
            
outf.close()
 # DR12Q DLA catalog from Noterdaeme, et al.
name = "dr12q_noterdaeme"
cat = os.path.join(catalog, name)
direct = "dr12q_noterdaeme/distfiles"
add = os.path.join(catalog, direct)

try:
    os.makedirs(add)
    print("\nDirectory '%s' created successfully\n" %direct)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")
    
url = "http://www2.iap.fr/users/noterdae/DLA/DLA_DR12_v2.tgz"
wget.download(url, add)
nam = "DLA_DR12_v2.tgz"
comp = os.path.join(add, nam)
tar = tarfile.open(comp)
tar.extractall(path = add)
tar.close()

direct = "processed"
add = os.path.join(cat, direct)

try:
    os.makedirs(add)
    print("\nDirectory '%s' created successfully\n" %direct)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")

#more gawking here
direct = "distfiles/DLA_DR12_v2.dat"
base = os.path.join(cat, direct)
other = "processed/dla_catalog"
exp = os.path.join(cat, other)
outf = open(exp, 'w')
with open(base) as fpin:
    for nr, line in enumerate(fpin):
        parts = line.rstrip('\n').split()
        #len(parts) is equal to nf or number of fields
        if (nr > 1 and len(parts) > 0):
            outf.write(parts[0])
            outf.write(' ')
            outf.write(parts[9])
            outf.write(' ')
            outf.write(parts[10])
            outf.write("\n")
            
outf.close()

direct = "distfiles/LOS_DR12_v2.dat"
base = os.path.join(cat, direct)
other = "processed/los_catalog"
exp = os.path.join(cat, other)
outf = open(exp, 'w')
with open(base) as fpin:
    for nr, line in enumerate(fpin):
        parts = line.rstrip('\n').split()
        if (nr > 1 and len(parts) > 0):
            outf.write(parts[0])
            outf.write("\n")
            
outf.close()

# DR12Q DLA visual survey, extracted from Noterdaeme, et al.
name = "dr12q_visual"
cat = os.path.join(catalog, name)
direct = "dr12q_visual/distfiles"
add = os.path.join(catalog, direct)

try:
    os.makedirs(add)
    print("\nDirectory '%s' created successfully\n" %direct)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")
    
url = "http://www2.iap.fr/users/noterdae/DLA/DLA_DR12_v2.tgz"
wget.download(url, add)
nam = "DLA_DR12_v2.tgz"
comp = os.path.join(add, nam)
tar = tarfile.open(comp)
tar.extractall(path = add)
tar.close()

direct = "processed"
add = os.path.join(cat, direct)

try:
    os.makedirs(add)
    print("\nDirectory '%s' created successfully\n" %direct)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")

#more gawking here
# redshifts and column densities are not available for visual
# survey, so fill in with z_QSO and DLA threshold density
# (log_10 N_HI = 20.3)
direct = "distfiles/LOS_DR12_v2.dat"
base = os.path.join(cat, direct)
other = "processed/dla_catalog"
exp = os.path.join(cat, other)
outf = open(exp, 'w')
with open(base) as fpin:
    for nr, line in enumerate(fpin):
        parts = line.rstrip('\n').split()
        #len(parts) is equal to nf or number of fields
        if (nr > 1 and len(parts) > 0 and not parts[5].startswith('0')):
            outf.write(parts[0])
            outf.write(' ')
            outf.write(parts[4])
            outf.write(' ')
            outf.write("20.3")
            outf.write("\n")
            
outf.close()

direct = "distfiles/LOS_DR12_v2.dat"
base = os.path.join(cat, direct)
other = "processed/los_catalog"
exp = os.path.join(cat, other)
outf = open(exp, 'w')
with open(base) as fpin:
    for nr, line in enumerate(fpin):
        parts = line.rstrip('\n').split()
        if (nr > 1 and len(parts) > 0):
            outf.write(parts[0])
            outf.write("\n")
            
outf.close()

import pickle
import dill
import numpy as np

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
        self.all_transition_wavelengths = [1215.6701, 1025.7223, 972.5368, 949.7431, 937.8035,
                                           930.7483, 926.2257, 923.1504, 920.9631, 919.3514,
                                           918.1294, 917.1806, 916.429, 915.824, 915.329, 914.919,
                                           914.576, 914.286, 914.039,913.826, 913.641, 913.480,
                                           913.339, 913.215, 913.104, 913.006, 912.918, 912.839,
                                           912.768, 912.703, 912.645] # transition wavelengths, Å
        self.all_oscillator_strengths = [0.416400, 0.079120, 0.029000, 0.013940, 0.007799, 0.004814, 0.003183, 0.002216, 0.001605,
                            0.00120, 0.000921, 0.0007226, 0.000577, 0.000469, 0.000386, 0.000321, 0.000270, 0.000230,
                            0.000197, 0.000170, 0.000148, 0.000129, 0.000114, 0.000101, 0.000089, 0.000080,
                            0.000071, 0.000064, 0.000058, 0.000053, 0.000048]
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

dill.dump_session('parameters.pkl')



import dill
import pickle
from astropy.io import fits
from astropy.io.fits import getdata
from pathlib import Path
import os
import numpy as np
import astropy
import pandas as pd
#dill.load_session('parameters.pkl')
#astropy.io.fits.Conf.use_memmap = True
#lazy_load_hdus=False
#disable_image_compression=True

preParams = preproccesing_params()
normParams = normalization_params()
loading = file_loading()

#def build_catalogs(preParams, normParams, loading):
p = Path(os.getcwd())
#gets parent path
base_directory = p.parent
# Parent Directory path
parent_dir = str(p.parent)
#original project has this function up one directory and needs data/dr9q/distfiles/DR9Q.fits
#no need to go up one level either since data directory is at same level
filepath = "dr9q/distfiles/DR9Q.fits"
filename = os.path.join(parent_dir, filepath)
with fits.open(filename) as hdl:
    dr9_catalog = hdl[1].data # assuming the first extension is a table
#hdul.close()
#dr9_catalog = fits.getdata(filename, 'binarytable')

filepath = "dr10q/distfiles/DR10Q_v2.fits"
filename = os.path.join(parent_dir, filepath)
with fits.open(filename) as hdl:
    dr10_catalog = hdl[1].data # assuming the first extension is a table
#hdul.close()
#dr10_catalog = fits.getdata(filename, 'binarytable')

filepath = "dr12q/distfiles/DR12Q.fits"
filename = os.path.join(parent_dir, filepath)

with fits.open(filename) as hdl:
    dr12_catalog = hdl[1].data # assuming the first extension is a table
print(dr12_catalog.dtype.names)
#hdul.close()
#dr12_catalog = fits.getdata(filename, 'binarytable')
#print(data[0])

#extract basic QSO information from DR12Q catalog
sdss_names       =  dr12_catalog.field('SDSS_NAME')
ras              =  dr12_catalog.field('RA')
decs             =  dr12_catalog.field('DEC')
thing_ids        =  dr12_catalog.field('THING_ID')
plates           =  dr12_catalog.field('PLATE')
mjds             =  dr12_catalog.field('MJD')
fiber_ids        =  dr12_catalog.field('FIBERID')
z_qsos           =  dr12_catalog.field('Z_VI')
zwarning         =  dr12_catalog.field('ZWARNING')
snrs             =  dr12_catalog.field('SNR_SPEC')
bal_visual_flags = (dr12_catalog.field('BAL_FLAG_VI') > 0)

num_quasars = len(z_qsos)

# determine which objects in DR12Q are in DR10Q and DR9Q, using SDSS
# thing IDs
thingID9 = dr9_catalog.field('THING_ID')
thingID10 = dr10_catalog.field('THING_ID')
in_dr9  = np.isin(thing_ids,  thingID9)
in_dr10 = np.isin(thing_ids, thingID10)

# to track reasons for filtering out QSOs
filter_flags = np.zeros(num_quasars, np.uint8)
comp = np.zeros(num_quasars, np.uint8)
other = np.zeros(num_quasars, np.uint8)

# filtering bit 0: z_QSO < 2.15
ind = (z_qsos < preParams.z_qso_cut)
#filter_flags = np.bitwise_or(filter_flags, ind)
#filter_flags[ind] = filter_flags[ind] | 0x00000001
filter_flags[ind] = 1
#filter_flags(ind) = bitset(filter_flags(ind), 1, true)

# filtering bit 1: BAL
#check list comprehension bs later
#newlist = [x if x != "banana" else "orange" for x in fruits]
#for idx, a in enumerate(foo):
    #foo[idx] = a + 42
ind = (bal_visual_flags)
#comp = np.bitwise_or(comp, ind)
#comp = comp | ind
#comp[val == 1] = 2
#comp[ind] = 2
#comp = [val if val != 1 else 2 for val in comp]
#for idx, val in enumerate(comp):
#    if (val == 1):
#        comp[idx] = 2

#filter_flags = filter_flags + comp
#filter_flags(ind) = bitset(filter_flags(ind), 2, true)
#filter_flags[ind] = filter_flags[ind] | 0x00000010
filter_flags[ind] = 2

# filtering bit 4: ZWARNING
ind = (zwarning > 0) #and zwarning <= 16)
print("ind", ind, ind.shape, np.count_nonzero(ind))
# but include `MANY_OUTLIERS` in our samples (bit: 1000)
ind_many_outliers = (zwarning == 16)
ind = ind & np.logical_not(ind_many_outliers)
#ind_many_outliers      = (zwarning == int('10000', 2))
#temp idea (only works for 3/4 of cases)
#ind = np.bitwise_xor(ind, ind_many_outliers)
#other = np.bitwise_or(other, ind)
#filter_flags[ind] = filter_flags[ind] | 0x00010000
filter_flags[ind] = 5
#filter_flags = filter_flags + other
#ind(ind_many_outliers) = 0
#filter_flags(ind) = bitset(filter_flags(ind), 5, true)

#print("ind_many_outliers")
#print(ind_many_outliers)
#print(ind_many_outliers.shape)
#print(np.count_nonzero(ind_many_outliers))
#print("ind")
#print(ind)
#print(ind.shape)
#print(np.count_nonzero(ind))
#temp = zwarning[zwarning>0]
#print("temp")
#print(temp)
#print(temp.shape)

los_inds = {}
dla_inds = {}
z_dlas = {}
log_nhis = {}

# load available DLA catalogs
catalog_name = {'dr9q_concordance', 'dr12q_noterdaeme', 'dr12q_visual'}
for cat in catalog_name:
    filepath = "dla_catalogs/{name}/processed/los_catalog".format(name = cat)
    filename = os.path.join(parent_dir, filepath)
    los_catalog = pd.read_csv(filename, sep = " ", header=None)
    los_catalog = np.array(los_catalog)
    los_inds[cat] = np.isin(thing_ids, los_catalog)
    
    filepath = "dla_catalogs/{name}/processed/dla_catalog".format(name = cat)
    filename = os.path.join(parent_dir, filepath)
    dla_catalog = pd.read_csv(filename, sep = " ", header=None)
    dla_catalog = np.array(dla_catalog)
    dla_inds[cat] = np.isin(thing_ids, dla_catalog)
    ind = np.isin(thing_ids, dla_catalog[:,0])
    ind = np.where(ind > 0)
    ind = np.squeeze(ind)
    this_z_dlas = np.zeros(num_quasars)
    this_log_nhis = np.zeros(num_quasars)
    this_z_dlas = this_z_dlas.tolist()
    this_log_nhis = this_log_nhis.tolist()
    
    for i in range(len(ind)):
        this_dla_ind = (dla_catalog[:,0] == thing_ids[ind[i]])
        #print('\nloopy')
        #print(i)
        #print(dla_catalog[:,0] == thing_ids[i])
        this_z_dlas[ind[i]]   = dla_catalog[this_dla_ind, 1]
        this_log_nhis[ind[i]] = dla_catalog[this_dla_ind, 2]
        
    z_dlas[cat] = this_z_dlas
    log_nhis[cat] = this_log_nhis

#need to append data at front in real program
release = "dr12q/processed"
filename = os.path.join(parent_dir, release)
variables_to_save = {'sdss_names': sdss_names, 'ras': ras, 'decs': decs, 'thing_ids': thing_ids, 'plates': plates, 
                     'mjds': mjds, 'fiber_ids': fiber_ids, 'z_qsos': z_qsos, 'snrs': snrs, 
                     'bal_visual_flags': bal_visual_flags, 'in_dr9': in_dr9, 'in_dr10': in_dr10, 
                     'filter_flags': filter_flags, 'los_inds': los_inds, 'dla_inds': dla_inds,
                     'z_dlas': z_dlas, 'log_nhis': log_nhis}

try:
    os.makedirs(filename)
    print("\nDirectory '%s' created successfully\n" %release)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")
    
# Open a file for writing data
filepath = os.path.join(filename, "catalog")
file_handler = open(filepath, 'wb')

# Dump the data of the object into the file
pickle.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()
#np.save(filename, variables_to_save)

# these plates use the 5.7.2 processing pipeline in SDSS DR12
v_5_7_2_plates = np.array([7339, 7340, 7386, 7388, 7389, 7391, 7396, 7398, 7401, 
                  7402, 7404, 7406, 7407, 7408, 7409, 7411, 7413, 7416, 
                  7419, 7422, 7425, 7426, 7428, 7455, 7512, 7513, 7515,
                  7516, 7517, 7562, 7563, 7564, 7565])

v_5_7_2_ind = np.isin(plates, v_5_7_2_plates)

# build file list for SDSS DR12Q spectra to download (i.e., the ones
# that are not yet removed from the catalog according to the filtering flags)
#fid = fopen(sprintf('%s/file_list', spectra_directory(release)), 'w');
#originally has data in it
release = "dr12q/spectra"
filename = os.path.join(parent_dir, release)
filelist = os.path.join(filename, "file_list")

try:
    os.makedirs(filename)
    print("\nDirectory '%s' created successfully\n" %release)
except OSError as error:
    print("\nDirectory '%s' can not be created\n")
    

plate_data = []
for i in range(num_quasars):
    if (filter_flags[i] > 0):
        continue
    
    #for 5.7.2 plates, simply print greedily print both 5.7.0 and 5.7.2 paths
    if (v_5_7_2_ind[i]):
        fid =  "v5_7_2/spectra/lite/./{plate}/spec-{plate}-{mjd}-{:04d}.fits".format(fiber_ids[i], plate=plates[i], mjd=mjds[i])
        plate_data.append(fid)

    fid =  "v5_7_0/spectra/lite/./{plate}/spec-{plate}-{mjd}-{:04d}.fits".format(fiber_ids[i], plate=plates[i], mjd=mjds[i])
    plate_data.append(fid)
    
#plate_data.append(fid)

outf = open(filelist, 'w')
for line in plate_data:
    outf.write(line)
    outf.write("\n")
            
outf.close()
    
print(sdss_names)
print('\nRAS')
print(ras)
print('\ndecs')
print(decs)
print('\nthing_ids')
print(thing_ids)
print('\nplates')
print(plates)
print('\nmjds')
print(mjds)
print('\nfiber_ids')
print(fiber_ids)
print('\nz_qso\n')
print("variable next")
print(z_qsos)
print('\n\nsnrs')
print(snrs)
print('\nzwarning')
print(zwarning)
print('\nbal_visual_flags')
print(bal_visual_flags)
print('\nnum_quasars')
print(num_quasars)
print('\nin_dr9')
print(in_dr9)
print('\nin_dr10')
print(in_dr10)
print('\nfilter_flags')
print(filter_flags)
print('\n')
#if (np.array_equal(filter_flags, comp)):
#    print("EQUAL")
print('\nother')
print(other)
print('\nlos_catalog')
print(los_catalog)
print('\nlos_inds')
print(los_inds)
print('\ndla_catalog')
print(dla_catalog)
print('\ndla_inds')
print(dla_inds)
print('\nind')
print(ind)
#print('\nz_dlas')
#print(z_dlas)
#print('\nlog_nhis')
#print(log_nhis)
print('\nvariables_to_save')
print(variables_to_save)
print("v_5_7_2_ind")
print(v_5_7_2_ind)
print(v_5_7_2_ind.shape)
#print('\nplate_data')
#print(plate_data)
print('\n\n\ni')


#read_spec: loads data from SDSS DR12Q coadded "speclite" FITS file;
#see
#https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
#for a complete description of the data format

#function [wavelengths, flux, noise_variance, pixel_mask] = read_spec(filename)

#mask bits to consider
#  BRIGHTSKY = 24;

#  measurements = fitsread(filename, ...
#          'binarytable',  1, ...
#          'tablecolumns', 1:4);

#coadded calibrated flux  10^-17 erg s^-1 cm^-2 A^-1
#  flux = measurements{1};

 #log_10 wavelength        log A
 # log_wavelengths = measurements{2};

#inverse noise variance of flux measurements
#  inverse_noise_variance = measurements{3};

#"and" mask
#  and_mask = measurements{4};

#convert log_10 wavelengths to wavelengths
#  wavelengths = 10.^log_wavelengths;

#derive noise variance
#  noise_variance = 1 ./ (inverse_noise_variance);

#derive bad pixel mask, remove pixels considered very bad
#(FULLREJECT, NOSKY, NODATA); additionally remove pixels with BRIGHTSKY set
#  pixel_mask = ...
#      (inverse_noise_variance == 0) | ...
#      (bitget(and_mask, BRIGHTSKY));

#end

# preload_qsos: loads spectra from SDSS FITS files, applies further
# filters, and applies some basic preprocessing such as normalization
# and truncation to the region of interest

# load QSO catalog
#variables_to_load = {'z_qsos', 'plates', 'mjds', 'fiber_ids', 'filter_flags'};
#load(sprintf('%s/catalog', processed_directory(release)), ...
#    variables_to_load{:});

#num_quasars = numel(z_qsos);

#all_wavelengths    =  cell(num_quasars, 1);
#all_flux           =  cell(num_quasars, 1);
#all_noise_variance =  cell(num_quasars, 1);
#all_pixel_mask     =  cell(num_quasars, 1);
#all_normalizers    = zeros(num_quasars, 1);

#for i = 1:num_quasars
  #if (filter_flags(i) > 0)
   # continue;
  #end

  #[this_wavelengths, this_flux, this_noise_variance, this_pixel_mask] ...
  #    = file_loader(plates(i), mjds(i), fiber_ids(i));

  # do not normalize flux: this is done in the learning and processing code.

  #ind = (this_wavelengths >= (min_lambda * (z_qso_cut) + 1)) & ...
  #      (this_wavelengths <= (max_lambda * (z_qso_training_max_cut) + 1)) & ...
  #      (~this_pixel_mask);
  # bit 3: not enough pixels available
  #if (nnz(ind) < min_num_pixels)
  #  filter_flags(i) = bitset(filter_flags(i), 4, true);
  #  continue;
  #end

 # all_wavelengths{i}    =    this_wavelengths;
 # all_flux{i}           =           this_flux;
 # all_noise_variance{i} = this_noise_variance;
 # all_pixel_mask{i}     =     this_pixel_mask;

#  fprintf('loaded quasar %i of %i (%i/%i/%04i)\n', ...
#          i, num_quasars, plates(i), mjds(i), fiber_ids(i));
#end

#variables_to_save = {'loading_min_lambda', 'loading_max_lambda', ...
#                     'normalization_min_lambda', 'normalization_max_lambda', ...
#                     'min_num_pixels', 'all_wavelengths', 'all_flux', ...
#                     'all_noise_variance', 'all_pixel_mask', ...
#                     'all_normalizers'};
#save(sprintf('%s/preloaded_qsos', processed_directory(release)), ...
#     variables_to_save{:}, '-v7.3');

# write new filter flags to catalog
#save(sprintf('%s/catalog', processed_directory(release)), ...
#     'filter_flags', '-append');
from astropy.io import fits
import pickle
import dill
from pathlib import Path
import numpy as np
import os

#dill.load_session("parameters.pkl")

def read_spec(plate, mjd, fiber_id, directory):
    fileName = "{direc}/{plates}/spec-{plates}-{mjds}-{:04d}.fits".format(fiber_id, direc=directory, plates=plate, mjds=mjd)
    #mask bits to consider
    BRIGHTSKY = 23
    
    #measurements = fitsread(filename, 'binarytable',  1, 'tablecolumns', 1:4)
    with fits.open(fileName) as hdl:
        measurements = hdl[1].data # assuming the first extension is a table
    #print(measurements.dtype.names)
    
    # coadded calibrated flux  10^-17 erg s^-1 cm^-2 A^-1
    flux = measurements.field('flux')

    # log_10 wavelength        log A
    log_wavelengths = measurements.field('loglam')

    # inverse noise variance of flux measurements
    inverse_noise_variance = measurements.field('ivar')

    # "and" mask
    and_mask = measurements.field('and_mask')

    # convert log_10 wavelengths to wavelengths
    wavelengths = np.power(10, log_wavelengths)
    #print("log_wavelengths")
    #print(log_wavelengths)

    # derive noise variance
    noise_variance = 1.0 / (inverse_noise_variance)

    # derive bad pixel mask, remove pixels considered very bad
    # (FULLREJECT, NOSKY, NODATA); additionally remove pixels with BRIGHTSKY set
    pixel_mask = (inverse_noise_variance==0) | (and_mask & pow(2,BRIGHTSKY))
    
    return [wavelengths, flux, noise_variance, pixel_mask];

#p = Path(os.getcwd())
#parent_dir = str(p.parent)
#release = "dr12q/spectra"
#directory = os.path.join(parent_dir, release)

#def file_loader(plate, mjd, fiber_id):
#    fileName = "%{direc}/{plates}/spec-{plates}-{mjds}-{:04d}.fits".format(fiber_id, direc=directory, plates=plate, mjds=mjd)
#    [wavelen, flu, noise_var, pix_mask] = read_spec(fileName)
#    return [wavelen, flu, noise_var, pix_mask];

# preload_qsos: loads spectra from SDSS FITS files, applies further
# filters, and applies some basic preprocessing such as normalization
# and truncation to the region of interest
preParams = preproccesing_params()
preParams.min_num_pixels = 200
normParams = normalization_params()
normParams.normalization_min_lambda = 1176
normParams.normalization_max_lambda = 1256
loading = file_loading()
loading.loading_min_lambda = 910
loading.loading_max_lambda = 1217
nullParams = null_params()

p = Path(os.getcwd())
parent_dir = str(p.parent)
release = "dr12q/processed/catalog"
filename = os.path.join(parent_dir, release)
#getting back pickled data
try:
    with open(filename,'rb') as f:
        variables_to_load = pickle.load(f)
except:
    print(variables_to_load)
    
z_qsos = variables_to_load["z_qsos"]
plates = variables_to_load["plates"]
mjds = variables_to_load["mjds"]
fiber_ids = variables_to_load["fiber_ids"]
filter_flags = variables_to_load["filter_flags"]

num_quasars = len(z_qsos)
#for debugging purposes, will be shortened to 5000
num_quasars = 5000

all_wavelengths    =  []
all_flux           =  []
all_noise_variance =  []
all_pixel_mask     =  []
all_normalizers    = np.zeros(num_quasars, 'uint8')

release = "dr12q/spectra"
directory = os.path.join(parent_dir, release)

for i in range(num_quasars):
    if (filter_flags[i] > 0):
        #print("skipped at ", i)
        all_wavelengths.append(0)
        all_flux.append(0)
        all_noise_variance.append(0)
        all_pixel_mask.append(0)
        continue
    #file_load pain
    #print('\nplates')
    #print(plates[0])
    #print('\nmjds')
    #print(mjds[0])
    #print('\nfiber_ids')
    #print(fiber_ids[0])
    [this_wavelengths, this_flux, this_noise_variance, this_pixel_mask] = read_spec(plates[i], mjds[i], fiber_ids[i], directory)
     # do not normalize flux: this is done in the learning and processing code.
    ind = (this_wavelengths >= (nullParams.min_lambda * (preParams.z_qso_cut) + 1)) & (this_wavelengths <= (nullParams.max_lambda * (preParams.z_qso_training_max_cut) + 1)) & (np.logical_not(this_pixel_mask))
    
    # bit 3: not enough pixels available
    if (np.count_nonzero(ind) < preParams.min_num_pixels):
        #filter_flags[i] = filter_flags[i] | 0x00001000
        filter_flags[i] = 4
        print("added at", i)
        all_wavelengths.append(0)
        all_flux.append(0)
        all_noise_variance.append(0)
        all_pixel_mask.append(0)
        continue
    
    all_wavelengths.append(this_wavelengths)
    all_flux.append(this_flux)
    all_noise_variance.append(this_noise_variance)
    all_pixel_mask.append(this_pixel_mask)
    
    message = "loaded quasar {num} of {numQ} ({plate}/{mjd}/{:04d})\n".format(fiber_ids[i], num=i, numQ=num_quasars, plate=plates[i], mjd=mjds[i])
    print(message)

all_wavelengths = np.array(all_wavelengths)
all_flux = np.array(all_flux)
all_noise_variance = np.array(all_noise_variance)
all_pixel_mask = np.array(all_pixel_mask)

variables_to_save = {'loading_min_lambda' : loading.loading_min_lambda, 'loading_max_lambda' : loading.loading_max_lambda,
                     'normalization_min_lambda': normParams.normalization_min_lambda,
                     'normalization_max_lambda': normParams.normalization_max_lambda,
                     'min_num_pixels': preParams.min_num_pixels, 'all_wavelengths': all_wavelengths, 'all_flux': all_flux,
                     'all_noise_variance': all_noise_variance, 'all_pixel_mask': all_pixel_mask,
                     'all_normalizers': all_normalizers}
    
# Open a file for writing data
filename = os.path.join(parent_dir, "dr12q/processed")
filepath = os.path.join(filename, "preloaded_qsos")
file_handler = open(filepath, 'wb')

# Dump the data of the object into the file
pickle.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()

new_filter_flags = filter_flags

# write new filter flags to catalog
filepath = os.path.join(filename, "catalog")
#getting back pickled data
#try:
with open(filepath,'rb') as f:
    temp = pickle.load(f)
#except:
temp['new_filter_flags'] = new_filter_flags

# Open a file for writing data
file_handler = open(filepath, 'wb')

# Dump the data of the object into the file
pickle.dump(temp, file_handler)

# close the file handler to release the resources
file_handler.close()
#np.save(filename, variables_to_save)

print("\nz_qsos")
print(z_qsos)
print("\nplates")
print(plates)
print("\nmjds")
print(mjds)
print("\nfiber_ids")
print(fiber_ids)
print("\nfilter_flags")
print(filter_flags)
print("\nall_normalizers")
print(all_normalizers)
print("\nvariables_to_save")
print(variables_to_save)
print("\ntemp")
print(temp)
print("\n\n\nn")