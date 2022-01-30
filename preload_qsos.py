#read_spec: loads data from SDSS DR12Q coadded "speclite" FITS file;
#see
#https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
#for a complete description of the data format

from astropy.io import fits
import pickle
import dill
from pathlib import Path
import numpy as np
import os
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

def read_spec(plate, mjd, fiber_id, directory):
    fileName = "{direc}/{plates}/spec-{plates}-{mjds}-{:04d}.fits".format(fiber_id, direc=directory, plates=plate, mjds=mjd)
    #mask bits to consider
    BRIGHTSKY = 23
    
    #measurements = fitsread(filename, 'binarytable',  1, 'tablecolumns', 1:4)
    try:
        hdl = fits.open(fileName, memmap=False)
        measurements = hdl[1].data
    finally:
        hdl.close()

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

    # derive noise variance
    noise_variance = 1.0 / (inverse_noise_variance)

    # derive bad pixel mask, remove pixels considered very bad
    # (FULLREJECT, NOSKY, NODATA); additionally remove pixels with BRIGHTSKY set
    pixel_mask = (inverse_noise_variance==0) | (and_mask & pow(2,BRIGHTSKY))
    
    return [wavelengths, flux, noise_variance, pixel_mask]

# preload_qsos: loads spectra from SDSS FITS files, applies further
# filters, and applies some basic preprocessing such as normalization
# and truncation to the region of interest
with open('parameters.pkl', 'rb') as handle:
    params = dill.load(handle)

preParams = params['preParams']
preParams.min_num_pixels = 200
normParams = params['normParams']
normParams.normalization_min_lambda = 1176
normParams.normalization_max_lambda = 1256
loading = params['loadParams']
loading.loading_min_lambda = 910
loading.loading_max_lambda = 1217
nullParams = params['nullParams']

p = Path(os.getcwd())
parent_dir = str(p.parent)
release = "dr12q/processed/catalog"
filename = os.path.join(parent_dir, release)
#getting back pickled data
#try:
with open(release,'rb') as f:
    variables_to_load = pickle.load(f)
#except:
#    print(variables_to_load)
    
z_qsos = variables_to_load["z_qsos"]
plates = variables_to_load["plates"]
mjds = variables_to_load["mjds"]
fiber_ids = variables_to_load["fiber_ids"]
filter_flags = variables_to_load["filter_flags"]

num_quasars = len(z_qsos)

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
    
    [this_wavelengths, this_flux, this_noise_variance, this_pixel_mask] = read_spec(plates[i], mjds[i], fiber_ids[i], release)
     # do not normalize flux: this is done in the learning and processing code.
    ind = (this_wavelengths >= (nullParams.min_lambda * (preParams.z_qso_cut) + 1)) & (this_wavelengths <= (nullParams.max_lambda * (preParams.z_qso_training_max_cut) + 1)) & (np.logical_not(this_pixel_mask))
    
    # bit 3: not enough pixels available
    if (np.count_nonzero(ind) < preParams.min_num_pixels):
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
filepath = "dr12q/processed/preloaded_qsos"

with open(filepath, 'wb') as handle:
    dill.dump(variables_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

new_filter_flags = filter_flags

# write new filter flags to catalog
filepath = os.path.join(filename, "catalog")
filepath = "dr12q/processed/catalog"
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
