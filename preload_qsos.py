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

dill.load_session("parameters.pkl")

def read_spec(plate, mjd, fiber_id, directory):
    fileName = "{direc}/{plates}/spec-{plates}-{mjds}-{:04d}.fits".format(fiber_id, direc=directory, plates=plate, mjds=mjd)
    #mask bits to consider
    BRIGHTSKY = 24
    
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

    # derive noise variance
    noise_variance = 1 / (inverse_noise_variance)

    # derive bad pixel mask, remove pixels considered very bad
    # (FULLREJECT, NOSKY, NODATA); additionally remove pixels with BRIGHTSKY set
    pixel_mask = (inverse_noise_variance == 0) | and_mask[BRIGHTSKY]
    
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
with open(filename,'rb') as f:
     variables_to_load = pickle.load(f)
print(variables_to_load)

z_qsos = variables_to_load["z_qsos"]
plates = variables_to_load["plates"]
mjds = variables_to_load["mjds"]
fiber_ids = variables_to_load["fiber_ids"]
filter_flags = variables_to_load["filter_flags"]

num_quasars = len(z_qsos)

all_wavelengths    =  {}
all_flux           =  {}
all_noise_variance =  {}
all_pixel_mask     =  {}
all_normalizers    = np.zeros(num_quasars, 'uint8')

release = "dr12q/spectra"
directory = os.path.join(parent_dir, release)

for i in range(num_quasars):
    if (filter_flags[i] > 0):
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
    ind = (this_wavelengths >= (nullParams.min_lambda * (preParams.z_qso_cut) + 1)) & (this_wavelengths <= (nullParams.max_lambda * (preParams.z_qso_training_max_cut) + 1)) & (~this_pixel_mask)
    
    # bit 3: not enough pixels available
    if (np.count_nonzero(ind) < preParams.min_num_pixels):
        filter_flags[i] = 4
        continue
    
    all_wavelengths[i]    =    this_wavelengths
    all_flux[i]           =           this_flux
    all_noise_variance[i] = this_noise_variance
    all_pixel_mask[i]     =     this_pixel_mask
    
    message = "loaded quasar {num} of {numQ} ({plate}/{mjd}/{:04d})\n".format(fiber_ids[i], num=i, numQ=num_quasars, plate=plates[i], mjd=mjds[i])
    print(message)

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

# write new filter flags to catalog
filepath = os.path.join(filename, "catalog")
file_handler = open(filepath, 'a')

# Dump the data of the object into the file
pickle.dump(filter_flags, file_handler)

# close the file handler to release the resources
file_handler.close()

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
print("\n\n\nn")
