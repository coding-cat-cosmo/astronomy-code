from pathlib import Path
import os
import ssl
import wget
import tarfile
import dill
import pickle
from astropy.io import fits
from pathlib import Path
import os
import numpy as np
import pandas as pd

#function creates directory given a valid path as an input
def makeDir(directory):
    try:
        os.makedirs(directory)
        print("Directory '%s' created successfully" %directory)
    except OSError as error:
        print("Directory '%s' can not be created" %directory)

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
makeDir(directory)
    
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
url = "http://data.sdss3.org/sas/dr9/env/BOSS_QSO/DR9Q/DR9Q.fits"
wget.download(url, directory)
 
# By setting exist_ok as True
# error caused due already
# existing directory can be suppressed
# but other OSError may be raised
# due to other error like
# invalid path name
#DR10Q
directory='dr10q/distfiles'
#add = os.path.join(parent_dir, directory)

makeDir(directory)
    
url = "http://data.sdss3.org/sas/dr10/boss/qso/DR10Q/DR10Q_v2.fits"
wget.download(url, directory)

#DR12Q
directory='dr12q/distfiles'
#add = os.path.join(parent_dir, directory)

makeDir(directory)
    
url = "http://data.sdss3.org/sas/dr12/boss/qso/DR12Q/DR12Q.fits"
wget.download(url, directory)

directory = "dla_catalogs"
catalog = os.path.join(parent_dir, directory)
# concordance catalog from BOSS DR9 Lyman-alpha forest catalog
name = "dr9q_concordance"
cat = os.path.join(directory, name)
direct = "dr9q_concordance/distfiles"
add = os.path.join(directory, direct)

makeDir(add)
    
url = "http://data.sdss3.org/sas/dr9/boss/lya/cat/BOSSLyaDR9_cat.txt"
wget.download(url, add)

direct = "processed"
add = os.path.join(cat, direct)

makeDir(add)
    
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
exp = os.path.join(directory, other)
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
cat = os.path.join(directory, name)
direct = "dr12q_noterdaeme/distfiles"
add = os.path.join(directory, direct)

makeDir(add)
    
url = "http://www2.iap.fr/users/noterdae/DLA/DLA_DR12_v2.tgz"
wget.download(url, add)
nam = "DLA_DR12_v2.tgz"
comp = os.path.join(add, nam)
tar = tarfile.open(comp)
tar.extractall(path = add)
tar.close()

direct = "processed"
add = os.path.join(cat, direct)

makeDir(add)

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
cat = os.path.join(directory, name)
direct = "dr12q_visual/distfiles"
add = os.path.join(directory, direct)

makeDir(add)
    
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

with open('parameters.pkl', 'rb') as handle:
    params = dill.load(handle)

preParams = params['preParams']
normParams = params['normParams']
loading = params['loadParams']

#def build_catalogs(preParams, normParams, loading):
p = Path(os.getcwd())
#gets parent path
base_directory = p.parent
# Parent Directory path
parent_dir = str(p.parent)
#original project has this function up one directory and needs data/dr9q/distfiles/DR9Q.fits
#no need to go up one level either since data directory is at same level
filepath = "dr9q/distfiles/DR9Q.fits"
#filename = os.path.join(parent_dir, filepath)
with fits.open(filepath) as hdl:
    dr9_catalog = hdl[1].data # assuming the first extension is a table

filepath = "dr10q/distfiles/DR10Q_v2.fits"
#filename = os.path.join(parent_dir, filepath)
with fits.open(filepath) as hdl:
    dr10_catalog = hdl[1].data # assuming the first extension is a table

filepath = "dr12q/distfiles/DR12Q.fits"
#filename = os.path.join(parent_dir, filepath)

with fits.open(filepath) as hdl:
    dr12_catalog = hdl[1].data # assuming the first extension is a table
print(dr12_catalog.dtype.names)

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
filter_flags[ind] = 1

# filtering bit 1: BAL
ind = (bal_visual_flags)
filter_flags[ind] = 2

# filtering bit 4: ZWARNING
ind = (zwarning > 0) #and zwarning <= 16)
print("ind", ind, ind.shape, np.count_nonzero(ind))
# but include `MANY_OUTLIERS` in our samples (bit: 1000)
ind_many_outliers = (zwarning == 16)
ind = ind & np.logical_not(ind_many_outliers)
filter_flags[ind] = 5

los_inds = {}
dla_inds = {}
z_dlas = {}
log_nhis = {}

# load available DLA catalogs
catalog_name = {'dr9q_concordance', 'dr12q_noterdaeme', 'dr12q_visual'}
for cat in catalog_name:
    filepath = "dla_catalogs/{name}/processed/los_catalog".format(name = cat)
    filename = os.path.join(parent_dir, filepath)
    los_catalog = pd.read_csv(filepath, sep = " ", header=None)
    los_catalog = np.array(los_catalog)
    los_inds[cat] = np.isin(thing_ids, los_catalog)
    
    filepath = "dla_catalogs/{name}/processed/dla_catalog".format(name = cat)
    filename = os.path.join(parent_dir, filepath)
    dla_catalog = pd.read_csv(filepath, sep = " ", header=None)
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

makeDir(release)
    
# Open a file for writing data
filepath = os.path.join(release, "catalog")
file_handler = open(filepath, 'wb')

# Dump the data of the object into the file
pickle.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()

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
#filename = os.path.join(parent_dir, release)
filelist = os.path.join(release, "file_list")

makeDir(release)   

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

outf = open(filelist, 'w')
for line in plate_data:
    outf.write(line)
    outf.write("\n")
            
outf.close()

# Path
directory = 'checkpoints'
#add = os.path.join(parent_dir, directory)
 
# Create the directory
# 'distfiles' 
makeDir(directory)
    