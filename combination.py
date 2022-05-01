import dill
import pickle
import numpy as np
#rel = "dr12q/processed/"
#direc = os.path.join(parent_dir, rel)
#lease = "dla_samples"
#lease = "dlaSamples"
#filename = os.path.join(direc, lease)
filename = "first_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
training_release = process["training_release"]
training_set_name = process['training_set_name']
dla_catalog_name = process['dla_catalog_name']
release = process['release']
test_set_name = process['test_set_name']
test_ind = process['test_ind']
prior_z_qso_increase = process['prior_z_qso_increase']
max_z_cut = process['max_z_cut']
num_lines = process['num_lines']
min_z_dlas = process['min_z_dlas']
max_z_dlas = process['max_z_dlas']
#sample_log_posteriors_no_dla = process['sample_log_posteriors_no_dla']
#sample_log_posteriors_dla = process['sample_log_posteriors_dla']
#sample_log_posteriors_dla_sub = process['sample_log_posteriors_dla_sub']
#sample_log_posteriors_dla_sup = process['sample_log_posteriors_dla_sup']
log_posteriors_no_dla = process["log_posteriors_no_dla"]
log_posteriors_dla = process['log_posteriors_dla']
log_posteriors_dla_sub = process['log_posteriors_dla_sub']
log_posteriors_dla_sup = process['log_posteriors_dla_sup']
model_posteriors = process['model_posteriors']
p_no_dlas = process['p_no_dlas']
p_dlas = process['p_dlas']
z_map = process['z_map']
z_true = process['z_true']
dla_true = process['dla_true']
z_dla_map = process['z_dla_map']
n_hi_map = process['n_hi_map']
log_nhi_map = process['log_nhi_map']
signal_to_noise = process['signal_to_noise']
all_thing_ids = process['all_thing_ids']
all_posdeferrors = process['all_posdeferrors']
all_exceptions = process['all_exceptions']
qso_ind = process['qso_ind']

#SAME: training_release, training_set_name, dla_catalog_name, release, test_set_name, test_ind,
#prior_z_qso_increase, max_z_cut, num_lines, all_thing_ids, qso_ind
#DIFFERENT: min_z_dlas, max_z_dlas, sample_log_posteriors_no_dla, sample_log_posteriors_dla,
#sample_log_posteriors_dla_sub, sample_log_posteriors_dla_sup, log_posteriors_no_dla, log_posteriors_dla,
#log_posteriors_dla_sub, log_posteriors_dla_sup, model_posteriors, p_no_dlas, p_dlas, z_map, z_true,
#dla_true, z_dla_map, n_hi_map, log_nhi_map, signal_to_noise, all_posdeferrors, all_exceptions
#PROBLEM: sample_log_posteriors_dla_sub, sample_log_posteriors_dla_sup



filename = "second_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "third_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "fourth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "fifth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "sixth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]
filename = "seventh_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]
filename = "eigth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]
filename = "ninth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "tenth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "eleventh_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "twelfth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "thirteenth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "fourteenth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "fifteenth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

filename = "sixteenth_part"
with open(filename, 'rb') as f:
    process = pickle.load(f)
    
#arr = [1]
arr = process['arr']
for num in arr:

    min_z_dlas[num] = process['min_z_dlas'][num]
    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
    model_posteriors[num] = process['model_posteriors'][num]
    p_no_dlas[num] = process['p_no_dlas'][num]
    p_dlas[num] = process['p_dlas'][num]
    z_map[num] = process['z_map'][num]
    z_true[num] = process['z_true'][num]
    dla_true[num] = process['dla_true'][num]
    z_dla_map[num] = process['z_dla_map'][num]
    n_hi_map[num] = process['n_hi_map'][num]
    log_nhi_map[num] = process['log_nhi_map'][num]
    signal_to_noise[num] = process['signal_to_noise'][num]
    all_posdeferrors[num] = process['all_posdeferrors'][num]
    all_exceptions[num] = process['all_exceptions'][num]

#with open(filename, 'rb') as f:
#    process = pickle.load(f)
    
#arr = [1]
#arr = process['arr']
#for num in arr:
    
#    min_z_dlas[num] = process['min_z_dlas'][num]
#    max_z_dlas[num] = process['max_z_dlas'][num]
    #sample_log_posteriors_no_dla[num] = process['sample_log_posteriors_no_dla'][num]
    #sample_log_posteriors_dla[num] = process['sample_log_posteriors_dla'][num]
#    log_posteriors_no_dla[num] = process["log_posteriors_no_dla"][num]
#    log_posteriors_dla[num] = process['log_posteriors_dla'][num]
#    log_posteriors_dla_sub[num] = process['log_posteriors_dla_sub'][num]
#    log_posteriors_dla_sup[num] = process['log_posteriors_dla_sup'][num]
#    model_posteriors[num] = process['model_posteriors'][num]
#    p_no_dlas[num] = process['p_no_dlas'][num]
#    p_dlas[num] = process['p_dlas'][num]
#    z_map[num] = process['z_map'][num]
#    z_true[num] = process['z_true'][num]
#    dla_true[num] = process['dla_true'][num]
#    z_dla_map[num] = process['z_dla_map'][num]
#    n_hi_map[num] = process['n_hi_map'][num]
#    log_nhi_map[num] = process['log_nhi_map'][num]
#    signal_to_noise[num] = process['signal_to_noise'][num]
#    all_posdeferrors[num] = process['all_posdeferrors'][num]
#    all_exceptions[num] = process['all_exceptions'][num]
    
print("\nmin_z_dlas")
print(min_z_dlas)
print(min_z_dlas.shape)
print("\nmax_z_dlas")
print(max_z_dlas)
print(max_z_dlas.shape)
#print("\nsample_log_posteriors_no_dla")
#print(sample_log_posteriors_no_dla)
#print(sample_log_posteriors_no_dla.shape)
#print("\nsample_log_posteriors_dla")
#print(sample_log_posteriors_dla)
#print(sample_log_posteriors_dla.shape)
print("\nlog_posteriors_dla")
print(log_posteriors_dla)
print(log_posteriors_dla.shape)
print("\nlog_posteriors_no_dla")
print(log_posteriors_no_dla)
print(log_posteriors_no_dla.shape)
print("\nlog_posteriors_dla_sub")
print(log_posteriors_dla_sub)
print(log_posteriors_dla_sub.shape)
print("\nlog_posteriors_dla_sup")
print(log_posteriors_dla_sup)
print(log_posteriors_dla_sup.shape)
print("\nmodel_posteriors")
print(model_posteriors)
print(model_posteriors.shape)
print("\np_no_dlas")
print(p_no_dlas)
print(p_no_dlas.shape)
print("\np_dlas")
print(p_dlas)
print(p_dlas.shape)
print("\nz_map")
print(z_map)
print(z_map.shape)
print("\nz_true")
print(z_true)
print(z_true.shape)
print("\ndla_true")
print(dla_true)
print(dla_true.shape)
print("\nz_dla_map")
print(z_dla_map)
print(z_dla_map.shape)
print("\nhi_map")
print(n_hi_map)
print(n_hi_map.shape)
print("\nlog_nhi_map")
print(log_nhi_map)
print(log_nhi_map.shape)
print("\nsignal_to_noise")
print(signal_to_noise)
print(signal_to_noise.shape)
print("\nall_posdeferrors")
print(all_posdeferrors)
print(all_posdeferrors.shape)
print("\nall_exceptions")
print(all_exceptions)
print(all_exceptions.shape)

filename = 'combo'

variables_to_save = {'training_release':training_release, 'training_set_name':training_set_name,
                    'dla_catalog_name':dla_catalog_name, 'release':release, 'test_set_name':test_set_name,
                    'test_ind':test_ind, 'prior_z_qso_increase':prior_z_qso_increase, 'max_z_cut':max_z_cut,
                    'num_lines':num_lines, 'min_z_dlas':min_z_dlas, 'max_z_dlas':max_z_dlas, # you need to save DLA search length to compute CDDF
                    #'sample_log_posteriors_no_dla':sample_log_posteriors_no_dla, 'sample_log_posteriors_dla':sample_log_posteriors_dla,
                    #'sample_log_posteriors_dla_sub':sample_log_posteriors_dla_sub, 'sample_log_posteriors_dla_sup':sample_log_posteriors_dla_sup,
                    'log_posteriors_no_dla':log_posteriors_no_dla, 'log_posteriors_dla':log_posteriors_dla,
                    'log_posteriors_dla_sub':log_posteriors_dla_sub, 'log_posteriors_dla_sup':log_posteriors_dla_sup,
                    'model_posteriors':model_posteriors, 'p_no_dlas':p_no_dlas, 'p_dlas':p_dlas, 'z_map':z_map,
                    'z_true':z_true, 'dla_true':dla_true, 'z_dla_map':z_dla_map, 'n_hi_map':n_hi_map, 'log_nhi_map':log_nhi_map,
                    'signal_to_noise':signal_to_noise, 'all_thing_ids':all_thing_ids, 'all_posdeferrors':all_posdeferrors,
                    'all_exceptions':all_exceptions, 'qso_ind':qso_ind}

print("\nfilename")
print(filename)
#fileName = os.path.join(parent_dir, filename)
#save(filename, variables_to_save{:}, '-v7.3');
# Open a file for writing data
file_handler = open(filename, 'wb')

# Dump the data of the object into the file
pickle.dump(variables_to_save, file_handler)

# close the file handler to release the resources
file_handler.close()
#pr.print_stats(sort='tottime')