import numpy as np
from itertools import combinations
import os
#from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
DPRIME_DIR = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
CACHE_PATH = None # you can cache results locally, if you want...

import charlieTools.nat_sounds_ms.decoding as decoding

site = 'TAR010c'
batch = 289
recache = False # For recaching decoding results locally

X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
ncells = X.shape[0]
nreps = X.shape[1]
nstim = X.shape[2]  # this should be = len(epochs)
nbins = X.shape[3]  # for each epoch(stim) there are nbins
nstim = nstim * nbins


# FIGURE OUT STIMULUS COMBINATION INDEXING REL TO EPOCH NAMES
# stim combos (with epoch names)
offsets = {e: v for e, v in zip(epochs, np.arange(0, nbins*nstim, nbins))}
ep_bin_str = np.concatenate([[e+'_'+str(idx+offsets[e]) for idx in range(nbins)] for e in epochs])
ep_combos = ['_'.join([c[0], c[1]]) for c in list(combinations(ep_bin_str, 2))]

# stim combos (by index #, after reshaping)
idx_combos = list(combinations(range(nstim), 2))

# LOAD DECODING RESULTS
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)

# extract results for an example epoch / index pair
epoch1 = 'STIM_00Oxford_male2b.wav'
epoch2 = 'STIM_00Oxford_male2b.wav'
tbin1 = 8
tbin2 = 18

# find the results index
idx = str(tbin1+offsets[epoch1])+'_'+str(tbin2+offsets[epoch2])

# numeric results (d-prime, mag(dU) etc.)
df = results.numeric_results
ex_dprime, sem = results.get_result('dp_opt_test', stim_pair=idx, n_components=2)



# get a "confusion" matrix








# to get delta-dprime
delta = df['bp_dp'] - df['sp_dp']

# only for evoked stimulus pairs
d_evoked = df.loc[results.evoked_stimulus_pairs, 'dp_opt_test']

# example extraction of array results (e.g. cos(dU, noise) -- two-D because two noise dims)
# get the alignement with the first noise pc (idx = [0, None])
cos_dU_evec_mean, cos_dU_evec_sem = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, n_components=2, idx=[0, None])
