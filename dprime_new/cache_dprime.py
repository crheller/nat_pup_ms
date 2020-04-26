"""
Procedure:
    1) Load data
    2) Generate list of stimulus pairs
    3) Make pupil mask, classify pupil for each stim pair
    4) Generate est/val sets (each est/val dataset should be shape Neuron X Rep X Stim)
    5) Preprocess est set (apply same preprocessing to val)
    --- For each stim pair ---
        6) Dimensionality reduction (on est, apply same to val)
        7) Compute dprime, save metrics
        8) Split data into large / small pupil, based on pupil mask generated at the beginning
"""
import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import os
import pandas as pd 
import pickle
import sys

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.dim_reduction as dr

import nems
import nems_lbhb.baphy as nb
import nems.db as nd
import logging

log = logging.getLogger(__name__)

# ================================ SET RNG STATE ===================================
np.random.seed(123)

# ============================== SAVE PARAMETERS ===================================
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'

# ============================ set up dbqueue stuff ================================
if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick

else:
    queueid = 0

if queueid:
    log.info("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

# ========================== read and parse system arguments ========================
site = sys.argv[1]  
batch = int(sys.argv[2])
modelname = sys.argv[3]
options = modelname.split('_')

njacks = 10
zscore = False
for op in options:
    if 'jk' in op:
        njacks = int(op[2:])
    if 'zscore' in op:
        zscore = True

# ================================= load recording ==================================
X, sp_bins, X_pup = decoding.load_site(site=site, batch=batch)
ncells = X.shape[0]
nreps = X.shape[1]
nstim = X.shape[2]
nbins = X.shape[3]
X = X.reshape(ncells, nreps, nstim * nbins)
sp_bins = sp_bins.reshape(1, nreps, nstim * nbins)
nstim = nstim * nbins

# =========================== generate a list of stim pairs ==========================
all_combos = list(combinations(range(nstim), 2))
spont_bins = np.argwhere(sp_bins[0, 0, :])
spont_combos = [c for c in all_combos if (c[0] in spont_bins) & (c[1] in spont_bins)]
ev_ev_combos = [c for c in all_combos if (c[0] not in spont_bins) & (c[1] not in spont_bins)]
spont_ev_combos = [c for c in all_combos if (c not in ev_ev_combos) & (c not in spont_combos)]

# =============================== make pupil mask ===================================
# mask pupil per stimulus, rather than overall
X_pup = X_pup.reshape(1, nreps, nstim)
pup_mask = X_pup >= np.tile(np.median(X_pup, axis=1), [1, X_pup.shape[1], 1])
nstim = nstim * nbins

# ============================== get pupil variance ==================================
# figure out pupil variance per stimulus
pupil_range = nat_preproc.get_pupil_range(X_pup, pup_mask)

# =========================== generate list of est/val sets ==========================
log.info("Generate list of {0} est / val sets".format(njacks))
est, val, p_est, p_val = nat_preproc.get_est_val_sets(X, pup_mask=pup_mask, njacks=njacks)
nreps_train = est[0].shape[1]
nreps_test = val[0].shape[1]

# determine number of dim reduction components (bounded by ndim in dataset) 
components = np.min([ncells, nreps_train])

# ============================ preprocess est / val sets =============================
if zscore:
    log.info("z-score est / val sets")
    est, val = nat_preproc.scale_est_val(est, val)
else:
    # just center data
    log.info("center est / val sets")
    est, val = nat_preproc.scale_est_val(est, val, sd=False)

# set up data frames to save results
pca_results = pd.DataFrame()
pls_results = pd.DataFrame()

# ============================== Loop over stim pairs ================================
for stim_pair_idx, combo in enumerate(all_combos):
    # print every 500th pair. Don't want to overwhelm log
    if (stim_pair_idx % 500) == 0:
        log.info("Analyzing stimulus pair {0} / {1}".format(stim_pair_idx, len(all_combos)))
    if combo in spont_combos:
        category = 'spont_spont'
    elif combo in spont_ev_combos:
        category = 'spont_evoked'
    elif combo in ev_ev_combos:
        category = 'evoked_evoked'

    for ev_set in range(njacks):
        X_train = est[ev_set][:, :, [combo[0], combo[1]]] 
        X_test = val[ev_set][:, :, [combo[0], combo[1]]]
        ptrain_mask = p_est[ev_set][:, :, [combo[0], combo[1]]]
        ptest_mask = p_val[ev_set][:, :, [combo[0], combo[1]]]

        xtrain = nat_preproc.flatten_X(X_train[:, :, :, np.newaxis])
        xtest = nat_preproc.flatten_X(X_test[:, :, :, np.newaxis])

        # ============================== PCA ANALYSIS ===============================
        _pca_results = decoding.do_pca_dprime_analysis(xtrain, 
                                                       xtest, 
                                                       nreps_train,
                                                       nreps_test,
                                                       ptrain_mask=ptrain_mask,
                                                       ptest_mask=ptest_mask)
        _pca_results.update({
            'n_components': 2,
            'jack_idx': ev_set,
            'combo': combo,
            'category': category,
            'site': site
        })
        pca_results = pca_results.append([_pca_results])

        # ============================== PLS ANALYSIS ===============================
        for n_components in range(2, components):

            _pls_results = decoding.do_pls_dprime_analysis(xtrain, 
                                                           xtest, 
                                                           nreps_train,
                                                           nreps_test,
                                                           ptrain_mask=ptrain_mask,
                                                           ptest_mask=ptest_mask,
                                                           n_components=n_components)
            _pls_results.update({
                'n_components': n_components,
                'jack_idx': ev_set,
                'combo': combo,
                'category': category,
                'site': site
            })
            pls_results = pls_results.append([_pls_results])

 
# convert columns to str
pca_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in pca_results.combo.values]
pls_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in pls_results.combo.values]

# convert to correct dtypes
pca_results = decoding.cast_dtypes(pca_results)
pls_results = decoding.cast_dtypes(pls_results)

# collapse over results to save disk space by packing into "DecodingResults object"
log.info("Compressing results into DecodingResults object... ")
pca_results = decoding.DecodingResults(pca_results, pupil_range=pupil_range)
pls_results = decoding.DecodingResults(pls_results, pupil_range=pupil_range)

# save results
log.info("Saving results to {}".format(path))
if not os.path.isdir(os.path.join(path, site)):
    os.mkdir(os.path.join(path, site))

pls_results.save_pickle(os.path.join(path, site, modelname+'_PLS.pickle'))
pca_results.save_pickle(os.path.join(path, site, modelname+'_PCA.pickle'))

if queueid:
    nd.update_job_complete(queueid)
