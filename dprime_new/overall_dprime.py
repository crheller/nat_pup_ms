"""
Script to calculate overall dprime, and cache the results.
    Results to save:
        For each site / stimulus pair / dimensionality (and for both train / test sets)
            dprime_raw      (LDA)
            dprime_diagonal (NULL)
            wopt            (optimal decoding axis) 
            wopt_diag       (diagonal decoding axis) 
            evals           (eigenvalues of the mean covariance matrix)
            evecs           (eigenvectors of the mean covariance matrix)

Procedure:
    1) Load data
    2) Generate est/val sets (each est/val dataset should be shape Neuron X Rep X Stim)
    3) Preprocess est set (apply same preprocessing to val)
    4) Generate list of stimulus pairs
    --- For each stim pair ---
        5) Dimensionality reduction (on est, apply same to val)
        6) Compute dprime, save metrics
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
options = {'cellid': site, 'rasterfs': 4, 'batch': batch, 'pupil': True, 'stim': False}
if batch == 294:
    options['runclass'] = 'VOC'
rec = nb.baphy_load_recording_file(**options)
rec['resp'] = rec['resp'].rasterize()
if 'cells_to_extract' in rec.meta.keys():
    if rec.meta['cells_to_extract'] is not None:
        log.info("Extracting cellids: {0}".format(rec.meta['cells_to_extract']))
        rec['resp'] = rec['resp'].extract_channels(rec.meta['cells_to_extract'])

# remove post stim silence (keep prestim so that can get a baseline dprime on each sound)
rec = rec.and_mask(['PostStimSilence'], invert=True)
if batch == 294:
    epochs = [epoch for epoch in rec.epochs.name.unique() if 'STIM_' in epoch]
else:
    epochs = [epoch for epoch in rec.epochs.name.unique() if 'STIM_00' in epoch]
rec = rec.and_mask(epochs)
resp_dict = rec['resp'].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
spont_signal = rec['resp'].epoch_to_signal('PreStimSilence')
sp_dict = spont_signal.extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)

# create response matrix, X
X = nat_preproc.dict_to_X(resp_dict)
sp_bins = nat_preproc.dict_to_X(sp_dict)
ncells = X.shape[0]
nreps = X.shape[1]
nstim = X.shape[2]
nbins = X.shape[3]
X = X.reshape(ncells, nreps, nstim * nbins)
sp_bins = sp_bins.reshape(1, nreps, nstim * nbins)
nstim = nstim * nbins

# =========================== generate list of est/val sets ==========================
log.info("Generate list of {0} est / val sets".format(njacks))
est, val = nat_preproc.get_est_val_sets(X, njacks=njacks)
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


# =========================== generate a list of stim pairs ==========================
all_combos = list(combinations(range(nstim), 2))
spont_bins = np.argwhere(sp_bins[0, 0, :])
spont_combos = [c for c in all_combos if (c[0] in spont_bins) & (c[1] in spont_bins)]
ev_ev_combos = [c for c in all_combos if (c[0] not in spont_bins) & (c[1] not in spont_bins)]
spont_ev_combos = [c for c in all_combos if (c not in ev_ev_combos) & (c not in spont_combos)]

# set up data frames to save results
# for pca, only one component organization (bc trial average PCA and pairwise comparisons)
columns = ['dp_opt_test', 'dp_diag_test', 'wopt_test', 'wdiag_test', 'var_explained_test',
           'evals_test', 'evecs_test', 'dU_test',
           'dp_opt_train', 'dp_diag_train', 'wopt_train', 'wdiag_train', 'var_explained_train',
           'evals_train', 'evecs_train', 'dU_train', 
           'dU_mag_test', 'dU_dot_evec_test', 'cos_dU_wopt_test', 'dU_dot_evec_sq_test', 'evec_snr_test', 'cos_dU_evec_test',
           'dU_mag_train', 'dU_dot_evec_train', 'cos_dU_wopt_train', 'dU_dot_evec_sq_train', 'evec_snr_train', 'cos_dU_evec_train',
           'jack_idx', 'n_components', 'combo', 'category', 'site']
pca_index = range(len(all_combos) * njacks)
pca_results = pd.DataFrame(columns=columns, index=pca_index)

pls_index = range(len(all_combos) * njacks * (components-2))
pls_results = pd.DataFrame(columns=columns, index=pls_index)


# ============================== Loop over stim pairs ================================
pca_idx = 0
pls_idx = 0
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

        xtrain = nat_preproc.flatten_X(X_train[:, :, :, np.newaxis])
        xtest = nat_preproc.flatten_X(X_test[:, :, :, np.newaxis])

        # ============================== PCA ANALYSIS ===============================
        # perform trial averaged PCA (bc pairwise comparison, can only do PCA dim = 2)
        xtrain_trial_average = X_train.mean(axis=1)[:, np.newaxis, :, np.newaxis]
        xtrain_trial_average = xtrain_trial_average.reshape(ncells, -1)
        #xtrain_trial_average = nat_preproc.flatten_X(xtrain_trial_average)

        pca = PCA(n_components=2)
        pca.fit(xtrain_trial_average.T)
        pca_weights = pca.components_

        xtrain_pca = (xtrain.T @ pca_weights.T).T
        xtest_pca = (xtest.T @ pca_weights.T).T

        xtrain_pca = nat_preproc.fold_X(xtrain_pca, nreps=nreps_train, nstim=2, nbins=1).squeeze()
        xtest_pca = nat_preproc.fold_X(xtest_pca, nreps=nreps_test, nstim=2, nbins=1).squeeze()

        pca_train_var = np.var(xtrain_pca.T @ pca_weights)  / np.var(xtrain)
        pca_test_var = np.var(xtest_pca.T @ pca_weights)  / np.var(xtest)

        # compute dprime metrics raw 
        pca_dp_train, pca_wopt_train, pca_evals_train, pca_evecs_train, pca_dU_train = \
                                decoding.compute_dprime(xtrain_pca[:, :, 0], xtrain_pca[:, :, 1])
        pca_dp_test, pca_wopt_test, pca_evals_test, pca_evecs_test, pca_dU_test = \
                                decoding.compute_dprime(xtest_pca[:, :, 0], xtest_pca[:, :, 1])

        # compute dprime metrics diag decoder
        pca_dp_train_diag, pca_wopt_train_diag, _, _, x = \
                                decoding.compute_dprime(xtrain_pca[:, :, 0], xtrain_pca[:, :, 1], diag=True)
        pca_dp_test_diag, pca_wopt_test_diag, _, _, _ = \
                                decoding.compute_dprime(xtest_pca[:, :, 0], xtest_pca[:, :, 1], diag=True)

        # caculate additional metrics
        dU_mag_train         = np.linalg.norm(pca_dU_train)
        dU_dot_evec_train    = pca_dU_train.dot(pca_evecs_train)
        cos_dU_wopt_train    = abs(decoding.unit_vectors(pca_dU_train.T).T.dot(decoding.unit_vectors(pca_wopt_train)))
        dU_dot_evec_sq_train = pca_dU_train.dot(pca_evecs_train) ** 2
        evec_snr_train       = dU_dot_evec_sq_train / pca_evals_train
        cos_dU_evec_train    = abs(decoding.unit_vectors(pca_dU_train.T).T.dot(pca_evecs_train))

        dU_mag_test         = np.linalg.norm(pca_dU_test)
        dU_dot_evec_test    = pca_dU_test.dot(pca_evecs_test)
        cos_dU_wopt_test    = abs(decoding.unit_vectors(pca_dU_test.T).T.dot(decoding.unit_vectors(pca_wopt_test)))
        dU_dot_evec_sq_test = pca_dU_test.dot(pca_evecs_test) ** 2
        evec_snr_test       = dU_dot_evec_sq_test / pca_evals_test
        cos_dU_evec_test    = abs(decoding.unit_vectors(pca_dU_test.T).T.dot(pca_evecs_test))

        pca_results.loc[pca_idx] = [pca_dp_test, pca_dp_test_diag, pca_wopt_test, pca_wopt_test_diag, pca_test_var,
                                    pca_evals_test, pca_evecs_test, pca_dU_test,
                                    pca_dp_train, pca_dp_train_diag, pca_wopt_train, pca_wopt_train_diag, pca_train_var,
                                    pca_evals_train, pca_evecs_train, pca_dU_train,
                                    dU_mag_test, dU_dot_evec_test, cos_dU_wopt_test, dU_dot_evec_sq_test, evec_snr_test, cos_dU_evec_test,
                                    dU_mag_train, dU_dot_evec_train, cos_dU_wopt_train, dU_dot_evec_sq_train, evec_snr_train, cos_dU_evec_train,
                                    ev_set, 2, combo, category, site]
        
        pca_idx += 1


        # ============================== PLS ANALYSIS ===============================
        for n_components in range(2, components):
            #log.info("PLS component {0} / {1}".format(n_components, components))
            # pls 
            try:
                Y = dr.get_one_hot_matrix(ncategories=2, nreps=nreps_train)
                pls = PLSRegression(n_components=n_components, max_iter=500, tol=1e-7)
                pls.fit(xtrain.T, Y.T)
                pls_weights = pls.x_weights_
                pad = False
            except np.linalg.LinAlgError:
                # deflated matrix on this iteration of NIPALS was ~0. e.g. the overall matrix rank may have been 6, but
                # by the time it gets to this iteration, the last couple of indpendent dims are so small, that matrix
                # is essentially 0 and PLS can't converge.
                log.info("PLS can't converge. No more dimensions in the deflated matrix. Pad with nan and continue. \n"
                              "jack_idx: {0} \n"
                              "n_components: {1} \n "
                              "stim category: {2} \n "
                              "stim combo: {3}".format(ev_set, n_components, category, combo))
                pad = True

            if not pad:
                xtrain_pls = (xtrain.T @ pls_weights).T
                xtest_pls = (xtest.T @ pls_weights).T

                if np.linalg.matrix_rank(xtrain_pls) < n_components:
                    # add one more check - this will cause singular matrix. dprime fn handles this,
                    # but to prevent a barrage of log messages, check here to prevent even attempting dprime calc.
                    # what this means is that the last dim(s) of x_weights are 0. i.e. there is no more explainable 
                    # information about Y in X.
                    evec_nan = np.nan * np.ones((n_components, n_components))
                    eval_nan = np.nan * np.ones(n_components)
                    dU_nan = np.nan * np.ones((1, n_components))
                    wopt_nan = np.nan * np.ones((n_components, 1)) 
                    pls_results.loc[pls_idx] = [np.nan, np.nan, wopt_nan, wopt_nan, np.nan,
                                eval_nan, evec_nan, dU_nan,
                                np.nan, np.nan, wopt_nan, wopt_nan, np.nan,
                                eval_nan, evec_nan, dU_nan,
                                np.nan, dU_nan, np.nan, dU_nan, dU_nan, dU_nan,
                                np.nan, dU_nan, np.nan, dU_nan, dU_nan, dU_nan,
                                ev_set, n_components, combo, category, site]


                else:
                    xtrain_pls = nat_preproc.fold_X(xtrain_pls, nreps=nreps_train, nstim=2, nbins=1).squeeze()
                    xtest_pls = nat_preproc.fold_X(xtest_pls, nreps=nreps_test, nstim=2, nbins=1).squeeze()
        
                    pls_train_var = np.var(xtrain_pls.T @ pls_weights.T)  / np.var(xtrain)
                    pls_test_var = np.var(xtest_pls.T @ pls_weights.T)  / np.var(xtest)

                    # compute dprime metrics raw 
                    pls_dp_train, pls_wopt_train, pls_evals_train, pls_evecs_train, pls_dU_train = \
                                            decoding.compute_dprime(xtrain_pls[:, :, 0], xtrain_pls[:, :, 1])
                    pls_dp_test, pls_wopt_test, pls_evals_test, pls_evecs_test, pls_dU_test = \
                                            decoding.compute_dprime(xtest_pls[:, :, 0], xtest_pls[:, :, 1])

                    # compute dprime metrics diag decoder
                    pls_dp_train_diag, pls_wopt_train_diag, _, _, _ = \
                                            decoding.compute_dprime(xtrain_pls[:, :, 0], xtrain_pls[:, :, 1], diag=True)
                    pls_dp_test_diag, pls_wopt_test_diag, _, _, _ = \
                                            decoding.compute_dprime(xtest_pls[:, :, 0], xtest_pls[:, :, 1], diag=True)

                    # caculate additional metrics
                    dU_mag_train         = np.linalg.norm(pls_dU_train)
                    dU_dot_evec_train    = pls_dU_train.dot(pls_evecs_train)
                    cos_dU_wopt_train    = abs(decoding.unit_vectors(pls_dU_train.T).T.dot(decoding.unit_vectors(pls_wopt_train)))
                    dU_dot_evec_sq_train = pls_dU_train.dot(pls_evecs_train) ** 2
                    evec_snr_train       = dU_dot_evec_sq_train / pls_evals_train
                    cos_dU_evec_train    = abs(decoding.unit_vectors(pls_dU_train.T).T.dot(pls_evecs_train))

                    dU_mag_test         = np.linalg.norm(pls_dU_test)
                    dU_dot_evec_test    = pls_dU_test.dot(pls_evecs_test)
                    cos_dU_wopt_test    = abs(decoding.unit_vectors(pls_dU_test.T).T.dot(decoding.unit_vectors(pls_wopt_test)))
                    dU_dot_evec_sq_test = pls_dU_test.dot(pls_evecs_test) ** 2
                    evec_snr_test       = dU_dot_evec_sq_test / pls_evals_test
                    cos_dU_evec_test    = abs(decoding.unit_vectors(pls_dU_test.T).T.dot(pls_evecs_test))

                    pls_results.loc[pls_idx] = [pls_dp_test, pls_dp_test_diag, pls_wopt_test, pls_wopt_test_diag, pls_test_var,
                                                pls_evals_test, pls_evecs_test, pls_dU_test,
                                                pls_dp_train, pls_dp_train_diag, pls_wopt_train, pls_wopt_train_diag, pls_train_var,
                                                pls_evals_train, pls_evecs_train, pls_dU_train,
                                                dU_mag_test, dU_dot_evec_test, cos_dU_wopt_test, dU_dot_evec_sq_test, evec_snr_test, cos_dU_evec_test,
                                                dU_mag_train, dU_dot_evec_train, cos_dU_wopt_train, dU_dot_evec_sq_train, evec_snr_train, cos_dU_evec_train,
                                                ev_set, n_components, combo, category, site]


            else:
                evec_nan = np.nan * np.ones((n_components, n_components))
                eval_nan = np.nan * np.ones(n_components)
                dU_nan = np.nan * np.ones((1, n_components))
                wopt_nan = np.nan * np.ones((n_components, 1)) 
                pls_results.loc[pls_idx] = [np.nan, np.nan, wopt_nan, wopt_nan, np.nan,
                            eval_nan, evec_nan, dU_nan,
                            np.nan, np.nan, wopt_nan, wopt_nan, np.nan,
                            eval_nan, evec_nan, dU_nan,
                            np.nan, dU_nan, np.nan, dU_nan, dU_nan, dU_nan,
                            np.nan, dU_nan, np.nan, dU_nan, dU_nan, dU_nan,
                            ev_set, n_components, combo, category, site]

            pls_idx += 1

# convert columns to str
pca_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in pca_results.combo.values]
pls_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in pls_results.combo.values]

# convert to correct dtypes
dtypes = {'dp_opt_test': 'float64',
            'dp_diag_test': 'float64',
            'wopt_test': 'object',
            'wdiag_test': 'object',
            'var_explained_test': 'float64',
            'evals_test': 'object',
            'evecs_test': 'object',
            'dU_test': 'object',
            'dp_opt_train': 'float64',
            'dp_diag_train': 'float64',
            'wopt_train': 'object',
            'wdiag_train': 'object',
            'var_explained_train': 'float64',
            'evals_train': 'object',
            'evecs_train': 'object',
            'dU_train': 'object',
            'dU_mag_test': 'float64',
            'dU_dot_evec_test': 'object',
            'cos_dU_wopt_test': 'float64',
            'dU_dot_evec_sq_test': 'object',
            'evec_snr_test': 'object', 
            'cos_dU_evec_test': 'object',
            'dU_mag_train': 'float64',
            'dU_dot_evec_train': 'object',
            'cos_dU_wopt_train': 'float64',
            'dU_dot_evec_sq_train': 'object',
            'evec_snr_train': 'object', 
            'cos_dU_evec_train': 'object',
            'category': 'category',
            'jack_idx': 'category',
            'n_components': 'category',
            'combo': 'category',
            'site': 'category'}
pca_results = pca_results.astype(dtypes)
pls_results = pls_results.astype(dtypes)

# collapse over results to save disk space by packing into "DecodingResults object"
log.info("Compressing results into DecodingResults object... ")
pca_results = decoding.DecodingResults(pca_results)
pls_results = decoding.DecodingResults(pls_results)

# save results
log.info("Saving results to {}".format(path))
if not os.path.isdir(os.path.join(path, site)):
    os.mkdir(os.path.join(path, site))

pls_results.save_pickle(os.path.join(path, site, modelname+'_PLS.pickle'))
pca_results.save_pickle(os.path.join(path, site, modelname+'_PCA.pickle'))

if queueid:
    nd.update_job_complete(queueid)
