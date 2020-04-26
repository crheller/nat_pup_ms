"""
For a given site:
    Project into a low-D space(s) (from 1 dim to N dims where N = number of neurons)
"""

import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.baphy as nb
from sklearn.cross_decomposition import PLSRegression
from sklearn .decomposition import PCA
from sklearn.preprocessing import scale
from charlieTools.plotting import compute_ellipse
import dim_reduction.helpers as hp
import pandas as pd
from itertools import combinations, product

# =============================== set analysis parameters ===============================
site = 'TAR010c'
batch = 289
fs = 4
z_score = False
jk_sets = 10
single_stim = None 

"""
[('STIM_00ferretmixed41.wav', 0, 1),
                     ('STIM_00ferretmixed41.wav', 1, 2),
                     ('STIM_00ferretmixed41.wav', 2, 5),
                     ('STIM_00ferretmixed41.wav', 3, 6),
                     ('STIM_00ferretmixed41.wav', 6, 7),
                     ('STIM_00ferretmixed41.wav', 10, 11)]
"""
# =======================================================================================
# load recording
options = {'cellid': site, 'rasterfs': fs, 'batch': batch, 'pupil': True, 'stim': False}
rec = nb.baphy_load_recording_file(**options)
rec['resp'] = rec['resp'].rasterize()

# add mask for removing pre/post stim silence
rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

# extract folded matrices and build single response matrix, where each bin is a "stimulus"
epochs = [epoch for epoch in rec.epochs.name.unique() if 'STIM_00' in epoch]
rec = rec.and_mask(epochs)
resp_dict_all= rec['resp'].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
if single_stim is None:
    pairs = [None]
else:
    pairs = single_stim

for single_stim in pairs:
    if single_stim is not None:
        resp_dict = {single_stim[0]: resp_dict_all[single_stim[0]][:, :, [single_stim[1], single_stim[2]]]}
    X = hp.dict_to_X(resp_dict)
    ncells = X.shape[0]
    nreps = X.shape[1]
    nstim = X.shape[2]
    nbins = X.shape[3]

    # split X into train / test set... do we need two training sets?? 
    # One for dim reduction, one for decoding axis? For now, stick w/ 50-50 split
    train_idx = np.random.choice(range(0, nreps), int(nreps / 2), replace=False)
    test_idx = np.array(list(set(range(0, nreps)) - set(train_idx)))
    nreps_train = len(train_idx)
    nreps_test = len(test_idx)
    components = min([ncells, nreps_train])  # nreps is constraining here bc it determines the max rank of the covariance matrix
    columns = ['test_dp', 'test_dp_diag', 'test_var_explained', 'train_dp', 'train_dp_diag', 'train_var_explained', 'n_components', 'jk_idx']
    pls_results = pd.DataFrame(index=range(components*jk_sets), columns=columns)
    pca_results = pd.DataFrame(index=range(components*jk_sets), columns=columns)
    for jk_set in range(jk_sets):
        print("'jackknife set' {0} / {1}".format(jk_set, jk_sets))
        train_idx = np.random.choice(range(0, nreps), int(nreps / 2), replace=False)
        test_idx = np.array(list(set(range(0, nreps)) - set(train_idx)))
        nreps_train = len(train_idx)
        nreps_test = len(test_idx)

        X_train = hp.flatten_X(X[:, train_idx, :, :])
        X_test = hp.flatten_X(X[:, test_idx, :, :])

        # ============================= preprocess train / test data =============================
        # preprocess training data, save params to do the same to test data
        u = X_train.mean(axis=-1)
        sd = X_train.std(axis=-1)
        if z_score:
            xtrain = X_train - u[:, np.newaxis]
            xtrain = xtrain / sd[:, np.newaxis]

            xtest = X_test - u[:, np.newaxis]
            xtest = xtest / sd[:, np.newaxis]
        else:
            # center only
            xtrain = X_train - u[:, np.newaxis]
            xtest = X_test - u[:, np.newaxis]

        xtrain_trial_average = hp.fold_X(xtrain, nreps=nreps_train, nstim=nstim, nbins=nbins).mean(axis=1)[:, np.newaxis, :, :]
        xtrain_trial_average = hp.flatten_X(xtrain_trial_average)

        # loop over number of components and save results
        for n_components in range(2, components):
            print('dimensionality reduction {0} / {1}'.format(n_components, components))
            # ======================= perform dimensionality reduction ===========================
            # pls 
            Y = hp.get_one_hot_matrix(ncategories = nbins * nstim, nreps=nreps_train)
            pls = PLSRegression(n_components=n_components, max_iter=500, tol=1e-7)
            pls.fit(xtrain.T, Y.T)
            pls_weights = pls.x_weights_

            if (single_stim is not None) & (n_components > 2):
                pass
            else:
                # pca (on trial averaged train data)
                pca = PCA(n_components=n_components)
                pca.fit(xtrain_trial_average.T)
                pca_weights = pca.components_

            # ================================= transform data ====================================
            xtrain_pls = (xtrain.T @ pls_weights).T
            xtest_pls = (xtest.T @ pls_weights).T

            if (single_stim is not None) & (n_components > 2):
                pass
            else:
                xtrain_pca = (xtrain.T @ pca_weights.T).T
                xtest_pca = (xtest.T @ pca_weights.T).T

            # get variance explained
            pls_train_var = np.var(xtrain_pls.T @ pls_weights.T)  / np.var(xtrain)
            pls_test_var = np.var(xtest_pls.T @ pls_weights.T)  / np.var(xtest)

            if (single_stim is not None) & (n_components > 2):
                pass
            else:
                pca_train_var = np.var(xtrain_pca.T @ pls_weights.T)  / np.var(xtrain)
                pca_test_var = np.var(xtest_pca.T @ pls_weights.T)  / np.var(xtest)

            # refold, and then reshape to Neurons X Reps X "Stimuli"
            xtrain_pls = hp.fold_X(xtrain_pls, nreps=nreps_train, nstim=nstim, nbins=nbins).reshape(n_components, nreps_train, nstim * nbins)
            xtest_pls = hp.fold_X(xtest_pls, nreps=nreps_test, nstim=nstim, nbins=nbins).reshape(n_components, nreps_test, nstim * nbins)

            if (single_stim is not None) & (n_components > 2):
                pass
            else:
                xtrain_pca = hp.fold_X(xtrain_pca, nreps=nreps_train, nstim=nstim, nbins=nbins).reshape(n_components, nreps_train, nstim * nbins)
                xtest_pca = hp.fold_X(xtest_pca, nreps=nreps_test, nstim=nstim, nbins=nbins).reshape(n_components, nreps_test, nstim * nbins)


            # ================================= compute dprime =====================================
            # get all combinations of stimuli
            combos = list(combinations(range(nstim * nbins), 2))
            pls_dp_train = np.zeros(len(combos))
            pca_dp_train = np.zeros(len(combos))
            pls_dp_test = np.zeros(len(combos))
            pca_dp_test = np.zeros(len(combos))
            pls_dp_train_diag = np.zeros(len(combos))
            pca_dp_train_diag = np.zeros(len(combos))
            pls_dp_test_diag = np.zeros(len(combos))
            pca_dp_test_diag = np.zeros(len(combos))
            for i, combo in enumerate(combos):
                # raw data pls
                atrain_pls = xtrain_pls[:, :, combo[0]]
                btrain_pls = xtrain_pls[:, :, combo[1]]
                wopt, pls_dp_train[i] = hp.compute_dprime(atrain_pls, btrain_pls)

                atest_pls = xtest_pls[:, :, combo[0]]
                btest_pls = xtest_pls[:, :, combo[1]]
                _, pls_dp_test[i] = hp.compute_dprime(atest_pls, btest_pls)                

                # diagonal decoder pls
                wopt, pls_dp_train_diag[i] = hp.compute_dprime(atrain_pls, btrain_pls, diag=True)
                _, pls_dp_test_diag[i] = hp.compute_dprime(atest_pls, btest_pls, diag=True)

                if np.isnan(pls_dp_test[i]) |  np.isnan(pls_dp_train[i]):
                    pls_dp_test[i] = np.nan
                    pls_dp_train[i] = np.nan
                    pls_dp_test[i] = np.nan
                    pls_dp_train_diag[i] = np.nan

                if (single_stim is not None) & (n_components > 2):
                    pass
                else:
                    # raw data pca
                    atrain_pca = xtrain_pca[:, :, combo[0]]
                    btrain_pca = xtrain_pca[:, :, combo[1]]
                    wopt, pca_dp_train[i] = hp.compute_dprime(atrain_pca, btrain_pca)

                    atest_pca = xtest_pca[:, :, combo[0]]
                    btest_pca = xtest_pca[:, :, combo[1]]
                    _, pca_dp_test[i] = hp.compute_dprime(atest_pca, btest_pca)

                    # diagonal decoder pca
                    wopt, pca_dp_train_diag[i] = hp.compute_dprime(atrain_pca, btrain_pca, diag=True)
                    _, pca_dp_test_diag[i] = hp.compute_dprime(atest_pca, btest_pca, diag=True)

                    if np.isnan(pca_dp_test[i]) |  np.isnan(pca_dp_train[i]):
                        pca_dp_test[i] = np.nan
                        pca_dp_train[i] = np.nan
                        pca_dp_test[i] = np.nan
                        pca_dp_train_diag[i] = np.nan


            # ================================ Save results ===========================================
            idx_offset = components * jk_set
            pls_results.loc[n_components + idx_offset] = [np.nanmean(pls_dp_test), np.nanmean(pls_dp_test_diag), pls_test_var,
                                                        np.nanmean(pls_dp_train), np.nanmean(pls_dp_train_diag), pls_train_var,
                                                        n_components, jk_set]
            if (single_stim is not None) & (n_components > 2):
                pass
            else:
                pca_results.loc[n_components + idx_offset] = [np.nanmean(pca_dp_test), np.nanmean(pca_dp_test_diag), pca_test_var,
                                                            np.nanmean(pca_dp_train), np.nanmean(pca_dp_train_diag), pca_train_var, 
                                                            n_components, jk_set]


    # convert to numeric for some weird reason
    pca_results = pca_results.apply(pd.to_numeric).groupby(by=['jk_idx', 'n_components']).mean()  
    pls_results = pls_results.apply(pd.to_numeric).groupby(by=['jk_idx', 'n_components']).mean()  
    f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    pca_vals = pca_results.groupby(by='n_components').mean()
    pca_sem = pca_results.groupby(by='n_components').sem()
    ax[0].set_title('PCA')
    ax[0].plot(pca_vals.index, pca_vals['test_dp'], 'o-', label='test')
    ax[0].fill_between(pca_vals.index, pca_vals['test_dp']-pca_sem['test_dp'], 
                                    pca_vals['test_dp']+pca_sem['test_dp'], alpha=0.5)
    ax[0].plot(pca_vals.index, pca_vals['train_dp'], 'o-', label='train')
    ax[0].fill_between(pca_vals.index, pca_vals['train_dp']-pca_sem['train_dp'], 
                                    pca_vals['train_dp']+pca_sem['train_dp'], alpha=0.5)
    ax[0].set_xlabel('n components')
    ax[0].set_ylabel('dprime')


    pls_vals = pls_results.groupby(by='n_components').mean()
    pls_sem = pls_results.groupby(by='n_components').sem()
    ax[1].set_title('PLS')
    ax[1].plot(pls_vals.index, pls_vals['test_dp'], 'o-', label='test')
    ax[1].fill_between(pls_vals.index, pls_vals['test_dp']-pls_sem['test_dp'], 
                                    pls_vals['test_dp']+pls_sem['test_dp'], alpha=0.5)
    ax[1].plot(pls_vals.index, pls_vals['train_dp'], 'o-', label='train')
    ax[1].fill_between(pls_vals.index, pls_vals['train_dp']-pls_sem['train_dp'], 
                                    pls_vals['train_dp']+pls_sem['train_dp'], alpha=0.5)
    ax[1].set_xlabel('n components')
    ax[1].set_ylabel('dprime')

    # diag decoder results
    ax[0].plot(pca_vals.index, pca_vals['test_dp_diag'], 'o--', label='test')
    ax[0].fill_between(pca_vals.index, pca_vals['test_dp_diag']-pca_sem['test_dp_diag'], 
                                    pca_vals['test_dp_diag']+pca_sem['test_dp_diag'], alpha=0.5)
    ax[0].plot(pca_vals.index, pca_vals['train_dp_diag'], 'o--', label='train')
    ax[0].fill_between(pca_vals.index, pca_vals['train_dp_diag']-pca_sem['train_dp_diag'], 
                                    pca_vals['train_dp_diag']+pca_sem['train_dp_diag'], alpha=0.5)
    ax[0].set_xlabel('n components')
    ax[0].set_ylabel('dprime')
    ax[0].axhline(0, linestyle='--', lw=2, color='lightgrey')

    ax[1].plot(pls_vals.index, pls_vals['test_dp_diag'], 'o--', label='test, diag dec.')
    ax[1].fill_between(pls_vals.index, pls_vals['test_dp_diag']-pls_sem['test_dp_diag'], 
                                    pls_vals['test_dp_diag']+pls_sem['test_dp_diag'], alpha=0.5)
    ax[1].plot(pls_vals.index, pls_vals['train_dp_diag'], 'o--', label='train, diag dec.')
    ax[1].fill_between(pls_vals.index, pls_vals['train_dp_diag']-pls_sem['train_dp_diag'], 
                                    pls_vals['train_dp_diag']+pls_sem['train_dp_diag'], alpha=0.5)
    ax[1].set_xlabel('n components')
    ax[1].set_ylabel('dprime')
    ax[1].axhline(0, linestyle='--', lw=2, color='lightgrey')

    ax[1].legend()

    f.tight_layout()

    if single_stim is not None:
        f.canvas.set_window_title("{0}, {1}, {2}".format(single_stim[0], single_stim[1], single_stim[2]))

plt.show()