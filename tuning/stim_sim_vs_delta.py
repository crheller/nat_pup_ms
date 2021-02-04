"""
predict change in d-prime based on the stim spectrogram stats
"""


from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
from regression_helper import fit_OLS_model
from tuning.helper import plot_confusion_matrix

import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import os
import pandas as pd 
import pickle
import sys
import matplotlib.pyplot as plt
import scipy.stats as ss
import statsmodels.api as sm
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.plotting as cplt

from nems_lbhb.xform_wrappers import generate_recording_uri
from nems.recording import load_recording


modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
recache = False
site = 'TAR010c' #'DRX008b.e1:64' #'TAR010c' #'DRX007a.e65:128'
batch = 289

for site in HIGHR_SITES:
    if 'BOL' in site:
        batch = 294
        pass
    else:
        batch = 289

        # get decoding results
        loader = decoding.DecodingResults()
        fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
        results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
        df = results.numeric_results.loc[results.evoked_stimulus_pairs]
        df['noiseAlign'] = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, 2, idx=(0, 0))[0]

        X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
        ncells = X.shape[0]
        nreps = X.shape[1]
        nstim = X.shape[2]
        nbins = X.shape[3]
        sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
        nstim = nstim * nbins

        # =========================== generate a list of stim pairs ==========================
        # these are the indices of the decoding results dataframes
        all_combos = list(combinations(range(nstim), 2))
        spont_bins = np.argwhere(sp_bins[0, 0, :])
        spont_combos = [c for c in all_combos if (c[0] in spont_bins) & (c[1] in spont_bins)]
        ev_ev_combos = [c for c in all_combos if (c[0] not in spont_bins) & (c[1] not in spont_bins)]
        spont_ev_combos = [c for c in all_combos if (c not in ev_ev_combos) & (c not in spont_combos)]

        X = X.reshape(ncells, nreps, nstim)
        ev_bins = list(set(range(X.shape[-1])).difference(set(spont_bins.squeeze())))
        Xev = X[:, :, ev_bins]

        # get the overall noise axis and see how it aligned with each dU
        # NOTE! this is imperfect because of cross validation...
        tdr2_axes = nat_preproc.get_first_pc_per_est([X])[0]
        du = results.slice_array_results('dU_all', results.evoked_stimulus_pairs, 2, idx=None)[0].apply(lambda x: x/np.linalg.norm(x))
        df['allNoiseAlign'] = du.apply(lambda x: np.abs(x.dot(tdr2_axes.T))[0][0])
        # ============================= DO PCA ================================
        Xu = Xev.mean(axis=1)
        spont = X[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
        Xu_center = Xu - spont # subtract spont
        pca = PCA()
        pca.fit(Xu_center.T)

        Xm = X.mean(axis=1)

        #plt.bar(range(X.shape[0]), pca.explained_variance_ratio_, edgecolor='k', width=0.5, color='lightgrey')

        spont = spont[:, :, np.newaxis] # for subtracting from single trial data
        X_spont = X - spont
        proj = (X_spont).T.dot(pca.components_.T)

        # load spectrogram
        loadkey = "ozgf.fs100.ch18.pup"
        loadkey = "ozgf.fs4.ch18.pup"
        uri = generate_recording_uri(cellid=site[:7], batch=batch, loadkey=loadkey)
        rec = load_recording(uri)
        # excise poststim
        postim_bins = rec['pupil'].extract_epoch('PostStimSilence').shape[-1]
        stim = []
        for e in epochs:
            stim.append(rec['stim']._data[e][:, :(-1 * postim_bins)])
        stim = np.concatenate(stim, axis=-1)

        stim_var = np.sum((stim - stim.mean(axis=-1, keepdims=True))**2)
        stim_var = np.sum((Xm - Xm.mean(axis=-1, keepdims=True))**2)
        df['delta'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
        #df['stim_sim'] = [sum((stim[:, int(p.split('_')[0])] - stim[:, int(p.split('_')[1])])**2) / stim_var for p in df.index.get_level_values(0)]
        df['stim_sim'] = [sum((Xm[:, int(p.split('_')[0])] - Xm[:, int(p.split('_')[1])])**2) / stim_var for p in df.index.get_level_values(0)]
        df['du_diff'] = df['bp_dU_mag'] - df['sp_dU_mag']
        df['noise_diff'] = results.slice_array_results('sp_evals', results.evoked_stimulus_pairs, 2)[0].apply(lambda x: x.sum()) - \
                                            results.slice_array_results('bp_evals', results.evoked_stimulus_pairs, 2)[0].apply(lambda x: x.sum())

        mask = (df['noise_diff']>-2) & (df['noise_diff'] < 0) & (df['du_diff'] > -.6) & (df['du_diff']<1.2)
        dfplot = df.copy() #df[mask]
        # plot stimulus similarity as a function of delta dU and delta noise
        nbins = 10
        f, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)

        dfplot.plot.hexbin(x='noise_diff', y='du_diff', C='stim_sim', ax=ax[0], gridsize=nbins)
        r = np.corrcoef(df['noise_diff'], df['stim_sim'])[0, 1]
        ax[0].set_xlabel(f"noise: r: {r:.4f}")
        r = np.corrcoef(df['du_diff'], df['stim_sim'])[0, 1]
        ax[0].set_ylabel(f"dU: r: {r:.4f}")

        dfplot.plot.hexbin(x='noise_diff', y='du_diff', C=None, ax=ax[1], gridsize=nbins, cmap='viridis')

        f.tight_layout()

plt.show()