"""
For each site, plot all response distributions (ellipses) for large / small pupil condtions on:
    the top two PCs of evoked responses?
    PC1 and noise axis?
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
import colors
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

np.random.seed(123)
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
recache = False
site = 'TAR010c'
batch = 289

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
pup_mask = pup_mask.reshape(1, nreps, nstim).squeeze()
ev_bins = list(set(range(X.shape[-1])).difference(set(spont_bins.squeeze())))
Xev = X[:, :, ev_bins]

# ============================= DO PCA ================================
Xu = Xev.mean(axis=1)
spont = X[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
Xu_center = Xu - spont # subtract spont
pca = PCA()
pca.fit(Xu_center.T)

spont = spont[:, :, np.newaxis] # for subtracting from single trial data
X_spont = X - spont
proj = (X_spont).T.dot(pca.components_[0:2, :].T)

# only keep evoked data of projection
proj = proj[ev_bins, :, :]

# update pupil mask
pup_mask = pup_mask[:, ev_bins]

# get single trial variance explained along evoked PCs
#      also, for big/small single trial separately
#      and, for big/small evoked
st_Xev = Xev - Xev.mean(axis=1, keepdims=True)
ev_Xev = Xev.mean(axis=1, keepdims=True)

# single trial variance
zm = st_Xev - st_Xev.mean(axis=0, keepdims=True)
tot_var = (zm**2).sum(axis=(1, 2)).sum()
var_explained = np.zeros(pca.components_.shape[0])
bp_var_explained = np.zeros(pca.components_.shape[0])
sp_var_explained = np.zeros(pca.components_.shape[0])

# evoked variance
zm = ev_Xev - ev_Xev.mean(axis=-1, keepdims=True)
ev_tot_var = (zm**2).sum(axis=(1, 2)).sum()
ev_var_explained = np.zeros(pca.components_.shape[0])
ev_bp_var_explained = np.zeros(pca.components_.shape[0])
ev_sp_var_explained = np.zeros(pca.components_.shape[0])

for i in range(0, pca.components_.shape[0]):
    # ==================== SINGLE TRIAL =======================
    # all trials
    fp = st_Xev.T.dot(pca.components_[i, :])
    # big / small pupil
    bp_fp = np.stack([fp[i, pup_mask[:, i]] for i in range(fp.shape[0])])
    sp_fp = np.stack([fp[i, ~pup_mask[:, i]] for i in range(fp.shape[0])])
    
    # all trials
    fp -= fp.mean(axis=(0, 1), keepdims=True)
    var_explained[i] = (fp**2).sum() / tot_var

    # big pupil
    bp_fp -= bp_fp.mean(axis=(0, 1), keepdims=True)
    bp_var_explained[i] = (bp_fp**2).sum() / tot_var

    # small pupil
    sp_fp -= sp_fp.mean(axis=(0, 1), keepdims=True)
    sp_var_explained[i] = (sp_fp**2).sum() / tot_var

    # ====================== EVOKED =======================
    # all trials
    fpev = ev_Xev.T.dot(pca.components_[i, :])
    # big / small pupil
    bp_fp = np.stack([Xev[:, pup_mask[:, i], i] for i in range(0, fp.shape[0])]).transpose([1, -1, 0]).T.dot(pca.components_[i, :]).mean(axis=-1, keepdims=True)
    sp_fp = np.stack([Xev[:, ~pup_mask[:, i], i] for i in range(0, fp.shape[0])]).transpose([1, -1, 0]).T.dot(pca.components_[i, :]).mean(axis=-1, keepdims=True)
    #bp_fp = np.stack([fp[i, pup_mask[:, i]].mean(axis=-1, keepdims=True) for i in range(fp.shape[0])])
    #sp_fp = np.stack([fp[i, ~pup_mask[:, i]].mean(axis=-1, keepdims=True) for i in range(fp.shape[0])])
    
    # all trials
    fpev -= fpev.mean(axis=(0, 1), keepdims=True)
    ev_var_explained[i] = (fpev**2).sum() / ev_tot_var

    # big pupil
    bp_fp -= bp_fp.mean(axis=(0, 1), keepdims=True)
    ev_bp_var_explained[i] = (bp_fp**2).sum() / ev_tot_var

    # small pupil
    sp_fp -= sp_fp.mean(axis=(0, 1), keepdims=True)
    ev_sp_var_explained[i] = (sp_fp**2).sum() / ev_tot_var


# noise pca
Xnoise = X - X.mean(axis=-1, keepdims=True)
Xnoise = Xnoise.reshape(ncells, -1)
npca = PCA()
npca.fit(Xnoise.T)


# for each stimulus plot ellipse
f, ax = plt.subplots(2, 2, figsize=(8, 8))

sidx = np.argsort((proj**2).sum(axis=-1).mean(axis=1))
samples = sidx[::-1][[1, 7]]#[[0, 6]]
idx = 0
for i in range(proj.shape[0]):
    if i in samples:
        if idx==0:
            color = 'tab:blue'
            lw = 2
            zorder = 10
            idx += 1
        elif idx==1:
            color = 'tab:orange'
            lw = 2
            zorder = 10
    else:
        color = 'lightgrey'
        lw = 0.7
        zorder = -1
    r = proj[i, :, :]
    bp = pup_mask[:, i]
    el = cplt.compute_ellipse(r[bp, 0], r[bp, 1])
    ax[0, 0].plot(el[0], el[1], lw=lw, color=color, zorder=zorder)
    el = cplt.compute_ellipse(r[~bp, 0], r[~bp, 1])
    ax[0, 1].plot(el[0], el[1], lw=lw, color=color, zorder=zorder)

ax[0, 0].axhline(0, linestyle='--', color='k', zorder=-1); ax[0, 0].axvline(0, linestyle='--', color='k', zorder=-1)
ax[0, 1].axhline(0, linestyle='--', color='k', zorder=-1); ax[0, 1].axvline(0, linestyle='--', color='k', zorder=-1)

ax[0, 0].set_xlabel(r"Stim $PC_1$")
ax[0, 0].set_ylabel(r"Stim $PC_2$")
ax[0, 0].set_title("Large pupil")
ax[0, 1].set_xlabel(r"Stim $PC_1$")
ax[0, 1].set_ylabel(r"Stim $PC_2$")
ax[0, 1].set_title("Small pupil")

# share axes
extents = np.array(ax[0, 0].get_xlim() + ax[0, 1].get_xlim() + ax[0, 0].get_ylim() + ax[0, 1].get_ylim())
ax[0, 0].set_ylim((extents.min(), extents.max()))
ax[0, 1].set_ylim((extents.min(), extents.max()))
ax[0, 0].set_xlim((extents.min(), extents.max()))
ax[0, 1].set_xlim((extents.min(), extents.max()))

# plot scree plot
ax[1, 0].bar(range(10), pca.explained_variance_ratio_[:10] * 100, edgecolor='k', color='lightgrey', width=0.5, label='Stimulus Activity')
ax[1, 0].bar(range(10), var_explained[:10] * 100, edgecolor='k', color='tab:orange', width=0.5, label='Single Trial Variance')
ax[1, 0].set_ylabel('% Variance explained')
ax[1, 0].set_xlabel(r"Stim $PC$")
ax[1, 0].legend(frameon=False)

# plot pupil-dependent variance on the stim PCs
ax[1, 1].plot(range(10), bp_var_explained[:10] * 100, color=colors.LARGE, label='Single Trial (Large)')
ax[1, 1].plot(range(10), sp_var_explained[:10] * 100, color=colors.SMALL, label='Single Trial (Small)')
ax[1, 1].plot(range(10), ev_bp_var_explained[:10] * 100 / 2, color=colors.LARGE, linestyle='--', label='Stimulus activity (Large)')
ax[1, 1].plot(range(10), ev_sp_var_explained[:10] * 100 / 2, color=colors.SMALL, linestyle='--', label='Stimulus active (Small)')

ax[1, 1].set_ylabel('% Variance explained')
ax[1, 1].set_xlabel(r"Stim $PC$")
ax[1, 1].legend(frameon=False)

f.tight_layout()

plt.show()