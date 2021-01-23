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


# for each stimulus plot ellipse
f, ax = plt.subplots(1, 3, figsize=(12, 4))

sidx = np.argsort((proj**2).sum(axis=-1).mean(axis=1))
samples = sidx[::-1][[1, 7]]#[[0, 6]]
idx = 0
for i in range(nstim):
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
    ax[0].plot(el[0], el[1], lw=lw, color=color, zorder=zorder)
    el = cplt.compute_ellipse(r[~bp, 0], r[~bp, 1])
    ax[1].plot(el[0], el[1], lw=lw, color=color, zorder=zorder)

ax[0].axhline(0, linestyle='--', color='k', zorder=-1); ax[0].axvline(0, linestyle='--', color='k', zorder=-1)
ax[1].axhline(0, linestyle='--', color='k', zorder=-1); ax[1].axvline(0, linestyle='--', color='k', zorder=-1)

ax[0].set_xlabel(r"$PC_1$")
ax[0].set_ylabel(r"$PC_2$")
ax[0].set_title("Large pupil")
ax[1].set_xlabel(r"$PC_1$")
ax[1].set_ylabel(r"$PC_2$")
ax[1].set_title("Small pupil")

# share axes
extents = np.array(ax[0].get_xlim() + ax[1].get_xlim() + ax[0].get_ylim() + ax[1].get_ylim())
ax[0].set_ylim((extents.min(), extents.max()))
ax[1].set_ylim((extents.min(), extents.max()))
ax[0].set_xlim((extents.min(), extents.max()))
ax[1].set_xlim((extents.min(), extents.max()))

# plot scree plot
ax[2].bar(range(10), pca.explained_variance_ratio_[:10] * 100, edgecolor='k', color='lightgrey', width=0.5)
ax[2].set_ylabel('Variance explained')
ax[2].set_xlabel(r"$PC$")

f.tight_layout()

plt.show()