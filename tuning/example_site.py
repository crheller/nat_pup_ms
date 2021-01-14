"""
Show diverse responses in PC space for single site. Maybe with example rasters? Idea would
be to illustrate that different PCs (axes) correspond to different neurons being active
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
from regression_helper import fit_OLS_model

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


site = 'DRX006b.e1:64'
batch = 289
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'

# get decoding results
loader = decoding.DecodingResults()
fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
df = results.numeric_results.loc[results.evoked_stimulus_pairs]

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

# ============================= DO PCA ================================
Xu = Xev.mean(axis=1)
spont = X[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
Xu_center = Xu - spont # subtract spont
pca = PCA()
pca.fit(Xu_center.T)

#plt.bar(range(X.shape[0]), pca.explained_variance_ratio_, edgecolor='k', width=0.5, color='lightgrey')

spont = spont[:, :, np.newaxis] # for subtracting from single trial data
X_spont = X - spont
proj = (X_spont).T.dot(pca.components_.T)

# load spectrogram
loadkey = "ozgf.fs100.ch18.pup"
uri = generate_recording_uri(cellid=site[:7], batch=batch, loadkey=loadkey)
rec = load_recording(uri)
# excise poststim
postim_bins = rec['pupil'].extract_epoch('PostStimSilence').shape[-1]
stim = []
for e in epochs:
    stim.append(rec['stim']._data[e][:, :(-1 * postim_bins)])
stim = np.concatenate(stim, axis=-1)

f = plt.figure(figsize=(16, 4))

spec = plt.subplot2grid((2, 9), (0, 0), colspan=5)
resp = plt.subplot2grid((2, 9), (1, 0), colspan=5)
beta = plt.subplot2grid((2, 9), (1, 5), colspan=4)
scree = plt.subplot2grid((2, 9), (0, 5), colspan=4)

spec.set_title('Spectrogram')
spec.imshow(np.sqrt(stim), cmap='Greys', aspect='auto')

resp.set_title("Response")
resp.plot(range(proj.shape[0]), proj.mean(axis=1)[:, 0], label='PC1', lw=2)
resp.plot(range(proj.shape[0]), proj.mean(axis=1)[:, 1], label='PC2', lw=2)
resp.plot(range(proj.shape[0]), proj.mean(axis=1)[:, 2], label='PC3', lw=2)
resp.set_xlim((0, proj.shape[0]))
resp.legend(frameon=False)
resp.axhline(0, linestyle='--', color='grey', lw=2)

scree.stem(range(X.shape[0]), pca.explained_variance_ratio_, markerfmt='.', basefmt=" ")
scree.set_ylim((0, None))
scree.set_ylabel('Var explained')
scree.set_xlabel('PC')

# plot weights
markerline, stemlines, baseline = beta.stem(np.arange(0, X.shape[0]), pca.components_[0, :], linefmt='tab:blue', markerfmt='.', basefmt=" ")
markerline.set_markerfacecolor('tab:blue')
markerline.set_markeredgecolor('None')
markerline, stemlines, baseline = beta.stem(np.arange(0, X.shape[0])+0.3, pca.components_[1, :], linefmt='tab:orange', markerfmt='.', basefmt=" ")
markerline.set_markerfacecolor('tab:orange')
markerline.set_markeredgecolor('None')
markerline, stemlines, baseline = beta.stem(np.arange(0, X.shape[0])+0.6, pca.components_[2, :], linefmt='tab:green', markerfmt='.', basefmt=" ")
markerline.set_markerfacecolor('tab:green')
markerline.set_markeredgecolor('None')

beta.axhline(0, linestyle='--', color='grey')
beta.set_xlabel("Unit")
beta.set_ylabel("Loading weight")

f.tight_layout()

plt.show()

