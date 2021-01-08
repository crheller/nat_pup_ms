"""
Can we characterize the 'tuning curves' of each population?
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH

import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import os
import pandas as pd 
import pickle
import sys
import matplotlib.pyplot as plt

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.dim_reduction as dr

import nems
import nems_lbhb.baphy as nb
import nems.db as nd


site = 'TAR010c'
batch = 289
recache = False # decoding results

X, sp_bins, X_pup, pup_mask = decoding.load_site(site=site, batch=batch)
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

# ========================= GET FIRST PC OF EVOKED RESPONSES ============================
Xu = Xev.mean(axis=1)
Xu_center = Xu - Xu.mean(axis=0, keepdims=True)
pca = PCA(n_components=1)
pca.fit(Xu_center.T)

# single trials projected onto PC1
Xpc1 = X.T.dot(pca.components_.T).squeeze()

Xpc1u = Xpc1.mean(axis=1)

loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
df = results.numeric_results.loc[results.evoked_stimulus_pairs]

# add pc1 resps to df
df.at[:, 'pc1_c1'] = [Xpc1u[int(c.split('_')[0])] for c in df.index.get_level_values('combo')]
df.at[:, 'pc1_c2'] = [Xpc1u[int(c.split('_')[1])] for c in df.index.get_level_values('combo')]

cc = np.corrcoef(np.abs(df['pc1_c1'].values - df['pc1_c2'].values), df['dp_opt_test'].values)[0, 1]
# diff in resp along first PC predict overall dprime
f, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.scatter(np.abs(df['pc1_c1'].values - df['pc1_c2'].values), df['dp_opt_test'].values, s=20, edgecolor='white')
ax.set_xlabel('r1 - r2 (on PC1')
ax.set_ylabel('d-prime')

f.tight_layout()
plt.show()