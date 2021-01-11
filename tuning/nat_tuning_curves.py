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
import scipy.stats as ss
import statsmodels.api as sm

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.plotting as cplt

import nems
import nems_lbhb.baphy as nb
import nems.db as nd
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems.recording import load_recording
from nems.epoch import epoch_names_matching


site = 'TAR010c' #'DRX006b.e1:64'
batch = 289
recache = False # decoding results
pc = 0 #[0, 1] # 0, 1, 2, [0, 1] (which PC to project on for resp characterization)
# if list, combine them (geometric mean of projection)

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

# ===================== get spectrogram power ==========================
batch=322
loadkey = "ozgf.fs100.ch93.pup"
uri = generate_recording_uri(cellid=site, batch=batch, loadkey=loadkey)
rec = load_recording(uri)

# calculate power in each stim bin (4Hz bins)
bin_len = rec['resp'].fs / 4 # to 4Hz
m = rec['resp'].epoch_to_signal('PostStimSilence').extract_epoch(epochs[0])[0, 0, :]
stim = rec['stim']._data[epochs[0]][:, ~m]
edges = np.arange(0, stim.shape[1]+bin_len, bin_len).astype(int)
pwr = []
for epoch in epochs:
    m = rec['resp'].epoch_to_signal('PostStimSilence').extract_epoch(epoch)[0, 0, :]
    stim = rec['stim']._data[epoch][:, ~m]
    for i in range(len(edges)-1):
        pwr.append((stim[:, edges[i]:edges[i+1]]**1).mean())

# ========================= GET FIRST PC OF EVOKED RESPONSES ============================
Xu = Xev.mean(axis=1)
spont = X[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
Xu_center = Xu - spont # subtract spont
pca = PCA(n_components=10)
pca.fit(Xu_center.T)

spont = X[:, :, spont_bins.squeeze()].mean(axis=(1,2), keepdims=True)

xproj12 = (X-spont).T.dot(pca.components_[0:2, :].T)

# plot PCA results to get "raw" sense of data
f, ax = plt.subplots(1, 2, figsize=(8, 4))


u = xproj12[ev_bins, :, :].mean(axis=1) # mean evoked responses
ax[0].scatter(u[:, 0], u[:, 1])
ax[0].set_xlabel(r"$PC_1$")
ax[0].set_ylabel(r"$PC_2$")
ax[0].set_title('Mean evoked responses')
ax[0].axis('square')

ax[1].set_title("Scree plot")
ax[1].plot(pca.explained_variance_ratio_, 'o-')
ax[1].set_xlabel(r"$PC$")
ax[1].set_ylabel("Var. Explained")


f.tight_layout()

# single trials projected onto PC1
if type(pc) is list:
    Xpc1 = (X - spont).T.dot(pca.components_[pc[0]:pc[-1]+1, :].T).squeeze()
    Xpc1 = np.sqrt(np.sum(Xpc1**2, axis=-1))
else:
    Xpc1 = (X - spont).T.dot(pca.components_[pc, :].T).squeeze()

Xpc1u = Xpc1.mean(axis=1)

loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
df = results.numeric_results.loc[results.evoked_stimulus_pairs]
cosDU = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, 2, [None,0])[0]

# add pc1 resps to df
df.at[:, 'pc1_c1'] = [Xpc1u[int(c.split('_')[0])] for c in df.index.get_level_values('combo')]
df.at[:, 'pc1_c2'] = [Xpc1u[int(c.split('_')[1])] for c in df.index.get_level_values('combo')]
df.at[:, 'spec1'] = [pwr[int(c.split('_')[0])] for c in df.index.get_level_values('combo')]
df.at[:, 'spec2'] = [pwr[int(c.split('_')[1])] for c in df.index.get_level_values('combo')]

# ==========================================================================
# multiple linear regression stim1/2 power and resp1/2 strength / variance(?) vs. dprime stats
X = df[['pc1_c1', 'pc1_c2', 'spec1', 'spec2']]
X = X.rename(columns={'pc1_c1': 'r1', 
                           'pc1_c2': 'r2', 
                           'spec1': 's1',
                           'spec2': 's2'})
X['r1*r2'] = X['r1'] * X['r2']
X['s1*s2'] = X['s1'] * X['s2']
X = sm.add_constant(X)

y = df['bp_dp'] - df['sp_dp']

reg = sm.OLS(y, X).fit()

# crude approach, splitting based on resp alone (top 25 vs. bottom 25)
R = pd.concat([X['r1'], X['r2']])
r1Big = X['r1'] > np.quantile(R, 0.75)
r2Big = X['r2'] > np.quantile(R, 0.75)
r1Small = X['r1'] < np.quantile(R, 0.25)
r2Small = X['r2'] < np.quantile(R, 0.25)

f, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

ax[0].bar([0, 1], [df[r1Big&r2Big]['bp_dp'].mean(),
                    df[r1Big&r2Big]['sp_dp'].mean()], 
                    yerr=[df[r1Big&r2Big]['bp_dp'].sem(),
                    df[r1Big&r2Big]['sp_dp'].sem()],
                    edgecolor='k', width=0.5)
ax[0].set_title('good / good')

ax[1].bar([0, 1], [df[(r1Big&~r2Big) | (~r1Big&r2Big)]['bp_dp'].mean(),
                    df[(r1Big&~r2Big) | (~r1Big&r2Big)]['sp_dp'].mean()],
                    yerr=[df[(r1Big&~r2Big) | (~r1Big&r2Big)]['bp_dp'].sem(),
                    df[(r1Big&~r2Big) | (~r1Big&r2Big)]['sp_dp'].sem()], 
                    edgecolor='k', width=0.5)
ax[1].set_title('good / bad')

ax[2].bar([0, 1], [df[~r1Big&~r2Big]['bp_dp'].mean(),
                    df[~r1Big&~r2Big]['sp_dp'].mean()], 
                    yerr=[df[~r1Big&~r2Big]['bp_dp'].sem(),
                    df[~r1Big&~r2Big]['sp_dp'].sem()],
                    edgecolor='k', width=0.5)
ax[2].set_title('bad / bad')

f.tight_layout()

plt.show()

# ==========================================================================
cc = np.corrcoef(np.abs(df['pc1_c1'].values - df['pc1_c2'].values), df['dp_opt_test'].values)[0, 1]


# resorted df index for heatmaps
vals = np.vstack([df['pc1_c1'].values, df['pc1_c2'].values])
unq = np.unique(vals)
combos = list(combinations(unq, 2))
idx = [np.argwhere((((df['pc1_c1']==c[0]) & (df['pc1_c2']==c[1])) | \
                ((df['pc1_c2']==c[0]) & (df['pc1_c1']==c[1]))).values)[0][0] for c in combos]
df = df.iloc[idx]
cosDU = cosDU.iloc[idx]

flip = [True if (df.iloc[ix]['pc1_c1']==c[0]) & (df.iloc[ix]['pc1_c2']==c[1]) 
                        else False for ix, c in enumerate(combos)]
flv1 = df.loc[flip, 'pc1_c1'] 
flv2 = df.loc[flip, 'pc1_c2']
df.loc[flip, 'pc1_c1'] = flv2
df.loc[flip, 'pc1_c2'] = flv1


# pc1 resp for c1/c2 vs. pupil effects
delta = df['bp_dp'] - df['sp_dp']
delta_norm = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])

cmap = 'PiYG'
nbins = 10
f, ax = plt.subplots(2, 2, figsize=(8, 8))

val = df['dp_opt_test']
heatmap = ss.binned_statistic_2d(x=df['pc1_c1'], 
                        y=df['pc1_c2'],
                        values=val,
                        statistic='mean',
                        bins=nbins)
ax[0, 0].imshow(heatmap.statistic, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[heatmap.x_edge[0], heatmap.x_edge[-1], 
                                            heatmap.y_edge[0], heatmap.y_edge[-1]], norm=cplt.MidpointNormalize(midpoint=0))
ax[0, 0].set_xlabel(r"$PC_1$, $Stim_1$ resp")
ax[0, 0].set_ylabel(r"$PC_1$, $Stim_2$ resp")
ax[0, 0].set_title(r"$d'^2$ (overall)")

val = cosDU.apply(lambda x: x[0]) 
heatmap = ss.binned_statistic_2d(x=df['pc1_c1'], 
                        y=df['pc1_c2'],
                        values=val,
                        statistic='mean',
                        bins=nbins)
ax[0, 1].imshow(heatmap.statistic, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[heatmap.x_edge[0], heatmap.x_edge[-1], 
                                            heatmap.y_edge[0], heatmap.y_edge[-1]], norm=cplt.MidpointNormalize(midpoint=0))
ax[0, 1].set_xlabel(r"$PC_1$, $Stim_1$ resp")
ax[0, 1].set_ylabel(r"$PC_1$, $Stim_2$ resp")
ax[0, 1].set_title(r"Noise / signal alignement")


heatmap = ss.binned_statistic_2d(x=df['pc1_c1'], 
                        y=df['pc1_c2'],
                        values=delta,
                        statistic='mean',
                        bins=nbins)
ax[1, 0].imshow(heatmap.statistic, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[heatmap.x_edge[0], heatmap.x_edge[-1], 
                                            heatmap.y_edge[0], heatmap.y_edge[-1]], norm=cplt.MidpointNormalize(midpoint=0))
ax[1, 0].set_xlabel(r"$PC_1$, $Stim_1$ resp")
ax[1, 0].set_ylabel(r"$PC_1$, $Stim_2$ resp")
ax[1, 0].set_title(r"$\Delta d'^2$ (raw)")

heatmap = ss.binned_statistic_2d(x=df['pc1_c1'], 
                        y=df['pc1_c2'],
                        values=delta_norm,
                        statistic='mean',
                        bins=nbins)
ax[1, 1].imshow(heatmap.statistic, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[heatmap.x_edge[0], heatmap.x_edge[-1], 
                                            heatmap.y_edge[0], heatmap.y_edge[-1]], norm=cplt.MidpointNormalize(midpoint=0))
ax[1, 1].set_xlabel(r"$PC_1$, $Stim_1$ resp")
ax[1, 1].set_ylabel(r"$PC_1$, $Stim_2$ resp")
ax[1, 1].set_title(r"$\Delta d'^2$ (normalized)")
f.tight_layout()

plt.show()