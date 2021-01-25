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
site = 'DRX007a.e65:128'
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

f = plt.figure(figsize=(16, 8))

spec = plt.subplot2grid((4, 9), (0, 0), colspan=5)
resp = plt.subplot2grid((4, 9), (1, 0), colspan=5)
beta = plt.subplot2grid((4, 9), (1, 5), colspan=4)
scree = plt.subplot2grid((4, 9), (0, 5), colspan=4)
confusion = plt.subplot2grid((4, 9), (2, 0), colspan=3, rowspan=3)
confusion2 = plt.subplot2grid((4, 9), (2, 3), colspan=3, rowspan=3)
confusion3 = plt.subplot2grid((4, 9), (2, 6), colspan=3, rowspan=3)

spec.set_title('Spectrogram')
spec.imshow(np.sqrt(stim), origin='lower', cmap='Greys', aspect='auto')

resp.set_title("Response")
resp.plot(range(proj.shape[0]), proj.mean(axis=1)[:, 0], label='PC1', lw=2)
resp.plot(range(proj.shape[0]), proj.mean(axis=1)[:, 1], label='PC2', lw=2)
resp.plot(range(proj.shape[0]), proj.mean(axis=1)[:, 2], label='PC3', lw=2)
resp.set_xlim((0, proj.shape[0]))
resp.legend(frameon=False)
resp.axhline(0, linestyle='--', color='grey', lw=2)

scree.stem(range(pca.components_.shape[0]), pca.explained_variance_ratio_, markerfmt='.', basefmt=" ")
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

# plot confusion matrix
df['delta'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
plot_confusion_matrix(df, 
                    metric='delta',
                    spectrogram=np.sqrt(stim),
                    resp_fs=4,
                    stim_fs=100,
                    ax=confusion
                    )
confusion.set_title(r"$\Delta d'^2$")

plot_confusion_matrix(df, 
                    metric='noiseAlign',
                    spectrogram=np.sqrt(stim),
                    resp_fs=4,
                    stim_fs=100,
                    ax=confusion2,
                    vmin=0,
                    vmax=1,
                    cmap='Reds'
                    )
confusion2.set_title(r"$cos(\theta_{\Delta \mu, e_1})$")

plot_confusion_matrix(df, 
                    metric='dU_mag_test',
                    spectrogram=np.sqrt(stim),
                    resp_fs=4,
                    stim_fs=100,
                    ax=confusion3,
                    vmin=0,
                    cmap='Reds'
                    )
confusion3.set_title(r"$|\Delta \mu|$")

f.tight_layout()

# look at a couple example pairs that do very different things
# 51 vs. many others (dprime decrease)
# 46 vs. many others (dprime increase)
# 46 / 51 have similar, big, PC1 responses

pairs = [
    [46, 47],
    [47, 51],
    [46, 48],
    [48, 51],
    [46, 49],
    [49, 51],
    [46, 50],
    [50, 51]
]
ylim = (-7, 7)
xlim = (-7, 7)
f, ax = plt.subplots(4, 2, figsize=(6, 12))

for a, p in zip(ax.flatten(), pairs):

    decoding.plot_stimulus_pair(site, batch, pair=p,
                                axlabs=[r"$dDR_1$", r"$dDR_2$"],
                                ylim=ylim,
                                xlim=xlim,
                                ellipse=True, 
                                pup_split=True,
                                ax=a, 
                                title_string=f"{p[0]} vs. {p[1]}")
f.tight_layout()

plt.show()