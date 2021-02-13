"""
Predict delta dprime from first order response stats / 
    more complicated population metrics
"""

from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
from regression_helper import fit_OLS_model
import figure_scripts2.helper as chelp

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
from charlieTools.statistics import get_direct_prob

import pandas as pd
import seaborn as sns
import os
import statsmodels.api as sm
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.io import wavfile
import nems.analysis.gammatone.gtgram as gt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

nPCs = 2
remove_ns_changes = False
recache = False
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'

regressors = ['pc1_mean', 'pc1_diff',
                    'dU_mag_test', 'noiseAlign']
xlab = [r'$PC_1$', r'$PC_1$ (diff)',
            r"$|\Delta \mu|$", r"$|cos(\theta_{e_{1}, \Delta \mu})|$"]

big_df = []
for i, site in enumerate(HIGHR_SITES):
    if 'BOL' in site:
        batch = 294
    else:
        batch = 289

    # get decoding results
    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    df = results.numeric_results.loc[results.evoked_stimulus_pairs]
    bp = results.slice_array_results('bp_evals', results.evoked_stimulus_pairs, 2, idx=None)[0]
    sp = results.slice_array_results('sp_evals', results.evoked_stimulus_pairs, 2, idx=None)[0]
    df['noise_diff'] = bp.apply(lambda x: x.sum()) - sp.apply(lambda x: x.sum())
    df['site'] = site

    # mask df based on significant change in dprime
    err = np.sqrt(df['bp_dp_sem']**2 + df['sp_dp_sem']**2)
    if remove_ns_changes:
        mask = err < abs(df['bp_dp'] - df['sp_dp'])
        df = df[mask]
    
    X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
    ncells = X.shape[0]
    nreps = X.shape[1]
    nstim = X.shape[2]
    nbins = X.shape[3]
    sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
    nstim = nstim * nbins
    spont_bins = np.argwhere(sp_bins[0, 0, :])
    X = X.reshape(ncells, nreps, nstim)
    pup_mask = pup_mask.reshape(1, nreps, nstim)
    ev_bins = list(set(range(X.shape[-1])).difference(set(spont_bins.squeeze())))
    Xev = X[:, :, ev_bins]

    # ========================== get alignement with overall noise axis ================
    df['noiseAlign'] = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, 2, idx=(0, 0))[0]

    # ============================= DO PCA ================================
    Xu = Xev.mean(axis=1)
    spont = X[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
    Xu_center = Xu - spont # subtract spont
    pca = PCA()
    pca.fit(Xu_center.T)

    spont = spont[:, :, np.newaxis] # for subtracting from single trial data
    X_spont = X - spont
    proj = (X_spont).T.dot(pca.components_.T)

    # ====================== GET RESPONSES ON PCs ==========================
    # for example site alone
    for pc in range(nPCs):
        df[f'pc{pc+1}_r1'] = [proj[int(idx.split('_')[0]), :, pc].mean() for idx in df.index.get_level_values(0)]
        df[f'pc{pc+1}_r2'] = [proj[int(idx.split('_')[1]), :, pc].mean() for idx in df.index.get_level_values(0)]
        df[f'pc{pc+1}_diff'] = np.abs(df[f'pc{pc+1}_r1'] - df[f'pc{pc+1}_r2'])
        df[f'pc{pc+1}_mean'] = (df[f'pc{pc+1}_r1'] + df[f'pc{pc+1}_r2']) / 2

    big_df.append(df)

big_df = pd.concat(big_df)
big_df['delt'] = (big_df['bp_dp']-big_df['sp_dp']) / (big_df['bp_dp'] + big_df['sp_dp'])

# instead of doing regression per-site, resample with hierarchical bootstrap to 
# get distribution over params. For each resample, fit the OLS model and save rsq / param values
np.random.seed(123)
njack = 10
even_sample = True
nboot = 500
for i in np.arange(nboot):
    print(f"Bootstrap {i} / {nboot}")
    temp = []
    num_lev1 = len(big_df.site.unique()) # n animals
    num_lev2 = max([big_df[big_df.site==s].shape[0] for s in big_df.site.unique()]) # min number of observations sampled for an animal
    rand_lev1 = np.random.choice(num_lev1, num_lev1)
    lev1_keys = np.array(list(big_df.site.unique()))[rand_lev1]
    for k in lev1_keys:
        # for this animal, how many obs to choose from?
        this_n_range = big_df[big_df.site==k].shape[0]
        if even_sample:
            rand_lev2 = np.random.choice(this_n_range, num_lev2, replace=True)
        else:
            rand_lev2 = np.random.choice(this_n_range, this_n_range, replace=True)
        
        temp.append(big_df[big_df.site==k].iloc[rand_lev2]) 
    
    # fit cross-validated model on the resampled data (no need for resampling here, because we did it above)
    _df = pd.concat(temp)
    X = _df[regressors]
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)

    y = _df['delt']
    y -= y.mean()
    y /= y.std()

    output = fit_OLS_model(X, y, replace=False, nboot=1, njacks=njack)

    # save r2 values and coefficients
    if i == 0:
        r2 = pd.DataFrame(output['r2'], index=[i])
        coef = pd.DataFrame(output['coef'], index=[i])
    else:
        r2 = pd.concat((r2, pd.DataFrame(output['r2'], index=[i])))
        coef = pd.concat((coef, pd.DataFrame(output['coef'], index=[i])))


f, ax = plt.subplots(1, 2, figsize=(8, 4))

# plot r2
r2p = r2[[k for k in r2 if (k=='full') | (k.startswith('u'))]]
sns.boxplot(x='variable', y='value', data=r2p.melt(), 
                        color='lightgrey', width=0.3, showfliers=False, linewidth=2, ax=ax[0])
ax[0].axhline(0, linestyle='--', color='k')
ax[0].set_xlabel("Regressor")
ax[0].set_ylabel(r"$R^2$ (unique)")
ax[0].set_xticks(range(r2p.shape[1]))
ax[0].set_xticklabels(['Full Model'] + xlab, rotation=45)
# add pvalue for each regressor
ym = ax[0].get_ylim()[-1]
for i, r in enumerate(r2p.keys()):
    p = get_direct_prob(r2[r], np.zeros(nboot))[0]
    ax[0].text(i-0.3, ym, f"p={p:.4f}", fontsize=6)

# plot coefficients
sns.boxplot(x='variable', y='value', data=coef.melt(), 
                         color='lightgrey', width=0.3, showfliers=False, linewidth=2, ax=ax[1])
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_xlabel("Regressor")
ax[1].set_ylabel("Regression coefficient (normalized)")
ax[1].set_xticks(range(coef.shape[1]))
ax[1].set_xticklabels(xlab, rotation=45)
# add pvalue for each regressor
ym = ax[1].get_ylim()[-1]
for i, r in enumerate(coef.keys()):
    p = get_direct_prob(coef[r], np.zeros(nboot))[0]
    ax[1].text(i-0.3, ym, f"p={p:.4f}", fontsize=6)

f.tight_layout()

plt.show()