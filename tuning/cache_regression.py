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

nPCs = 1
nboot = 1
replace = False # don't resample the raw data if false / nboot=1.
njack = 10
interactions = True
null_params = False # if true, include DU diff / noise diff in model
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
recache = False

regressors = ['dU_mag_test', 'noiseAlign'] #['r1', 'r2', 'dU_mag_test', 'noiseAlign', 'mean_pupil_range']

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

    # mask df based on significant change in dprime
    err = np.sqrt(df['bp_dp_sem']**2 + df['sp_dp_sem']**2)
    mask = err < abs(df['bp_dp'] - df['sp_dp'])
    df = df[mask]
    
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
    pup_mask = pup_mask.reshape(1, nreps, nstim)
    ev_bins = list(set(range(X.shape[-1])).difference(set(spont_bins.squeeze())))
    Xev = X[:, :, ev_bins]

    # ========================== get alignement with overall noise axis ================
    df['noiseAlign'] = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, 2, idx=(0, 0))[0]
    #tdr2_axes = nat_preproc.get_first_pc_per_est([X])[0].T
    #fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/nc_zscore_lvs.pickle'
    #with open(fn, 'rb') as handle:
    #    lv_dict = pickle.load(handle)
    #tdr2_axes = lv_dict[site]['beta2'].T
    #du = results.slice_array_results('dU_all', results.evoked_stimulus_pairs, 2, idx=None)[0].apply(lambda x: x/np.linalg.norm(x))
    #df['noiseAlign'] = du.apply(lambda x: np.abs(x.dot(tdr2_axes))[0][0])

    # ============================= DO PCA ================================
    Xu = Xev.mean(axis=1)
    spont = X[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
    Xu_center = Xu - spont # subtract spont
    pca = PCA()
    pca.fit(Xu_center.T)

    spont = spont[:, :, np.newaxis] # for subtracting from single trial data
    X_spont = X - spont
    proj = (X_spont).T.dot(pca.components_.T)

    # ====================== DO REGRESSION ANALYSIS WITH CROSS-VAL ==========================
    # for example site alone
    if nPCs > 1:
        raise ValueError("Can't do this yet -- need to update...")
    
    df['r1'] = [proj[int(idx.split('_')[0]), :, 0].mean() for idx in df.index.get_level_values(0)]
    df['r2'] = [proj[int(idx.split('_')[1]), :, 0].mean() for idx in df.index.get_level_values(0)]
    df['du_diff'] = df['bp_dU_mag'] - df['sp_dU_mag']

    y = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
    if null_params:
        X = df[['r1', 'r2', 'dU_mag_test', 'noiseAlign', 'mean_pupil_range', 'du_diff', 'noise_diff']]
    else:
        X = df[regressors]

    if interactions:
        X['interaction'] = X['dU_mag_test'] * X['noiseAlign']
        #X['interaction'] = X['noiseAlign'] * X['mean_pupil_range']
        #X['interaction'] = X['r1'] * X['r2']
        #X['interaction2'] = X['dU_mag_test'] * X['noiseAlign']

    # duplicate r1/r2 for symmetry
    if 'r1' in X.keys():
        X1 = X.copy()
        X1['r1'] = X['r2']
        X1['r2'] = X['r1']
        X = pd.concat([X, X1])

    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    y -= y.mean()
    y /= y.std()
    y = pd.concat([y, y])

    X = sm.add_constant(X)
    
    output = fit_OLS_model(X, y, nboot=nboot, njacks=njack, replace=replace)

    big_df.append(df)

    if i == 0:
        rsq = {k: [v] for k, v in output['r2'].items()}
        coeff = {k: [v] for k, v in output['coef'].items()}
        rsq_ci = {k: [v] for k, v in output['ci'].items()}
        coeff_ci = {k: [v] for k, v in output['ci_coef'].items()}

    else:
        for k in rsq.keys():
            rsq[k].append(output['r2'][k])
            rsq_ci[k].append(output['ci'][k])
        for k in coeff.keys():    
            coeff[k].append(output['coef'][k])
            coeff_ci[k].append(output['ci_coef'][k])

big_df = pd.concat(big_df)

big_df['delt'] = (big_df['bp_dp']-big_df['sp_dp']) / (big_df['bp_dp'] + big_df['sp_dp'])
# do stepwise regression on the big data frame with long list of regressors
reg = ['r1', 'r2', 'dU_mag_test', 'noiseAlign', 'mean_pupil_range', 'du_diff', 'noise_diff']

perf = []
ci = []
for i, r in enumerate(reg):
    y = big_df['delt']
    y -= y.mean()
    y /= y.std()

    # set up regressors
    X = big_df[reg[0:(i+1)]]
    X -= X.mean(axis=0)
    X /= X.std()

    # add constant
    X = sm.add_constant(X)

    # fit model
    out = fit_OLS_model(X, y, nboot=10, njacks=5, replace=False)

    # save performance
    perf.append(out['r2']['full'])
    ci.append(out['ci']['full'])

f, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.plot(range(len(reg)), perf, '.-', color='k')

for i, _ci in enumerate(ci):
    ax.plot([i, i], [_ci[0], _ci[1]], 'k')
    ax.plot([i-0.02, i+0.02], [_ci[0], _ci[0]], 'k-')
    ax.plot([i-0.02, i+0.02], [_ci[1], _ci[1]], 'k-')

ax.axhline(0, linestyle='--', color='grey')

ax.set_xticks(np.arange(0, len(reg)))
ax.set_xticklabels(reg, rotation=45)

f.tight_layout()

plt.show()

keys = ['r1', 'r2', 'dU', 'noise', 'pvar']
keys = regressors + ['interaction']

f, ax = plt.subplots(1, 2, figsize=(8, 4))

d = pd.DataFrame(np.array(list(rsq.values()))[1:-1:2,:].T, columns=keys)
sns.stripplot(x='variable', y='value', data=d.melt(), ax=ax[0])
ax[0].axhline(0, linestyle='--', color='k') 

ax[0].set_ylabel(r"Unique $cvR^2$")
ax[0].set_xlabel("Regressor")

d=pd.DataFrame(np.array(list(coeff.values()))[:7,:].T, columns=keys)
sns.stripplot(x='variable', y='value', data=d.melt(), ax=ax[1])
ax[1].axhline(0, linestyle='--', color='k') 

ax[1].set_ylabel(r"Coefficient")
ax[1].set_xlabel("Regressor")

f.tight_layout()

plt.show()
