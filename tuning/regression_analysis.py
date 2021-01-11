"""
Regress delta dprime against first order stimulus / response characteristics
    Do certain sounds / responsiveness lead to bigger / smaller pupil changes?
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
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42

recache = False # decoding results
pc = [0, 1, 2] #[0, 1] # 0, 1, 2, [0, 1] (which PC to project on for resp characterization)
collapse = True
combine_combinations = True
norm = True # normalize the delta dprime
savefig = True
fig_fn = PY_FIGURES_DIR.split('/py_figures/')[0] + '/svd_scripts/regression.pdf'

# save full model / coefficients, unique R2, independent R2
cols = ['rsq', 'mean_r', 'diff_r', 'mean_rSD', 'diff_rSD', 'cos_r1r2', 'mean_s', 'diff_s'] 

# hold parameter values
dp = pd.DataFrame(columns=cols)
delta = pd.DataFrame(columns=cols)
# hold unique r2
dp_ur2 = pd.DataFrame(columns=cols)
delta_ur2 = pd.DataFrame(columns=cols)
# hold independent r2
dp_ir2 = pd.DataFrame(columns=cols)
delta_ir2 = pd.DataFrame(columns=cols)

df_all = pd.DataFrame()
sk = 0
for batch in [289, 294]:
    if batch == 289:
        # list of sites with > 10 reps of each stimulus
        sites = ['TAR010c', 'TAR017b', 
                'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
                'DRX007a.e1:64', 'DRX007a.e65:128', 
                'DRX008b.e1:64', 'DRX008b.e65:128']
                
    elif batch == 294:
        sites = ['BOL005c', 'BOL006b']

    for site in sites:
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
        loadkey = "ozgf.fs100.ch18.pup"
        uri = generate_recording_uri(cellid=site[:7], batch=batch, loadkey=loadkey)
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

        # ========================= GET MAGNITUDE/SD OF EVOKED RESPONSES ============================
        Xu = Xev.mean(axis=1)
        spont = X[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
        Xu_center = Xu - spont # subtract spont
        pca = PCA(n_components=10)
        pca.fit(Xu_center.T)

        spont = X[:, :, spont_bins.squeeze()].mean(axis=(1,2), keepdims=True)
        if type(pc) is list:
            Xpc1_full = (X - spont).T.dot(pca.components_[pc[0]:pc[-1]+1, :].T)
            if collapse:
                Xpc1 = np.sqrt(np.sum(Xpc1_full**2, axis=-1))
            else:
                Xpc1 = Xpc1_full.squeeze()
        else:
            Xpc1_full = (X - spont).T.dot(pca.components_[pc, :].T)
            Xpc1 = Xpc1_full.squeeze()

        # resp mag
        Xpc1u = Xpc1.mean(axis=1)
        # resp variance
        Xpc1sd = Xpc1.var(axis=1)

        # ============================ LOAD DPRIME RESULTS =============================
        loader = decoding.DecodingResults()
        modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
        fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
        results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
        df = results.numeric_results.loc[results.evoked_stimulus_pairs]
        df['noiseAlign'] = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, 2, [None,0])[0].apply(lambda x: x[0])

        # get alignment of r1/r2 in resp space (if resp space dim=1), set to constant
        if Xpc1_full.shape[-1] == 1:
            df.at[:, 'cos_r1r2'] = 0
        else:
            Xpc1u_full = Xpc1_full.mean(axis=1)
            r1 = [Xpc1u_full[int(c.split('_')[0])] / np.linalg.norm(Xpc1u_full[int(c.split('_')[0])]) for c in df.index.get_level_values('combo')]
            r2 = [Xpc1u_full[int(c.split('_')[1])] / np.linalg.norm(Xpc1u_full[int(c.split('_')[1])]) for c in df.index.get_level_values('combo')]
            cos = [abs(_r1.dot(_r2.T)) for _r1, _r2 in zip(r1, r2)]
            df.at[:, 'cos_r1r2'] = cos

        # Run regression model -- predict delta dprime / dprime from resp stats / spectrogram stats
        df.at[:, 'r1'] = [Xpc1u[int(c.split('_')[0])] for c in df.index.get_level_values('combo')]
        df.at[:, 'r2'] = [Xpc1u[int(c.split('_')[1])] for c in df.index.get_level_values('combo')]
        df.at[:, 'r1_sd'] = [Xpc1sd[int(c.split('_')[0])] for c in df.index.get_level_values('combo')]
        df.at[:, 'r2_sd'] = [Xpc1sd[int(c.split('_')[1])] for c in df.index.get_level_values('combo')]
        df.at[:, 's1'] = [pwr[int(c.split('_')[0])] for c in df.index.get_level_values('combo')]
        df.at[:, 's2'] = [pwr[int(c.split('_')[1])] for c in df.index.get_level_values('combo')]
        df.at[:, 'site'] = site
        df.at[:, 'site_key'] = sk
        sk += 1

        #if not combine_combinations:
        #    # flip combos things for symmetry
        #    df2 = df.copy()
        #    df2['r1'] = df['r2'].copy()
        #    df2['r2'] = df['r1'].copy()
        #    df2['r1_sd'] = df['r2_sd'].copy()
        #    df2['r2_sd'] = df['r1_sd'].copy()
        #    df2['s1'] = df['s2'].copy()
        #    df2['s2'] = df['s1'].copy()

        #    df = pd.concat([df, df2])

        # combine r1/r2 and s1/s2 stats into single columns
        df['mean_r'] = df[['r1', 'r2']].mean(axis=1)
        df['diff_r'] = abs(df['r1'] - df['r2'])
        df['mean_s'] = df[['s1', 's2']].mean(axis=1)
        df['diff_s'] = abs(df['s1'] - df['s2'])
        df['mean_rSD'] = df[['r1_sd', 'r2_sd']].mean(axis=1)
        df['diff_rSD'] = abs(df['r1_sd'] - df['r2_sd'])

        df_all = df_all.append(df)

        X = df[cols[1:]]
        X = X - X.mean()
        X = X / X.std()
        X = sm.add_constant(X)

        if norm:
            y = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
        else:
            y = (df['bp_dp'] - df['sp_dp'])
        y -= y.mean()
        y /= y.std()
        #delta_reg = sm.OLS(y, X).fit()
        results = fit_OLS_model(X, y, nboot=10)
        _delta = pd.DataFrame(columns=cols+['site'], data=[[results['r2']['full']] + list(results['coef'].values()) + [site]])
        delta = delta.append(_delta)
        _delta_ur2 = pd.DataFrame(columns=cols+['site'], data=[[results['r2']['full']] + [v for k, v in results['r2'].items() if k.startswith('u')] + [site]])
        delta_ur2 = delta_ur2.append(_delta_ur2)
        _delta_ir2 = pd.DataFrame(columns=cols+['site'], data=[[results['r2']['full']] + [v for k, v in results['r2'].items() \
                                                                                if (k.startswith('u')==False) & (k!='full')] + [site]])
        delta_ir2 = delta_ir2.append(_delta_ir2)

        y = df['dp_opt_test']
        y -= y.mean()
        y /= y.std()
        #reg = sm.OLS(y, X).fit()
        results = fit_OLS_model(X, y, nboot=10)
        _dp = pd.DataFrame(columns=cols+['site'], data=[[results['r2']['full']] + list(results['coef'].values()) + [site]])
        dp = dp.append(_dp)
        _dp_ur2 = pd.DataFrame(columns=cols+['site'], data=[[results['r2']['full']] + [v for k, v in results['r2'].items() if k.startswith('u')] + [site]])
        dp_ur2 = dp_ur2.append(_dp_ur2)
        _dp_ir2 = pd.DataFrame(columns=cols+['site'], data=[[results['r2']['full']] + [v for k, v in results['r2'].items() \
                                                                                if (k.startswith('u')==False) & (k!='full')] + [site]])
        dp_ir2 = dp_ir2.append(_dp_ir2)

xvals = cols

f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

sns.stripplot(x='variable', y='value', data=dp[xvals].melt(), ax=ax[0])
ax[0].axhline(0, linestyle='--', color='k', lw=2)
ax[0].set_title(r"Overall $d'^2$")

sns.stripplot(x='variable', y='value', data=delta[xvals].melt(), ax=ax[1])
ax[1].axhline(0, linestyle='--', color='k', lw=2)
ax[1].set_title(r"$\Delta d'^2$")

f.tight_layout()

if savefig:
    f.savefig(fig_fn)


f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

sns.stripplot(x='variable', y='value', data=dp_ur2[xvals].melt(), ax=ax[0])
ax[0].axhline(0, linestyle='--', color='k', lw=2)
ax[0].set_title(r"Overall $d'^2$")
ax[0].set_ylabel(r"$cvR^2$ (unique)")

sns.stripplot(x='variable', y='value', data=delta_ur2[xvals].melt(), ax=ax[1])
ax[1].axhline(0, linestyle='--', color='k', lw=2)
ax[1].set_title(r"$\Delta d'^2$")
ax[1].set_ylabel(r"$cvR^2$ (unique)")

f.tight_layout()

f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

sns.stripplot(x='variable', y='value', data=dp_ir2[xvals].melt(), ax=ax[0])
ax[0].axhline(0, linestyle='--', color='k', lw=2)
ax[0].set_title(r"Overall $d'^2$")
ax[0].set_ylabel(r"$cvR^2$ (independent)")

sns.stripplot(x='variable', y='value', data=delta_ir2[xvals].melt(), ax=ax[1])
ax[1].axhline(0, linestyle='--', color='k', lw=2)
ax[1].set_title(r"$\Delta d'^2$")
ax[1].set_ylabel(r"$cvR^2$ (independent)")

f.tight_layout()

plt.show()


# regression overall
X = df_all[cols[1:]+['noiseAlign']+['dU_mag_test']]
X = X - X.mean()
X = X / X.std()
X = sm.add_constant(X)

if norm:
    y = (df_all['bp_dp'] - df_all['sp_dp']) / (df_all['bp_dp'] + df_all['sp_dp'])
else:
    y = (df_all['bp_dp'] - df_all['sp_dp'])
y -= y.mean()
y /= y.std()
#delta_reg = sm.OLS(y, X).fit()
results = fit_OLS_model(X, y, nboot=10)

# regression for delta mu / noise angle
X = df_all[['dU_mag_test', 'noiseAlign', 'site_key']]
X['interaction'] = X['dU_mag_test'] * X['noiseAlign']
X = X - X.mean()
X = X / X.std()
X = sm.add_constant(X)
reg2 = sm.OLS(y, X).fit()