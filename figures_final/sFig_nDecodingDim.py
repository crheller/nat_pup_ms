'''
Supplemental figure showing choice of n decoding dimensions
'''
import sys
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms/')
sys.path.append('/home/charlie/lbhb/code/projects/nat_pup_ms/')
from path_settings import DPRIME_DIR, PY_FIGURES_DIR3, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, CPN_SITES
import charlieTools.nat_sounds_ms.decoding as decoding
import load_results as ld
import nems.db as nd
import charlieTools.nat_sounds_ms.preprocessing as nat_preproc

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

savefig = True
fig_fn = PY_FIGURES_DIR3 + 'S_nDecodingDims.svg'

display_dU = False
modelnames = ['dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-dU',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-1',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-2',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-3',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-4',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-5',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-6'
    ]
ndims = [1, 2, 3, 4, 5, 6, 7, 8]

path = DPRIME_DIR
loader = decoding.DecodingResults()
recache = False
sites = HIGHR_SITES + CPN_SITES
batches = [289] * len(HIGHR_SITES) + [331]*len(CPN_SITES)

df = []; df1 = []; df2 = []; df3 = []; df4 = []; df5 = []; df6 = []; df7 = []
for i, (batch, site) in enumerate(zip(batches, sites)):
    for mn, ddf, ndim in zip(modelnames, [df, df1, df2, df3, df4, df5, df6, df7], ndims):
        if site in ['BOL005c', 'BOL006b']:
            batch = 294
        try:
            if batch in [289, 294]:
                mn = mn.replace('mvm-25-2_', '')
            fn = os.path.join(path, str(batch), site, mn+'_TDR.pickle')
            results = loader.load_results(fn, cache_path=None, recache=recache)
            _df = results.numeric_results
        except:
            raise ValueError(f"WARNING!! NOT LOADING SITE {site}")

        # only use epochs with reliable noise distros
        if batch in [289, 294]:
            fn = f'/auto/users/hellerc/results/nat_pupil_ms/reliable_epochs/{batch}/{site}.pickle'
            reliable_epochs = pickle.load(open(fn, "rb"))
            reliable_epochs = np.array(reliable_epochs['sorted_epochs'])[reliable_epochs['reliable_mask']]
            reliable_epochs = ['_'.join(e) for e in reliable_epochs]
            stim = results.evoked_stimulus_pairs
            stim = [s for s in stim if (results.mapping[s][0] in reliable_epochs) & (results.mapping[s][1] in reliable_epochs)]
        else:
            stim = results.evoked_stimulus_pairs

        _df = _df.loc[pd.IndexSlice[stim, ndim], :]
        _df['site'] = site
        _df['batch'] = batch
        _df['delta_dprime'] = (_df['bp_dp'] - _df['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
        _df.index = _df.index.get_level_values(0)

        ddf.append(_df)

df = pd.concat(df); df1 = pd.concat(df1); df2 = pd.concat(df2); df3 = pd.concat(df3); df4 = pd.concat(df4); df5 = pd.concat(df5); df6 = pd.concat(df6); df7 = pd.concat(df7)

overall = pd.concat([df['dp_opt_test'], df1['dp_opt_test'], df2['dp_opt_test'], df3['dp_opt_test'], df4['dp_opt_test'], 
                    df5['dp_opt_test'], df6['dp_opt_test'], df7['dp_opt_test'], df['site']], axis=1)
overall.columns = [r"$\Delta \mu$", r'$dDR$', r'$dDR_1$', r'$dDR_2$', r'$dDR_3$', r'$dDR_4$', r'$dDR_5$', r'$dDR_6$', 'site']
delta = pd.concat([df['delta_dprime'], df1['delta_dprime'], df2['delta_dprime'], df3['delta_dprime'], df4['delta_dprime'], 
                    df5['delta_dprime'], df6['delta_dprime'], df7['delta_dprime'], df['site']], axis=1)
delta.columns = [r"$\Delta \mu$", r'$dDR$', r'$dDR_1$', r'$dDR_2$', r'$dDR_3$', r'$dDR_4$', r'$dDR_5$', r'$dDR_6$', 'site']

if not display_dU:
    overall = overall[[c for c in overall.columns if c!=r"$\Delta \mu$"]]
    delta = delta[[c for c in delta.columns if c!=r"$\Delta \mu$"]]

# plot fraction change in dprime for each site as function of dims, split into subplots by CPN / NAT data
f, ax = plt.subplots(2, 2, figsize=(4.2, 4), sharex=True)
for i, site in enumerate(overall.site.unique()):
    vals = overall[overall.site==site].mean().values
    if site in CPN_SITES:
        ax[0, 0].plot(vals / vals[0], color='tab:blue', alpha=0.5, zorder=-1)
    else:
        ax[0, 1].plot(vals / vals[0], color='tab:orange', alpha=0.5, zorder=-1)
vals = overall[overall.site.isin(CPN_SITES)].groupby(by='site').mean().values
ax[0, 0].errorbar(range(vals.shape[1]), (vals.T / vals[:, 0]).T.mean(axis=0), 
                                yerr=(vals.T / vals[:, 0]).T.std(axis=0) / np.sqrt(vals.shape[0]),
                                capsize=3, lw=2, color='k')
vals = overall[~overall.site.isin(CPN_SITES)].groupby(by='site').mean().values
ax[0, 1].errorbar(range(vals.shape[1]), (vals.T / vals[:, 0]).T.mean(axis=0), 
                                yerr=(vals.T / vals[:, 0]).T.std(axis=0) / np.sqrt(vals.shape[0]),
                                capsize=3, lw=2, color='k')

ax[0, 0].set_ylabel(r"$d'^2$ (normalized)")
ax[0, 1].set_ylabel(r"$d'^2$ (normalized)")
xticks = delta.columns[:-1]
ax[0, 0].set_xticks(range(vals.shape[1]))
ax[0, 0].set_xticklabels(np.arange(1, vals.shape[1]+1))
ax[0, 1].set_xticks(range(vals.shape[1]))
ax[0, 1].set_xticklabels(np.arange(1, vals.shape[1]+1))

# Now plot delta dprime for each site in the same way (but not normalized)
for i, site in enumerate(overall.site.unique()):
    vals = delta[delta.site==site].mean().values
    if site in CPN_SITES:
        ax[1, 0].plot(vals, color='tab:blue', alpha=0.5, zorder=-1)
    else:
        ax[1, 1].plot(vals, color='tab:orange', alpha=0.5, zorder=-1)
vals = delta[delta.site.isin(CPN_SITES)].groupby(by='site').mean().values
ax[1, 0].errorbar(range(vals.shape[1]), (vals.T).T.mean(axis=0), 
                                yerr=(vals.T).T.std(axis=0) / np.sqrt(vals.shape[0]),
                                capsize=3, lw=2, color='k')
vals = delta[~delta.site.isin(CPN_SITES)].groupby(by='site').mean().values
ax[1, 1].errorbar(range(vals.shape[1]), (vals.T).T.mean(axis=0), 
                                yerr=(vals.T).T.std(axis=0) / np.sqrt(vals.shape[0]),
                                capsize=3, lw=2, color='k')

ax[1, 0].set_ylabel(r"$\Delta d'^2$")
ax[1, 1].set_ylabel(r"$\Delta d'^2$")
xticks = delta.columns[:-1]
ax[1, 0].set_xticks(range(vals.shape[1]))
ax[1, 0].set_xticklabels(np.arange(1, vals.shape[1]+1))
ax[1, 1].set_xticks(range(vals.shape[1]))
ax[1, 1].set_xticklabels(np.arange(1, vals.shape[1]+1))
ax[1, 0].set_xlabel("Number of noise dimensions")
ax[1, 1].set_xlabel("Number of noise dimensions")

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()