"""
Compare models with 0, 1, 2, or 3 extra dDR dims for the data with high reps (CPN)

Idea is to show that overall dprime improves, but delta dprime is low dimensional.
""" 
import sys
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms/')
sys.path.append('/home/charlie/lbhb/code/projects/nat_pup_ms/')
from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, CPN_SITES
import charlieTools.nat_sounds_ms.decoding as decoding
import load_results as ld
import nems.db as nd
import charlieTools.nat_sounds_ms.preprocessing as nat_preproc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

modelnames = ['dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-dU',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-1',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-2',
            'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-3'
    ]
ndims = [1, 2, 3, 4, 5]

path = DPRIME_DIR
loader = decoding.DecodingResults()
recache = False
sites = CPN_SITES
batches = [331] * len(CPN_SITES)

df = []; df1 = []; df2 = []; df3 = []; df4 = []
for i, (batch, site) in enumerate(zip(batches, sites)):
    for mn, ddf, ndim in zip(modelnames, [df, df1, df2, df3, df4], ndims):
        if site in ['BOL005c', 'BOL006b']:
            batch = 294
        try:
            fn = os.path.join(path, str(batch), site, mn+'_TDR.pickle')
            results = loader.load_results(fn, cache_path=None, recache=recache)
            _df = results.numeric_results
        except:
            raise ValueError(f"WARNING!! NOT LOADING SITE {site}")

        stim = results.evoked_stimulus_pairs
        _df = _df.loc[pd.IndexSlice[stim, ndim], :]
        _df['site'] = site
        _df['batch'] = batch
        _df['delta_dprime'] = (_df['bp_dp'] - _df['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
        _df.index = _df.index.get_level_values(0)

        ddf.append(_df)

df = pd.concat(df); df1 = pd.concat(df1); df2 = pd.concat(df2); df3 = pd.concat(df3); df4 = pd.concat(df4)

overall = pd.concat([df['dp_opt_test'], df1['dp_opt_test'], df2['dp_opt_test'], df3['dp_opt_test'], df4['dp_opt_test'], df['site']], axis=1)
overall.columns = [r"$\Delta \mu$", r'$dDR$', r'$dDR_1$', r'$dDR_2$', r'$dDR_3$', 'site']
delta = pd.concat([df['delta_dprime'], df1['delta_dprime'], df2['delta_dprime'], df3['delta_dprime'], df4['delta_dprime'], df['site']], axis=1)
delta.columns = [r"$\Delta \mu$", r'$dDR$', r'$dDR_1$', r'$dDR_2$', r'$dDR_3$', 'site']

f, ax = plt.subplots(2, 2, figsize=(9, 8))

sns.stripplot(data=overall.melt('site'), x='variable', y='value', **{'s': 2}, ax=ax[0, 0], zorder=-1)
ax[0, 0].set_ylabel(r"$d'^2$")
ax[0, 0].set_xlabel(r"$dDR$ Noise Dimensions")

sns.stripplot(data=delta.melt('site'), x='variable', y='value', **{'s': 2}, ax=ax[0, 1])
ax[0, 1].set_ylabel(r"$\Delta d'^2$")
ax[0, 1].set_xlabel(r"$dDR$ Noise Dimensions")

cols = plt.get_cmap('tab10', len(overall.site.unique()))
for i, site in enumerate(overall.site.unique()):
    data = overall[overall.site==site].melt('site').groupby(by='variable').mean()['value']
    data = data / data[0]
    ax[1, 0].plot(range(5), data, '-', color=cols(i))
    data = delta[delta.site==site].melt('site').groupby(by='variable').mean()['value']
    data = data #- data[0]
    ax[1, 1].plot(range(5), data, '-', color=cols(i), label=site)
#ax[1, 1].set_ylim((-.1, .1))
ax[1, 1].legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
ax[1, 1].set_xlabel(r"$dDR$ Noise Dimensions")
ax[1, 0].set_xlabel(r"$dDR$ Noise Dimensions")
ax[1, 0].set_ylabel(r"Normalized $d'^2$")
ax[1, 1].set_ylabel(r"Normalized $\Delta d'^2$")
f.tight_layout()

plt.show()