"""
Scratch file to look at CPN decoding results
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, CPN_SITES
import charlieTools.nat_sounds_ms.decoding as decoding

import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

path = DPRIME_DIR
loader = decoding.DecodingResults()
n_components = 2
recache = False
df_all = []
sites = CPN_SITES
batches = [331]*len(CPN_SITES)
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
modelname = modelname.replace('loadpred', 'loadpred.cpn')

for batch, site in zip(batches, sites):
    if (site in LOWR_SITES) & (batch != 331):
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    try:
        fn = os.path.join(path, str(batch), site, mn+'_TDR.pickle')
        results = loader.load_results(fn, cache_path=None, recache=recache)
        _df = results.numeric_results
    except:
        raise ValueError(f"WARNING!! NOT LOADING SITE {site}")

    stim = results.evoked_stimulus_pairs
    _df = _df.loc[pd.IndexSlice[stim, 2], :]
    _df['site'] = site
    _df['sp_noise_mag'] = results.array_results['sp_evals'].loc[pd.IndexSlice[stim, 2], 'mean'].apply(lambda x: x.sum())
    _df['bp_noise_mag'] = results.array_results['bp_evals'].loc[pd.IndexSlice[stim, 2], 'mean'].apply(lambda x: x.sum())
    _df['noise_alignment'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=(0,0))[0]
    _df['delta_dprime'] = (_df['bp_dp'] - _df['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    df_all.append(_df)

df = pd.concat(df_all)

# delta mu, colored by site
f, ax = plt.subplots(1, 5, figsize=(15, 3))

sns.scatterplot(x='bp_dU_mag', y='sp_dU_mag', data=df, hue='site', ax=ax[0], **{'s': 10, 'edgecolor': 'none'})
ll, ul = (np.min(ax[0].get_ylim()+ax[0].get_xlim()), np.max(ax[0].get_ylim()+ax[0].get_xlim()))
ax[0].plot([ll, ul], [ll, ul], 'k--')
ax[0].set_title(r"$\Delta \mu$")

sns.scatterplot(x='bp_noise_mag', y='sp_noise_mag', data=df, hue='site', ax=ax[1], **{'s': 10, 'edgecolor': 'none'})
ll, ul = (np.min(ax[1].get_ylim()+ax[1].get_xlim()), np.max(ax[1].get_ylim()+ax[1].get_xlim()))
ax[1].plot([ll, ul], [ll, ul], 'k--')
ax[1].set_title(r"$\Delta$ noise (overall in $dDR$ space)")

sns.histplot(df, x="noise_alignment", hue="site", element="step", lw=0, ax=ax[2], fill=False, kde=True)
ax[2].set_title(r"Noise Alignment")

sns.scatterplot(x='bp_dp', y='sp_dp', data=df, hue='site', ax=ax[3], **{'s': 10, 'edgecolor': 'none'})
ll, ul = (np.min(ax[3].get_ylim()+ax[3].get_xlim()), np.max(ax[3].get_ylim()+ax[3].get_xlim()))
ax[3].plot([ll, ul], [ll, ul], 'k--')
ax[3].set_title(r"$d'^2$")

sns.barplot(data=df, x='site', y='delta_dprime', ax=ax[4])
ax[4].set_title(r"$\Delta d'^2$")

f.tight_layout()

plt.show()