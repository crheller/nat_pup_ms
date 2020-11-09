"""
Supplemental figure (previously a more detailed figure in the main text) simply showing 
that second order changes are required to reproduce the actual data.
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, DU_MAG_CUT, NOISE_INTERFERENCE_CUT
import colors as color
import ax_labels as alab

from nems_lbhb.baphy import parse_cellid

import charlieTools.statistics as stats
import charlieTools.preprocessing as preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
import scipy.ndimage.filters as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = True

compute_bootstats = True
recache = False
ALL_TRAIN_DATA = False  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = HIGHR_SITES
path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR + 'supp_simulations.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
sim1 = 'dprime_simInTDR_sim1_jk10_zscore_nclvz_fixtdr2'
sim12 = 'dprime_simInTDR_sim12_jk10_zscore_nclvz_fixtdr2'
estval = '_test'
n_components = 2

mi_norm = True

# where to crop the data
if estval == '_train':
    x_cut = (2.5, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = None
    y_cut = None

# ========================================= Load results ====================================================
df = []
df_sim1 = []
df_sim12 = []
for site in sites:
    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results

    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = sim1.replace('_jk10', '_jk1_eev') 
    else: mn = sim1
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim1 = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df_sim1 = results_sim1.numeric_results

    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = sim12.replace('_jk10', '_jk1_eev') 
    else: mn = sim12
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim12 = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df_sim12 = results_sim12.numeric_results

    stim = results.evoked_stimulus_pairs

    _df = _df.loc[pd.IndexSlice[stim, n_components], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, n_components, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, n_components, idx=[0, 0])[0]
    _df['state_diff'] = (_df['bp_dp'] - _df['sp_dp']) / _df['dp_opt_test']
    _df['state_MI'] = (_df['bp_dp'] - _df['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df['site'] = site
    df.append(_df)

    _df_sim1 = _df_sim1.loc[pd.IndexSlice[stim, n_components], :]
    _df_sim1['state_diff'] = (_df_sim1['bp_dp'] - _df_sim1['sp_dp']) / _df['dp_opt_test']
    _df_sim1['state_MI'] = (_df_sim1['bp_dp'] - _df_sim1['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df_sim1['site'] = site
    df_sim1.append(_df_sim1)

    _df_sim12 = _df_sim12.loc[pd.IndexSlice[stim, n_components], :]
    _df_sim12['state_diff'] = (_df_sim12['bp_dp'] - _df_sim12['sp_dp']) / _df['dp_opt_test']
    _df_sim12['state_MI'] = (_df_sim12['bp_dp'] - _df_sim12['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df_sim12['site'] = site
    df_sim12.append(_df_sim12)

df_all = pd.concat(df)
df_sim1_all = pd.concat(df_sim1)
df_sim12_all = pd.concat(df_sim12)

# filter based on x_cut / y_cut
if (x_cut is not None) & (y_cut is not None):
    mask1 = (df_all['dU_mag'+estval] < x_cut[1]) & (df_all['dU_mag'+estval] > x_cut[0])
    mask2 = (df_all['cos_dU_evec'+estval] < y_cut[1]) & (df_all['cos_dU_evec'+estval] > y_cut[0])
else:
    mask1 = (True * np.ones(df_all.shape[0])).astype(bool)
    mask2 = (True * np.ones(df_all.shape[0])).astype(bool)
df = df_all[mask1 & mask2]
df_sim1 = df_sim1_all[mask1 & mask2]
df_sim12 = df_sim12_all[mask1 & mask2]

if mi_norm:
    df['state_diff'] = df['state_MI']
    df['sim1'] = df_sim1['state_MI']
    df['sim12'] = df_sim12['state_MI']
else:
    df['sim1'] = df_sim1['state_diff']
    df['sim12'] = df_sim12['state_diff']
# ========================================= Plot data =====================================================
# set up subplots
f, ax = plt.subplots(1, 1, figsize=(3, 6))

# plot dprime per site for the raw simulations
for i, s in zip([0, 1, 2], ['state_diff', 'sim1', 'sim12']):
    try:
        vals = df.loc[df.site.isin(LOWR_SITES)].groupby(by='site').mean()[s]
        ax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.0, len(vals)),
                    vals, color='grey', marker='D', edgecolor='white', s=30, zorder=2)
    except:
        pass
    vals = df.loc[df.site.isin(HIGHR_SITES)].groupby(by='site').mean()[s]
    ax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.0, len(vals)),
                vals, color='k', marker='o', edgecolor='white', s=50, zorder=3)

    # now, for each site draw lines between points in each model. Color red if 2nd order hurts, blue if helps
    line_colors = []
    for s in df.site.unique():
        vals = df.groupby(by='site').mean()[['state_diff', 'sim1', 'sim12']].loc[s].values
        if vals[1] < vals[2]:
            ax.plot([0, 1, 2], vals, color='grey', alpha=0.5, zorder=1)
            line_colors.append('blue')
        else:
            ax.plot([0, 1, 2], vals, color='grey', alpha=0.5, zorder=1)
            line_colors.append('red')

ax.axhline(0, linestyle='--', color='grey', lw=2)     
ax.set_xticks([0, 1, 2])
ax.set_xlim((-0.5, 2.5))
ax.set_xticklabels(['Actual', 'Ind.\nvariability only', 'Full simulation'], rotation=45)
ax.set_xlabel('Dataset')
if mi_norm:
    ax.set_ylabel(r"$\Delta d'^{2}$")    
else:
    ax.set_ylabel(r"$\Delta d'^{2}$")
ax.set_title('Discriminability Change')
if not mi_norm:
    ax.set_ylim((-1, 2))
else:
    ax.set_ylim((-.1, .5))

# bootstrap test instead
if compute_bootstats:
    print("generating bootstrap stats for dprime models. Could be slow...")

    d = {s: df[df.site==s]['sim12'].values-df[df.site==s]['state_diff'].values for s in df.site.unique()}
    bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
    p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
    print("Raw delta dprime vs. full simulation, p={0}".format(p))

    d = {s: df[df.site==s]['state_diff'].values-df[df.site==s]['sim1'].values for s in df.site.unique()}
    bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
    p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
    print("Raw delta dprime vs. first order simulation, p={0}".format(p))

    d = {s: df[df.site==s]['sim12'].values-df[df.site==s]['sim1'].values for s in df.site.unique()}
    bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
    p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
    print("Full delta dprime vs. first order simulation, p={0}".format(p))

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()