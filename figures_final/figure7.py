"""
Look at delta dprime as a function of the response magnitude (z-scored) for each stimulus.

Idea is that this is a negative result showing that the "goodness" of the stimulus (its ability to drive neurons)
doesn't predict the diversity in decoding changes.
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR3
from global_settings import HIGHR_SITES, CPN_SITES

import charlieTools.nat_sounds_ms.decoding as decoding
from charlieTools.statistics import get_direct_prob, get_bootstrapped_sample
from regression_helper import fit_OLS_model

import statsmodels.api as sm
import scipy.stats as ss
import pickle
import os
import pandas as pd
from itertools import combinations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

modelname = 'dprime_jk10_zscore_nclvz_fixtdr2-fa'
nComponents = 2
modelname331 = 'dprime_mvm-25-1_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-6'
nComponents331 = 8
sig_pairs_only = False
recache = True
savefig = True
fig_fn = PY_FIGURES_DIR3 + 'fig7.svg'

np.random.seed(123)

# ============================= LOAD DPRIME =========================================
path = DPRIME_DIR
loader = decoding.DecodingResults()
df_all = []
sites = CPN_SITES
batches = [331]*len(CPN_SITES)
for batch, site in zip(batches, sites):
    if batch == 331:
        mn = modelname331
        n_components = nComponents331
    else:
        mn = modelname
        n_components = nComponents
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    try:
        fn = os.path.join(path, str(batch), site, mn+'_TDR.pickle')
        results = loader.load_results(fn, cache_path=None, recache=recache)
        _df = results.numeric_results
    except:
        print(f"WARNING!! NOT LOADING SITE {site}")

    if (batch in [289, 294]) & (sig_pairs_only):
        fn = f'/auto/users/hellerc/results/nat_pupil_ms/reliable_epochs/{batch}/{site}.pickle'
        reliable_epochs = pickle.load(open(fn, "rb"))
        reliable_epochs = np.array(reliable_epochs['sorted_epochs'])[reliable_epochs['reliable_mask']]
        reliable_epochs = ['_'.join(e) for e in reliable_epochs]
        stim = results.evoked_stimulus_pairs
        stim = [s for s in stim if (results.mapping[s][0] in reliable_epochs) & (results.mapping[s][1] in reliable_epochs)]
    else:
        stim = results.evoked_stimulus_pairs
    _df = _df.loc[pd.IndexSlice[stim, n_components], :]
    _df['cos_dU'] = results.slice_array_results('cos_dU_evec_test', stim, n_components, idx=(0,0))[0] #.apply(lambda x: np.arccos(x)*180/np.pi)
    _df['site'] = site

    # zscore the resp mag within site to prevent bias from number of cells
    alldata = pd.concat([_df['r1mag_test'], _df['r2mag_test']])
    m = alldata.mean()
    sd = alldata.std()
    _df['r1mag_test'] = (_df['r1mag_test'] - m) / sd
    _df['r2mag_test'] = (_df['r2mag_test'] - m) / sd

    df_all.append(_df)

df_all = pd.concat(df_all)

df_all['delta'] = (df_all['bp_dp'] - df_all['sp_dp']) / (df_all['bp_dp'] + df_all['sp_dp'])
df_all['delta_dU'] = df_all['bp_dU_mag'] - df_all['sp_dU_mag']

c1 = []
c2 = []
c12 = []
c1_ci = []
c2_ci = []
c12_ci = []
for s in sites:
    print(f"Running regression for site {s}")
    df_dup = df_all.copy()
    df_dup = df_dup[df_dup.site==s]
    df_dup = pd.concat([df_all, df_all])
    df_dup['r1mag_test'] = pd.concat([df_all['r1mag_test'], df_all['r2mag_test']])
    df_dup['r2mag_test'] = pd.concat([df_all['r2mag_test'], df_all['r1mag_test']])

    X = df_dup[['r1mag_test', 'r2mag_test']]
    X['interaction'] = X['r1mag_test'] * X['r2mag_test']
    X -= X.mean()
    X /= X.std()
    X = sm.add_constant(X)

    y = df_dup['delta']

    res = fit_OLS_model(X, y, replace=True, nboot=10, njacks=2)
    c1.append(res['coef']['r1mag_test'])
    c1_ci.append(res['ci_coef']['r1mag_test'])
    c2.append(res['coef']['r2mag_test'])
    c2_ci.append(res['ci_coef']['r2mag_test'])
    c12.append(res['coef']['interaction'])
    c12_ci.append(res['ci_coef']['interaction'])

bins = 40
mm = 0.75
f, ax = plt.subplots(1, 2, figsize=(5, 2))

df_all.plot.hexbin(x='r1mag_test',
                   y='r2mag_test',
                   C='delta',
                   gridsize=bins,
                   vmin=-mm, vmax=mm, cmap='bwr', ax=ax[0])
ax[0].set_xlabel(r"$|\mathbf{r}_a|$ (normalized)")
ax[0].set_ylabel(r"$|\mathbf{r}_b|$ (normalized)")
ax[0].set_title(r"$\Delta d'^2$")

o1 = 0.2
o2 = 0.4
s = 10
ax[1].scatter(c1, np.arange(0, len(c1)), edgecolor='tab:blue', color='white', label=r"$|\mathbf{r}_a|$", s=s)
ax[1].scatter(c2, np.arange(o1, len(c1)+o1), edgecolor='tab:orange', color='white', label=r"$|\mathbf{r}_b|$", s=s)
ax[1].scatter(c12, np.arange(o2, len(c1)+o2), edgecolor='k', color='white', label=r"$|\mathbf{r}_a|*|\mathbf{r}_b|$", s=s)
for i, cf in enumerate(c1_ci):
    ax[1].plot([cf[0], cf[1]], [i, i], zorder=-1, color='tab:blue')
    ax[1].plot([c2_ci[i][0], c2_ci[i][1]], [i+o1, i+o1], zorder=-1, color='tab:orange')
    ax[1].plot([c12_ci[i][0], c12_ci[i][1]], [i+o2, i+o2], zorder=-1, color='k')

ax[1].legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
ax[1].set_ylabel("Site")
ax[1].set_xlabel(r"Regression coefficient")
ax[1].set_title(r"$\Delta d'^2$ ~ $|\mathbf{r}_a| + |\mathbf{r}_b|$")
ax[1].axvline(0, linestyle='--', color='lightgrey', zorder=-1)

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()