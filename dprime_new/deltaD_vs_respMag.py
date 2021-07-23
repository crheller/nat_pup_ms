"""
Look at delta dprime as a function of the response magnitude (z-scored) for each stimulus.

Idea is that this is a negative result showing that the "goodness" of the stimulus (its ability to drive neurons)
doesn't predict the diversity.
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
    df_all.append(_df)

df_all = pd.concat(df_all)

df_all['delta'] = (df_all['bp_dp'] - df_all['sp_dp']) / (df_all['bp_dp'] + df_all['sp_dp'])
df_all['delta_dU'] = df_all['bp_dU_mag'] - df_all['sp_dU_mag']

bins = 40
mm = 0.75
f, ax = plt.subplots(1, 3, figsize=(9, 3))

df_all.plot.hexbin(x='r1mag_test',
                   y='r2mag_test',
                   C='delta',
                   gridsize=bins,
                   vmin=-mm, vmax=mm, cmap='bwr', ax=ax[0])

# color by site
sns.scatterplot(data=df_all, x='r1mag_test', y='r2mag_test', hue='site', ax=ax[1])

# for each site, plot the fraction pos/neg for good/bad stim pairs
goodprop = []
badprop = []
for s in df_all.site.unique():
    r1 = df_all[df_all.site==s]['r1mag_test']
    r2 = df_all[df_all.site==s]['r2mag_test']
    gm = np.sqrt(r1*r2)
    gm = (r1 + r2) / 2
    med = np.median(gm)
    g = (df_all[df_all.site==s]['delta'][gm>=med] > 0).sum() / sum(gm>=med)
    b = (df_all[df_all.site==s]['delta'][gm<med] > 0).sum() / sum(gm<med)
    ax[2].plot([0, 1], [b, g], color='k')

    goodprop.append(g)
    badprop.append(b)

f.tight_layout()
plt.show()


df_dup = df_all.copy()
df_dup = pd.concat([df_all, df_all])
df_dup['r1mag_test'] = pd.concat([df_all['r1mag_test'], df_all['r2mag_test']])
df_dup['r2mag_test'] = pd.concat([df_all['r2mag_test'], df_all['r1mag_test']])

X = df_dup[['r1mag_test', 'r2mag_test']]
X['interaction'] = X['r1mag_test'] * X['r2mag_test']
X -= X.mean()
X /= X.std()
X = sm.add_constant(X)

y = df_dup['delta']

fit_OLS_model(X, y, replace=True, nboot=100, njacks=10)