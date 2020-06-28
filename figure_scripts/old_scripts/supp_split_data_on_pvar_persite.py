"""
Goals:
1) Show that dprime results for a given stimulus pair depends on the variance in pupil 
    for that pair. Use this as motivation to split up data based on pupil variance.
2) Plot pupil variance for all stim pairs. Fit bimodal distribution, split
    pairs based on this. Cache the split for later analyses.

Plots generated here probably would make a nice supp. figure
"""
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR

import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

all_sites = True
loader = decoding.DecodingResults()
path = DPRIME_DIR
cache_file = path + 'high_pvar_stim_combos.csv'
figsave = PY_FIGURES_DIR + 'supp_split_data.svg'
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
n_components = 2
savefig = True

# list of sites with > 10 reps of each stimulus
if all_sites:
    sites = ALL_SITES

else:
    sites = HIGHR_SITES

# for each site extract dprime and site. Concat into master df
import timeit
start = timeit.default_timer()
dfs = []
for site in sites:
    if site in LOWR_SITES:
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn)
    #results = loader.load_json(fn.replace('.pickle', '.json'))
    bp = results.get_result('bp_dp', results.evoked_stimulus_pairs, n_components)[0]
    sp = results.get_result('sp_dp', results.evoked_stimulus_pairs, n_components)[0]

    _df = pd.concat([bp, sp], axis=1)
    _df['site'] = site

    _df['p_range'] = results.get_result('mean_pupil_range', results.evoked_stimulus_pairs, n_components)[0]
    dfs.append(_df)
    del _df
end = timeit.default_timer()
print(end - start)

df = pd.concat(dfs)


mask = np.array([False] * df.shape[0])
# for each site, split into high / low pupil variance
for s in df.site.unique():
    mask = mask | ((df['site']==s) & (df['p_range'] > df[df.site==s]['p_range'].median()))

# save combos where mask is True
df[mask][['site']].to_csv(cache_file)

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.scatter(df[~mask].groupby(by='site').mean()['sp_dp'], 
           df[~mask].groupby(by='site').mean()['bp_dp'],
           color='b', edgecolor='white', s=50, label='small pupil variance')
ax.scatter(df[mask].groupby(by='site').mean()['sp_dp'], 
           df[mask].groupby(by='site').mean()['bp_dp'],
           color='r', edgecolor='white', s=50, label='large pupil variance')
ax.plot([df[~mask].groupby(by='site').mean()['sp_dp'], df[mask].groupby(by='site').mean()['sp_dp']],
           [df[~mask].groupby(by='site').mean()['bp_dp'], df[mask].groupby(by='site').mean()['bp_dp']], color='grey')
ax.plot([0, 100], [0, 100], '--', color='grey')
ax.axhline(0, linestyle='--', color='grey')
ax.axvline(0, linestyle='--', color='grey')

ax.set_xlabel(r"$d'^2_{small}$")
ax.set_ylabel(r"$d'^2_{big}$")
ax.legend(frameon=False)

f.tight_layout()

if savefig:
    f.savefig(figsave)

plt.show()