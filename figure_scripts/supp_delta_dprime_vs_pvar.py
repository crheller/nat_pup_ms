"""
Show that delta dprime tightly correlated with mean per stim pupil variance
"""

from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR

import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
from scipy.optimize import curve_fit
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True
all_sites = True
loader = decoding.DecodingResults()
path = DPRIME_DIR
cache_file = path + 'high_pvar_stim_combos.csv'
fig_fn = PY_FIGURES_DIR + 'supp_pvar_vs_delta_dprime.svg'
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
n_components = 2
norm_state_diff = True

x_cut = (1.5, 6)
y_cut = (0, 1)

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
    dp = results.get_result('dp_opt_test', results.evoked_stimulus_pairs, n_components)[0]   
    _df = pd.concat([bp, sp, dp], axis=1)
    _df['site'] = site
    _df['dU_mag_test'] = results.get_result('dU_mag_test', results.evoked_stimulus_pairs, n_components)[0]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, n_components, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', results.evoked_stimulus_pairs, n_components, idx=[0, 0])[0]

    _df['p_range'] = results.get_result('mean_pupil_range', results.evoked_stimulus_pairs, n_components)[0]
    _df['p_var'] = results.pupil_range.iloc[-1]['range']
    dfs.append(_df)
    del _df
end = timeit.default_timer()
print(end - start)

df_all = pd.concat(dfs)

# filter based on x_cut / y_cut
mask1 = (df_all['dU_mag_test'] < x_cut[1]) & (df_all['dU_mag_test'] > x_cut[0])
mask2 = (df_all['cos_dU_evec_test'] < y_cut[1]) & (df_all['cos_dU_evec_test'] > y_cut[0])
df = df_all[mask1 & mask2]


df['state_diff'] = (df['bp_dp'] - df['sp_dp']) / df['dp_opt_test']
df['abs_diff'] = (df['bp_dp'] - df['sp_dp'])
norm_pvar = df.groupby(by='site').mean()['p_range']
pvar = df.groupby(by='site').mean()['p_var']
state_diff = df.groupby(by='site').mean()['state_diff']
abs_diff = df.groupby(by='site').mean()['abs_diff']

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.axhline(0, linestyle='--', color='grey', lw=2, zorder=1)

if norm_state_diff:
    cc = ss.pearsonr(norm_pvar, state_diff)
    ax.scatter(norm_pvar.loc[HIGHR_SITES], state_diff.loc[HIGHR_SITES], marker='o', edgecolor='white', color='k', s=50,
                    label=r"$r = %s, p = %s$" % (round(cc[0], 2), round(cc[1], 3)), zorder=3)
    ax.legend(frameon=False)
    ax.scatter(norm_pvar.loc[LOWR_SITES], state_diff.loc[LOWR_SITES], marker='D', edgecolor='white', color='grey', s=30, zorder=2)
else:
    cc = ss.pearsonr(norm_pvar, abs_diff)
    ax.scatter(norm_pvar.loc[HIGHR_SITES], abs_diff.loc[HIGHR_SITES], marker='o', edgecolor='white', color='k', s=50,
                    label=r"$r = %s, p = %s$" % (round(cc[0], 2), round(cc[1], 3)), zorder=3)
    ax.legend(frameon=False)
    ax.scatter(norm_pvar.loc[LOWR_SITES], abs_diff.loc[LOWR_SITES], marker='D', edgecolor='white', color='grey', s=30, zorder=2)

ax.set_xlabel('Pupil variance')
ax.set_ylabel(r"$\Delta d'^2$")

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()

