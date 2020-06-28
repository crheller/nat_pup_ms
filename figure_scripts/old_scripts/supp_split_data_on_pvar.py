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
    dp = results.get_result('dp_opt_test', results.evoked_stimulus_pairs, n_components)[0]

    _df = pd.concat([bp, sp, dp], axis=1)
    _df['site'] = site

    _df['p_range'] = results.get_result('mean_pupil_range', results.evoked_stimulus_pairs, n_components)[0]
    dfs.append(_df)
    del _df
end = timeit.default_timer()
print(end - start)

df = pd.concat(dfs)
# distribution of pupil ranges per stimulus pair
f, ax = plt.subplots(1, 2, figsize=(6, 3))

bins = np.arange(0, 0.6, 0.01)
y, x, _ = ax[0].hist(df['p_range'], bins=bins, color='lightgray', edgecolor='k', label='data')
x = (x[1:] + x[:-1]) / 2
ax[0].set_xlabel(r"$\frac{\bar p_{big, s} - \bar p_{small, s}}{\sigma^{2}_{p, all}}$", fontsize=12)

# fit bimodal distribution
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)
def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

#expected = (0.25, 0.1, 400, 0.4, 0.1, 400)
#params, cov = curve_fit(bimodal, x, y, expected)

# add fit to the plot
#ax[0].plot(x, bimodal(x, *params), color='k', lw=2)
#ax[0].plot(x, gauss(x, *params[:3]), color='blue', lw=2, label='low variance')
#ax[0].plot(x, gauss(x, *params[3:]), color='red', lw=2, label='high variance')
div = np.median(df['p_range'])
ax[0].axvline(div, color='red', lw=2)
ax[0].legend(frameon=False)

# keep only data in the right-most hump of the distribution
#mean = params[3]
#sd = params[4] * 3
#mask = ((mean - abs(sd)) <= df['p_range']) & (df['p_range'] < (mean + abs(sd)))

mask = df['p_range'] > div
# save sites / combos where mask is True
df[mask][['site']].to_csv(cache_file)

ax[1].scatter(df[~mask].groupby(by='site').mean()['sp_dp'], 
           df[~mask].groupby(by='site').mean()['bp_dp'],
           color='b', edgecolor='white', s=50, label='small pupil variance')
ax[1].scatter(df[mask].groupby(by='site').mean()['sp_dp'], 
           df[mask].groupby(by='site').mean()['bp_dp'],
           color='r', edgecolor='white', s=50, label='large pupil variance')
ax[1].plot([0, 70], [0, 70], '--', color='grey')
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0, linestyle='--', color='grey')

ax[1].set_xlabel(r"$d'^2_{small}$")
ax[1].set_ylabel(r"$d'^2_{big}$")
ax[1].legend(frameon=False)

f.tight_layout()

if savefig:
    f.savefig(figsave)

plt.show()