"""
Goals:
1) Show that dprime results for a given stimulus pair depends on the variance in pupil 
    for that pair. Use this as motivation to split up data based on pupil variance.
2) Plot pupil variance for all stim pairs. Fit bimodal distribution, split
    pairs based on this. Cache the split for later analyses.

Plots generated here probably would make a nice supp. figure
"""
import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
import os


loader = decoding.DecodingResults()
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
cache_file = path + 'high_pvar_stim_combos.csv'
figsave = 'inkscape_figures/supp_fig1_split_data.svg'
modelname = 'dprime_jk10'
n_components = 2

# list of sites with > 10 reps of each stimulus
sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']

# for each site extract dprime and site. Concat into master df
df = pd.DataFrame()
for site in sites:
    fn = os.path.join(path, site, modelname+'_PLS.pickle')
    results = loader.load_results(fn)

    bp = results.get_result('bp_dp', results.evoked_stimulus_pairs, n_components)[0]
    sp = results.get_result('sp_dp', results.evoked_stimulus_pairs, n_components)[0]

    _df = pd.concat([bp, sp], axis=1)
    _df['site'] = site

    # mean pupil range
    combos = [(int(c.split('_')[0]), int(c.split('_')[1])) for c in _df.index.get_level_values('combo')]
    pr = results.pupil_range
    pr_range = [np.mean([pr[pr.stim==c[0]]['range'], pr[pr.stim==c[1]]['range']]) for c in combos]

    _df['p_range'] = pr_range

    df = df.append(_df)

# distribution of pupil ranges per stimulus pair
f, ax = plt.subplots(1, 2, figsize=(8, 4))

bins = np.arange(0, 0.6, 0.01)
y, x, _ = ax[0].hist(df['p_range'], bins=bins, color='lightgray', edgecolor='k', label='data')
x = (x[1:] + x[:-1]) / 2
ax[0].set_xlabel(r"$\frac{\bar p_{big, s} - \bar p_{small, s}}{\sigma^{2}_{p, all}}$", fontsize=12)

# fit bimodal distribution
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)
def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

expected = (0.25, 0.1, 400, 0.4, 0.1, 400)
params, cov = curve_fit(bimodal, x, y, expected)

# add fit to the plot
ax[0].plot(x, bimodal(x, *params), color='k', lw=2)
ax[0].plot(x, gauss(x, *params[:3]), color='blue', lw=2, label='low variance')
ax[0].plot(x, gauss(x, *params[3:]), color='red', lw=2, label='high variance')
ax[0].legend(frameon=False)

# keep only data in the right-most hump of the distribution
mean = params[3]
sd = params[4]
mask = ((mean - abs(sd)) <= df['p_range']) & (df['p_range'] < (mean + abs(sd)))
# save sites / combos where mask is True
df[mask][['site']].to_csv(cache_file)


# add ax.scatter(df[df['p_range']>median].groupby(by='site').mean()['sp_dp'] ** (1/2), 
ax[1].scatter(df[mask].groupby(by='site').mean()['sp_dp'] ** (1/2), 
           df[mask].groupby(by='site').mean()['bp_dp'] ** (1/2),
           color='r', edgecolor='white', s=50, label='large pupil variance')
ax[1].scatter(df[~mask].groupby(by='site').mean()['sp_dp'] ** (1/2), 
           df[~mask].groupby(by='site').mean()['bp_dp'] ** (1/2),
           color='b', edgecolor='white', s=50, label='small pupil variance')
ax[1].plot([0, 10], [0, 10], '--', color='grey')
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0, linestyle='--', color='grey')

ax[1].set_xlabel(r"$d'_{small}$")
ax[1].set_ylabel(r"$d'_{big}$")
ax[1].legend(frameon=False)

f.tight_layout()

f.savefig(figsave)

plt.show()