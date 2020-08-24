'''
Supp. figure for high rep sites only showing that for d' test, there's no more information 
gained including additional noise dims
'''

import colors as color
import ax_labels as alab
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.statistics as stats
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True

path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR+'supp_extra_TDR.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
val = 'dp_opt_test'
estval = '_test'

# only crop the dprime value. Show count for everything
if estval == '_train':
    x_cut = (3, 8.5)
    y_cut = (0.1, .45) 
elif estval == '_test':
    #x_cut = (1, 8)
    #y_cut = (0.2, 1) 
    x_cut = (1.5, 6)
    y_cut = (0, 1)

sites = HIGHR_SITES  # need true cross validation for this figure

df = []
dfn1 = []
dfn2 = []
for site in sites:
    mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn)

    stim = results.evoked_stimulus_pairs

    _df = results.numeric_results
    _df = _df.loc[pd.IndexSlice[stim, 2], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
    _df['site'] = site
    df.append(_df)

    mn2 = mn+'_noiseDim1'
    fn = os.path.join(path, site, mn2+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results
    _df = _df.loc[pd.IndexSlice[stim, 3], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 3, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 3, idx=[0, 0])[0]
    _df['site'] = site
    dfn1.append(_df)

    mn3 = mn+'_noiseDim2'
    fn = os.path.join(path, site, mn3+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results
    _df = _df.loc[pd.IndexSlice[stim, 4], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 4, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 4, idx=[0, 0])[0]
    _df['site'] = site
    dfn2.append(_df)

df = pd.concat(df)
dfn1 = pd.concat(dfn1)
dfn2 = pd.concat(dfn2)

# filter based on x_cut / y_cut
mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
df_dp = df.loc[mask1 & mask2]
dfn1_dp = dfn1.loc[(mask1 & mask2).values]
dfn2_dp = dfn2.loc[(mask1 & mask2).values]

# very simple plot -- line plot for each site of mean dp_test. Show that it's flat

f, ax = plt.subplots(1, 1)

colors = plt.cm.get_cmap('Blues', len(sites))
for i, s in enumerate(sites):
    norm = df_dp[df_dp.site==s]['dp_opt_test'].mean()
    ax.plot([0, 1, 2], [0,
                        dfn1_dp[dfn1_dp.site==s]['dp_opt_test'].mean() / norm - 1, 
                        dfn2_dp[dfn2_dp.site==s]['dp_opt_test'].mean() / norm - 1], 'o-', color=colors(i))

ax.set_ylabel(r"Fraction $d'^2$ improvement")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['TDR', 'TDR+1', 'TDR+2'])
ax.set_xlabel('Number of dimensions')

f.tight_layout()

f.savefig(fig_fn)

# compute stats
np.random.seed(123)
d = {s: dfn1_dp.loc[dfn1_dp.site==s, 'dp_opt_test'].values - df_dp.loc[df_dp.site==s, 'dp_opt_test'].values
                    for s in df_dp.site.unique()}
bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]

print("TDR+1 vs. TDR: pval: {0}".format(p))

print("Mean percent improvement: {0}".format((100 * (dfn1_dp.groupby(by='site').mean()['dp_opt_test'] - \
                                                    df_dp.groupby(by='site').mean()['dp_opt_test']) / \
                                                       df_dp.groupby(by='site').mean()['dp_opt_test'] ).mean()))

d = {s: dfn2_dp.loc[dfn2_dp.site==s, 'dp_opt_test'].values - dfn1_dp.loc[dfn1_dp.site==s, 'dp_opt_test'].values
                    for s in df_dp.site.unique()}
bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]

print("TDR+2 vs. TDR+1: pval: {0}".format(p))

plt.show()