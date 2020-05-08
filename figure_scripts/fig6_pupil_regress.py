"""
Illustrate that first and second order effects are independent by regressing out
pupil-explained variance. Show that delta noise correlations and decoding improvement
persist. 

Show that overall noise correlations can be predicted from first order effects, while
delta noise correlations cannot.
"""


import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = True
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig6_pupil_regression.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_pr_jk10_zscore'
sim1 = 'dprime_sim1_pr_jk10_zscore'
sim2 = 'dprime_sim2_pr_jk10_zscore'
estval = '_train'
nbins = 20
cmap = 'PRGn'
high_var_only = True

# where to crop the data
if estval == '_train':
    x_cut = (2, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 9)
    y_cut = (0.35, 1) 

# set up subplots
f = plt.figure(figsize=(4, 4))

dbax = plt.subplot2grid((1, 1), (0, 0))


sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
        'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
        'DRX007a.e1:64', 'DRX007a.e65:128', 
        'DRX008b.e1:64', 'DRX008b.e65:128']
df = []
df_sim1 = []
df_sim2 = []
for site in sites:
    fn = os.path.join(path, site, modelname+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

    fn = os.path.join(path, site, sim1+'_TDR.pickle')
    results_sim1 = loader.load_results(fn)
    _df_sim1 = results_sim1.numeric_results

    fn = os.path.join(path, site, sim2+'_TDR.pickle')
    results_sim2 = loader.load_results(fn)
    _df_sim2 = results_sim2.numeric_results

    stim = results.evoked_stimulus_pairs
    high_var_pairs = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/high_pvar_stim_combos.csv', index_col=0)
    high_var_pairs = high_var_pairs[high_var_pairs.site==site].index.get_level_values('combo')
    if high_var_only:
        stim = [s for s in stim if s in high_var_pairs]
    if len(stim) == 0:
        pass
    else:
        _df = _df.loc[pd.IndexSlice[stim, 2], :]
        _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
        _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
        _df['state_diff'] = (_df['bp_dp'] - _df['sp_dp']) / _df['dp_opt_test']
        _df['site'] = site
        df.append(_df)

        _df_sim1 = _df_sim1.loc[pd.IndexSlice[stim, 2], :]
        _df_sim1['state_diff'] = (_df_sim1['bp_dp'] - _df_sim1['sp_dp']) / _df['dp_opt_test']
        _df_sim1['site'] = site
        df_sim1.append(_df_sim1)

        _df_sim2 = _df_sim2.loc[pd.IndexSlice[stim, 2], :]
        _df_sim2['state_diff'] = (_df_sim2['bp_dp'] - _df_sim2['sp_dp']) / _df['dp_opt_test']
        _df_sim2['site'] = site
        df_sim2.append(_df_sim2)

df = pd.concat(df)
df_sim1 = pd.concat(df_sim1)
df_sim2 = pd.concat(df_sim2)

# filter based on x_cut / y_cut
mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
df = df[mask1 & mask2]
df_sim1 = df_sim1[mask1 & mask2]
df_sim2 = df_sim2[mask1 & mask2]

# append the simulation results as columns in the raw dataframe
df['sim1'] = df_sim1['state_diff']
df['sim2'] = df_sim2['state_diff']

# bar plot of delta dprime for raw data, 1st order, and 2nd order simulation
dbax.bar([0, 1, 2], 
        [df['state_diff'].mean(), df['sim1'].mean(), df['sim2'].mean()],
        yerr=[df['state_diff'].sem(), df['sim1'].sem(), df['sim2'].sem()],
        edgecolor='k', color=['k', 'orange', 'blue'], lw=2, width=0.5)
dbax.set_xticks([0, 1, 2])
dbax.set_xticklabels(['Raw', '1st order', '2nd order'])
dbax.set_ylabel(r"$\Delta d'^{2}$")


f.tight_layout()

plt.show()