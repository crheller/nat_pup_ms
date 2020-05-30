"""
Demonstrate that both first and second order differences between large and small pupil 
contibute. 
Show that effects are in different areas of the heatmap.
    (with heatmap? Or with bar plots per quadrant? Or with linear regression model?)
"""

import colors as color
import ax_labels as alab

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
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig5_decoding_simulation.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
sim1 = 'dprime_sim1_jk10_zscore_nclvz_fixtdr2'
sim2 = 'dprime_sim2_jk10_zscore_nclvz_fixtdr2'
sim12 = 'dprime_sim12_jk10_zscore_nclvz_fixtdr2'
estval = '_test'
nbins = 20
cmap = 'PRGn'
high_var_only = True

# where to crop the data
if estval == '_train':
    x_cut = (2.5, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 8)
    y_cut = (0.2, 1) 

# set up subplots
f = plt.figure(figsize=(9, 3))

bax = plt.subplot2grid((1, 3), (0, 0))
s1ax = plt.subplot2grid((1, 3), (0, 1))
s2ax = plt.subplot2grid((1, 3), (0, 2))

sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
        'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
        'DRX007a.e1:64', 'DRX007a.e65:128', 
        'DRX008b.e1:64', 'DRX008b.e65:128']
df = []
df_sim1 = []
df_sim2 = []
df_sim12 = []
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

    fn = os.path.join(path, site, sim12+'_TDR.pickle')
    results_sim12 = loader.load_results(fn)
    _df_sim12 = results_sim12.numeric_results

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

        _df_sim12 = _df_sim12.loc[pd.IndexSlice[stim, 2], :]
        _df_sim12['state_diff'] = (_df_sim12['bp_dp'] - _df_sim12['sp_dp']) / _df['dp_opt_test']
        _df_sim12['site'] = site
        df_sim12.append(_df_sim12)

df = pd.concat(df)
df_sim1 = pd.concat(df_sim1)
df_sim2 = pd.concat(df_sim2)
df_sim12 = pd.concat(df_sim12)

# filter based on x_cut / y_cut
mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
df = df[mask1 & mask2]
df_sim1 = df_sim1[mask1 & mask2]
df_sim2 = df_sim2[mask1 & mask2]
df_sim12 = df_sim12[mask1 & mask2]

# append the simulation results as columns in the raw dataframe
df['sim1'] = df_sim1['state_diff']
df['sim2'] = df_sim2['state_diff']
df['sim12'] = df_sim12['state_diff']

# bar plot of delta dprime for raw data, 1st order, and 2nd order simulation
bax.bar([0, 1, 2, 3], 
        [df['state_diff'].mean(), df['sim1'].mean(), df['sim2'].mean(), df['sim12'].mean()],
        yerr=[df['state_diff'].sem(), df['sim1'].sem(), df['sim2'].sem(), df['sim12'].sem()],
        edgecolor='k', color=['lightgrey'], lw=2, width=0.5)
bax.set_xticks([0, 1, 2, 3])
bax.set_xticklabels(['Raw', '1st order', '2nd order', '1st + 2nd'], rotation=45)
bax.set_ylabel(r"$\Delta d'^{2}$")
bax.set_ylim((-0.1, 1.5))

# plot delta dprime heatmaps for 1st and 2nd order
df.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='sim1', 
                  gridsize=nbins, ax=s1ax, cmap=cmap, vmin=-3, vmax=3) 
s1ax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
s1ax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
s1ax.spines['bottom'].set_color(color.SIGNAL)
s1ax.spines['bottom'].set_lw(2)
s1ax.xaxis.label.set_color(color.SIGNAL)
s1ax.tick_params(axis='x', colors=color.SIGNAL)
s1ax.spines['left'].set_color(color.COSTHETA)
s1ax.spines['left'].set_lw(2)
s1ax.yaxis.label.set_color(color.COSTHETA)
s1ax.tick_params(axis='y', colors=color.COSTHETA)
s1ax.set_title(r"$\Delta d'^2$, 1st order")

df.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='sim2', 
                  gridsize=nbins, ax=s2ax, cmap=cmap, vmin=-3, vmax=3) 
s2ax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
s2ax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
s2ax.spines['bottom'].set_color(color.SIGNAL)
s2ax.spines['bottom'].set_lw(2)
s2ax.xaxis.label.set_color(color.SIGNAL)
s2ax.tick_params(axis='x', colors=color.SIGNAL)
s2ax.spines['left'].set_color(color.COSTHETA)
s2ax.spines['left'].set_lw(2)
s2ax.yaxis.label.set_color(color.COSTHETA)
s2ax.tick_params(axis='y', colors=color.COSTHETA)
s2ax.set_title(r"$\Delta d'^2$, 2nd order")

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()