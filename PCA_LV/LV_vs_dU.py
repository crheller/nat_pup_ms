"""
For each site, load decoding results, extact dU for each stimulus pair,
investigate how LVs (beta1 / beta2) related to dU for each stimulus pair.
Map this back onto areas where first order simulation shows big state 
effects vs. where second order simulation shows big state effects
"""

import ax_labels as alab
import colors as color

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


nbins = 20
high_var_only = True
estval = '_train'

sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']

# only crop the dprime value. Show count for everything
if estval == '_train':
    x_cut = (2, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 9)
    y_cut = (0.35, 1) 

df = []
for site in sites:
    loader = decoding.DecodingResults()
    path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
    modelname = 'dprime_jk10_zscore_nclv'
    fn = os.path.join(path, site, modelname+'_TDR.pickle')
    decoding_results = loader.load_results(fn)
    modelname = 'dprime_sim2_jk10_zscore_nclv'
    fn = os.path.join(path, site, modelname+'_TDR.pickle')
    sim2_results = loader.load_results(fn)

    stim = decoding_results.evoked_stimulus_pairs
    high_var_pairs = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/high_pvar_stim_combos.csv', index_col=0)
    high_var_pairs = high_var_pairs[high_var_pairs.site==site].index.get_level_values('combo')
    if high_var_only:
        stim = [s for s in stim if s in high_var_pairs]

    if len(stim) == 0:
        pass
    else:
        _df = sim2_results.numeric_results
        _df = _df.loc[pd.IndexSlice[stim, 2], :]
        _df['cos_dU_evec_test'] = decoding_results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
        _df['cos_dU_evec_train'] = decoding_results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]

        dU = sim2_results.get_result('wopt_all', stim, 2)[0]

        fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/pca_regression_lvs.pickle'
        # load results from pickle file
        with open(fn, 'rb') as handle:
            lv_results = pickle.load(handle)

        beta1 = lv_results[site]['beta1']
        beta2 = lv_results[site]['beta2']

        _df['cos_dU_b1'] = 0
        _df['cos_dU_b2'] = 0
        for idx in _df.index:
            i = pd.IndexSlice[idx[0], idx[1]]
            _dU = dU.loc[i].T
            _dU = _dU / np.linalg.norm(dU)

            cos_b1 = abs(_dU.dot(beta1)[0])
            cos_b2 = abs(_dU.dot(beta2)[0])

            _df.loc[i, 'cos_dU_b1'] = cos_b1
            _df.loc[i, 'cos_dU_b2'] = cos_b2

        _df['site'] = site
        df.append(_df)

df = pd.concat(df)

# filter based on x_cut / y_cut
mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
#mask3 = (df['beta1_snr'] < 2000) & (df['beta2_snr'] < 2000)
df_dp = df[mask1 & mask2]

df_dp['state_diff'] = ((df_dp['bp_dp'] - df_dp['sp_dp']) / df_dp['dp_opt_test']).values

# plot 4 heatmaps
# overall dprime
# delta dprime
# cos b1
# cos b2

f, ax = plt.subplots(2, 2, figsize=(6, 6))

df_dp.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='dp_opt_test', 
                  gridsize=nbins, ax=ax[0, 0], cmap='Greens', vmin=0) 
ax[0, 0].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[0, 0].set_ylabel(alab.COSTHETA, color=color.COSTHETA)
ax[0, 0].spines['bottom'].set_color(color.SIGNAL)
ax[0, 0].spines['bottom'].set_lw(2)
ax[0, 0].xaxis.label.set_color(color.SIGNAL)
ax[0, 0].tick_params(axis='x', colors=color.SIGNAL)
ax[0, 0].spines['left'].set_color(color.COSTHETA)
ax[0, 0].spines['left'].set_lw(2)
ax[0, 0].yaxis.label.set_color(color.COSTHETA)
ax[0, 0].tick_params(axis='y', colors=color.COSTHETA)
ax[0, 0].set_title(r"$d'^2$")

df_dp.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='state_diff', 
                  gridsize=nbins, ax=ax[0, 1], cmap='PRGn', vmin=-1.5, vmax=1.5) 
ax[0, 1].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[0, 1].set_ylabel(alab.COSTHETA, color=color.COSTHETA)
ax[0, 1].spines['bottom'].set_color(color.SIGNAL)
ax[0, 1].spines['bottom'].set_lw(2)
ax[0, 1].xaxis.label.set_color(color.SIGNAL)
ax[0, 1].tick_params(axis='x', colors=color.SIGNAL)
ax[0, 1].spines['left'].set_color(color.COSTHETA)
ax[0, 1].spines['left'].set_lw(2)
ax[0, 1].yaxis.label.set_color(color.COSTHETA)
ax[0, 1].tick_params(axis='y', colors=color.COSTHETA)
ax[0, 1].set_title(r"$\Delta d'^2$")

df_dp.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='cos_dU_beta1', 
                  gridsize=nbins, ax=ax[1, 0], cmap='Greens', vmin=0, vmax=0.1) 
ax[1, 0].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[1, 0].set_ylabel(alab.COSTHETA, color=color.COSTHETA)
ax[1, 0].spines['bottom'].set_color(color.SIGNAL)
ax[1, 0].spines['bottom'].set_lw(2)
ax[1, 0].xaxis.label.set_color(color.SIGNAL)
ax[1, 0].tick_params(axis='x', colors=color.SIGNAL)
ax[1, 0].spines['left'].set_color(color.COSTHETA)
ax[1, 0].spines['left'].set_lw(2)
ax[1, 0].yaxis.label.set_color(color.COSTHETA)
ax[1, 0].tick_params(axis='y', colors=color.COSTHETA)
ax[1, 0].set_title(r"$cos(\theta_{\beta_{1}, \Delta \mu})$")

df_dp.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='beta2_snr', 
                  gridsize=nbins, ax=ax[1, 1], cmap='Greens') #, vmin=0, vmax=.1) 
ax[1, 1].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[1, 1].set_ylabel(alab.COSTHETA, color=color.COSTHETA)
ax[1, 1].spines['bottom'].set_color(color.SIGNAL)
ax[1, 1].spines['bottom'].set_lw(2)
ax[1, 1].xaxis.label.set_color(color.SIGNAL)
ax[1, 1].tick_params(axis='x', colors=color.SIGNAL)
ax[1, 1].spines['left'].set_color(color.COSTHETA)
ax[1, 1].spines['left'].set_lw(2)
ax[1, 1].yaxis.label.set_color(color.COSTHETA)
ax[1, 1].tick_params(axis='y', colors=color.COSTHETA)
ax[1, 1].set_title(r"$cos(\theta_{\beta_{2}, \Delta \mu})$")

f.tight_layout()

plt.show()