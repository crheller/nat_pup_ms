"""
Determine how first (beta1) and second order (beta2) latent variables map onto
decoding space.
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
mpl.rcParams.update({'svg.fonttype': 'none'})

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclv'
sim1_mn = 'dprime_sim1_jk10_zscore_nclv'
sim2_mn = 'dprime_sim2_jk10_zscore_nclv'
estval = '_train'
nbins = 20

high_var_only = True
recache = True

# only crop the dprime value. Show count for everything
if estval == '_train':
    x_cut = (3, 8.5)
    y_cut = (0.1, .45) 
elif estval == '_test':
    x_cut = (1, 9)
    y_cut = (0.35, 1) 

sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']

df = []
sim1 = []
sim2 = []
if recache:
    for site in sites:
        fn = os.path.join(path, site, modelname+'_TDR.pickle')
        results = loader.load_results(fn)
        _df = results.numeric_results

        fn = os.path.join(path, site, sim1_mn+'_TDR.pickle')
        sim1_results = loader.load_results(fn)
        _sim1 = sim1_results.numeric_results

        fn = os.path.join(path, site, sim2_mn+'_TDR.pickle')
        sim2_results = loader.load_results(fn)
        _sim2 = sim2_results.numeric_results

        stim = results.evoked_stimulus_pairs
        high_var_pairs = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/high_pvar_stim_combos.csv', index_col=0)
        high_var_pairs = high_var_pairs[high_var_pairs.site==site].index.get_level_values('combo')
        if high_var_only:
            hv = 'highvar'
            stim = [s for s in stim if s in high_var_pairs]
        else:
            hv = 'all'
        if len(stim) == 0:
            pass
        else:
            _df = _df.loc[pd.IndexSlice[stim, 2], :]
            _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
            _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
            _df['site'] = site
            df.append(_df)

            _sim1 = _sim1.loc[pd.IndexSlice[stim, 2], :]
            _sim1['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
            _sim1['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
            _sim1['site'] = site
            sim1.append(_sim1)

            _sim2 = _sim2.loc[pd.IndexSlice[stim, 2], :]
            _sim2['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
            _sim2['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
            _sim2['site'] = site
            sim2.append(_sim2)

    df = pd.concat(df)
    sim1 = pd.concat(sim1)
    sim2 = pd.concat(sim2)

    print('caching concatenated results')
    df.to_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/{0}_{1}.csv'.format(modelname, hv))
    sim1.to_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/{0}_{1}.csv'.format(sim1_mn, hv))
    sim2.to_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/{0}_{1}.csv'.format(sim2_mn, hv))


else:
    if high_var_only:
        hv = 'highvar'
    else:
        hv = 'all'
    print('loading results from cache...')
    df = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/{0}_{1}.csv'.format(modelname, hv), index_col=0)
    sim1 = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/{0}_{1}.csv'.format(sim1_mn, hv), index_col=0)
    sim2 = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/{0}_{1}.csv'.format(sim2_mn, hv), index_col=0)


# filter based on x_cut / y_cut
mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
df_dp = df[mask1 & mask2]
sim1_dp = sim1[mask1 & mask2]
sim2_dp = sim2[mask1 & mask2]


# first, plot beta1_snr / beta2_snr onto decoding space
f, ax = plt.subplots(2, 2, figsize=(8, 8))

df_dp.plot.hexbin(x='dU_mag'+estval, 
               y='cos_dU_evec'+estval, 
               C='beta1_snr', 
               gridsize=nbins, ax=ax[0, 0], cmap='Blues') 
ax[0, 0].set_title(r"$\beta_{1, SNR}$")
ax[0, 0].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[0, 0].set_ylabel(alab.COSTHETA, color=color.COSTHETA)

df_dp.plot.hexbin(x='dU_mag'+estval, 
               y='cos_dU_evec'+estval, 
               C='beta2_snr', 
               gridsize=nbins, ax=ax[0, 1], cmap='Blues') 
ax[0, 1].set_title(r"$\beta_{2, SNR}$")
ax[0, 1].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[0, 1].set_ylabel(alab.COSTHETA, color=color.COSTHETA)

df_dp.plot.hexbin(x='dU_mag'+estval, 
               y='cos_dU_evec'+estval, 
               C='cos_dU_beta1', 
               gridsize=nbins, ax=ax[1, 0], cmap='Blues') 
ax[1, 0].set_title(r"$\beta_{1} \cdot \frac{\Delta \mu}{||\Delta \mu||}$")
ax[1, 0].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[1, 0].set_ylabel(alab.COSTHETA, color=color.COSTHETA)

df_dp.plot.hexbin(x='dU_mag'+estval, 
               y='cos_dU_evec'+estval, 
               C='cos_dU_beta2', 
               gridsize=nbins, ax=ax[1, 1], cmap='Blues') 
ax[1, 1].set_title(r"$\beta_{2} \cdot \frac{\Delta \mu}{||\Delta \mu||}$")
ax[1, 1].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[1, 1].set_ylabel(alab.COSTHETA, color=color.COSTHETA)

f.tight_layout()

# predict delta dprime for 1st / 2nd order simulation as function of 
# overlap between dU and beta
df_dp['state_diff'] = ((df_dp['bp_dp'] - df_dp['sp_dp']) / df_dp['dp_opt_test']).values
sim1_dp['state_diff'] = ((sim1_dp['bp_dp'] - sim1_dp['sp_dp']) / df_dp['dp_opt_test']).values
sim2_dp['state_diff'] = ((sim2_dp['bp_dp'] - sim2_dp['sp_dp']) / df_dp['dp_opt_test']).values

df_dp['state_diff_sim1'] = sim1_dp['state_diff']
df_dp['state_diff_sim2'] = sim2_dp['state_diff']

mask = (df_dp['beta1_mag']<0.5)
mask = mask & (df_dp['cos_dU_beta1']>0.2)
dft = df_dp[mask]

vmin = -3
vmax = 3

f, ax = plt.subplots(1, 2, figsize=(8, 4))

dft.plot.hexbin(x='beta1_mag', 
               y='cos_dU_beta1', 
               C='state_diff_sim1', 
               gridsize=nbins, ax=ax[0], cmap='PuOr', vmin=vmin, vmax=vmax) 
ax[0].set_title(r"$\Delta d'$ first order")
ax[0].set_xlabel(r"$||\beta_{1}||$")
ax[0].set_ylabel(r"$cos(\theta_{\Delta \mu, \beta_{1}})$")

dft.plot.hexbin(x='beta1_mag', 
               y='cos_dU_beta1', 
               C=None, 
               gridsize=nbins, ax=ax[1], cmap='Reds', vmin=0) 
ax[1].set_xlabel(r"$||\beta_{1}||$")
ax[1].set_ylabel(r"$cos(\theta_{\Delta \mu, \beta_{1}})$")

f.tight_layout()

mask = (df_dp['beta2_mag']<0.5)
mask = mask & (df_dp['cos_dU_beta2']>0.2)
dft2 = df_dp[mask]

f, ax = plt.subplots(1, 2, figsize=(8, 4))

dft2.plot.hexbin(x='beta2_mag', 
               y='cos_dU_beta2', 
               C='state_diff_sim2', 
               gridsize=nbins, ax=ax[0], cmap='PuOr', vmin=vmin, vmax=vmax) 
ax[0].set_title(r"$\Delta d'$ second order")
ax[0].set_xlabel(r"$||\beta_{2}||$")
ax[0].set_ylabel(r"$cos(\theta_{\Delta \mu, \beta_{2}})$")

dft2.plot.hexbin(x='beta2_mag', 
               y='cos_dU_beta2', 
               C=None, 
               gridsize=nbins, ax=ax[1], cmap='Reds', vmin=0) 
ax[1].set_xlabel(r"$||\beta_{2}||$")
ax[1].set_ylabel(r"$cos(\theta_{\Delta \mu, \beta_{2}})$")

f.tight_layout()

# plot state_diff as function of beta1 mag and beta2 mag
dft = df_dp[(df_dp['beta1_mag']<0.6) & (df_dp['beta2_mag']<0.6)]
f, ax = plt.subplots(1, 2, figsize=(8, 4))

dft.plot.hexbin(x='beta1_mag',
                  y='beta2_mag',
                  C='state_diff_sim1',
                  gridsize=nbins, ax=ax[0], cmap='PuOr', vmin=vmin, vmax=vmax)
ax[0].set_xlabel(r"$||\beta_{1}||$")
ax[0].set_ylabel(r"$||\beta_{2}||$")
ax[0].set_title(r"$\Delta d'$")

dft.plot.hexbin(x='beta1_mag',
                  y='beta2_mag',
                  C=None,
                  gridsize=nbins, ax=ax[1], cmap='Reds', vmin=0)
ax[1].set_xlabel(r"$||\beta_{1}||$")
ax[1].set_ylabel(r"$||\beta_{2}||$")

f.tight_layout()

# plot binned 1D values
mask = (df_dp['beta1_dot_wopt']<0.3) & (df_dp['beta2_dot_wopt']<0.3)
mask = mask & (df_dp['beta1_dot_wopt']>0) & (df_dp['beta2_dot_wopt']>0)
dft = df_dp[mask]

nbins = 10

out_1 = ss.binned_statistic(dft['beta1_dot_wopt'], dft['state_diff'], statistic='mean', bins=nbins)
out_2 = ss.binned_statistic(dft['beta2_dot_wopt'], dft['state_diff'], statistic='mean', bins=nbins)

out_sim1 = ss.binned_statistic(dft['beta1_dot_wopt'], dft['state_diff_sim1'], statistic='mean', bins=nbins)
out_b1_sim2 = ss.binned_statistic(dft['beta1_dot_wopt'], dft['state_diff_sim2'], statistic='mean', bins=nbins)
out_sim2 = ss.binned_statistic(dft['beta2_dot_wopt'], dft['state_diff_sim2'], statistic='mean', bins=nbins)
out_b2_sim1 = ss.binned_statistic(dft['beta2_dot_wopt'], dft['state_diff_sim1'], statistic='mean', bins=nbins)

sim1_count = ss.binned_statistic(dft['beta1_dot_wopt'], dft['state_diff'], statistic='count', bins=nbins).statistic
sim2_count = ss.binned_statistic(dft['beta2_dot_wopt'], dft['state_diff'], statistic='count', bins=nbins).statistic

sf = 10
ylim = (0, 2)
f, ax = plt.subplots(1, 3, figsize=(12, 4))

r = np.round(np.corrcoef(out_1.bin_edges[1:], out_1.statistic)[0,1], 2)
ax[0].scatter(out_1.bin_edges[1:], out_1.statistic, s=sim1_count / sf, label='raw, r={}'.format(r))
r = np.round(np.corrcoef(out_1.bin_edges[1:], out_sim1.statistic)[0,1], 2)
ax[0].scatter(out_sim1.bin_edges[1:], out_sim1.statistic, s=sim1_count / sf, label='1st order sim, r={}'.format(r))
r = np.round(np.corrcoef(out_1.bin_edges[1:], out_b1_sim2.statistic)[0,1], 2)
ax[0].scatter(out_sim1.bin_edges[1:], out_b1_sim2.statistic, s=sim1_count / sf, label='2nd order sim, r={}'.format(r))
ax[0].legend(fontsize=6, frameon=False)
ax[0].set_ylabel(r"$\Delta d'$")
ax[0].set_xlabel(r"$\beta_{1} \cdot w_{opt}$")
ax[0].set_title('Decoding improvement vs. \n 1st-order overlap with decoding space', fontsize=8)
ax[0].set_ylim(ylim)

r = np.round(np.corrcoef(out_2.bin_edges[1:], out_2.statistic)[0,1], 2)
ax[1].scatter(out_2.bin_edges[1:], out_2.statistic, s=sim1_count / sf, label='raw, r={}'.format(r))
r = np.round(np.corrcoef(out_2.bin_edges[1:], out_b2_sim1.statistic)[0,1], 2)
ax[1].scatter(out_sim2.bin_edges[1:], out_b2_sim1.statistic, s=sim1_count / sf, label='1st order sim, r={}'.format(r))
r = np.round(np.corrcoef(out_2.bin_edges[1:], out_sim2.statistic)[0,1], 2)
ax[1].scatter(out_sim2.bin_edges[1:], out_sim2.statistic, s=sim1_count / sf, label='2nd order sim, r={}'.format(r))
ax[1].legend(fontsize=6, frameon=False)
ax[1].set_xlabel(r"$\beta_{2} \cdot w_{opt}$")
ax[1].set_ylim(ylim)
ax[1].set_title('Decoding improvement vs. \n 2nd-order overlap with decoding space', fontsize=8)

# plot beta1 vs beta2 correlation
out_b1b2 = ss.binned_statistic(dft['beta1_dot_wopt'], dft['beta2_dot_wopt'], statistic='mean', bins=nbins)
count_b1b2 = ss.binned_statistic(dft['beta1_dot_wopt'], dft['beta2_dot_wopt'], statistic='count', bins=nbins)
r = np.round(np.corrcoef(out_b1b2.bin_edges[1:], out_b1b2.statistic)[0,1], 2)
ax[2].scatter(out_b1b2.bin_edges[1:], out_b1b2.statistic, s=count_b1b2.statistic/sf, label="r={}".format(r))
ax[2].set_ylabel(r"$\beta_{2} \cdot w_{opt}$")
ax[2].set_xlabel(r"$\beta_{1} \cdot w_{opt}$")
ax[2].legend(fontsize=6, frameon=False)

f.tight_layout()

plt.show()