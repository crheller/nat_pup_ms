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
    x_cut = (1, 8)
    y_cut = (0.4, 1) 

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

df_dp = df.copy()
sim1_dp = sim1.copy()
sim2_dp = sim2.copy()
df_dp['state_diff'] = (df_dp['bp_dp'] - df_dp['sp_dp']) / df_dp['dp_opt_test']
df_dp['state_diff_sim1'] = (sim1_dp['bp_dp'] - sim1_dp['sp_dp']) / df_dp['dp_opt_test']
df_dp['state_diff_sim2'] = (sim2_dp['bp_dp'] - sim2_dp['sp_dp']) / df_dp['dp_opt_test']

# reminder of cos sim vs. theta relationship (nonlinear)
theta = np.arange(0, 90)
theta_rad = theta * (np.pi / 180)
cos_theta = np.cos(theta_rad)

f, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(theta, cos_theta)
ax.axhline(0.5, linestyle='--', color='grey')
ax.axvline(45, linestyle='--', color='grey')
ax.set_xlabel(r'$\theta$ (deg)')
ax.set_ylabel(r'$cos(\theta)$')
f.tight_layout()


# first illustrate distribution of cos sim between beta/dU and beta/w_opt.
bins = np.arange(0, 1, 0.01)
f, ax = plt.subplots(2, 3, figsize=(9, 6))

ax[0, 0].hist(df_dp['beta1_dot_dU'], bins=bins)
ax[0, 0].set_xlabel(r'$\beta_{1} \cdot \Delta \mu$')

ax[0, 1].hist(df_dp['beta1_dot_wopt'], bins=bins)
ax[0, 1].set_xlabel(r'$\beta_{1} \cdot w_{opt}$')

ax[0, 2].scatter(df_dp['beta1_dot_dU'], df_dp['beta1_dot_wopt'], s=5, alpha=0.3)
ax[0, 2].set_xlabel(r'$\beta_{1} \cdot \Delta \mu$')
ax[0, 2].set_ylabel(r'$\beta_{1} \cdot w_{opt}$')

ax[1, 0].hist(df_dp['beta2_dot_dU'], bins=bins)
ax[1, 0].set_xlabel(r'$\beta_{2} \cdot \Delta \mu$')

ax[1, 1].hist(df_dp['beta2_dot_wopt'], bins=bins)
ax[1, 1].set_xlabel(r'$\beta_{2} \cdot w_{opt}$')

ax[1, 2].scatter(df_dp['beta2_dot_dU'], df_dp['beta2_dot_wopt'], s=5, alpha=0.3)
ax[1, 2].set_xlabel(r'$\beta_{2} \cdot \Delta \mu$')
ax[1, 2].set_ylabel(r'$\beta_{2} \cdot w_{opt}$')

f.tight_layout()

# plot distribution of similarity between dU and wopt, and overlap between e1 and dU
f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].hist(df_dp['cos_dU_wopt_test'], bins=bins)
ax[0].set_xlabel(r'$\Delta \mu \cdot w_{opt}$')

ax[1].hist(df_dp['cos_dU_evec_test'], bins=bins)
ax[1].set_xlabel(r'$\Delta \mu \cdot e_{1}$')

f.tight_layout()


# linear models of state difference for sim1 and sim2 as fn of beta_dot_dU and e1_dot_dU
# filter based on x_cut / y_cut
mask1 = (df_dp['dU_mag'+estval] < x_cut[1]) & (df_dp['dU_mag'+estval] > x_cut[0])
mask2 = (df_dp['cos_dU_evec'+estval] < y_cut[1]) & (df_dp['cos_dU_evec'+estval] > y_cut[0])
df_dp = df_dp[mask1 & mask2]
sim1_dp = sim1_dp[mask1 & mask2]
sim2_dp = sim2_dp[mask1 & mask2]

test_train = '_test'
X = df_dp[['beta1_dot_dU', 'cos_dU_evec'+test_train]]
X['interaction'] = df_dp['beta1_dot_dU'] * df_dp['cos_dU_evec'+test_train]
X = sm.add_constant(X)
y = df_dp['state_diff_sim1']
ols = sm.OLS(y, X)
results1 = ols.fit()
X['pred'] = ols.predict(results1.params)

X2 = df_dp[['beta2_dot_dU', 'cos_dU_evec'+test_train]]
X2['interaction'] = df_dp['beta2_dot_dU'] * df_dp['cos_dU_evec'+test_train]
X2 = sm.add_constant(X2)
y = df_dp['state_diff_sim2']
ols = sm.OLS(y, X2)
results2 = ols.fit()
X2['pred'] = ols.predict(results2.params)

f, ax = plt.subplots(2, 2, figsize=(8.5, 8))

X.plot.hexbin(x='beta1_dot_dU',
              y='cos_dU_evec'+test_train,
              C='pred',
              cmap='PRGn',
              vmin=-1,
              vmax=1,
              gridsize=nbins,
              ax=ax[0, 0])
ax[0, 0].set_xlabel(r'$\beta_{1} \cdot \Delta \mu$')
ax[0, 0].set_ylabel(r'$e_{1} \cdot \Delta \mu$')
ax[0, 0].set_title(r"$\Delta \hat{d'}$ 1st-order sim")

X2.plot.hexbin(x='beta2_dot_dU',
              y='cos_dU_evec'+test_train,
              C='pred',
              cmap='PRGn',
              vmin=-1,
              vmax=1,
              gridsize=nbins,
              ax=ax[0, 1])
ax[0, 1].set_xlabel(r'$\beta_{2} \cdot \Delta \mu$')
ax[0, 1].set_ylabel(r'$e_{1} \cdot \Delta \mu$')
ax[0, 1].set_title(r"$\Delta \hat{d'}$ 2nd-order sim")

df_dp.plot.hexbin(x='beta1_dot_dU',
              y='cos_dU_evec'+test_train,
              C='state_diff_sim1',
              cmap='PRGn',
              vmin=-3,
              vmax=3,
              gridsize=nbins,
              ax=ax[1, 0])
ax[1, 0].set_xlabel(r'$\beta_{1} \cdot \Delta \mu$')
ax[1, 0].set_ylabel(r'$e_{1} \cdot \Delta \mu$')
ax[1, 0].set_title(r"$\Delta d'$ 1st-order sim")

df_dp.plot.hexbin(x='beta2_dot_dU',
              y='cos_dU_evec'+test_train,
              C='state_diff_sim2',
              cmap='PRGn',
              vmin=-2,
              vmax=2,
              gridsize=nbins,
              ax=ax[1, 1])
ax[1, 1].set_xlabel(r'$\beta_{2} \cdot \Delta \mu$')
ax[1, 1].set_ylabel(r'$e_{1} \cdot \Delta \mu$')
ax[1, 1].set_title(r"$\Delta d'$ 2nd-order sim")

f.tight_layout()

plt.show()
