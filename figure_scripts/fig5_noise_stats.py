"""
Idea is to replace current figure 5 (10-2-2020). 

Want to come up with a simpler way to differentiate between first / 
second order (population stuff) effects that doesn't require using the simulated data.

Think it can be done with combo of eigenvec / dU stats and (maybe) pupil regression
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
import colors as color
import ax_labels as alab

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.statistics as stats
import os
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
import scipy.ndimage.filters as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

np.random.seed(123)  # for reproducible bootstrap standard error generation

savefig = False

recache = False
ALL_TRAIN_DATA = False  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = HIGHR_SITES
path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR + 'fig4_modeldprime.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
n_components = 2
#val = 'dp_opt_test'
bp_dp = 'bp_dp'
sp_dp = 'sp_dp'
estval = '_test'
cmap = 'Greens'
nline_bins = 6
smooth = True
sigma = 1.2
nbins = 20
cmap = 'Greens'
vmin = None #0.1 #-.1
vmax = None #0.3 #.1

# where to crop the data
x_cut = DU_MAG_CUT
y_cut = NOISE_INTERFERENCE_CUT

df = []
for site in sites:
    if (site in LOWR_SITES) | (ALL_TRAIN_DATA): mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results
    stim = results.evoked_stimulus_pairs
    _df = _df.loc[_df.index.get_level_values('combo').isin(stim)]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, n_components, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, n_components, idx=[0, 0])[0]
    _df['state_diff'] = (_df[bp_dp] - _df[sp_dp]) / _df['dp_opt_test']
    _df['state_diff_abs'] = (_df[bp_dp] - _df[sp_dp])
    _df['state_MI'] = (_df[bp_dp] - _df[sp_dp]) / (_df[bp_dp] + _df[sp_dp])
    _df['bp_dU_dot_evec_sq'] = results.slice_array_results('bp_dU_dot_evec_sq', stim, 2, idx=[0, 0])[0]
    _df['sp_dU_dot_evec_sq'] = results.slice_array_results('sp_dU_dot_evec_sq', stim, 2, idx=[0, 0])[0]
    _df['bp_evec_snr'] = results.slice_array_results('bp_evec_snr', stim, 2, idx=[0, 0])[0]
    _df['sp_evec_snr'] = results.slice_array_results('sp_evec_snr', stim, 2, idx=[0, 0])[0]
    _df['bp_lambda'] = results.slice_array_results('bp_evals', stim, 2, idx=[0])[0]
    _df['sp_lambda'] = results.slice_array_results('sp_evals', stim, 2, idx=[0])[0]
    _df['bp_cos_dU_evec'] = results.slice_array_results('bp_cos_dU_evec', stim, 2, idx=[0, 0])[0]
    _df['sp_cos_dU_evec'] = results.slice_array_results('sp_cos_dU_evec', stim, 2, idx=[0, 0])[0]
    _df['snr_diff'] = _df['bp_evec_snr'] - _df['sp_evec_snr']
    _df['site'] = site
    df.append(_df)

df_all = pd.concat(df)
df_all['lambda_diff'] = df_all['bp_lambda'] - df_all['sp_lambda']
df_all['mag_diff'] = df_all['bp_dU_mag'] - df_all['sp_dU_mag']
df_all['cos_dU_evec_diff'] = df_all['bp_cos_dU_evec'] - df_all['sp_cos_dU_evec']

df_cut = df_all[df_all['dp_opt_test'] > 5]

# KDE plot of signal to noise ratio along principal noise dimension for each stim pair
f, ax = plt.subplots(1, 4, figsize=(12, 3))

# SNR
x = df_all['sp_evec_snr'].values
y = df_all['bp_evec_snr'].values
xy = np.vstack((x, y))
m = np.max(xy)
z = ss.gaussian_kde(xy)(xy)
ax[0].scatter(x, y, c=z, s=10, edgecolor='')
ax[0].plot([0, m], [0, m], 'k--')
ax[0].set_title("Signal-to-noise ratio on noise axis\n"+r"$\frac{(\Delta \mathbf{\mu} \cdot \mathbf{e}_1)^2}{\lambda_1}$")
ax[0].set_xlabel(r"Small Pupil")
ax[0].set_ylabel(r"Big pupil")

# dU mag
x = df_all['sp_dU_mag'].values
y = df_all['bp_dU_mag'].values
xy = np.vstack((x, y))
m = np.max(xy)
z = ss.gaussian_kde(xy)(xy)
ax[1].scatter(x, y, c=z, s=10, edgecolor='')
ax[1].plot([0, m], [0, m], 'k--')
ax[1].set_title("Signal magnitude\n"+r"$|\Delta \mathbf{\mu}|$")
ax[1].set_xlabel(r"Small Pupil")
ax[1].set_ylabel(r"Big pupil")

# noise mag
x = df_all['sp_lambda'].values
y = df_all['bp_lambda'].values
xy = np.vstack((x, y))
m = np.max(xy)
z = ss.gaussian_kde(xy)(xy)
ax[2].scatter(x, y, c=z, s=10, edgecolor='')
ax[2].plot([0, m], [0, m], 'k--')
ax[2].set_title("Noise variance\n"+r"$\lambda_1$")
ax[2].set_xlabel(r"Small Pupil")
ax[2].set_ylabel(r"Big pupil")

# noise signal alignment
x = df_all['sp_cos_dU_evec'].values
y = df_all['bp_cos_dU_evec'].values
xy = np.vstack((x, y))
m = np.max(xy)
z = ss.gaussian_kde(xy)(xy)
ax[3].scatter(x, y, c=z, s=10, edgecolor='')
ax[3].plot([0, m], [0, m], 'k--')
ax[3].set_title("Noise interference\n"+r"$|cos(\theta_{\Delta \mathbf{\mu}, \mathbf{e}_1})|$")
ax[3].set_xlabel(r"Small Pupil")
ax[3].set_ylabel(r"Big pupil")

f.tight_layout()

# decompose the SNR measure into the relevant parts:
# noise/signal alignement (interaction), noise magnitude (second-order), signal magnitude (first-order)
# model changes in dprime as function of these elements
r2_all = []
r2_sig_unique = []
r2_noise_unique = []
r2_interference_unique = []
coefs = np.zeros((len(df_all.site.unique()), 4))
for i, s in enumerate(df_all.site.unique()):
    df = df_cut[df_cut.site==s]
    y = copy.deepcopy(df['state_MI'])
    #y -= y.mean()
    #y /= y.std()

    # full model
    X = copy.deepcopy(df[['cos_dU_evec_diff', 'lambda_diff', 'mag_diff']]) 
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    full_model = sm.OLS(y, X).fit() 
    r2_all.append(full_model.rsquared)
    coefs[i, :] = full_model.params.values[::-1]

    # signal unique
    X = copy.deepcopy(df[['cos_dU_evec_diff', 'lambda_diff', 'mag_diff']])
    X['mag_diff'] = np.random.permutation(X['mag_diff'].values)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    y = copy.deepcopy(df['state_MI'])
    model1 = sm.OLS(y, X).fit() 
    r2_sig_unique.append(full_model.rsquared - model1.rsquared)

    # noise unique
    X = copy.deepcopy(df[['cos_dU_evec_diff', 'lambda_diff', 'mag_diff']])
    X['lambda_diff'] = np.random.permutation(X['lambda_diff'].values)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    y = copy.deepcopy(df['state_MI'])
    model2 = sm.OLS(y, X).fit() 
    r2_noise_unique.append(full_model.rsquared - model2.rsquared)

    # interaction unique
    X = copy.deepcopy(df[['cos_dU_evec_diff', 'lambda_diff', 'mag_diff']])
    X['cos_dU_evec_diff'] = np.random.permutation(X['cos_dU_evec_diff'].values)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    y = copy.deepcopy(df['state_MI'])
    model3 = sm.OLS(y, X).fit() 
    r2_interference_unique.append(full_model.rsquared - model3.rsquared)

r2 = pd.DataFrame(columns=['Full', 'Signal', 'Noise', 'Interference'], data=np.vstack([r2_all, r2_sig_unique, r2_noise_unique, r2_interference_unique]).T) 
r2 = r2.melt()
r2['reg'] = np.concatenate([['Full Model']*len(r2_all),
                                ['Signal']*len(r2_all),
                               ['Noise']*len(r2_all),
                               ['Interference']*len(r2_all)])
r2 = r2.rename(columns={'value': r"$R^2$ unique", 'variable': 'Regressor'})

coefs = pd.DataFrame(columns=['Signal', 'Noise', 'Interference'], data=coefs[:, :-1]) 
coefs = coefs.melt()
coefs['reg'] = np.concatenate([['Signal']*len(r2_all),
                               ['Noise']*len(r2_all),
                               ['Interference']*len(r2_all)])
coefs = coefs.rename(columns={'value': 'Coefficient', 'variable': 'Regressor'})

f, ax = plt.subplots(1, 2, figsize=(8, 4))

sns.stripplot(y=r"$R^2$ unique", x='Regressor', data=r2, dodge=True, ax=ax[0], alpha=0.6)

sns.stripplot(y="Regressor", x='Coefficient', data=coefs, dodge=True, ax=ax[1], alpha=0.6,
                    palette={'Signal': 'tab:orange', 'Noise': 'tab:green', 'Interference': 'tab:red'})

ax[0].axhline(0, linestyle='--', color='k')
ax[1].axvline(0, linestyle='--', color='k')

f.tight_layout()

plt.show()