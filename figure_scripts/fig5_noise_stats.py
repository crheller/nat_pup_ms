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

savefig = True
fig_fn1 = PY_FIGURES_DIR + 'fig5_noise_stats1.svg'
fig_fn2 = PY_FIGURES_DIR + 'fig5_noise_stats2.svg'
fig_fn3 = PY_FIGURES_DIR + 'fig5_regression.svg'

recache = False
ALL_TRAIN_DATA = False  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = HIGHR_SITES
path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR + 'fig5_something.svg' 
loader = decoding.DecodingResults()
modelname_pr = 'dprime_pr_jk10_zscore_nclvz_fixtdr2'   # rm2 doesn't seem to work well for this analysis
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
n_components = 2
evals_idx = None # None (total variance)
bp_dp = 'bp_dp'
sp_dp = 'sp_dp'
estval = '_test'

# where to crop the data
x_cut = None #DU_MAG_CUT
y_cut = None #NOISE_INTERFERENCE_CUT

df = []
df_pr = []
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
    _df['bp_lambda'] = results.slice_array_results('bp_evals', stim, 2, idx=evals_idx)[0]
    _df['sp_lambda'] = results.slice_array_results('sp_evals', stim, 2, idx=evals_idx)[0]
    _df['bp_cos_dU_evec'] = results.slice_array_results('bp_cos_dU_evec', stim, 2, idx=[0, 0])[0]
    _df['sp_cos_dU_evec'] = results.slice_array_results('sp_cos_dU_evec', stim, 2, idx=[0, 0])[0]
    _df['snr_diff'] = _df['bp_evec_snr'] - _df['sp_evec_snr']
    _df['site'] = site
    df.append(_df)

    # pupil-corrected results:
    if (site in LOWR_SITES) | (ALL_TRAIN_DATA): mn = modelname_pr.replace('_jk10', '_jk1_eev') 
    else: mn = modelname_pr
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
    _df['bp_lambda'] = results.slice_array_results('bp_evals', stim, 2, idx=evals_idx)[0]
    _df['sp_lambda'] = results.slice_array_results('sp_evals', stim, 2, idx=evals_idx)[0]
    _df['bp_cos_dU_evec'] = results.slice_array_results('bp_cos_dU_evec', stim, 2, idx=[0, 0])[0]
    _df['sp_cos_dU_evec'] = results.slice_array_results('sp_cos_dU_evec', stim, 2, idx=[0, 0])[0]
    _df['snr_diff'] = _df['bp_evec_snr'] - _df['sp_evec_snr']
    _df['site'] = site
    df_pr.append(_df)

df_all = pd.concat(df)
df_all['lambda_diff'] = df_all['bp_lambda'] - df_all['sp_lambda']
df_all['mag_diff'] = df_all['bp_dU_mag'] - df_all['sp_dU_mag']
df_all['cos_dU_evec_diff'] = df_all['bp_cos_dU_evec'] - df_all['sp_cos_dU_evec']

df_all_pr = pd.concat(df_pr)
df_all_pr['lambda_diff'] = df_all_pr['bp_lambda'] - df_all_pr['sp_lambda']
df_all_pr['mag_diff'] = df_all_pr['bp_dU_mag'] - df_all_pr['sp_dU_mag']
df_all_pr['cos_dU_evec_diff'] = df_all_pr['bp_cos_dU_evec'] - df_all_pr['sp_cos_dU_evec']

# filter based on x_cut / y_cut
if (x_cut is not None) & (y_cut is not None):
    mask1 = (df_all['dU_mag'+estval] < x_cut[1]) & (df_all['dU_mag'+estval] > x_cut[0])
    mask2 = (df_all['cos_dU_evec'+estval] < y_cut[1]) & (df_all['cos_dU_evec'+estval] > y_cut[0])
else:
    mask1 = (True * np.ones(df_all.shape[0])).astype(np.bool)
    mask2 = (True * np.ones(df_all.shape[0])).astype(np.bool)
df_cut = df_all[mask1 & mask2]
df_cut_pr = df_all_pr[mask1 & mask2]

if evals_idx is None:
    df_cut['bp_lambda'] = df_cut['bp_lambda'].apply(lambda x: x.sum())
    df_cut['sp_lambda'] = df_cut['sp_lambda'].apply(lambda x: x.sum())
    df_cut['lambda_diff'] = df_cut['bp_lambda'] - df_cut['sp_lambda']

    df_cut_pr['bp_lambda'] = df_cut_pr['bp_lambda'].apply(lambda x: x.sum())
    df_cut_pr['sp_lambda'] = df_cut_pr['sp_lambda'].apply(lambda x: x.sum())
    df_cut_pr['lambda_diff'] = df_cut_pr['bp_lambda'] - df_cut_pr['sp_lambda']

# KDE plot of signal to noise ratio along principal noise dimension for each stim pair
f, ax = plt.subplots(1, 4, figsize=(12, 3))

# SNR
x = df_cut['sp_evec_snr'].values
y = df_cut['bp_evec_snr'].values
xy = np.vstack((x, y))
m = np.max(xy)
z = ss.gaussian_kde(xy)(xy)
ax[0].scatter(x, y, c=z, s=10, edgecolor='')
ax[0].plot([0, m], [0, m], 'k--')
ax[0].set_title("Signal-to-noise ratio on noise axis\n"+r"$\frac{(\Delta \mathbf{\mu} \cdot \mathbf{e}_1)^2}{\lambda_1}$")
ax[0].set_xlabel(r"Small Pupil")
ax[0].set_ylabel(r"Big pupil")

# dU mag
x = df_cut['sp_dU_mag'].values
y = df_cut['bp_dU_mag'].values
xy = np.vstack((x, y))
m = np.max(xy)
z = ss.gaussian_kde(xy)(xy)
ax[1].scatter(x, y, c=z, s=10, edgecolor='')
ax[1].plot([0, m], [0, m], 'k--')
ax[1].set_title("Signal magnitude\n"+r"$|\Delta \mathbf{\mu}|$")
ax[1].set_xlabel(r"Small Pupil")
ax[1].set_ylabel(r"Big pupil")

# noise mag
x = df_cut['sp_lambda'].values
y = df_cut['bp_lambda'].values
xy = np.vstack((x, y))
m = np.max(xy)
z = ss.gaussian_kde(xy)(xy)
ax[2].scatter(x, y, c=z, s=10, edgecolor='')
ax[2].plot([0, m], [0, m], 'k--')
ax[2].set_title("Noise variance\n"+r"$\lambda_1$")
ax[2].set_xlabel(r"Small Pupil")
ax[2].set_ylabel(r"Big pupil")

# noise signal alignment
x = df_cut['sp_cos_dU_evec'].values
y = df_cut['bp_cos_dU_evec'].values
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
coefs = np.zeros((len(df_cut.site.unique()), 4))
for i, s in enumerate(df_cut.site.unique()):
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

coefs = pd.DataFrame(columns=[r"$\Delta$ Signal magnitude", 
                                r"$\Delta$ Noise variance", 
                                r"$\Delta$ Noise interference"], data=coefs[:, :-1]) 
coefs = coefs.melt()
coefs['reg'] = np.concatenate([['Signal']*len(r2_all),
                               ['Noise']*len(r2_all),
                               ['Interference']*len(r2_all)])
coefs = coefs.rename(columns={'value': 'Coefficient', 'variable': 'Regressor'})

f, ax = plt.subplots(1, 2, figsize=(8, 4))

sns.stripplot(y=r"$R^2$ unique", x='Regressor', data=r2, dodge=True, ax=ax[0], alpha=0.3,
                    palette={'Full': 'tab:blue', 'Signal': 'tab:orange', 'Noise': 'tab:green', 'Interference': 'tab:red'})
sns.pointplot(y=r"$R^2$ unique", x='Regressor', data=r2, dodge=0.4, join=False, ci=95, ax=ax[0], errwidth=1, scale=0.7, capsize=0.05,
                    palette={'Full': 'tab:blue', 'Signal': 'tab:orange', 'Noise': 'tab:green', 'Interference': 'tab:red'})

sns.stripplot(y="Regressor", x='Coefficient', data=coefs, dodge=True, ax=ax[1], alpha=0.3,
                    palette={r"$\Delta$ Signal magnitude": 'tab:orange', 
                    r"$\Delta$ Noise variance": 'tab:green', 
                    r"$\Delta$ Noise interference": 'tab:red'})
sns.pointplot(y="Regressor", x='Coefficient', data=coefs, dodge=0.4, join=False, ci=95, ax=ax[1], errwidth=1, scale=0.7, capsize=0.05,
                    palette={r"$\Delta$ Signal magnitude": 'tab:orange', 
                             r"$\Delta$ Noise variance": 'tab:green', 
                             r"$\Delta$ Noise interference": 'tab:red'})

ax[1].axvline(0, linestyle='--', color='grey')
ax[0].axhline(0, linestyle='--', color='grey')
ax[1].set_title(r"$\Delta d'^2$"+"\nRegression coefficients")

f.tight_layout()

# plot the change in stats on the same axis as before
nbins = 20
cmap_first = 'PuOr'
cmap_second = 'PuOr'
vmax = None
f, ax = plt.subplots(1, 2, figsize=(9, 4))

df_cut.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='mag_diff', 
                  gridsize=nbins, ax=ax[0], cmap=cmap_first, vmin=-3, vmax=3) 
ax[0].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[0].set_ylabel(alab.COSTHETA, color=color.COSTHETA)
ax[0].spines['bottom'].set_color(color.SIGNAL)
ax[0].xaxis.label.set_color(color.SIGNAL)
ax[0].tick_params(axis='x', colors=color.SIGNAL)
ax[0].spines['left'].set_color(color.COSTHETA)
ax[0].yaxis.label.set_color(color.COSTHETA)
ax[0].tick_params(axis='y', colors=color.COSTHETA)
ax[0].set_title(r"$\Delta$ Signal Magnitude ($|\Delta \mathbf{\mu}|$)")

df_cut.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='lambda_diff', 
                  gridsize=nbins, ax=ax[1], cmap=cmap_second, vmin=-4, vmax=4) 
ax[1].set_xlabel(alab.SIGNAL, color=color.SIGNAL)
ax[1].set_ylabel(alab.COSTHETA, color=color.COSTHETA)
ax[1].spines['bottom'].set_color(color.SIGNAL)
ax[1].xaxis.label.set_color(color.SIGNAL)
ax[1].tick_params(axis='x', colors=color.SIGNAL)
ax[1].spines['left'].set_color(color.COSTHETA)
ax[1].yaxis.label.set_color(color.COSTHETA)
ax[1].tick_params(axis='y', colors=color.COSTHETA)
ax[1].set_title(r"$\Delta$ Noise Variance ($\lambda_1$)")

f.tight_layout()

# summary of signal vs. noise variance changes
ndf1 = df_cut[['bp_lambda', 'bp_dU_mag']].rename(columns={'bp_lambda': 'noise', 'bp_dU_mag': 'signal'})
ndf1['state'] = 'big'
ndf2 = df_cut[['sp_lambda', 'sp_dU_mag']].rename(columns={'sp_lambda': 'noise', 'sp_dU_mag': 'signal'})
ndf2['state'] = 'small'
ndf = pd.concat([ndf1, ndf2])

ndf = ndf[ndf['noise']<15]

g = sns.jointplot(data=ndf, x="noise", y="signal", hue="state", s=10, alpha=0.2)

# state_MI as fn of predictors
dd = df_cut[df_cut['lambda_diff']<15]
dd.plot.hexbin(x='mag_diff',  
                   y='lambda_diff',  
                   C='state_MI',  
                   gridsize=nbins, cmap=cmap_second, vmin=-1, vmax=1)    

plt.show()

# stats tests
# noise variance change
np.random.seed(123)
nboots = 1000
ds = {s: df_cut[(df_cut.site==s)]['lambda_diff'].values for s in df_cut.site.unique()}
ds_boot = stats.get_bootstrapped_sample(ds, nboot=nboots)

p = 1 - stats.get_direct_prob(ds_boot, np.zeros(nboots))[0]

print("big pupil variance vs. small pupil variance: \n" + \
                f"p = {p}\n" + \
                f"mean = {np.mean(ds_boot)}\n" + \
                f"sem  = {np.std(ds_boot)/np.sqrt(nboots)}\n")

nboots = 1000
ds = {s: df_cut[(df_cut.site==s)]['mag_diff'].values for s in df_cut.site.unique()}
ds_boot = stats.get_bootstrapped_sample(ds, nboot=nboots)

p = 1 - stats.get_direct_prob(np.zeros(nboots), ds_boot)[0]

print("big pupil dU vs. small pupil dU change: \n" + \
                f"p = {p}\n" + \
                f"mean = {np.mean(ds_boot)}\n" + \
                f"sem  = {np.std(ds_boot)/np.sqrt(nboots)}\n")

nboots = 1000
ds = {s: df_cut[(df_cut.site==s)]['cos_dU_evec_diff'].values for s in df_cut.site.unique()}
ds_boot = stats.get_bootstrapped_sample(ds, nboot=nboots)

p = 1 - stats.get_direct_prob(np.zeros(nboots), ds_boot)[0]

print("big pupil cos(dU, e) vs. small pupil cos(dU, e) change: \n" + \
                f"p = {p}\n" + \
                f"mean = {np.mean(ds_boot)}\n" + \
                f"sem  = {np.std(ds_boot)/np.sqrt(nboots)}")


# =============================================== FINAL SUMMARY FIGURES ================================================

# Final summary. Show that signal / noise / alignment all change (but alignment is negligible)
# Show that for pupil corrected, lamdba diff still has predictive power for delta dprime
# but mag diff does not. Suggests that first order, not variance stuff can be accounted for by single pupil dim

# seaborn jointplot is a massive pain in the ass, and creates new figure every time, so can't put all on the same plot

# jointplot summaries

# raw data
ndf1 = pd.concat([np.sqrt(df_cut['bp_lambda']), df_cut['bp_dU_mag']], axis=1).rename(columns={'bp_lambda': r"Shared noise variance ($\lambda$)", 
                    'bp_dU_mag': r"Signal Magnitude ($|\Delta \mathbf{\mu}|$)"})
ndf1['state'] = 'Big'
ndf2 = pd.concat([np.sqrt(df_cut['sp_lambda']), df_cut['sp_dU_mag']], axis=1).rename(columns={'sp_lambda': r"Shared noise variance ($\lambda$)", 
                    'sp_dU_mag': r"Signal Magnitude ($|\Delta \mathbf{\mu}|$)"})
ndf2['state'] = 'Small'
ndf = pd.concat([ndf1, ndf2])

g1 = sns.jointplot(data=ndf, x=r"Shared noise variance ($\lambda$)", 
                    y=r"Signal Magnitude ($|\Delta \mathbf{\mu}|$)", hue="state", s=10, alpha=0.2,
                    palette={'Big': color.LARGE, 'Small': color.SMALL}, ylim=(0, 16), xlim=(0, 6), rasterized=True)
g1.fig.canvas.set_window_title("Raw data")
g1.fig.set_size_inches(4, 4)
g1.fig.tight_layout()

# pupil corrected data
ndf1 = pd.concat([np.sqrt(df_cut_pr['bp_lambda']), df_cut_pr['bp_dU_mag']], axis=1).rename(columns={'bp_lambda': r"Shared noise variance ($\lambda$)", 
                'bp_dU_mag': r"Signal Magnitude ($|\Delta \mathbf{\mu}|$)"})
ndf1['state'] = 'Big'
ndf2 = pd.concat([np.sqrt(df_cut_pr['sp_lambda']), df_cut_pr['sp_dU_mag']], axis=1).rename(columns={'sp_lambda': r"Shared noise variance ($\lambda$)",
                'sp_dU_mag': r"Signal Magnitude ($|\Delta \mathbf{\mu}|$)"})
ndf2['state'] = 'Small'
ndf = pd.concat([ndf1, ndf2])
g2 = sns.jointplot(data=ndf, x=r"Shared noise variance ($\lambda$)", 
                    y=r"Signal Magnitude ($|\Delta \mathbf{\mu}|$)", hue="state", s=10, alpha=0.2,
                    palette={'Big': color.LARGE, 'Small': color.SMALL}, ylim=(0, 16), xlim=(0, 6), rasterized=True)
g2.fig.canvas.set_window_title("Pupil-corrected data")
g2.fig.set_size_inches(4, 4)
g2.fig.tight_layout()

if savefig:
    g1.ax_joint.figure.savefig(fig_fn1)
    g2.ax_joint.figure.savefig(fig_fn2)


# regression model results
r2_all = []
r2_sig = []
r2_sig_pr = []
r2_noise = []
r2_noise_pr = []
r2_interference = []
r2_interference_pr = []
coefs = np.zeros((len(df_cut.site.unique()), 4))
for i, s in enumerate(df_cut.site.unique()):
    df = df_cut[df_cut.site==s]
    df_pr = df_cut_pr[df_cut_pr.site==s]
    y = copy.deepcopy(df['state_MI'])

    # full model
    X = copy.deepcopy(df[['cos_dU_evec_diff', 'lambda_diff', 'mag_diff']]) 
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    full_model = sm.OLS(y, X).fit() 
    r2_all.append(full_model.rsquared)
    coefs[i, :] = full_model.params.values[::-1]

    # signal only
    X = copy.deepcopy(df[['mag_diff']])
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    model1 = sm.OLS(y, X).fit() 
    r2_sig.append(model1.rsquared)

    X = copy.deepcopy(df_pr[['mag_diff']])
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    model1 = sm.OLS(y, X).fit() 
    r2_sig_pr.append(model1.rsquared)

    # noise only
    X = copy.deepcopy(df[['lambda_diff']])
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    model2 = sm.OLS(y, X).fit() 
    r2_noise.append(model2.rsquared)

    X = copy.deepcopy(df_pr[['lambda_diff']])
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    model2 = sm.OLS(y, X).fit() 
    r2_noise_pr.append(model2.rsquared)

    # interaction only
    X = copy.deepcopy(df[['cos_dU_evec_diff']])
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    model3 = sm.OLS(y, X).fit() 
    r2_interference.append(model3.rsquared)

    X = copy.deepcopy(df_pr[['cos_dU_evec_diff']])
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X = sm.add_constant(X)
    model3 = sm.OLS(y, X).fit() 
    r2_interference_pr.append(model3.rsquared)

# R2 values for each model
r2 = pd.concat([pd.DataFrame(columns=[r"$\Delta$ Signal magnitude", r"$\Delta$ Shared noise variance", r"$\Delta$ Noise interference"], 
                             data=np.stack([r2_sig, r2_noise, r2_interference]).T),
                pd.DataFrame(columns=[r"$\Delta$ Signal magnitude", r"$\Delta$ Shared noise variance", r"$\Delta$ Noise interference"], 
                             data=np.stack([r2_sig_pr, r2_noise_pr, r2_interference_pr]).T)])
r2 = r2.melt()
r2['corrected'] = np.tile(np.concatenate(((False * np.ones(len(r2_sig)).astype(bool), (True * np.ones(len(r2_sig)).astype(bool))))), [3])
r2 = r2.rename(columns={'value': r'$R^2$', 'variable': 'Regressor'})

# model coefficients
coefs = pd.DataFrame(columns=[r"$\Delta$ Signal"+"\nmagnitude", 
                                r"$\Delta$ Shared"+"\nnoise variance", 
                                r"$\Delta$ Noise" +"\ninterference"], data=coefs[:, :-1]) 
coefs = coefs.melt()
coefs = coefs.rename(columns={'value': 'Coefficient', 'variable': 'Regressor'})

f, ax = plt.subplots(2, 2, figsize=(8, 8))

sns.stripplot(y="Regressor", x='Coefficient', data=coefs, dodge=True, ax=ax[1, 0], alpha=0.3, color='k')
sns.pointplot(y="Regressor", x='Coefficient', data=coefs, dodge=0.4, join=False, ci=95, ax=ax[1, 0], errwidth=1, scale=0.7, capsize=0.05, color='k')

g = sns.stripplot(x="Regressor", y=r'$R^2$', data=r2, dodge=True, ax=ax[1, 1], alpha=0.3, hue='corrected', palette={False: color.RAW, True: color.CORRECTED})
sns.pointplot(x="Regressor", y=r'$R^2$', data=r2, dodge=0.4, join=False, ci=95, errwidth=1, scale=0.7, capsize=0.05,
                    ax=ax[1, 1], alpha=0.3, hue='corrected', palette={False: color.RAW, True: color.CORRECTED})
g.axes.legend([], frameon=False)
g.axes.set_xticks(range(3))
g.axes.set_xticklabels([r"$\Delta$ Signal"+"\nmagnitude", r"$\Delta$ Shared noise"+"\nvariance", r"$\Delta$ Noise"+"\ninterference"], rotation=45)
g.axes.set_title(r"$\Delta d'^2$"+"\nExplained variance")
ax[1, 0].axvline(0, linestyle='--', color='grey')
ax[1, 1].axhline(0, linestyle='--', color='grey')
ax[1, 0].set_title(r"$\Delta d'^2$"+"\nRegression coefficients")

# plot simulated data for large / small pupil to illustrate changes in dU and lambda

# small pupil
np.random.seed(123)
u1 = [-1, .1]
u2 = [1, -.2]
cov = np.array([[1, 0.5], [0.5, 1]])
A = np.random.multivariate_normal(u1, cov, (200,))
B = np.random.multivariate_normal(u2, cov, (200,))
Ael = cplt.compute_ellipse(A[:, 0], A[:, 1])
Bel = cplt.compute_ellipse(B[:, 0], B[:, 1])

ax[0, 0].scatter(B[:, 0].mean(), B[:, 1].mean(), edgecolor='k', s=50, color='tab:orange')
ax[0, 0].scatter(A[:, 0].mean(), A[:, 1].mean(), edgecolor='k', s=50, color='tab:blue')
ax[0, 0].plot(Ael[0], Ael[1], color='tab:blue', lw=2)
ax[0, 0].plot(Bel[0], Bel[1], color='tab:orange', lw=2)
ax[0, 0].set_title("Small Pupil", color=color.SMALL)
ax[0, 0].set_xlabel(r"$\Delta \mathbf{\mu} (TDR_1)$")
ax[0, 0].set_ylabel(r"$TDR_2$")

u1 = [-2, -.1]
u2 = [2, .2]
cov = np.array([[.8, 0.2], [0.2, .8]]) 
A = np.random.multivariate_normal(u1, cov, (200,))
B = np.random.multivariate_normal(u2, cov, (200,))
Ael = cplt.compute_ellipse(A[:, 0], A[:, 1])
Bel = cplt.compute_ellipse(B[:, 0], B[:, 1])

ax[0, 1].scatter(B[:, 0].mean(), B[:, 1].mean(), edgecolor='k', s=50, color='tab:orange')
ax[0, 1].scatter(A[:, 0].mean(), A[:, 1].mean(), edgecolor='k', s=50, color='tab:blue')
ax[0, 1].plot(Bel[0], Bel[1], color='tab:orange', lw=2)
ax[0, 1].plot(Ael[0], Ael[1], color='tab:blue', lw=2)
ax[0, 1].set_title("Large Pupil", color=color.LARGE)
ax[0, 1].set_xlabel(r"$\Delta \mathbf{\mu} (TDR_1)$")
ax[0, 1].set_ylabel(r"$TDR_2$")

ax[0, 1].axis('equal')
ax[0, 0].axis('equal')

ylims = (np.min([ax[0, 0].get_ylim()[0], ax[0, 1].get_ylim()[0]]), np.max([ax[0, 0].get_ylim()[1], ax[0, 1].get_ylim()[1]]))
xlims = (np.min([ax[0, 0].get_xlim()[0], ax[0, 1].get_xlim()[0]]), np.max([ax[0, 0].get_xlim()[1], ax[0, 1].get_xlim()[1]]))
ax[0, 0].set_xlim(xlims)
ax[0, 0].set_ylim(ylims)
ax[0, 1].set_xlim(xlims)
ax[0, 1].set_ylim(ylims)

f.tight_layout()

if savefig:
    f.savefig(fig_fn3)

plt.show()