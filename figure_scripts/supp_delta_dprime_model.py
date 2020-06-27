"""
Model the change in dprime for each site. Show consitency across sites.
"""

import colors as color
import ax_labels as alab
from path_settings import DPRIME_DIR, PY_FIGURES_DIR
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = False

path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR + 'supp_modeldprime.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
val = 'dp_opt_test'
estval = '_test'
high_var_only = True
all_sites = True
pred_heatmap = False
equi_density = False

# where to crop the data
if estval == '_train':
    x_cut = (3, 8.5)
    y_cut = (0.1, .45) 
elif estval == '_test':
    #x_cut = (1, 8)
    #y_cut = (0.2, 1) 
    x_cut = (2, 6)
    y_cut = (0, 1)

# set up subplots
if equi_density:
    f = plt.figure(figsize=(6, 3))
    scax = plt.subplot2grid((1, 2), (0, 0))
    cax = plt.subplot2grid((1, 2), (0, 1))
if pred_heatmap:
    f = plt.figure(figsize=(6, 3))
    scax = plt.subplot2grid((1, 2), (0, 0))
    hax = plt.subplot2grid((1, 2), (0, 1))
else:
    f = plt.figure(figsize=(4, 4))
    scax = plt.subplot2grid((1, 1), (0, 0))

if all_sites:
    sites = ALL_SITES
else:
    sites = HIGHR_SITES

df = []
for site in sites:
    if site in LOWR_SITES:
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

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

df_all = pd.concat(df)

# filter based on x_cut / y_cut
mask1 = (df_all['dU_mag'+estval] < x_cut[1]) & (df_all['dU_mag'+estval] > x_cut[0])
mask2 = (df_all['cos_dU_evec'+estval] < y_cut[1]) & (df_all['cos_dU_evec'+estval] > y_cut[0])
df = df_all[mask1 & mask2]

# linear model to predict delta dprime and overall dprime
# for each site (with data in this cropped window), fit the model(s)
beta_overall = []
beta_delta = []
ci_overall = []
ci_delta = []
pvals_overall = []
pvals_delta = []
highr_mask = []
rsquared = []
df['z_state_diff'] = np.nan
for s in df.site.unique():
    if s in HIGHR_SITES:
        highr_mask.append(True)
    else:
        highr_mask.append(False)
    X = df[df.site==s][['cos_dU_evec'+estval, 'dU_mag'+estval]]
    X['dU_mag'+estval] = X['dU_mag'+estval] - X['dU_mag'+estval].mean()
    X['dU_mag'+estval] /= X['dU_mag'+estval].std()
    X['cos_dU_evec'+estval] = X['cos_dU_evec'+estval] - X['dU_mag'+estval].mean()
    X['cos_dU_evec'+estval] /= X['cos_dU_evec'+estval].std()
    
    X = sm.add_constant(X)
    X['interaction'] = X['cos_dU_evec'+estval] * X['dU_mag'+estval]
    y = df[df.site==s]['state_diff']
    y -= y.mean()
    y /= y.std()
    df.loc[df.site==s, 'z_state_diff'] = y

    model = sm.OLS(y, X).fit()
    low_ci = model.conf_int().values[:,0]
    high_ci = model.conf_int().values[:,1]
    beta_delta.append(model.params.values)
    ci_delta.append(high_ci - low_ci)
    pvals_delta.append(model.pvalues)
    rsquared.append(model.rsquared)

    y = df[df.site==s]['dp_opt_test']
    y -= y.mean()
    y /= y.std()
    model = sm.OLS(y, X).fit()
    low_ci = model.conf_int().values[:,0]
    high_ci = model.conf_int().values[:,1]
    beta_overall.append(model.params.values)
    ci_overall.append(high_ci - low_ci)
    pvals_overall.append(model.pvalues)

beta_overall = np.stack(beta_overall)
beta_delta = np.stack(beta_delta)
pvals_overall = np.stack(pvals_overall)
pvals_delta = np.stack(pvals_delta)
highr_mask = np.array(highr_mask)
# plot beta weights
for bo, bd, po, pd, rs in zip(beta_overall, beta_delta, pvals_overall, pvals_delta, rsquared):
    scax.plot([bo[1], bo[2]], [bd[1], bd[2]], color='grey', zorder=1)

scax.scatter(beta_overall[highr_mask, 1], beta_delta[highr_mask, 1], color=color.COSTHETA, s=50, edgecolor='white', label=alab.COSTHETA_short, zorder=3)
scax.scatter(beta_overall[highr_mask, 2], beta_delta[highr_mask, 2], color=color.SIGNAL, s=50, edgecolor='white', label=alab.SIGNAL_short, zorder=3)
scax.legend(frameon=False)
scax.scatter(beta_overall[~highr_mask, 1], beta_delta[~highr_mask, 1], color=color.COSTHETA, marker='D', s=20, edgecolor='white', label=alab.COSTHETA_short, zorder=2)
scax.scatter(beta_overall[~highr_mask, 2], beta_delta[~highr_mask, 2], color=color.SIGNAL, marker='D', s=20, edgecolor='white', label=alab.SIGNAL_short, zorder=2)
scax.axhline(0, linestyle='--', color='k')
scax.axvline(0, linestyle='--', color='k')
scax.set_xlabel(r"$\beta$"
                " for "
                r"$d'^{2}$")
scax.set_ylabel(r"$\beta$"
                " for "
                r"$\Delta d'^{2}$")
scax.set_title("Regression coefficients")

# print statistics for reg. coefficients

if equi_density:
    # finally, get equi-density contours for each site to show distribution of data
    # for each experiment
    fd, a_dummy = plt.subplots(1, 1)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 15, 100)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    colors = plt.cm.get_cmap('Blues', len(df.site.unique()))
    for i, site in enumerate(df.site.unique()):
        # estimate kde and plot
        values = df[df.site==site][['cos_dU_evec'+estval, 'dU_mag'+estval]].values.T
        kde = ss.gaussian_kde(values)
        cont = np.reshape(kde(positions).T, xx.shape)
        cset = a_dummy.contour(xx, yy, cont, levels=1)
        seg = cset.allsegs[1][0]
        cax.plot(seg[:, 1], seg[:, 0], '-', color=colors(i), label=site, lw=2)

    cax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
    cax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
    cax.spines['bottom'].set_color(color.SIGNAL)
    cax.xaxis.label.set_color(color.SIGNAL)
    cax.tick_params(axis='x', colors=color.SIGNAL)
    cax.spines['left'].set_color(color.COSTHETA)
    cax.yaxis.label.set_color(color.COSTHETA)
    cax.tick_params(axis='y', colors=color.COSTHETA)
    cax.set_title("Equi-density contours")

    plt.close(fd)

if pred_heatmap:
    X = df[['cos_dU_evec'+estval, 'dU_mag'+estval]]
    #X['dU_mag'+estval] = X['dU_mag'+estval] - X['dU_mag'+estval].mean()
    #X['dU_mag'+estval] /= X['dU_mag'+estval].std()
    #X['cos_dU_evec'+estval] = X['cos_dU_evec'+estval] - X['dU_mag'+estval].mean()
    #X['cos_dU_evec'+estval] /= X['cos_dU_evec'+estval].std()
    X = sm.add_constant(X)
    X['interaction'] = X['cos_dU_evec'+estval] * X['dU_mag'+estval]
    y = df['z_state_diff']
    y -= y.mean()
    y /= y.std()
    model = sm.OLS(y, X).fit()

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()