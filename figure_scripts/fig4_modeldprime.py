"""
Heatmap of delta dprime (in same cropped space as fig 2)
Model overall dprime (and delta dprime) in cropped space from fig 2. 
Compare model weights for predicting delta vs. predicting overall.
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

savefig = True

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig3_modeldprime.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore'
val = 'dp_opt_test'
estval = '_test'
nbins = 20
cmap = 'PRGn'
high_var_only = True

# where to crop the data
if estval == '_train':
    x_cut = (3, 8.5)
    y_cut = (0.1, .45) 
elif estval == '_test':
    x_cut = (1, 8)
    y_cut = (0.4, 1) 

# set up subplots
f = plt.figure(figsize=(6, 6))

dpax = plt.subplot2grid((2, 2), (0, 0))
hax = plt.subplot2grid((2, 2), (0, 1))
scax = plt.subplot2grid((2, 2), (1, 0))
cax = plt.subplot2grid((2, 2), (1, 1))

sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']
df = []
for site in sites:
    fn = os.path.join(path, site, modelname+'_TDR.pickle')
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

df = pd.concat(df)

# filter based on x_cut / y_cut
mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
df = df[mask1 & mask2]

# plot large vs. small dprime per site
dfg = df.groupby(by='site').mean()
mi = np.min([dfg['sp_dp'].min(), dfg['bp_dp'].min()])
ma = np.max([dfg['sp_dp'].max(), dfg['bp_dp'].max()])
dpax.scatter(dfg['sp_dp'], dfg['bp_dp'], color='k', s=50, edgecolor='white')
dpax.plot([mi, ma], [mi, ma], color='grey', linestyle='--')
dpax.set_xlabel('Small pupil')
dpax.set_ylabel('Large pupil')
dpax.set_title(r"$d'^{2}$")

# plot delta dprime
df.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='state_diff', 
                  gridsize=nbins, ax=hax, cmap=cmap, vmin=-3, vmax=3) 
hax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
hax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
hax.spines['bottom'].set_color(color.SIGNAL)
hax.spines['bottom'].set_lw(2)
hax.xaxis.label.set_color(color.SIGNAL)
hax.tick_params(axis='x', colors=color.SIGNAL)
hax.spines['left'].set_color(color.COSTHETA)
hax.spines['left'].set_lw(2)
hax.yaxis.label.set_color(color.COSTHETA)
hax.tick_params(axis='y', colors=color.COSTHETA)
hax.set_title(r"$\Delta d'^2$")

# linear model to predict delta dprime and overall dprime
# for each site (with data in this cropped window), fit the model(s)
beta_overall = []
beta_delta = []
ci_overall = []
ci_delta = []
for s in df.site.unique():
    X = df[df.site==s][['cos_dU_evec'+estval, 'dU_mag'+estval]]
    X['dU_mag'+estval] = X['dU_mag'+estval] - X['dU_mag'+estval].mean()
    X['dU_mag'+estval] /= X['dU_mag'+estval].std()
    X['cos_dU_evec'+estval] = X['cos_dU_evec'+estval] - X['dU_mag'+estval].mean()
    X['cos_dU_evec'+estval] /= X['cos_dU_evec'+estval].std()
    
    X = sm.add_constant(X)
    y = df[df.site==s]['state_diff']
    model = sm.OLS(y, X).fit()
    low_ci = model.conf_int().values[:,0]
    high_ci = model.conf_int().values[:,1]
    beta_delta.append(model.params.values)
    ci_delta.append(high_ci - low_ci)


    y = df[df.site==s]['dp_opt_test']
    model = sm.OLS(y, X).fit()
    low_ci = model.conf_int().values[:,0]
    high_ci = model.conf_int().values[:,1]
    beta_overall.append(model.params.values)
    ci_overall.append(high_ci - low_ci)

beta_overall = np.stack(beta_overall)
beta_delta = np.stack(beta_delta)

# plot beta weights
for bo, bd in zip(beta_overall, beta_delta):
    scax.plot([bo[1], bo[2]], [bd[1], bd[2]], color='grey', zorder=1)
scax.scatter(beta_overall[:, 1], beta_delta[:, 1], color=color.COSTHETA, s=50, edgecolor='white', label=alab.COSTHETA, zorder=2)
scax.scatter(beta_overall[:, 2], beta_delta[:, 2], color=color.SIGNAL, s=50, edgecolor='white', label=alab.SIGNAL, zorder=2)
scax.axhline(0, linestyle='--', color='k')
scax.axvline(0, linestyle='--', color='k')
scax.set_xlabel(r"$\beta_{k}$"
                " for "
                r"$d'^{2}$")
scax.set_ylabel(r"$\beta_{k}$"
                " for "
                r"$\Delta d'^{2}$")
scax.legend(frameon=False)
scax.set_title("Regression coefficients")

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
cax.spines['bottom'].set_lw(2)
cax.xaxis.label.set_color(color.SIGNAL)
cax.tick_params(axis='x', colors=color.SIGNAL)
cax.spines['left'].set_color(color.COSTHETA)
cax.spines['left'].set_lw(2)
cax.yaxis.label.set_color(color.COSTHETA)
cax.tick_params(axis='y', colors=color.COSTHETA)
cax.set_title("Equi-density contours")
#cax.legend()

plt.close(fd)

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()

