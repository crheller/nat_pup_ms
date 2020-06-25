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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig4_modeldprime.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
val = 'dp_opt_test'
estval = '_test'
nbins = 5
cmap = 'Greens'
high_var_only = True
center = False
smooth = True
pred = False # plot prediction of delta dprime

# where to crop the data
if estval == '_train':
    x_cut = (3, 8.5)
    y_cut = (0.1, .45) 
elif estval == '_test':
    x_cut = (1, 8)
    y_cut = (0.2, 1) 

# set up subplots
f = plt.figure(figsize=(9, 6))

dpax = plt.subplot2grid((2, 3), (0, 0))
hax = plt.subplot2grid((2, 3), (1, 0))
lax1 = plt.subplot2grid((2, 3), (1, 1))
lax2 = plt.subplot2grid((2, 3), (0, 1))
lax3 = plt.subplot2grid((2, 3), (1, 2))
lax4 = plt.subplot2grid((2, 3), (0, 2))

#'bbl086b'
sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'DRX006b.e1:64', 'DRX006b.e65:128', 
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

# plot delta dprime (or prediction)
X = df[['dU_mag'+estval, 'cos_dU_evec'+estval]].copy()
X['interaction'] = X['dU_mag'+estval] * X['cos_dU_evec'+estval]
X = X - X.mean(axis=0)
X = X / X.std(axis=0)
X = sm.add_constant(X)
y = df['state_diff'].copy()
y -= y.mean()
y /= y.std()
ols = sm.OLS(y, X)
results = ols.fit()
df['pred'] = ols.predict(results.params)
if pred:
    val = 'pred'
    vmin = -0.25
    vmax = 0.1
    cmap = 'Greens'
else:
    val = 'state_diff'
    vmin = -4
    vmax = 4

# loop over each site, compute zscore of delta dprime,
# bin, then average across sites
hm = []
xbins = np.linspace(1, 8, nbins)
ybins = np.linspace(0.2, 1, nbins)
for s in df.site.unique():
        vals = df[df.site==s]['state_diff']
        vals -= vals.mean()
        vals /= vals.std()
        heatmap = ss.binned_statistic_2d(x=df[df.site==s]['dU_mag'+estval], 
                                    y=df[df.site==s]['cos_dU_evec'+estval],
                                    values=vals,
                                    statistic='mean',
                                    bins=[xbins, ybins])
        hm.append(heatmap.statistic.T / np.nanmax(heatmap.statistic))
t = np.nanmean(np.stack(hm), 0)

if smooth:
    im = hax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='gaussian', 
                                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=-.5, vmax=.5)
else:
    im = hax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='none', 
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=-.25, vmax=.25)
divider = make_axes_locatable(hax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')

hax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
hax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
hax.spines['bottom'].set_color(color.SIGNAL)
hax.xaxis.label.set_color(color.SIGNAL)
hax.tick_params(axis='x', colors=color.SIGNAL)
hax.spines['left'].set_color(color.COSTHETA)
hax.yaxis.label.set_color(color.COSTHETA)
hax.tick_params(axis='y', colors=color.COSTHETA)
hax.set_title(r"$\Delta d'^2$ (z-score)")

grouped = df.groupby(by='site')
cbins = 3 #np.linspace(0.3, 1, 4)
dbins = 3 #np.linspace(3, 8, 4)
du_dp = []
cos_dp = []
du_delta = []
cos_delta = []
cos_bin = []
du_bin = []
colors = plt.cm.get_cmap('Reds', len(df.site.unique()))
for i, g in enumerate(grouped):
    if center:
        g[1]['state_diff'] -= g[1]['state_diff'].mean()
        #g[1]['state_diff'] /= g[1]['state_diff'].std()
        g[1]['dp_opt_test'] -= g[1]['dp_opt_test'].mean()
        #g[1]['dp_opt_test'] /= g[1]['dp_opt_test'].std()
    cos = ss.binned_statistic(g[1]['cos_dU_evec_test'], g[1]['state_diff'], statistic='mean', bins=cbins)
    du = ss.binned_statistic(g[1]['dU_mag_test'], g[1]['state_diff'], statistic='mean', bins=dbins)
    du_val_delt = du.statistic 
    #du_val_delt -= du_val_delt.mean()
    cos_val_delt = cos.statistic 
    #cos_val_delt -= cos_val_delt.mean()
    #du_val_delt /= du_val_delt.std()
    #cos_val_delt /= cos_val_delt.std()

    du_delta.append(du_val_delt)
    cos_delta.append(cos_val_delt)

    lax1.plot(cos.bin_edges[1:], cos_val_delt, color=colors(i), zorder=2)
    lax3.plot(du.bin_edges[1:], du_val_delt, color=colors(i), zorder=2)

    cos = ss.binned_statistic(g[1]['cos_dU_evec_test'], g[1]['dp_opt_test'], statistic='mean', bins=cbins)
    du = ss.binned_statistic(g[1]['dU_mag_test'], g[1]['dp_opt_test'], statistic='mean', bins=dbins)
    du_val = du.statistic 
    #du_val -= du_val.mean()
    cos_val = cos.statistic 
    #cos_val -= cos_val.mean()
    #du_val /= du_val.std()
    #cos_val /= cos_val.std()

    du_dp.append(du_val)
    cos_dp.append(cos_val)

    lax2.plot(cos.bin_edges[1:], cos_val, color=colors(i), zorder=1)
    lax4.plot(du.bin_edges[1:], du_val, color=colors(i), zorder=1)

    cos_bin.append(cos.bin_edges[1:])
    du_bin.append(du.bin_edges[1:])

cos_bin = np.nanmean(np.stack(cos_bin), axis=0)
du_bin = np.nanmean(np.stack(du_bin), axis=0)

du_val_delt = np.nanmean(np.stack(du_delta), axis=0)
cos_val_delt = np.nanmean(np.stack(cos_delta), axis=0)

lax1.plot(cos_bin, cos_val_delt, '-o', lw=3, color='k', label=r"$\Delta d'^{2}$", zorder=4)
lax3.plot(du_bin, du_val_delt, '-o', lw=3, color='k', label=r"$\Delta d'^{2}$", zorder=4)

du_val = np.nanmean(np.stack(du_dp), axis=0)
cos_val = np.nanmean(np.stack(cos_dp), axis=0)

lax2.plot(cos_bin, cos_val, '-o', lw=3, color='k', label=r"$d'^{2}$", zorder=3)
lax4.plot(du_bin, du_val, '-o', lw=3, color='k', label=r"$d'^{2}$", zorder=3)

lax1.set_xlabel(alab.COSTHETA, color=color.COSTHETA)
lax1.spines['bottom'].set_color(color.COSTHETA)
lax1.xaxis.label.set_color(color.COSTHETA)
lax1.tick_params(axis='x', colors=color.COSTHETA)
lax1.set_title('Discriminability Improvement')

lax2.set_xlabel(alab.COSTHETA, color=color.COSTHETA)
lax2.spines['bottom'].set_color(color.COSTHETA)
lax2.xaxis.label.set_color(color.COSTHETA)
lax2.tick_params(axis='x', colors=color.COSTHETA)
lax2.set_title('Overall Discriminability')

lax3.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
lax3.spines['bottom'].set_color(color.SIGNAL)
lax3.xaxis.label.set_color(color.SIGNAL)
lax3.tick_params(axis='x', colors=color.SIGNAL)

lax4.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
lax4.spines['bottom'].set_color(color.SIGNAL)
lax4.xaxis.label.set_color(color.SIGNAL)
lax4.tick_params(axis='x', colors=color.SIGNAL)

lax1.set_ylabel(r"$\Delta d'^{2}$")
lax2.set_ylabel(r"$d'^{2}$")
lax3.set_ylabel(r"$\Delta d'^{2}$")
lax4.set_ylabel(r"$d'^{2}$")

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()

"""

f, ax = plt.subplots(1, 2, figsize=(8, 4))
nbins = 4
cos_center = []
du_center = []
cos_diff = []
du_diff = []
cos_dp = []
du_dp = []

for s in df.site.unique():
    dft = df[df.site==s]
    dft['state_diff'] -= dft['state_diff'].mean()
    dft['state_diff'] /= dft['state_diff'].std()
    dft['dp_opt_test'] -= dft['dp_opt_test'].mean()
    dft['dp_opt_test'] /= dft['dp_opt_test'].std()

    dfcos = dft.sort_values(by='cos_dU_evec_test')
    dfdu = dft.sort_values(by='dU_mag_test')

    win_size = int(dfcos.shape[0] / (nbins+1))
    rol_windows = np.arange(win_size, dfcos.shape[0], win_size)
    rol_windows = [int(x) for x in np.linspace(win_size, dfcos.shape[0], nbins)]
    cosx = [np.roll(dfcos['cos_dU_evec_test'].values, r)[:win_size].mean() for r in rol_windows]
    cosy_diff = [np.roll(dfcos['state_diff'].values, r)[:win_size].mean() for r in rol_windows]
    cosy_dp = [np.roll(dfcos['dp_opt_test'].values, r)[:win_size].mean() for r in rol_windows]

    dux = [np.roll(dfdu['dU_mag_test'].values, r)[:win_size].mean() for r in rol_windows]
    duy_diff = [np.roll(dfdu['state_diff'].values, r)[:win_size].mean() for r in rol_windows]
    duy_dp = [np.roll(dfdu['dp_opt_test'].values, r)[:win_size].mean() for r in rol_windows]

    ax[0].plot(cosx, cosy_dp, color='lightpink')
    ax[0].plot(cosx, cosy_diff, color='grey')

    ax[1].plot(dux, duy_dp, color='lightpink')
    ax[1].plot(dux, duy_diff, color='grey')

f.tight_layout()

plt.show()
"""