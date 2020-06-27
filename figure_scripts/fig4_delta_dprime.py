"""
Heatmap of delta dprime (in same cropped space as fig 2)
Model overall dprime (and delta dprime) in cropped space from fig 2. 
Compare model weights for predicting delta vs. predicting overall.
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
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

savefig = False

path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR + 'fig4_modeldprime.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
val = 'dp_opt_test'
estval = '_test'
cmap = 'Greens'
high_var_only = False
all_sites = True
plot_individual = False # for line plots
reds = False  # if individual lines are plotted, color code them, or not
center = True
nline_bins = 8
smooth = True
per_site_heatmap = True # z-score dprime within site first, then sum over sites for heatmap
nbins = 8
vmin = -.15
vmax = .15

if all_sites:
    sites = ALL_SITES
else:
    sites = HIGHR_SITES

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
f = plt.figure(figsize=(9, 6))

dpax = plt.subplot2grid((2, 3), (0, 0))
hax = plt.subplot2grid((2, 3), (1, 0))
lax1 = plt.subplot2grid((2, 3), (1, 1))
lax2 = plt.subplot2grid((2, 3), (0, 1))
lax3 = plt.subplot2grid((2, 3), (1, 2))
lax4 = plt.subplot2grid((2, 3), (0, 2))

df = []
for site in sites:
    if site in LOWR_SITES: mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

    stim = results.evoked_stimulus_pairs
    high_var_pairs = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/high_pvar_stim_combos.csv', index_col=0)
    high_var_pairs = high_var_pairs[high_var_pairs.site==site].index.get_level_values('combo')
    if high_var_only:
        stim = high_var_pairs

    if len(stim) == 0:
        pass
    else:
        _df = _df.loc[_df.index.get_level_values('combo').isin(stim)]
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

# plot large vs. small dprime per site
dfg = df.groupby(by='site').mean()
mi = np.min([dfg['sp_dp'].min(), dfg['bp_dp'].min()])
ma = np.max([dfg['sp_dp'].max(), dfg['bp_dp'].max()])
dpax.scatter(dfg.loc[LOWR_SITES]['sp_dp'], dfg.loc[LOWR_SITES]['bp_dp'], marker="D", color='grey', s=30, edgecolor='white')
dpax.scatter(dfg.loc[HIGHR_SITES]['sp_dp'], dfg.loc[HIGHR_SITES]['bp_dp'], color='k', s=50, edgecolor='white')
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

val = 'state_diff'

# loop over each site, compute zscore of delta dprime,
# bin, then average across sites
hm = []
xbins = np.linspace(x_cut[0], x_cut[1], nbins)
ybins = np.linspace(y_cut[0], y_cut[1], nbins)
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
                                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
else:
    im = hax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='none', 
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
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
if plot_individual:
    cbins = nline_bins
    dbins = nline_bins
else:
    cbins = np.linspace(y_cut[0], y_cut[1], nline_bins)
    dbins = np.linspace(x_cut[0], x_cut[1], nline_bins)
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
        g[1]['state_diff'] /= g[1]['state_diff'].std()
        g[1]['dp_opt_test'] -= g[1]['dp_opt_test'].mean()
        g[1]['dp_opt_test'] /= g[1]['dp_opt_test'].std()
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

    if reds:
        lcol = colors(i)
    else:
        lcol = 'lightgrey'

    if plot_individual:
        lax1.plot(cos.bin_edges[1:], cos_val_delt, color=lcol, zorder=2)
        lax3.plot(du.bin_edges[1:], du_val_delt, color=lcol, zorder=2)

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

    if plot_individual:
        lax2.plot(cos.bin_edges[1:], cos_val, color=lcol, zorder=1)
        lax4.plot(du.bin_edges[1:], du_val, color=lcol, zorder=1)

    cos_bin.append(cos.bin_edges[1:])
    du_bin.append(du.bin_edges[1:])

cos_bin = np.nanmean(np.stack(cos_bin), axis=0)
du_bin = np.nanmean(np.stack(du_bin), axis=0)

du_val_delt = np.nanmean(np.stack(du_delta), axis=0)
du_val_delt_sem = np.nanstd(np.stack(du_delta), axis=0) / np.sqrt((~np.isnan(np.stack(du_delta))).sum(axis=0))
cos_val_delt = np.nanmean(np.stack(cos_delta), axis=0)
cos_val_delt_sem = np.nanstd(np.stack(cos_delta), axis=0) / np.sqrt((~np.isnan(np.stack(cos_delta))).sum(axis=0))

if plot_individual:
    lax1.plot(cos_bin, cos_val_delt, '-o', lw=3, color='k', label=r"$\Delta d'^{2}$", zorder=4)
    lax3.plot(du_bin, du_val_delt, '-o', lw=3, color='k', label=r"$\Delta d'^{2}$", zorder=4)
else:
    lax1.errorbar(cos_bin, cos_val_delt, yerr=cos_val_delt_sem, marker='.', lw=1, color='k', label=r"$\Delta d'^{2}$", zorder=4)
    lax3.errorbar(du_bin, du_val_delt, yerr=du_val_delt_sem, marker='.', lw=1, color='k', label=r"$\Delta d'^{2}$", zorder=4)
    if center:
        lax1.set_ylim((-0.3, 0.3))
        lax3.set_ylim((-0.3, 0.3))

du_val = np.nanmean(np.stack(du_dp), axis=0)
du_val_sem = np.nanstd(np.stack(du_dp), axis=0) / np.sqrt((~np.isnan(np.stack(du_dp))).sum(axis=0))
cos_val = np.nanmean(np.stack(cos_dp), axis=0)
cos_val_sem = np.nanstd(np.stack(cos_dp), axis=0) / np.sqrt((~np.isnan(np.stack(cos_dp))).sum(axis=0))

if plot_individual:
    lax2.plot(cos_bin, cos_val, '-o', lw=3, color='k', label=r"$d'^{2}$", zorder=3)
    lax4.plot(du_bin, du_val, '-o', lw=3, color='k', label=r"$d'^{2}$", zorder=3)
else:
    lax2.errorbar(cos_bin, cos_val, yerr=cos_val_sem, marker='.', lw=1, color='k', label=r"$\Delta d'^{2}$", zorder=4)
    lax4.errorbar(du_bin, du_val, yerr=du_val_sem, marker='.', lw=1, color='k', label=r"$\Delta d'^{2}$", zorder=4)
    if center:
        lax2.set_ylim((-1.1, 1.1))
        lax4.set_ylim((-1.1, 1.1))

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

lax1.set_ylabel(r"$\Delta d'^{2}$ (z-score)")
lax2.set_ylabel(r"$d'^{2}$ (z-score)")
lax3.set_ylabel(r"$\Delta d'^{2}$ (z-score)")
lax4.set_ylabel(r"$d'^{2}$ (z-score)")

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()