"""
Demonstrate that both first and second order differences between large and small pupil 
contibute. 
Show that effects are in different areas of the heatmap.
    (with heatmap? Or with bar plots per quadrant? Or with linear regression model?)
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
import scipy.ndimage.filters as sf
import scipy.stats as ss
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = False

path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR + 'fig5_decoding_simulation.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
sim1 = 'dprime_simInTDR_sim1_jk10_zscore_nclvz_fixtdr2'
sim2 = 'dprime_simInTDR_sim2_jk10_zscore_nclvz_fixtdr2'
sim12 = 'dprime_simInTDR_sim12_jk10_zscore_nclvz_fixtdr2'
estval = '_test'

all_sites = True
nbins = 8
vmin = -.2
vmax = .2
high_var_only = False
persite = True
smooth = True
sigma = 2
cmap = 'Greens'

# where to crop the data
if estval == '_train':
    x_cut = (2.5, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    #x_cut = (1, 8)
    #y_cut = (0.2, 1) 
    x_cut = (1.5, 6)
    y_cut = (0, 1)

# set up subplots
f = plt.figure(figsize=(12, 3))

bax = plt.subplot2grid((1, 4), (0, 0))
s1ax = plt.subplot2grid((1, 4), (0, 1))
s2ax = plt.subplot2grid((1, 4), (0, 2))
s12ax = plt.subplot2grid((1, 4), (0, 3))

if all_sites:
    sites = ALL_SITES
else:
    sites = HIGHR_SITES

df = []
df_sim1 = []
df_sim2 = []
df_sim12 = []
for site in sites:
    if site in LOWR_SITES: mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

    if site in LOWR_SITES: mn = sim1.replace('_jk10', '_jk1_eev') 
    else: mn = sim1
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim1 = loader.load_results(fn)
    _df_sim1 = results_sim1.numeric_results

    if site in LOWR_SITES: mn = sim2.replace('_jk10', '_jk1_eev') 
    else: mn = sim2
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim2 = loader.load_results(fn)
    _df_sim2 = results_sim2.numeric_results

    if site in LOWR_SITES: mn = sim12.replace('_jk10', '_jk1_eev') 
    else: mn = sim12
    fn = os.path.join(path, site, mn+'_TDR.pickle')
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

df_all = pd.concat(df)
df_sim1_all = pd.concat(df_sim1)
df_sim2_all = pd.concat(df_sim2)
df_sim12_all = pd.concat(df_sim12)

# filter based on x_cut / y_cut
mask1 = (df_all['dU_mag'+estval] < x_cut[1]) & (df_all['dU_mag'+estval] > x_cut[0])
mask2 = (df_all['cos_dU_evec'+estval] < y_cut[1]) & (df_all['cos_dU_evec'+estval] > y_cut[0])
df = df_all[mask1 & mask2]
df_sim1 = df_sim1_all[mask1 & mask2]
df_sim2 = df_sim2_all[mask1 & mask2]
df_sim12 = df_sim12_all[mask1 & mask2]

# append the simulation results as columns in the raw dataframe
df['sim1'] = df_sim1['state_diff']
df['sim2'] = df_sim2['state_diff']
df['sim12'] = df_sim12['state_diff']
df['sim2'] = df['sim12'] - df['sim1']

# compute z-scored (within site) state_diff results
df['sim1_z'] = np.nan
df['sim2_z'] = np.nan
df['sd_z'] = np.nan
for s in df.site.unique():
    z_sim1 = df.loc[df.site==s, 'sim1']
    z_sim1 -= z_sim1.mean()
    z_sim1 /= z_sim1.std()
    df.loc[df.site==s, 'sim1_z'] = z_sim1

    z_sim2 = df.loc[df.site==s, 'sim2']
    z_sim2 -= z_sim2.mean()
    z_sim2 /= z_sim2.std()
    df.loc[df.site==s, 'sim2_z'] = z_sim2

    z_sd = df.loc[df.site==s, 'state_diff']
    z_sd -= z_sd.mean()
    z_sd /= z_sd.std()
    df.loc[df.site==s, 'sd_z'] = z_sd

# bar plot of delta dprime for raw data, 1st order, and 2nd order simulation
if not persite:
    bax.bar([0, 1, 2, 3], 
            [df['state_diff'].mean(), df['sim1'].mean(), df['sim2'].mean(), df['sim12'].mean()],
            yerr=[df['state_diff'].sem(), df['sim1'].sem(), df['sim2'].sem(), df['sim12'].sem()],
            edgecolor='k', color=['lightgrey'], lw=1, width=0.5)

else:
    #df.groupby(by='site').mean()[['state_diff', 'sim1', 'sim2', 'sim12']].T.plot(color='lightgrey', legend=False, ax=bax)
    #bax.bar([0, 1, 2, 3], df.groupby(by='site').mean()[['state_diff', 'sim1', 'sim2', 'sim12']].mean(), 
    #                    yerr=df.groupby(by='site').mean()[['state_diff', 'sim1', 'sim2', 'sim12']].sem(),
    #                    color='lightgrey', edgecolor='k', lw=1, zorder=1)
    for i, s in zip([0, 1, 2, 3], ['state_diff', 'sim1', 'sim2', 'sim12']):
        try:
            vals = df.loc[df.site.isin(LOWR_SITES)].groupby(by='site').mean()[s]
            bax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.05, len(vals)),
                        vals, color='grey', marker='D', edgecolor='white', s=30, zorder=2)
        except:
            pass
        vals = df.loc[df.site.isin(HIGHR_SITES)].groupby(by='site').mean()[s]
        bax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.05, len(vals)),
                    vals, color='k', marker='o', edgecolor='white', s=50, zorder=3)
    bax.axhline(0, linestyle='--', color='grey', lw=2)     

bax.set_xticks([0, 1, 2, 3])
bax.set_xticklabels(['Raw', '1st order', '2nd order', '1st + 2nd'], rotation=45)
bax.set_ylabel(r"$\Delta d'^{2}$")
bax.set_title('Discriminability Improvement')
if not persite:
    bax.set_ylim((-0, 1.5))

# plot delta dprime heatmaps for 1st and 2nd order
hm = []
xbins = np.linspace(x_cut[0], x_cut[1], nbins)
ybins = np.linspace(y_cut[0], y_cut[1], nbins)
for s in df.site.unique():
        vals = df[df.site==s]['sim1']
        vals -= vals.mean()
        vals /= vals.std()
        heatmap = ss.binned_statistic_2d(x=df[df.site==s]['dU_mag'+estval], 
                                    y=df[df.site==s]['cos_dU_evec'+estval],
                                    values=vals,
                                    statistic='mean',
                                    bins=[xbins, ybins])
        hm.append(heatmap.statistic.T / np.nanmax(heatmap.statistic))
t = np.nanmean(np.stack(hm), 0)

vmin=-0.1
vmax=0.1
if smooth:
    #im = s1ax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='gaussian', 
    #                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
    t = sf.gaussian_filter(t, sigma)
    im = s1ax.imshow(t, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
else:
    im = s1ax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='none', 
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
divider = make_axes_locatable(s1ax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')

#df.plot.hexbin(x='dU_mag'+estval, 
#                  y='cos_dU_evec'+estval, 
#                  C='sim1', 
#                  gridsize=nbins, ax=s1ax, cmap=cmap, vmin=-3, vmax=3) 
s1ax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
s1ax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
s1ax.spines['bottom'].set_color(color.SIGNAL)
s1ax.xaxis.label.set_color(color.SIGNAL)
s1ax.tick_params(axis='x', colors=color.SIGNAL)
s1ax.spines['left'].set_color(color.COSTHETA)
s1ax.yaxis.label.set_color(color.COSTHETA)
s1ax.tick_params(axis='y', colors=color.COSTHETA)
s1ax.set_title(r"$\Delta d'^2$ (z-score)"+"\n 1st-order")


hm = []
xbins = np.linspace(x_cut[0], x_cut[1], nbins)
ybins = np.linspace(y_cut[0], y_cut[1], nbins)
for s in df.site.unique():
        vals = df[df.site==s]['sim2']
        vals -= vals.mean()
        vals /= vals.std()
        heatmap = ss.binned_statistic_2d(x=df[df.site==s]['dU_mag'+estval], 
                                    y=df[df.site==s]['cos_dU_evec'+estval],
                                    values=vals,
                                    statistic='mean',
                                    bins=[xbins, ybins])
        hm.append(heatmap.statistic.T / np.nanmax(heatmap.statistic))
t = np.nanmean(np.stack(hm), 0)

vmin=-0.05
vmax=0.05
if smooth:
    #im = s2ax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='gaussian', 
    #                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
    t = sf.gaussian_filter(t, sigma)
    im = s2ax.imshow(t, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
else:
    im = s2ax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='none', 
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
divider = make_axes_locatable(s2ax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
#df.plot.hexbin(x='dU_mag'+estval, 
#                  y='cos_dU_evec'+estval, 
#                  C='sim2', 
#                  gridsize=nbins, ax=s2ax, cmap=cmap, vmin=-3, vmax=3) 
s2ax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
s2ax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
s2ax.spines['bottom'].set_color(color.SIGNAL)
s2ax.xaxis.label.set_color(color.SIGNAL)
s2ax.tick_params(axis='x', colors=color.SIGNAL)
s2ax.spines['left'].set_color(color.COSTHETA)
s2ax.yaxis.label.set_color(color.COSTHETA)
s2ax.tick_params(axis='y', colors=color.COSTHETA)
s2ax.set_title(r"$\Delta d'^2$ (z-score)"+"\n 2nd-order")


hm = []
xbins = np.linspace(x_cut[0], x_cut[1], nbins)
ybins = np.linspace(y_cut[0], y_cut[1], nbins)
for s in df.site.unique():
        vals = df[df.site==s]['sim12']
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
    #im = s12ax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='gaussian', 
    #                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
    t = sf.gaussian_filter(t, sigma)
    im = s12ax.imshow(t, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
else:
    im = s12ax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='none', 
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
divider = make_axes_locatable(s12ax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
#df.plot.hexbin(x='dU_mag'+estval, 
#                  y='cos_dU_evec'+estval, 
#                  C='sim2', 
#                  gridsize=nbins, ax=s2ax, cmap=cmap, vmin=-3, vmax=3) 
s12ax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
s12ax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
s12ax.spines['bottom'].set_color(color.SIGNAL)
s12ax.xaxis.label.set_color(color.SIGNAL)
s12ax.tick_params(axis='x', colors=color.SIGNAL)
s12ax.spines['left'].set_color(color.COSTHETA)
s12ax.yaxis.label.set_color(color.COSTHETA)
s12ax.tick_params(axis='y', colors=color.COSTHETA)
s12ax.set_title(r"$\Delta d'^2$ (z-score)"+"\n 1st+2nd-order")

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()