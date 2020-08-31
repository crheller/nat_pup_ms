"""
Combine the simualtion results and the pupil regression results into one figure. Idea is 
really that this becomes a figure focued on first order decoding effects, but shows that
there's stuff left over than can't be attributed to first order. Adding second order simulation
completely recovers the raw results (in both cases). Thus, first order can't explain second
order effects.

One big motivation for combining results here is to get rid of using the heatmaps, which 
didn't seem all that helpful. Address the "where" of decoding improvements in figure 4
and in the supplemental figures.
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, DU_MAG_CUT, NOISE_INTERFERENCE_CUT
import colors as color
import ax_labels as alab

from nems_lbhb.baphy import parse_cellid

import charlieTools.statistics as stats
import charlieTools.preprocessing as preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import os
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

savefig = True

recache = False
ALL_TRAIN_DATA = False  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = HIGHR_SITES
path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR + 'fig5_simulations.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
sim1 = 'dprime_simInTDR_sim1_jk10_zscore_nclvz_fixtdr2'
sim12 = 'dprime_simInTDR_sim12_jk10_zscore_nclvz_fixtdr2'
modelname_pr = 'dprime_pr_rm2_jk10_zscore_nclvz_fixtdr2'
sim1_pr = 'dprime_simInTDR_sim1_pr_rm2_jk10_zscore_nclvz_fixtdr2'
sim12_pr = 'dprime_simInTDR_sim12_pr_rm2_jk10_zscore_nclvz_fixtdr2'
estval = '_test'
n_components = 2

barplot = False
smooth = True
collapse_across_sites = True
second_order_unique = True  #just for heatmap
mi_norm = True
sigma = 1.2
nbins = 20
cmap = 'Greens'
vmin = 0.05  #-0.05
vmax = 0.25   #0.05

# where to crop the data
if estval == '_train':
    x_cut = (2.5, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = None
    y_cut = None

x_cut_plot = DU_MAG_CUT
y_cut_plot = NOISE_INTERFERENCE_CUT

# ========================================= Load results ====================================================
df = []
df_sim1 = []
df_sim12 = []
df_pr = []
df_sim1_pr = []
df_sim12_pr = []
for site in sites:
    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results

    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = sim1.replace('_jk10', '_jk1_eev') 
    else: mn = sim1
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim1 = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df_sim1 = results_sim1.numeric_results

    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = sim12.replace('_jk10', '_jk1_eev') 
    else: mn = sim12
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim12 = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df_sim12 = results_sim12.numeric_results

    # pr results
    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = modelname_pr.replace('_jk10', '_jk1_eev') 
    else: mn = modelname_pr
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_pr = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df_pr = results_pr.numeric_results

    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = sim1_pr.replace('_jk10', '_jk1_eev') 
    else: mn = sim1_pr
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim1 = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df_sim1_pr = results_sim1.numeric_results

    if (site in LOWR_SITES) | ALL_TRAIN_DATA: mn = sim12_pr.replace('_jk10', '_jk1_eev') 
    else: mn = sim12_pr
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim12 = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df_sim12_pr = results_sim12.numeric_results

    stim = results.evoked_stimulus_pairs

    _df = _df.loc[pd.IndexSlice[stim, n_components], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, n_components, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, n_components, idx=[0, 0])[0]
    _df['state_diff'] = (_df['bp_dp'] - _df['sp_dp']) / _df['dp_opt_test']
    _df['state_MI'] = (_df['bp_dp'] - _df['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df['site'] = site
    df.append(_df)

    _df_sim1 = _df_sim1.loc[pd.IndexSlice[stim, n_components], :]
    _df_sim1['state_diff'] = (_df_sim1['bp_dp'] - _df_sim1['sp_dp']) / _df['dp_opt_test']
    _df_sim1['state_MI'] = (_df_sim1['bp_dp'] - _df_sim1['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df_sim1['site'] = site
    df_sim1.append(_df_sim1)

    _df_sim12 = _df_sim12.loc[pd.IndexSlice[stim, n_components], :]
    _df_sim12['state_diff'] = (_df_sim12['bp_dp'] - _df_sim12['sp_dp']) / _df['dp_opt_test']
    _df_sim12['state_MI'] = (_df_sim12['bp_dp'] - _df_sim12['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df_sim12['site'] = site
    df_sim12.append(_df_sim12)

    # pr results
    _df_pr = _df_pr.loc[pd.IndexSlice[stim, n_components], :]
    _df_pr['state_diff'] = (_df_pr['bp_dp'] - _df_pr['sp_dp']) / _df['dp_opt_test']
    _df_pr['state_MI'] = (_df_pr['bp_dp'] - _df_pr['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df_pr['site'] = site
    df_pr.append(_df_pr)

    _df_sim1_pr = _df_sim1_pr.loc[pd.IndexSlice[stim, n_components], :]
    _df_sim1_pr['state_diff'] = (_df_sim1_pr['bp_dp'] - _df_sim1_pr['sp_dp']) / _df['dp_opt_test']
    _df_sim1_pr['state_MI'] = (_df_sim1_pr['bp_dp'] - _df_sim1_pr['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df_sim1_pr['site'] = site
    df_sim1_pr.append(_df_sim1_pr)

    _df_sim12_pr = _df_sim12_pr.loc[pd.IndexSlice[stim, n_components], :]
    _df_sim12_pr['state_diff'] = (_df_sim12_pr['bp_dp'] - _df_sim12_pr['sp_dp']) / _df['dp_opt_test']
    _df_sim12_pr['state_MI'] = (_df_sim12_pr['bp_dp'] - _df_sim12_pr['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])
    _df_sim12_pr['site'] = site
    df_sim12_pr.append(_df_sim12_pr)

df_all = pd.concat(df)
df_sim1_all = pd.concat(df_sim1)
df_sim12_all = pd.concat(df_sim12)
df_pr_all = pd.concat(df_pr)
df_sim1_pr_all = pd.concat(df_sim1_pr)
df_sim12_pr_all = pd.concat(df_sim12_pr)

# filter based on x_cut / y_cut
if (x_cut is not None) & (y_cut is not None):
    mask1 = (df_all['dU_mag'+estval] < x_cut[1]) & (df_all['dU_mag'+estval] > x_cut[0])
    mask2 = (df_all['cos_dU_evec'+estval] < y_cut[1]) & (df_all['cos_dU_evec'+estval] > y_cut[0])
else:
    mask1 = (True * np.ones(df_all.shape[0])).astype(bool)
    mask2 = (True * np.ones(df_all.shape[0])).astype(bool)
df = df_all[mask1 & mask2]
df_sim1 = df_sim1_all[mask1 & mask2]
df_sim12 = df_sim12_all[mask1 & mask2]
df_pr = df_pr_all[mask1 & mask2]
df_sim1_pr = df_sim1_pr_all[mask1 & mask2]
df_sim12_pr = df_sim12_pr_all[mask1 & mask2]

if mi_norm:
    df['state_diff'] = df['state_MI']
    df['sim1'] = df_sim1['state_MI']
    df['pr'] = df_pr['state_MI']
    df['sim1_pr'] = df_sim1_pr['state_MI']
    df['sim12'] = df_sim12['state_MI']
    df['sim12_pr'] = df_sim12_pr['state_MI']
else:
    df['sim1'] = df_sim1['state_diff']
    df['pr'] = df_pr['state_diff']
    df['sim1_pr'] = df_sim1_pr['state_diff']
    df['sim12'] = df_sim12['state_diff']
    df['sim12_pr'] = df_sim12_pr['state_diff']

# ========================================= Plot data =====================================================
# set up subplots
f = plt.figure(figsize=(9, 6.2))

dax = plt.subplot2grid((2, 3), (0, 0))
dprax = plt.subplot2grid((2, 3), (1, 0))
prax = plt.subplot2grid((2, 3), (1, 1))
s1ax = plt.subplot2grid((2, 3), (0, 1))
s12ax = plt.subplot2grid((2, 3), (0, 2))


# plot dprime per site for the raw simulations
if barplot:
    dfg = df.groupby(by='site').mean()
    dax.bar([0, 1, 2], [dfg['state_diff'].mean(), dfg['sim1'].mean(), dfg['sim12'].mean()],
                        yerr=[dfg['state_diff'].sem(), dfg['sim1'].sem(), dfg['sim12'].sem()],
                        color='lightgrey', edgecolor='k', width=0.5)
else:
    for i, s in zip([0, 1, 2], ['state_diff', 'sim1', 'sim12']):
        try:
            vals = df.loc[df.site.isin(LOWR_SITES)].groupby(by='site').mean()[s]
            dax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.0, len(vals)),
                        vals, color='grey', marker='D', edgecolor='white', s=30, zorder=2)
        except:
            pass
        vals = df.loc[df.site.isin(HIGHR_SITES)].groupby(by='site').mean()[s]
        dax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.0, len(vals)),
                    vals, color='k', marker='o', edgecolor='white', s=50, zorder=3)

    # now, for each site draw lines between points in each model. Color red if 2nd order hurts, blue if helps
    line_colors = []
    for s in df.site.unique():
        vals = df.groupby(by='site').mean()[['state_diff', 'sim1', 'sim12']].loc[s].values
        if vals[1] < vals[2]:
            dax.plot([0, 1, 2], vals, color='blue', alpha=0.5, zorder=1)
            line_colors.append('blue')
        else:
            dax.plot([0, 1, 2], vals, color='red', alpha=0.5, zorder=1)
            line_colors.append('red')


    dax.axhline(0, linestyle='--', color='grey', lw=2)     
dax.set_xticks([0, 1, 2])
dax.set_xticklabels(['None', '1st order', '1st + 2nd'], rotation=45)
dax.set_xlabel('Simulation')
if mi_norm:
    dax.set_ylabel(r"$\Delta d'^{2}$")    
else:
    dax.set_ylabel(r"$\Delta d'^{2}$")
dax.set_title('Discriminability Change \n Raw Data', color=color.RAW)
if not mi_norm:
    dax.set_ylim((-1, 2))
else:
    dax.set_ylim((-.3, .5))

# print statistics comparing full simulation vs. raw data
print("Raw delta dprime vs. full simulation,   p: {0}".format(ss.wilcoxon(df.groupby(by='site').mean()['sim12'], 
                                                                           df.groupby(by='site').mean()['state_diff'])))

# bootstrap test instead
d = {s: df[df.site==s]['sim12'].values-df[df.site==s]['state_diff'].values for s in df.site.unique()}
print("generating bootstrap stats for dprime models. Could be very slow...")
#bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
#p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
#print("Raw delta dprime vs. full simulation, p={0}".format(p))

# plot dprime per site for the pupil regress simulations
if barplot:
    dfg = df.groupby(by='site').mean()
    dprax.bar([0, 1], [dfg['sim1_pr'].mean(), dfg['sim12_pr'].mean()],
                        yerr=[dfg['sim1_pr'].sem(), dfg['sim12_pr'].sem()],
                        color='lightgrey', edgecolor='k', width=0.5)

else:
    for i, s in zip([0.5, 1.5], ['sim1_pr', 'sim12_pr']):
        try:
            vals = df.loc[df.site.isin(LOWR_SITES)].groupby(by='site').mean()[s]
            dprax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0, len(vals)),
                        vals, color='grey', marker='D', edgecolor='white', s=30, zorder=2)
        except:
            pass
        vals = df.loc[df.site.isin(HIGHR_SITES)].groupby(by='site').mean()[s]
        dprax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0, len(vals)),
                    vals, color='k', marker='o', edgecolor='white', s=50, zorder=3)

        # now, for each site draw lines between points in each model. Color red if 2nd order hurts, blue if helps
    for i, s in enumerate(df.site.unique()):
        vals = df.groupby(by='site').mean()[['sim1_pr', 'sim12_pr']].loc[s].values
        dprax.plot([0.5, 1.5], vals, color=line_colors[i], alpha=0.5, zorder=1)

    dprax.axhline(0, linestyle='--', color='grey', lw=2)     
dprax.set_xticks([0.5, 1.5])
dprax.set_xticklabels(['1st order', '1st + 2nd'], rotation=45)
dprax.set_xlabel('Simulation')
if mi_norm:
    dprax.set_ylabel(r"$\Delta d'^{2}$")
else:
    dprax.set_ylabel(r"$\Delta d'^{2}$")
dprax.set_title('Discriminability Change \n Pupil Corrected Data', color=color.CORRECTED)
dprax.set_ylim(dax.get_ylim())
dprax.set_xlim(dax.get_xlim())


# plot the residual correlation with pupil
raw_corr = []
pr_corr = []
pr_corr_by_site = {d: [] for d in sites}
raw_corr_by_site = {d: [] for d in sites}
for site in sites:
    print('Loading spike data for site {}'.format(site))
    batch = 289
    if site in ['BOL005c', 'BOL006b']:
         batch = 294
    fs = 4
    ops = {'batch': batch, 'cellid': site}
    xmodel = 'ns.fs{}.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'.format(fs)
    path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'
    cells, _ = parse_cellid(ops)
    rec = preproc.generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                                cache_path=path, recache=False)
    rec = rec.apply_mask()
    raw_residual = rec['resp']._data - rec['psth_sp']._data
    corr_residual = rec['resp']._data - rec['psth']._data
    pupil = rec['pupil']._data

    rc = []
    prc = []
    for i in range(raw_residual.shape[0]):
        rc.append(np.corrcoef(raw_residual[i, :], pupil)[0, 1])
        prc.append(np.corrcoef(corr_residual[i, :], pupil)[0, 1])
        pr_corr_by_site[site].append(np.corrcoef(corr_residual[i, :], pupil)[0, 1])
        raw_corr_by_site[site].append(np.corrcoef(raw_residual[i, :], pupil)[0, 1])
    raw_corr.extend(rc)
    pr_corr.extend(prc)

bins = np.arange(-0.45, 0.45, 0.05)
out = prax.hist([raw_corr, pr_corr], rwidth=0.8, edgecolor='k', 
                color=[color.RAW, color.CORRECTED], density=True, bins=bins)
prax.legend(['Raw', 'Pupil-corrected'], frameon=False, fontsize=8)

# plot pdfs
m, sd = ss.norm.fit(raw_corr)
xmin, xmax = prax.get_xlim()
x = np.linspace(xmin, xmax, 1000)
p = ss.norm.pdf(x, m, sd)
prax.plot(x, p, 'k', linewidth=2, color=color.RAW)

m, sd = ss.norm.fit(pr_corr)
xmin, xmax = prax.get_xlim()
x = np.linspace(xmin, xmax, 1000)
p = ss.norm.pdf(x, m, sd)
prax.plot(x, p, 'k', linewidth=2, color=color.CORRECTED)

prax.set_xlabel(r"Pearson's $r$")
prax.set_ylabel(r"Neuron Density")
prax.set_title('Residual spike count \n correlation with pupil')

# do statistical test. 
#Is corrected distriution's correlation with pupil diff than zero?
pr_corr_by_site = {d: np.array(v) for (d, v) in pr_corr_by_site.items()}
bootstat = stats.get_bootstrapped_sample(pr_corr_by_site, nboot=5000)
p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
print("corrected correlation coef. diff from zero? pvalue: {0}".format(p))
print("Mean corr.: {0}, sem: {1}".format(np.mean(pr_corr), np.std(bootstat)))
# is raw distribution diff from zero?
raw_corr_by_site = {d: np.array(v) for (d, v) in raw_corr_by_site.items()}
bootstat = stats.get_bootstrapped_sample(raw_corr_by_site, nboot=5000)
p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
print("raw correlation coef. diff from zero? pvalue: {0}".format(p))
print("Mean corr.: {0}, sem: {1}".format(np.mean(raw_corr), np.std(bootstat)))

# print mean / sem of n units per session
mean_un = np.mean([len(v) for k, v in raw_corr_by_site.items()])
sem_un = np.std([len(v) for k, v in raw_corr_by_site.items()]) / np.sqrt(len(sites))
print("Mean n units per site: {0} +/- {1}".format(mean_un, sem_un))

# plot heatmaps
hm = []
xbins = np.linspace(x_cut_plot[0], x_cut_plot[1], nbins)
ybins = np.linspace(y_cut_plot[0], y_cut_plot[1], nbins)
for s in df.site.unique():
        vals = df[df.site==s]['sim1']
        if not mi_norm:
            vals -= vals.mean()
            vals /= vals.std()
        heatmap = ss.binned_statistic_2d(x=df[df.site==s]['dU_mag'+estval], 
                                    y=df[df.site==s]['cos_dU_evec'+estval],
                                    values=vals,
                                    statistic='mean',
                                    bins=[xbins, ybins])
        hm.append(heatmap.statistic.T)# / np.nanmax(heatmap.statistic))
t = np.nanmean(np.stack(hm), 0)

if collapse_across_sites:
    #df.plot.hexbin(x='dU_mag'+estval, 
    #              y='cos_dU_evec'+estval, 
    #              C='sim1', 
    #              gridsize=nbins, ax=s1ax, cmap=cmap, vmin=vmin, vmax=vmax) 
    vals = df['sim1']
    heatmap = ss.binned_statistic_2d(x=df['dU_mag'+estval], 
                            y=df['cos_dU_evec'+estval],
                            values=vals,
                            statistic='mean',
                            bins=[xbins, ybins])
    hm = [heatmap.statistic.T] # / np.nanmax(heatmap.statistic))
    t = np.nanmean(np.stack(hm), 0)

if smooth:
    t = sf.gaussian_filter(t, sigma)
    im = s1ax.imshow(t, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
else:
    im = s1ax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='none', 
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
divider = make_axes_locatable(s1ax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')

s1ax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
s1ax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
s1ax.spines['bottom'].set_color(color.SIGNAL)
s1ax.xaxis.label.set_color(color.SIGNAL)
s1ax.tick_params(axis='x', colors=color.SIGNAL)
s1ax.spines['left'].set_color(color.COSTHETA)
s1ax.yaxis.label.set_color(color.COSTHETA)
s1ax.tick_params(axis='y', colors=color.COSTHETA)
if mi_norm:
    s1ax.set_title(r"$\Delta d'^2$"+"\n 1st-order unique")
else:
    s1ax.set_title(r"$\Delta d'^2$ (z-score)"+"\n 1st-order unique")


hm = []
xbins = np.linspace(x_cut_plot[0], x_cut_plot[1], nbins)
ybins = np.linspace(y_cut_plot[0], y_cut_plot[1], nbins)
for s in df.site.unique():
        if second_order_unique:
            vals = df[df.site==s]['sim12'] - df[df.site==s]['sim1']
        else:
            vals = df[df.site==s]['sim12']
        if not mi_norm:
            vals -= vals.mean()
            vals /= vals.std()
        heatmap = ss.binned_statistic_2d(x=df[df.site==s]['dU_mag'+estval], 
                                    y=df[df.site==s]['cos_dU_evec'+estval],
                                    values=vals,
                                    statistic='mean',
                                    bins=[xbins, ybins])
        hm.append(heatmap.statistic.T)# / np.nanmax(heatmap.statistic))
t = np.nanmean(np.stack(hm), 0)

if collapse_across_sites:
    #df['sim2'] = (df['sim12'] - df['sim1'])
    #df.plot.hexbin(x='dU_mag'+estval, 
    #              y='cos_dU_evec'+estval, 
    #              C='sim2', 
    #              gridsize=nbins, ax=s12ax, cmap=cmap, vmin=vmin, vmax=vmax) 
    vals = df['sim12'] - df['sim1']
    heatmap = ss.binned_statistic_2d(x=df['dU_mag'+estval], 
                            y=df['cos_dU_evec'+estval],
                            values=vals,
                            statistic='mean',
                            bins=[xbins, ybins])
    hm = [heatmap.statistic.T] # / np.nanmax(heatmap.statistic))
    t = np.nanmean(np.stack(hm), 0)

if smooth:
    t = sf.gaussian_filter(t, sigma)
    im = s12ax.imshow(t, aspect='auto', origin='lower', cmap=cmap,
                                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
else:
    im = s12ax.imshow(t, aspect='auto', origin='lower', cmap=cmap, interpolation='none', 
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin, vmax=vmax)
divider = make_axes_locatable(s12ax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
s12ax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
s12ax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
s12ax.spines['bottom'].set_color(color.SIGNAL)
s12ax.xaxis.label.set_color(color.SIGNAL)
s12ax.tick_params(axis='x', colors=color.SIGNAL)
s12ax.spines['left'].set_color(color.COSTHETA)
s12ax.yaxis.label.set_color(color.COSTHETA)
s12ax.tick_params(axis='y', colors=color.COSTHETA)
if second_order_unique: tag = '2nd-order unique' 
else: tag='1st+2nd-order'
if mi_norm:
    s12ax.set_title(r"$\Delta d'^2$"+"\n {}".format(tag))
else:
    s12ax.set_title(r"$\Delta d'^2$ (z-score)"+"\n {}".format(tag))

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()

# model delta dprime (for the first order sim and full - first) 
# as function of each axis in order to quantify differences between heatmaps.
beta_cos = []
beta_du = []
beta_int = []
for s in df.site.unique():
    X = df[df.site==s][['cos_dU_evec'+estval, 'dU_mag'+estval]].copy()
    X['dU_mag'+estval] = X['dU_mag'+estval] - X['dU_mag'+estval].mean()
    X['dU_mag'+estval] /= X['dU_mag'+estval].std()
    X['cos_dU_evec'+estval] = X['cos_dU_evec'+estval] - X['cos_dU_evec'+estval].mean()
    X['cos_dU_evec'+estval] /= X['cos_dU_evec'+estval].std()
    X = sm.add_constant(X)
    X['interaction'] = X['cos_dU_evec'+estval] * X['dU_mag'+estval]

    y = df[df.site==s]['sim1'].values.copy()
    y -= y.mean()
    y /= y.std()

    model = sm.OLS(y, X).fit()

    beta_cos.append(model.params.cos_dU_evec_test)
    beta_du.append(model.params.dU_mag_test)
    beta_int.append(model.params.interaction)

# print model coefficients / confidence intervals / pvals
print("First order simualtion results: \n")
print("noise intereference beta       mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_cos), 
                                                       np.std(beta_cos) / np.sqrt(len(beta_cos)), 
                                                       ss.ranksums(beta_cos, np.zeros(len(beta_cos))).pvalue,
                                                       ss.ranksums(beta_cos, np.zeros(len(beta_cos))).statistic))
print("discrimination magnitude beta  mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_du), 
                                                            np.std(beta_du) / np.sqrt(len(beta_du)), 
                                                       ss.ranksums(beta_du, np.zeros(len(beta_cos))).pvalue,
                                                       ss.ranksums(beta_du, np.zeros(len(beta_cos))).statistic))
print("interaction term beta          mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_int), 
                                                    np.std(beta_int) / np.sqrt(len(beta_int)), 
                                                       ss.ranksums(beta_int, np.zeros(len(beta_cos))).pvalue,
                                                       ss.ranksums(beta_int, np.zeros(len(beta_cos))).statistic))
print('\n \n')


beta_cos = []
beta_du = []
beta_int = []
for s in df.site.unique():
    X = df[df.site==s][['cos_dU_evec'+estval, 'dU_mag'+estval]].copy()
    X['dU_mag'+estval] = X['dU_mag'+estval] - X['dU_mag'+estval].mean()
    X['dU_mag'+estval] /= X['dU_mag'+estval].std()
    X['cos_dU_evec'+estval] = X['cos_dU_evec'+estval] - X['cos_dU_evec'+estval].mean()
    X['cos_dU_evec'+estval] /= X['cos_dU_evec'+estval].std()
    X = sm.add_constant(X)
    X['interaction'] = X['cos_dU_evec'+estval] * X['dU_mag'+estval]

    y = df[df.site==s]['sim12'].values.copy() - df[df.site==s]['sim1'].values.copy()
    y -= y.mean()
    y /= y.std()

    model = sm.OLS(y, X).fit()

    beta_cos.append(model.params.cos_dU_evec_test)
    beta_du.append(model.params.dU_mag_test)
    beta_int.append(model.params.interaction)

# print model coefficients / confidence intervals / pvals
print("Second order simualtion results: \n")
print("noise intereference beta       mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_cos), 
                                                       np.std(beta_cos) / np.sqrt(len(beta_cos)), 
                                                       ss.ranksums(beta_cos, np.zeros(len(beta_cos))).pvalue,
                                                       ss.ranksums(beta_cos, np.zeros(len(beta_cos))).statistic))
print("discrimination magnitude beta  mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_du), 
                                                            np.std(beta_du) / np.sqrt(len(beta_du)), 
                                                       ss.ranksums(beta_du, np.zeros(len(beta_cos))).pvalue,
                                                       ss.ranksums(beta_du, np.zeros(len(beta_cos))).statistic))
print("interaction term beta          mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_int), 
                                                    np.std(beta_int) / np.sqrt(len(beta_int)), 
                                                       ss.ranksums(beta_int, np.zeros(len(beta_cos))).pvalue,
                                                       ss.ranksums(beta_int, np.zeros(len(beta_cos))).statistic))
print('\n \n')
