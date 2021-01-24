"""
Heatmap of delta dprime (in same cropped space as fig 2)
Model overall dprime (and delta dprime) in cropped space from fig 2. 
Compare model weights for predicting delta vs. predicting overall.

CLEANING THIS UP AS OF 08.29.2020 (this is the new version, old version is in the old_scripts dir)
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT, HIGHR_PEG_SITES, PEG_SITES
import colors as color
import ax_labels as alab

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.statistics as stats
import os
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
ALL_TRAIN_DATA = True  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = PEG_SITES # HIGHR_SITES
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

# set up subplots
f = plt.figure(figsize=(12, 6))

dpax = plt.subplot2grid((2, 4), (0, 0))
hax = plt.subplot2grid((2, 4), (1, 0))
lax1 = plt.subplot2grid((2, 4), (1, 1))
lax2 = plt.subplot2grid((2, 4), (0, 1))
lax3 = plt.subplot2grid((2, 4), (1, 2))
lax4 = plt.subplot2grid((2, 4), (0, 2))
regcoef1 = plt.subplot2grid((2, 4), (0, 3))
regcoef2 = plt.subplot2grid((2, 4), (1, 3))

df = []
for site in sites:
    if (site in LOWR_SITES) | (ALL_TRAIN_DATA): mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results
    stim = results.evoked_stimulus_pairs
    _df = _df.loc[_df.index.get_level_values('combo').isin(stim)]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, n_components, idx=(0, 0))[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, n_components, idx=(0, 0))[0]
    _df['state_diff'] = (_df[bp_dp] - _df[sp_dp]) / _df['dp_opt_test']
    _df['state_diff_abs'] = (_df[bp_dp] - _df[sp_dp])
    _df['state_MI'] = (_df[bp_dp] - _df[sp_dp]) / (_df[bp_dp] + _df[sp_dp])
    _df['bp_dU_dot_evec_sq'] = results.slice_array_results('bp_dU_dot_evec_sq', stim, 2, idx=(0, 0))[0]
    _df['sp_dU_dot_evec_sq'] = results.slice_array_results('sp_dU_dot_evec_sq', stim, 2, idx=(0, 0))[0]
    _df['bp_evec_snr'] = results.slice_array_results('bp_evec_snr', stim, 2, idx=(0, 0))[0]
    _df['sp_evec_snr'] = results.slice_array_results('sp_evec_snr', stim, 2, idx=(0, 0))[0]
    _df['bp_lambda'] = results.slice_array_results('bp_evals', stim, 2, idx=(0))[0]
    _df['sp_lambda'] = results.slice_array_results('sp_evals', stim, 2, idx=(0))[0]
    _df['bp_cos_dU_evec'] = results.slice_array_results('bp_cos_dU_evec', stim, 2, idx=(0, 0))[0]
    _df['sp_cos_dU_evec'] = results.slice_array_results('sp_cos_dU_evec', stim, 2, idx=(0, 0))[0]
    _df['snr_diff'] = _df['bp_evec_snr'] - _df['sp_evec_snr']
    _df['site'] = site
    df.append(_df)

df_all = pd.concat(df)

# filter based on x_cut / y_cut
if (x_cut is not None) & (y_cut is not None):
    mask1 = (df_all['dU_mag'+estval] < x_cut[1]) & (df_all['dU_mag'+estval] > x_cut[0])
    mask2 = (df_all['cos_dU_evec'+estval] < y_cut[1]) & (df_all['cos_dU_evec'+estval] > y_cut[0])
else:
    mask1 = (True * np.ones(df_all.shape[0])).astype(bool)
    mask2 = (True * np.ones(df_all.shape[0])).astype(bool)
df = df_all[mask1 & mask2]

# plot large vs. small dprime per site
dfg = df_all.groupby(by='site').mean()
dfe = df_all.groupby(by='site').sem()
mi = np.min([dfg[sp_dp].min(), dfg[bp_dp].min()])
ma = np.max([dfg[sp_dp].max(), dfg[bp_dp].max()])
try:
    dpax.scatter(dfg.loc[LOWR_SITES][sp_dp], dfg.loc[LOWR_SITES][bp_dp], marker="D", color='grey', s=30, edgecolor='white')
except:
    pass
try:
    dpax.scatter(dfg.loc[HIGHR_SITES][sp_dp], dfg.loc[HIGHR_SITES][bp_dp], color='k', s=50, edgecolor='white')
except:
    dpax.scatter(dfg.loc[HIGHR_PEG_SITES][sp_dp], dfg.loc[HIGHR_PEG_SITES][bp_dp], color='k', s=50, edgecolor='white')
#dpax.errorbar(dfg.loc[HIGHR_SITES][sp_dp], dfg.loc[HIGHR_SITES][bp_dp], 
#              xerr=dfe.loc[HIGHR_SITES][sp_dp], yerr=dfe.loc[HIGHR_SITES][bp_dp], 
#                color='k', fmt='.')
dpax.plot([mi, ma], [mi, ma], color='grey', linestyle='--')
dpax.set_xlabel('Small pupil')
dpax.set_ylabel('Large pupil')
dpax.set_title(r"$d'^{2}$")

# print significance of group effect of scatter plot
print("Large vs. small pupil dprime       pval: {0} \n"
      "                                   n:    {1} \n"
      "                                   W stat: {2} \n".format(ss.wilcoxon(dfg[sp_dp], dfg[bp_dp]).pvalue, dfg.shape[0], ss.wilcoxon(dfg[sp_dp], dfg[bp_dp]).statistic))

# plot delta dprime
# make heatmap irrespective of site ID
vals = df['state_MI']
xbins = np.linspace(x_cut[0], x_cut[1], nbins)
ybins = np.linspace(y_cut[0], y_cut[1], nbins)
heatmap = ss.binned_statistic_2d(x=df['dU_mag'+estval], 
                        y=df['cos_dU_evec'+estval],
                        values=vals,
                        statistic='mean',
                        bins=[xbins, ybins])
hm = [heatmap.statistic.T]
t = np.nanmean(np.stack(hm), 0)

if smooth:
    t = sf.gaussian_filter(t, sigma)
    im = hax.imshow(t, aspect='auto', origin='lower', cmap=cmap,
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
hax.set_title(r"$\Delta d'^2$")    

# ============================ BINNED LINE PLOTS =========================================
grouped = df.groupby(by='site')
cbins = np.linspace(y_cut[0], y_cut[1], nline_bins)
dbins = np.linspace(x_cut[0], x_cut[1], nline_bins)
du_dp = []
cos_dp = []
du_delta = []
cos_delta = []
cos_bp_dp = []
cos_sp_dp = []
du_bp_dp = []
du_sp_dp = []
cos_bin = []
du_bin = []
colors = plt.cm.get_cmap('Reds', len(df.site.unique()))
for i, g in enumerate(grouped):
    # get binned delta dprime
    cos = ss.binned_statistic(g[1]['cos_dU_evec_test'], g[1]['state_MI'], statistic='mean', bins=cbins)
    du = ss.binned_statistic(g[1]['dU_mag_test'], g[1]['state_MI'], statistic='mean', bins=dbins)

    du_val_delt = du.statistic 
    cos_val_delt = cos.statistic 
    du_delta.append(du_val_delt)
    cos_delta.append(cos_val_delt)

    # get binned overall dprime
    cos = ss.binned_statistic(g[1]['cos_dU_evec_test'], g[1]['dp_opt_test'], statistic='mean', bins=cbins)
    du = ss.binned_statistic(g[1]['dU_mag_test'], g[1]['dp_opt_test'], statistic='mean', bins=dbins)
    du_val = du.statistic 
    cos_val = cos.statistic 
    du_dp.append(du_val)
    cos_dp.append(cos_val)

    cos_bp = ss.binned_statistic(g[1]['cos_dU_evec_test'], g[1][bp_dp], statistic='mean', bins=cbins)
    cos_sp = ss.binned_statistic(g[1]['cos_dU_evec_test'], g[1][sp_dp], statistic='mean', bins=cbins)
    du_bp = ss.binned_statistic(g[1]['dU_mag_test'], g[1][bp_dp], statistic='mean', bins=dbins)
    du_sp = ss.binned_statistic(g[1]['dU_mag_test'], g[1][sp_dp], statistic='mean', bins=dbins)
    cos_bp_dp.append(cos_bp.statistic)
    cos_sp_dp.append(cos_sp.statistic)
    du_bp_dp.append(du_bp.statistic)
    du_sp_dp.append(du_sp.statistic)

    cos_bin.append(cos.bin_edges[1:])
    du_bin.append(du.bin_edges[1:])

cos_bin = np.nanmean(np.stack(cos_bin), axis=0)
du_bin = np.nanmean(np.stack(du_bin), axis=0)

du_val_delt = np.nanmean(np.stack(du_delta), axis=0)
du_val_delt_sem = np.nanstd(np.stack(du_delta), axis=0) / np.sqrt((~np.isnan(np.stack(du_delta))).sum(axis=0))
cos_val_delt = np.nanmean(np.stack(cos_delta), axis=0)
cos_val_delt_sem = np.nanstd(np.stack(cos_delta), axis=0) / np.sqrt((~np.isnan(np.stack(cos_delta))).sum(axis=0))

# plot overall results (grand mean after grouping, or overall mean w/o grouping, if collapse_over_all_site=True)
# for delta dprime
lab = r"$\Delta d'^2$"

# plot by just collapsing over all sites
# generate error bars with hierarchical bootstrap
print('generating bootstrap SE, will take a moment')
cos_val_delt = ss.binned_statistic(df['cos_dU_evec_test'], df['state_MI'], statistic='mean', bins=cbins).statistic
ds = [{s: df[(df.site==s) & (df['cos_dU_evec_test']>cbins[i]) & (df['cos_dU_evec_test']<cbins[i+1])]['state_MI'].values for s in df.site.unique()} for i in range(len(cbins)-1)]
cos_val_delt_se = np.stack([np.std(stats.get_bootstrapped_sample(ds[i], nboot=100)) for i in range(len(ds))])

du_val_delt = ss.binned_statistic(df['dU_mag_test'], df['state_MI'], statistic='mean', bins=dbins).statistic
ds = [{s: df[(df.site==s) & (df['dU_mag_test']>dbins[i]) & (df['dU_mag_test']<dbins[i+1])]['state_MI'].values for s in df.site.unique()} for i in range(len(cbins)-1)]
du_val_delt_se = np.stack([np.std(stats.get_bootstrapped_sample(ds[i], nboot=100)) for i in range(len(ds))])

lax1.errorbar(cos_bin, cos_val_delt, yerr=cos_val_delt_se, marker='.', lw=1, color='k', label=r"$\Delta d'^{2}$", zorder=4)
lax3.errorbar(du_bin, du_val_delt, yerr=du_val_delt_se, marker='.', lw=1, color='k', label=r"$\Delta d'^{2}$", zorder=4)
 
lax1.set_ylim((0.1, 0.4))
lax3.set_ylim((0.1, 0.4))

du_val = np.nanmean(np.stack(du_dp), axis=0)
du_val_sem = np.nanstd(np.stack(du_dp), axis=0) / np.sqrt((~np.isnan(np.stack(du_dp))).sum(axis=0))
cos_val = np.nanmean(np.stack(cos_dp), axis=0)
cos_val_sem = np.nanstd(np.stack(cos_dp), axis=0) / np.sqrt((~np.isnan(np.stack(cos_dp))).sum(axis=0))

du_bp_val = np.nanmean(np.stack(du_bp_dp), axis=0)
du_bp_val_sem = np.nanstd(np.stack(du_bp_dp), axis=0) / np.sqrt((~np.isnan(np.stack(du_bp_dp))).sum(axis=0))
du_sp_val = np.nanmean(np.stack(du_sp_dp), axis=0)
du_sp_val_sem = np.nanstd(np.stack(du_sp_dp), axis=0) / np.sqrt((~np.isnan(np.stack(du_sp_dp))).sum(axis=0))

cos_bp_val = np.nanmean(np.stack(cos_bp_dp), axis=0)
cos_bp_val_sem = np.nanstd(np.stack(cos_bp_dp), axis=0) / np.sqrt((~np.isnan(np.stack(cos_bp_dp))).sum(axis=0))
cos_sp_val = np.nanmean(np.stack(cos_sp_dp), axis=0)
cos_sp_val_sem = np.nanstd(np.stack(cos_sp_dp), axis=0) / np.sqrt((~np.isnan(np.stack(cos_sp_dp))).sum(axis=0))

# plot overall results (grand mean after grouping, or overall mean w/o grouping, if collapse_over_all_site=True)
# for overall dprime

# big pupil
print("generating bootstrapped SE")
cos_val = ss.binned_statistic(df['cos_dU_evec_test'], df[bp_dp], statistic='mean', bins=cbins).statistic
ds = [{s: df[(df.site==s) & (df['cos_dU_evec_test']>cbins[i]) & (df['cos_dU_evec_test']<cbins[i+1])][bp_dp].values for s in df.site.unique()} for i in range(len(cbins)-1)]
cos_val_se = np.stack([np.std(stats.get_bootstrapped_sample(ds[i], nboot=100)) for i in range(len(ds))])

du_val = ss.binned_statistic(df['dU_mag_test'], df[bp_dp], statistic='mean', bins=dbins).statistic
ds = [{s: df[(df.site==s) & (df['dU_mag_test']>dbins[i]) & (df['dU_mag_test']<dbins[i+1])][bp_dp].values for s in df.site.unique()} for i in range(len(cbins)-1)]
du_val_se = np.stack([np.std(stats.get_bootstrapped_sample(ds[i], nboot=100)) for i in range(len(ds))])

lax2.errorbar(cos_bin, cos_val, yerr=cos_val_se, marker='.', lw=1, color='firebrick', label=r"$\Delta d'^{2}$", zorder=4)
lax4.errorbar(du_bin, du_val, lw=1,yerr=du_val_se, marker='.', color='firebrick', label=r"$\Delta d'^{2}$", zorder=4)

# small pupil
cos_val = ss.binned_statistic(df['cos_dU_evec_test'], df[sp_dp], statistic='mean', bins=cbins).statistic
ds = [{s: df[(df.site==s) & (df['cos_dU_evec_test']>cbins[i]) & (df['cos_dU_evec_test']<cbins[i+1])][sp_dp].values for s in df.site.unique()} for i in range(len(cbins)-1)]
cos_val_se = np.stack([np.std(stats.get_bootstrapped_sample(ds[i], nboot=100)) for i in range(len(ds))])

du_val = ss.binned_statistic(df['dU_mag_test'], df[sp_dp], statistic='mean', bins=dbins).statistic
ds = [{s: df[(df.site==s) & (df['dU_mag_test']>dbins[i]) & (df['dU_mag_test']<dbins[i+1])][sp_dp].values for s in df.site.unique()} for i in range(len(cbins)-1)]
du_val_se = np.stack([np.std(stats.get_bootstrapped_sample(ds[i], nboot=100)) for i in range(len(ds))])

lax2.errorbar(cos_bin, cos_val, lw=1, yerr=cos_val_se, marker='.', color='navy', label=r"$\Delta d'^{2}$", zorder=4)
lax4.errorbar(du_bin, du_val, lw=1, color='navy', yerr=du_val_se, marker='.', label=r"$\Delta d'^{2}$", zorder=4)

lax2.set_ylim((0, 65))
lax4.set_ylim((0, 65))


lax1.set_xlabel(alab.COSTHETA, color=color.COSTHETA)
lax1.spines['bottom'].set_color(color.COSTHETA)
lax1.xaxis.label.set_color(color.COSTHETA)
lax1.tick_params(axis='x', colors=color.COSTHETA)
lax1.set_title('Discriminability Change')

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
lax3.set_ylabel(r"$\Delta d'^{2}$")
    
lax2.set_ylabel(r"$d'^{2}$")
lax4.set_ylabel(r"$d'^{2}$")

# ============================================= MODEL DPRIME / DELTA =================================================

# fit models
beta_overall = []
beta_delta = []
ci_overall = []
ci_delta = []
pvals_overall = []
pvals_delta = []
highr_mask = []
rsquared = []
for s in df.site.unique():
    if s in HIGHR_SITES:
        highr_mask.append(True)
    else:
        highr_mask.append(False)

    X = df_all[df_all.site==s][['cos_dU_evec_test', 'dU_mag_test']]
    X['dU_mag_test'] = X['dU_mag_test'] - X['dU_mag_test'].mean()
    X['dU_mag_test'] /= X['dU_mag_test'].std()
    X['cos_dU_evec_test'] = X['cos_dU_evec_test'] - X['cos_dU_evec_test'].mean()
    X['cos_dU_evec_test'] /= X['cos_dU_evec_test'].std()

    
    X = sm.add_constant(X)
    X['interaction'] = X['cos_dU_evec_test'] * X['dU_mag_test']
    y = (df_all[df_all.site==s][bp_dp].values.copy() - df_all[df_all.site==s][sp_dp].values.copy()) / \
        (df_all[df_all.site==s][bp_dp].values.copy() + df_all[df_all.site==s][sp_dp].values.copy())
    y -= y.mean()
    y /= y.std()

    model = sm.OLS(y, X).fit()
    low_ci = model.conf_int().values[:,0]
    high_ci = model.conf_int().values[:,1]
    beta_delta.append(model.params.values)
    ci_delta.append(high_ci - low_ci)
    pvals_delta.append(model.pvalues)
    rsquared.append(model.rsquared)

    y = df_all[df_all.site==s]['dp_opt_test']
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

# print statistics for reg. coefficients
print("OVERALL D'")
print("noise intereference beta       mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_overall[:,1]), 
                                                       beta_overall[:,1].std() / np.sqrt(beta_overall.shape[0]), 
                                                       ss.ranksums(beta_overall[:, 1], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_overall[:, 1], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_overall[:, 1]<0.05).sum(), pvals_overall.shape[0]))

      
print("discrimination magnitude beta  mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_overall[:,2]), 
                                                            beta_overall[:,2].std() / np.sqrt(beta_overall.shape[0]), 
                                                       ss.ranksums(beta_overall[:, 2], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_overall[:, 2], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_overall[:, 2]<0.05).sum(), pvals_overall.shape[0]))


      
print("interaction term beta          mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_overall[:, 3]), 
                                                    beta_overall[:, 3].std() / np.sqrt(beta_overall.shape[0]), 
                                                       ss.ranksums(beta_overall[:, 3], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_overall[:, 3], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_overall[:, 3]<0.05).sum(), pvals_overall.shape[0]))

      
      
print("\n")
print("DELTA D'")
print("noise intereference beta       mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_delta[:,1]), 
                                                       beta_delta[:,1].std() / np.sqrt(beta_delta.shape[0]), 
                                                       ss.ranksums(beta_delta[:, 1], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_delta[:, 1], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_delta[:,1]<0.05).sum(), pvals_overall.shape[0]))

print("discrimination magnitude beta  mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_delta[:,2]), 
                                                            beta_delta[:,2].std() / np.sqrt(beta_delta.shape[0]), 
                                                       ss.ranksums(beta_delta[:, 2], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_delta[:, 2], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_delta[:,2]<0.05).sum(), pvals_overall.shape[0]))

      
print("interaction term beta          mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_delta[:,3]), 
                                                    beta_delta[:,3].std() / np.sqrt(beta_delta.shape[0]), 
                                                       ss.ranksums(beta_delta[:, 3], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_delta[:, 3], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_delta[:,3]<0.05).sum(), pvals_overall.shape[0]))

# plot
coefs = pd.DataFrame(columns=[r"$d'^2$", r"$d'^2$", r"$\Delta d'^2$", r"$\Delta d'^2$"], 
                    data=np.concatenate((beta_delta[:, [1,2]], beta_overall[:, [1,2]]), axis=1)[:,::-1])
coefs = coefs.melt()
coefs['regressor'] = np.concatenate([['Discrimination Magnitude']*beta_delta.shape[0],
                                ['Noise Interference']*beta_delta.shape[0],
                               ['Discrimination Magnitude']*beta_delta.shape[0],
                               ['Noise Interference']*beta_delta.shape[0]])
sns.stripplot(y='variable', x='value', data=coefs[coefs['variable']==r"$d'^2$"], hue='regressor', dodge=True, ax=regcoef1,
                                                     palette={'Noise Interference': color.COSTHETA, 'Discrimination Magnitude': color.SIGNAL}, alpha=0.3)
sns.pointplot(y='variable', x='value', data=coefs[coefs['variable']==r"$d'^2$"], hue='regressor', dodge=0.4, join=False, ci=95, ax=regcoef1, errwidth=1, scale=0.7, capsize=0.05,
                                                     palette={'Noise Interference': color.COSTHETA, 'Discrimination Magnitude': color.SIGNAL})
regcoef1.axvline(0, linestyle='--', color='grey')
regcoef1.legend(frameon=False, fontsize=10, title='Regressor')

sns.stripplot(y='variable', x='value', data=coefs[coefs['variable']==r"$\Delta d'^2$"], hue='regressor', dodge=True, ax=regcoef2,
                                                     palette={'Noise Interference': color.COSTHETA, 'Discrimination Magnitude': color.SIGNAL}, alpha=0.3)
sns.pointplot(y='variable', x='value', data=coefs[coefs['variable']==r"$\Delta d'^2$"], hue='regressor', dodge=0.4, join=False, ci=95, ax=regcoef2, errwidth=1, scale=0.7, capsize=0.05,
                                                     palette={'Noise Interference': color.COSTHETA, 'Discrimination Magnitude': color.SIGNAL})
regcoef2.axvline(0, linestyle='--', color='grey')
regcoef2.legend(frameon=False, fontsize=10, title='Regressor')

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()
