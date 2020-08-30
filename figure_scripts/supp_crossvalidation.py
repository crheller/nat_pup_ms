"""
Supplemental figure showing that key results do not depend on cross validation.
Show figures for key results with training data for ALL data, including low repetition sites.

Show fit results for:
    * heatmap/histogram from fig 3 (overall goes up, patterns don't change, data spans similar region)
    * scatter plot from fig4 (delta basically unchanged, still significant)
    * heatmap from fig4 and insets (i.e. delta dprime still depends on y-axis, not x-axis)
"""
import colors as color
import ax_labels as alab
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import statsmodels.api as sm
import scipy.stats as ss
import scipy.ndimage.filters as sf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True
recache = False # recache dprime results locally
ALL_TRAIN_DATA = True  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = ALL_SITES
path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR+'supp_crossvalidation.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
nbins = 20
cmap = 'Greens'
vmax_overall = None
vmax_delta = None
vmin_delta = None
hexscale = 'log' # or 'log'

# only crop the dprime heatmaps. Show count for everything, compute stats for everything
x_cut = (2, 8)
y_cut = (0.2, 1)

# ======================================== LOAD THE DATA ===================================
df = []
for site in sites:
    if (site in LOWR_SITES) | (ALL_TRAIN_DATA):
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results
    stim = results.evoked_stimulus_pairs
    _df = _df.loc[pd.IndexSlice[stim, 2], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
    _df['site'] = site
    df.append(_df)

df = pd.concat(df)

# ============================== FILTER DATAFRAME FOR PLOTTING =============================================
# filter based on x_cut / y_cut
if (x_cut is not None) & (y_cut is not None):
    mask1 = (df['dU_mag_test'] < x_cut[1]) & (df['dU_mag_test'] > x_cut[0])
    mask2 = (df['cos_dU_evec_test'] < y_cut[1]) & (df['cos_dU_evec_test'] > y_cut[0])
else:
    mask1 = (True * np.ones(df.shape[0])).astype(bool)
    mask2 = (True * np.ones(df.shape[0])).astype(bool)
df_plot = df[mask1 & mask2]

# ============================================== LAYOUT FIGURE ================================================
f = plt.figure(figsize=(9.5, 6))

histax = plt.subplot2grid((2, 3), (0, 0))
dpax = plt.subplot2grid((2, 3), (0, 1))
deltax = plt.subplot2grid((2, 3), (1, 1))
scax = plt.subplot2grid((2, 3), (1, 0))
regcoef1 = plt.subplot2grid((2, 3), (0, 2))
regcoef2 = plt.subplot2grid((2, 3), (1, 2))


# ================================= PLOT OVERALL DPRIME HEATMAP + HISTOGRAM ===================================
# plot dprime
im = df_plot.plot.hexbin(x='dU_mag_test', 
                  y='cos_dU_evec_test', 
                  C='dp_opt_test', 
                  gridsize=nbins, ax=dpax, cmap=cmap, vmax=vmax_overall) 
dpax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
dpax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
dpax.spines['bottom'].set_color(color.SIGNAL)
dpax.xaxis.label.set_color(color.SIGNAL)
dpax.tick_params(axis='x', colors=color.SIGNAL)
dpax.spines['left'].set_color(color.COSTHETA)
dpax.yaxis.label.set_color(color.COSTHETA)
dpax.tick_params(axis='y', colors=color.COSTHETA)
dpax.set_title(r"$d'^2$")

# plot count histogram
df.plot.hexbin(x='dU_mag_test', 
               y='cos_dU_evec_test', 
               C=None, 
               gridsize=nbins, ax=histax, cmap='Reds', bins=hexscale) 
# overlay box for data extracted
line = np.array([[x_cut[0], y_cut[0]], 
                 [x_cut[0], y_cut[1]], 
                 [x_cut[1], y_cut[1]], 
                 [x_cut[1], y_cut[0]], 
                 [x_cut[0], y_cut[0]]])
histax.plot(line[:, 0], line[:, 1], linestyle='--', lw=2, color='k')

histax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
histax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
histax.spines['bottom'].set_color(color.SIGNAL)
histax.xaxis.label.set_color(color.SIGNAL)
histax.tick_params(axis='x', colors=color.SIGNAL)
histax.spines['left'].set_color(color.COSTHETA)
histax.yaxis.label.set_color(color.COSTHETA)
histax.tick_params(axis='y', colors=color.COSTHETA)
histax.set_title('Count')


# ========================================== PLOT DELTA DPRIME ==========================================
sigma = 1.2
df['state_MI'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
# heatmap
xbins = np.linspace(x_cut[0], x_cut[1], nbins)
ybins = np.linspace(y_cut[0], y_cut[1], nbins)
vals = df['state_MI']
heatmap = ss.binned_statistic_2d(x=df['dU_mag_test'], 
                        y=df['cos_dU_evec_test'],
                        values=vals,
                        statistic='mean',
                        bins=[xbins, ybins])
hm = [heatmap.statistic.T] 
t = np.nanmean(np.stack(hm), 0)
t = sf.gaussian_filter(t, sigma)
im = deltax.imshow(t, aspect='auto', origin='lower', cmap=cmap,
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], vmin=vmin_delta, vmax=vmax_delta)
divider = make_axes_locatable(deltax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')

# make slightly thinner
#deltax.set_aspect(cplt.get_square_asp(deltax))

deltax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
deltax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
deltax.spines['bottom'].set_color(color.SIGNAL)
deltax.xaxis.label.set_color(color.SIGNAL)
deltax.tick_params(axis='x', colors=color.SIGNAL)
deltax.spines['left'].set_color(color.COSTHETA)
deltax.yaxis.label.set_color(color.COSTHETA)
deltax.tick_params(axis='y', colors=color.COSTHETA)

deltax.set_title(r"$\Delta d'^2$")

# scatter plot
dfg = df.groupby(by='site').mean()
mi = np.min([dfg['sp_dp'].min(), dfg['bp_dp'].min()])
ma = np.max([dfg['sp_dp'].max(), dfg['bp_dp'].max()])
scax.scatter(dfg['sp_dp'], dfg['bp_dp'], color='k', s=50, edgecolor='white')
scax.plot([mi, ma], [mi, ma], color='grey', linestyle='--')
scax.set_xlabel('Small pupil')
scax.set_ylabel('Large pupil')
scax.set_title(r"$d'^{2}$")

# print significance of group effect of scatter plot
print("Large vs. small pupil dprime       pval: {0} \n"
      "                                   n:    {1} \n"
      "                                   W stat: {2} \n".format(ss.wilcoxon(dfg['sp_dp'], dfg['bp_dp']).pvalue, dfg.shape[0], ss.wilcoxon(dfg['sp_dp'], dfg['bp_dp']).statistic))



# ======================================== MODEL DPRIME/DELTA DPRIME ====================================
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

    X = df[df.site==s][['cos_dU_evec_test', 'dU_mag_test']]
    X['dU_mag_test'] = X['dU_mag_test'] - X['dU_mag_test'].mean()
    X['dU_mag_test'] /= X['dU_mag_test'].std()
    X['cos_dU_evec_test'] = X['cos_dU_evec_test'] - X['cos_dU_evec_test'].mean()
    X['cos_dU_evec_test'] /= X['cos_dU_evec_test'].std()

    
    X = sm.add_constant(X)
    X['interaction'] = X['cos_dU_evec_test'] * X['dU_mag_test']
    y = (df[df.site==s]['bp_dp'].values.copy() - df[df.site==s]['sp_dp'].values.copy()) / \
        (df[df.site==s]['bp_dp'].values.copy() + df[df.site==s]['sp_dp'].values.copy())
    y -= y.mean()
    y /= y.std()

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