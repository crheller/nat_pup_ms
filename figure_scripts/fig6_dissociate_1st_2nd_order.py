"""
Dissociate first and second order effects. Do this by:
    showing bandpass filtered noise correlation results
    show that delta rsc not predicted from MI, but rsc is 
"""
import load_results as ld
import colors as color
from path_settings import DPRIME_DIR, PY_FIGURES_DIR
from global_settings import ALL_SITES

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.preprocessing as preproc
from nems_lbhb.baphy import parse_cellid
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = False
fig_fn = PY_FIGURES_DIR + 'fig6_dissociate_1st_2nd_order.svg'
mi_max = 0.3  # for plotting purposes only, not for models
mi_min = -0.2
vmin = -0.15
vmax = 0.15
cmap = 'PiYG'
# set up subplots
f = plt.figure(figsize=(9, 6))

ncax = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
bsax = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
mncax = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
mdncax = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)
rscax = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=1)
drscax = plt.subplot2grid((2, 3), (1, 1), colspan=1, rowspan=1)

# ============================= load and plot delta noise correlations across freq. bands =======================================
boxcar = True
evoked = True
fs4 = False

modelnames = ['rsc_fft0-0.5', 'rsc_pr_rm1_fft0-0.5',
              'rsc_fft0.5-2', 'rsc_pr_rm1_fft0.5-2',
              'rsc_fft2-4', 'rsc_pr_rm1_fft2-4',
              'rsc_fft4-10', 'rsc_pr_rm1_fft4-10',
              'rsc_fft10-25', 'rsc_pr_rm1_fft10-25',
              'rsc_fft25-50', 'rsc_pr_rm1_fft25-50'
              ]
if fs4:
    modelnames = ['rsc_fft0-0.5', 'rsc_pr_rm1_fft0-0.5',
                'rsc_fft0.5-2', 'rsc_pr_rm1_fft0.5-2']
if evoked:
    modelnames = [m.replace('rsc', 'rsc_ev') for m in modelnames]
if boxcar:
    modelnames = [m.replace('fft', 'boxcar_fft') for m in modelnames]
if fs4:
    modelnames = [m.replace('fft', 'fs4_fft') for m in modelnames]

raw = [m for m in modelnames if 'pr' not in m]
corr = [m for m in modelnames if 'pr' in m]
f_band = [m.split('fft')[-1] for m in raw]

raw_results = []
for r in raw:
    print('loading {}'.format(r))
    raw_results.append(ld.load_noise_correlation(r))

corr_results = []
for c in corr:
    print('loading {}'.format(c))
    corr_results.append(ld.load_noise_correlation(c))

# print pvalues for each relevant comparison (corr vs. raw, and big vs. small)
for b in range(0, len(raw_results)):
    print("bin {0} corr. vs. raw:     pval: {1}".format(b+1, ss.wilcoxon(corr_results[b].groupby(by='site').mean()['all'], 
                                                                        raw_results[b].groupby(by='site').mean()['all'])))

    print("bin {0} big vs. small:     pval: {1}".format(b+1, ss.wilcoxon(raw_results[b].groupby(by='site').mean()['bp'], 
                                                                        raw_results[b].groupby(by='site').mean()['sp'])))

    print('\n')

# plot results
xvals = range(len(raw_results))
raw_nc = np.array([r.groupby(by='site').mean()['all'].mean() for r in raw_results])
raw_sem = np.array([r.groupby(by='site').mean()['all'].sem() for r in raw_results])
corr_nc = np.array([c.groupby(by='site').mean()['all'].mean() for c in corr_results])
corr_sem = np.array([c.groupby(by='site').mean()['all'].sem() for c in corr_results])

bp_nc = np.array([r.groupby(by='site').mean()['bp'].mean() for r in raw_results])
bp_sem = np.array([r.groupby(by='site').mean()['bp'].sem() for r in raw_results])
sp_nc = np.array([r.groupby(by='site').mean()['sp'].mean() for r in raw_results])
sp_sem = np.array([r.groupby(by='site').mean()['sp'].sem() for r in raw_results])

ncax.plot(xvals, raw_nc, marker='.', color=color.RAW, label='Raw')
ncax.fill_between(xvals, raw_nc-raw_sem, raw_nc+raw_sem, alpha=0.2, color=color.RAW, lw=0)
ncax.plot(xvals, corr_nc, marker='.', color=color.CORRECTED, label='Corrected')
ncax.fill_between(xvals, corr_nc-corr_sem, corr_nc+corr_sem, alpha=0.2, color=color.CORRECTED, lw=0)
ncax.axhline(0, linestyle='--', lw=2, color='grey')
ncax.legend(frameon=False)
ncax.set_xticks(xvals)
ncax.set_xticklabels(f_band, rotation=45)
ncax.set_ylabel('Noise Correlation')
ncax.set_ylim((-0.01, 0.12))
ncax.set_xlabel('Frequency Band (Hz)')
#ncax.set_aspect(cplt.get_square_asp(ncax))

bsax.plot(xvals, bp_nc, marker='.', color=color.LARGE, label='Large')
bsax.fill_between(xvals, bp_nc-bp_sem, bp_nc+bp_sem, alpha=0.2, color=color.LARGE, lw=0)
bsax.plot(xvals, sp_nc, marker='.', color=color.SMALL, label='Small')
bsax.fill_between(xvals, sp_nc-sp_sem, sp_nc+sp_sem, alpha=0.2, color=color.SMALL, lw=0)
bsax.axhline(0, linestyle='--', lw=2, color='grey')
bsax.legend(frameon=False)
bsax.set_xticks(xvals)
bsax.set_xticklabels(f_band, rotation=45)
bsax.set_ylabel('Noise Correlation')
bsax.set_ylim((-0.01, 0.12))
bsax.set_xlabel('Frequency Band (Hz)')
#bsax.set_aspect(cplt.get_square_asp(bsax))


# =============================== load and plot first vs. second order effects ==================================
path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'
MI_pred = True
gain_pred = False
df = pd.concat([pd.read_csv(os.path.join(path,'d_289_pup_sdexp.csv'), index_col=0),
                pd.read_csv(os.path.join(path,'d_294_pup_sdexp.csv'), index_col=0)])
try:
    df['r'] = [np.float(r.strip('[]')) for r in df['r'].values]
    df['r_se'] = [np.float(r.strip('[]')) for r in df['r_se'].values]
except:
    pass

df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se'])
df['site'] = [idx[:7] for idx in df.index]
df = df[df.loc[:, 'site'].isin([s[:7] for s in ALL_SITES])]

MI = df.loc[:, pd.IndexSlice['MI', 'st.pup']]

rsc_path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'
rsc_df = ld.load_noise_correlation('rsc_ev', xforms_model='NULL', path=rsc_path)

# add column for the gain of each neuron
m1 = [MI.loc[p.split('_')[0]] for p in rsc_df.index]
m2 = [MI.loc[p.split('_')[1]] for p in rsc_df.index]
rsc_df['m1'] = m1
rsc_df['m2'] = m2
rsc_df['diff'] = rsc_df['sp'] - rsc_df['bp']

mask = ~rsc_df['diff'].isna()
# mask low bin count data (see supp figure for histogram)
rsc_df = rsc_df[mask]

# model results per site
rsc_df['m1*m2'] = rsc_df['m1'] * rsc_df['m2']

# lists to save model results
dnc_beta = []
overall_beta = []
dnc_pvals = []
overall_pvals = []
for s in rsc_df.site.unique():
    X = rsc_df[rsc_df.site==s][['m1', 'm2', 'm1*m2']]
    X = X - X.mean(axis=0)
    X = X / X.std(axis=0)
    X = sm.add_constant(X)
    y = rsc_df[rsc_df.site==s]['diff']
    model_dnc = sm.OLS(y, X).fit()

    dnc_beta.append(model_dnc.params.values)
    dnc_pvals.append(model_dnc.pvalues.values)

    y = rsc_df[rsc_df.site==s]['all']
    model_all = sm.OLS(y, X).fit()

    overall_beta.append(model_all.params.values)
    overall_pvals.append(model_all.pvalues.values)

dnc_beta = np.stack(dnc_beta)
overall_beta = np.stack(overall_beta)
dnc_pvals = np.stack(dnc_pvals)
overall_pvals = np.stack(overall_pvals)

# plot heatmaps of noise corr vs. mi
# plot overall noise correlation
xbins = np.linspace(mi_min, mi_max, 20)
ybins = np.linspace(mi_min, mi_max, 20)
heatmap_rsc = ss.binned_statistic_2d(x=rsc_df['m1'], 
                            y=rsc_df['m2'],
                            values=rsc_df['all'],
                            statistic='mean',
                            bins=[xbins, ybins])

im = rscax.imshow(heatmap_rsc[0], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                            origin='lower')
divider = make_axes_locatable(rscax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
rscax.set_title(r"Noise Correlation")
rscax.set_xlabel(r"$MI_j$")
rscax.set_ylabel(r"$MI_i$")

# plot diff
heatmap_drsc = ss.binned_statistic_2d(x=rsc_df['m1'], 
                            y=rsc_df['m2'],
                            values=rsc_df['diff'],
                            statistic='mean',
                            bins=[xbins, ybins])

im = drscax.imshow(heatmap_drsc[0], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                            origin='lower')
divider = make_axes_locatable(drscax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
drscax.set_title(r"$\Delta$ Noise Correlation")
drscax.set_xlabel(r"$MI_j$")
drscax.set_ylabel(r"$MI_i$")

beta = [r'$MI_{i}$', r'$MI_{j}$', r'$MI_{i*j}$']

# plot model coefficients
coefs = pd.DataFrame(columns=[r"$rsc$", r"$rsc$", r"$rsc$", r"$\Delta rsc$", r"$\Delta rsc$", r"$\Delta rsc$"], 
                    data=np.concatenate((overall_beta[:, [1,2,3]], dnc_beta[:, [1,2,3]]), axis=1))
coefs = coefs.melt()
coefs['regressor'] = np.concatenate([[r'$MI_i$']*dnc_beta.shape[0],
                                [r'$MI_j$']*dnc_beta.shape[0],
                               [r'$MI_{i*j}$']*dnc_beta.shape[0],
                               [r'$MI_i$']*dnc_beta.shape[0],
                               [r'$MI_j$']*dnc_beta.shape[0],
                               [r'$MI_{i*j}$']*dnc_beta.shape[0]])
mncax.axvline(0, linestyle='--', color='k', zorder=1)
sns.stripplot(y='regressor', x='value', data=coefs[coefs['variable']==r"$rsc$"], ax=mncax, linewidth=1, edgecolor='white',
                                                     color='lightgrey', zorder=1)
sns.pointplot(y='regressor', x='value', data=coefs[coefs['variable']==r"$rsc$"], join=False, ci=95, ax=mncax, errwidth=1, scale=0.7, capsize=0.05,
                                                     color='k', zorder=2)

mdncax.axvline(0, linestyle='--', color='k', zorder=1)
sns.stripplot(y='regressor', x='value', data=coefs[coefs['variable']==r"$\Delta rsc$"], ax=mdncax, linewidth=1, edgecolor='white',
                                                     color='lightgrey', zorder=1)
sns.pointplot(y='regressor', x='value', data=coefs[coefs['variable']==r"$\Delta rsc$"], join=False, ci=95, ax=mdncax, errwidth=1, scale=0.7, capsize=0.05,
                                                     color='k', zorder=2)

mncax.set_xlim((-0.1, 0.1))
mdncax.set_xlim((-0.1, 0.1))


# PRINT STATS FOR REGRESSION MODELS (over the group)
print("OVERALL rsc")
print("Mi       mean:  {0} \n"
      "         sem:   {1} \n"
      "         pval:  {2} \n"
      "         U stat: {3} \n".format(np.mean(overall_beta[:,1]), 
                                       overall_beta[:,1].std() / np.sqrt(overall_beta.shape[0]), 
                                       ss.ranksums(overall_beta[:, 1], np.zeros(dnc_beta.shape[0])).pvalue,
                                       ss.ranksums(overall_beta[:, 1], np.zeros(dnc_beta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((overall_pvals[:, 1]<0.05).sum(), overall_pvals.shape[0]))

print("Mj       mean:  {0} \n"
      "         sem:   {1} \n"
      "         pval:  {2} \n"
      "         U stat: {3} \n".format(np.mean(overall_beta[:,2]), 
                                       overall_beta[:,2].std() / np.sqrt(overall_beta.shape[0]), 
                                       ss.ranksums(overall_beta[:, 2], np.zeros(dnc_beta.shape[0])).pvalue,
                                       ss.ranksums(overall_beta[:, 2], np.zeros(dnc_beta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((overall_pvals[:, 2]<0.05).sum(), overall_pvals.shape[0]))

print("Mij       mean:  {0} \n"
      "         sem:   {1} \n"
      "         pval:  {2} \n"
      "         U stat: {3} \n".format(np.mean(overall_beta[:,3]), 
                                       overall_beta[:,3].std() / np.sqrt(overall_beta.shape[0]), 
                                       ss.ranksums(overall_beta[:, 3], np.zeros(dnc_beta.shape[0])).pvalue,
                                       ss.ranksums(overall_beta[:, 3], np.zeros(dnc_beta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((overall_pvals[:, 3]<0.05).sum(), overall_pvals.shape[0]))

print("DELTA rsc")
print("Mi       mean:  {0} \n"
      "         sem:   {1} \n"
      "         pval:  {2} \n"
      "         U stat: {3} \n".format(np.mean(dnc_beta[:,1]), 
                                       dnc_beta[:,1].std() / np.sqrt(dnc_beta.shape[0]), 
                                       ss.ranksums(dnc_beta[:, 1], np.zeros(dnc_beta.shape[0])).pvalue,
                                       ss.ranksums(dnc_beta[:, 1], np.zeros(dnc_beta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((dnc_pvals[:, 1]<0.05).sum(), dnc_pvals.shape[0]))

print("Mj       mean:  {0} \n"
      "         sem:   {1} \n"
      "         pval:  {2} \n"
      "         U stat: {3} \n".format(np.mean(dnc_beta[:,2]), 
                                       dnc_beta[:,2].std() / np.sqrt(dnc_beta.shape[0]), 
                                       ss.ranksums(dnc_beta[:, 2], np.zeros(dnc_beta.shape[0])).pvalue,
                                       ss.ranksums(dnc_beta[:, 2], np.zeros(dnc_beta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((dnc_pvals[:, 2]<0.05).sum(), dnc_pvals.shape[0]))

print("Mij       mean:  {0} \n"
      "         sem:   {1} \n"
      "         pval:  {2} \n"
      "         U stat: {3} \n".format(np.mean(dnc_beta[:,3]), 
                                       dnc_beta[:,3].std() / np.sqrt(dnc_beta.shape[0]), 
                                       ss.ranksums(dnc_beta[:, 3], np.zeros(dnc_beta.shape[0])).pvalue,
                                       ss.ranksums(dnc_beta[:, 3], np.zeros(dnc_beta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((dnc_pvals[:, 3]<0.05).sum(), dnc_pvals.shape[0]))

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()
