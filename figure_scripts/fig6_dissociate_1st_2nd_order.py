"""
Dissociate first and second order effects. Do this by:
    showing bandpass filtered noise correlation results
    show that delta rsc not predicted from MI, but rsc is 
"""
import load_results as ld
import colors as color
from path_settings import DPRIME_DIR, PY_FIGURES_DIR

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.preprocessing as preproc
from nems_lbhb.baphy import parse_cellid
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = True
fig_fn = PY_FIGURES_DIR + '/fig6_dissociate_1st_2nd_order.svg'
mi_max = 0.3
mi_min = -0.2
vmin = -0.1
vmax = 0.1
# set up subplots
f = plt.figure(figsize=(12, 7.5))

ncax = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)
bsax = plt.subplot2grid((4, 6), (0, 3), colspan=2, rowspan=2)
mncax = plt.subplot2grid((4, 6), (2, 2), colspan=1, rowspan=2)
mdncax = plt.subplot2grid((4, 6), (2, 5), colspan=1, rowspan=2)
rscax = plt.subplot2grid((4, 6), (2, 0), colspan=2, rowspan=2)
drscax = plt.subplot2grid((4, 6), (2, 3), colspan=2, rowspan=2)

# ============================= load and plot delta noise correlations across freq. bands =======================================
boxcar = True
evoked = True
fs4 = False

modelnames = ['rsc_fft0-0.5', 'rsc_pr_rm2_fft0-0.5',
              'rsc_fft0.5-2', 'rsc_pr_rm2_fft0.5-2',
              'rsc_fft2-4', 'rsc_pr_rm2_fft2-4',
              'rsc_fft4-10', 'rsc_pr_rm2_fft4-10',
              'rsc_fft10-25', 'rsc_pr_rm2_fft10-25',
              'rsc_fft25-50', 'rsc_pr_rm2_fft25-50'
              ]
if fs4:
    modelnames = ['rsc_fft0-0.5', 'rsc_pr_rm2_fft0-0.5',
                'rsc_fft0.5-2', 'rsc_pr_rm2_fft0.5-2']
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
                                                                        raw_results[b].groupby(by='site').mean()['all']).pvalue))

    print("bin {0} big vs. small:     pval: {1}".format(b+1, ss.wilcoxon(raw_results[b].groupby(by='site').mean()['bp'], 
                                                                        raw_results[b].groupby(by='site').mean()['sp']).pvalue))

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
MI = df.loc[:, pd.IndexSlice['MI', 'st.pup']]

rsc_path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'
rsc_df = ld.load_noise_correlation('rsc_bal', path=rsc_path)

# add column for the gain of each neuron
m1 = [MI.loc[p.split('_')[0]] for p in rsc_df.index]
m2 = [MI.loc[p.split('_')[1]] for p in rsc_df.index]
rsc_df['m1'] = m1
rsc_df['m2'] = m2
rsc_df['diff'] = rsc_df['sp'] - rsc_df['bp']
mask = (rsc_df['m1'] < mi_max) & (rsc_df['m1'] > mi_min) & (rsc_df['m2'] < mi_max) & (rsc_df['m2'] > mi_min)

# mask low bin count data (see supp figure for histogram)
rsc_df = rsc_df[mask]

# model results
rsc_df['m1*m2'] = rsc_df['m1'] * rsc_df['m2']
X = sm.add_constant(rsc_df[['m1', 'm2', 'm1*m2']])
y = rsc_df['diff']
model_dnc = sm.OLS(y, X).fit()

X = sm.add_constant(rsc_df[['m1', 'm2', 'm1*m2']])
y = rsc_df['all']
model_all = sm.OLS(y, X).fit()

# plot heatmaps of noise corr vs. mi
# plot overall noise correlation
xbins = np.linspace(mi_min, mi_max, 10)
ybins = np.linspace(mi_min, mi_max, 10)
heatmap_rsc = ss.binned_statistic_2d(x=rsc_df['m1'], 
                            y=rsc_df['m2'],
                            values=rsc_df['all'],
                            statistic='mean',
                            bins=[xbins, ybins])

im = rscax.imshow(heatmap_rsc[0], cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax,
                            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                            origin='lower', interpolation='gaussian')
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

im = drscax.imshow(heatmap_drsc[0], cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax,
                            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                            origin='lower', interpolation='gaussian')
divider = make_axes_locatable(drscax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
drscax.set_title(r"$\Delta$ Noise Correlation")
drscax.set_xlabel(r"$MI_j$")
drscax.set_ylabel(r"$MI_i$")

beta = [r'$MI_{i}$', r'$MI_{j}$', r'$MI_{i*j}$']

# plot
ci = abs(model_dnc.conf_int()[0] - model_dnc.conf_int()[1])
#mdncax.errorbar([0, 1, 2], model_dnc.params.values[1:], yerr=ci.values[1:], 
#                        color='k', marker='.', linestyle='none', lw=2, 
#                        label=r'$R^{2} = %s$' % round(model_dnc.rsquared, 3))
mdncax.errorbar(0, model_dnc.params.values[1], yerr=ci.values[1], marker='o', label=beta[0])
mdncax.errorbar(1, model_dnc.params.values[2], yerr=ci.values[2], marker='o', label=beta[1])
mdncax.errorbar(2, model_dnc.params.values[3], yerr=ci.values[3], marker='o', label=beta[2])

mdncax.axhline(0, linestyle='--', color='grey', lw=2)
mdncax.set_ylabel(r"$\Delta$ Noise correlation per unit $MI$")
mdncax.set_title(r"$R^2 = {}$".format(round(model_dnc.rsquared, 3)), fontsize=8)
mdncax.set_xlim((-0.5, 3.5))
mdncax.set_ylim((-0.2, 0.6))

ci = abs(model_all.conf_int()[0] - model_all.conf_int()[1])
#mncax.errorbar([0, 1, 2], model_all.params.values[1:], yerr=ci.values[1:], 
#                        color=['k', 'g', 'b'], marker='.', linestyle='none', lw=2, 
#                        label=r'$R^{2} = %s$' % round(model_all.rsquared, 3))

mncax.errorbar(0, model_all.params.values[1], yerr=ci.values[1], marker='o', label=beta[0])
mncax.errorbar(1, model_all.params.values[2], yerr=ci.values[2], marker='o', label=beta[1])
mncax.errorbar(2, model_all.params.values[3], yerr=ci.values[3], marker='o', label=beta[2])

mncax.axhline(0, linestyle='--', color='grey', lw=2)
mncax.set_ylabel(r"Noise correlation per unit $MI$")
mncax.set_xlim((-0.5, 3.5))
mncax.legend(frameon=False, fontsize=8)
mncax.set_title(r"$R^2 = {}$".format(round(model_all.rsquared, 3)), fontsize=8)
mncax.set_ylim((-0.2, 2))

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()