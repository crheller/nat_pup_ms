"""
Dissociate first and second order effects. Do this by:
    showing bandpass filtered noise correlation results
    show that delta rsc not predicted from MI, but rsc is 
"""
import load_results as ld
import colors as color

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.preprocessing as preproc
from nems_lbhb.baphy import parse_cellid
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = True
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig7_dissociate_1st_2nd_order.svg'

# set up subplots
f = plt.figure(figsize=(6, 6))

ncax = plt.subplot2grid((2, 3), (0, 0), colspan=2)
bsax = plt.subplot2grid((2, 3), (1, 0), colspan=2)
mncax = plt.subplot2grid((2, 3), (0, 2), colspan=1)
mdncax = plt.subplot2grid((2, 3), (1, 2), colspan=1)

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


# plot results
xvals = range(len(raw_results))
raw_nc = [r['all'].mean() for r in raw_results]
raw_sem = [r['all'].sem() for r in raw_results]
corr_nc = [c['all'].mean() for c in corr_results]
corr_sem = [c['all'].sem() for c in corr_results]

bp_nc = [r['bp'].mean() for r in raw_results]
bp_sem = [r['bp'].sem() for r in raw_results]
sp_nc = [r['sp'].mean() for r in raw_results]
sp_sem = [r['sp'].sem() for r in raw_results]

ncax.errorbar(xvals, raw_nc, yerr=raw_sem, marker='.', color=color.RAW, label='Raw')
ncax.errorbar(xvals, corr_nc, yerr=corr_sem, marker='.', color=color.CORRECTED, label='Corrected')
ncax.axhline(0, linestyle='--', lw=2, color='grey')
ncax.legend(frameon=False)
ncax.set_xticks(xvals)
ncax.set_xticklabels(f_band, rotation=45)
ncax.set_ylabel('Noise Correlation')
ncax.set_ylim((-0.01, 0.08))
ncax.set_xlabel('Frequency Band (Hz)')

bsax.errorbar(xvals, bp_nc, yerr=bp_sem, marker='.', color=color.LARGE, label='Large')
bsax.errorbar(xvals, sp_nc, yerr=sp_sem, marker='.', color=color.SMALL, label='Small')
bsax.axhline(0, linestyle='--', lw=2, color='grey')
bsax.legend(frameon=False)
bsax.set_xticks(xvals)
bsax.set_xticklabels(f_band, rotation=45)
bsax.set_ylabel('Noise Correlation')
bsax.set_ylim((-0.01, 0.08))
bsax.set_xlabel('Frequency Band (Hz)')


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
mask = (rsc_df['m1'] < .3) & (rsc_df['m1'] > -.2) & (rsc_df['m2'] < .3) & (rsc_df['m2'] > -.2)

# model results
rsc_df['m1*m2'] = rsc_df['m1'] * rsc_df['m2']
X = sm.add_constant(rsc_df[['m1', 'm2', 'm1*m2']])
y = rsc_df['diff']
model_dnc = sm.OLS(y, X).fit()

X = sm.add_constant(rsc_df[['m1', 'm2', 'm1*m2']])
y = rsc_df['all']
model_all = sm.OLS(y, X).fit()

beta = [r'$0$', r'$MI_{i}$', r'$MI_{j}$', r'$MI_{i}*MI_{j}$']

# plot
ci = abs(model_dnc.conf_int()[0] - model_dnc.conf_int()[1])
mdncax.errorbar([0, 1, 2, 3], model_dnc.params.values, yerr=ci.values, 
                        color='k', marker='.', linestyle='none', lw=2, 
                        label=r'$R^{2} = %s$' % round(model_dnc.rsquared, 3))
mdncax.axhline(0, linestyle='--', color='grey', lw=2)
mdncax.set_xlabel("Predictor")
mdncax.set_ylabel(r"$\beta$ coefficient")
mdncax.set_title(r"$\Delta$"+ "Noise \n Correlation", fontsize=10)
mdncax.set_xticks([0, 1, 2, 3])
mdncax.set_xticklabels(beta, rotation=45, fontsize=8)
mdncax.set_xlim((-0.5, 3.5))
mdncax.legend(frameon=False, fontsize=8)
mdncax.set_ylim((-0.15, 0.15))

ci = abs(model_all.conf_int()[0] - model_all.conf_int()[1])
mncax.errorbar([0, 1, 2, 3], model_all.params.values, yerr=ci.values, 
                        color='k', marker='.', linestyle='none', lw=2, 
                        label=r'$R^{2} = %s$' % round(model_all.rsquared, 3))
mncax.axhline(0, linestyle='--', color='grey', lw=2)
mncax.set_xlabel("Predictor")
mncax.set_ylabel(r"$\beta$ coefficient")
mncax.set_title("Noise \n Correlation", fontsize=10)
mncax.set_xticks([0, 1, 2, 3])
mncax.set_xticklabels(beta, rotation=45, fontsize=8)
mncax.set_xlim((-0.5, 3.5))
mncax.legend(frameon=False, fontsize=8)
mncax.set_ylim((-0.1, 0.8))

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()