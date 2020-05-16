import load_results as ld
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#nc = ld.load_noise_correlation('rsc')
#mask = (nc['p_all'] < 1) #& (nc['site']=='BRT026c')
#pairs = nc.loc[mask].index
boxcar = False
evoked = True

modelnames = ['rsc_fft0-0.05', 'rsc_pr_rm2_fft0-0.05',
            'rsc_fft0.05-0.1', 'rsc_pr_rm2_fft0.05-0.1',
            'rsc_fft0.1-0.25', 'rsc_pr_rm2_fft0.1-0.25',
            'rsc_fft0.25-0.5', 'rsc_pr_rm2_fft0.25-0.5',
            'rsc_fft0-0.1', 'rsc_pr_rm2_fft0-0.1',
            'rsc_fft0-0.25', 'rsc_pr_rm2_fft0-0.25',
            'rsc_fft0-0.5', 'rsc_pr_rm2_fft0-0.5',
            'rsc_fft0.5-2', 'rsc_pr_rm2_fft0.5-2',
            'rsc_fft0.1-4', 'rsc_pr_rm2_fft0.1-4',
            'rsc_fft0.25-4', 'rsc_pr_rm2_fft0.25-4',
            'rsc_fft0.5-4', 'rsc_pr_rm2_fft0.5-4',
            'rsc_fft4-10', 'rsc_pr_rm2_fft4-10',
            'rsc_fft10-25', 'rsc_pr_rm2_fft10-25',
            'rsc_fft25-50', 'rsc_pr_rm2_fft25-50']
if evoked:
    modelnames = [m.replace('rsc', 'rsc_ev') for m in modelnames]
if boxcar:
    modelnames = [m.replace('fft', 'boxcar_fft') for m in modelnames]

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

f, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].errorbar(xvals, raw_nc, yerr=raw_sem, marker='.', color='forestgreen', label='Raw')
ax[0].errorbar(xvals, corr_nc, yerr=corr_sem, marker='.', color='purple', label='Corrected')
ax[0].legend(frameon=False)
ax[0].set_xticks(xvals)
ax[0].set_xticklabels(f_band, rotation=45)
ax[0].set_ylabel('rsc')

ax[1].errorbar(xvals, bp_nc, yerr=bp_sem, marker='.', color='firebrick', label='Large')
ax[1].errorbar(xvals, sp_nc, yerr=sp_sem, marker='.', color='navy', label='Small')
ax[1].legend(frameon=False)
ax[1].set_xticks(xvals)
ax[1].set_xticklabels(f_band, rotation=45)
ax[1].set_ylabel('rsc')


f.tight_layout()

plt.show()