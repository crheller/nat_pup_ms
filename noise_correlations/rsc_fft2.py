import load_results as ld
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

nc = ld.load_noise_correlation('rsc')
mask = (nc['p_all'] < 1) #& (nc['site']=='BRT026c')
pairs = nc.loc[mask].index
boxcar = 'boxcar_'

nc1 = ld.load_noise_correlation('rsc_ev_{}fft0-0.25'.format(boxcar)).loc[pairs]
nc2 = ld.load_noise_correlation('rsc_ev_{}fft0.5-2'.format(boxcar)).loc[pairs]
nc3 = ld.load_noise_correlation('rsc_ev_{}fft0.25-4'.format(boxcar)).loc[pairs]
nc4 = ld.load_noise_correlation('rsc_ev_{}fft10-25'.format(boxcar)).loc[pairs]
nc5 = ld.load_noise_correlation('rsc_ev_{}fft25-50'.format(boxcar)).loc[pairs]
nc1pr = ld.load_noise_correlation('rsc_ev_pr_rm2_{}fft0-0.25'.format(boxcar)).loc[pairs]
nc2pr = ld.load_noise_correlation('rsc_ev_pr_rm2_{}fft0.5-2'.format(boxcar)).loc[pairs]
nc3pr = ld.load_noise_correlation('rsc_ev_pr_rm2_{}fft0.25-4'.format(boxcar)).loc[pairs]
nc4pr = ld.load_noise_correlation('rsc_ev_pr_rm2_{}fft10-25'.format(boxcar)).loc[pairs]
nc5pr = ld.load_noise_correlation('rsc_ev_pr_rm2_{}fft25-50'.format(boxcar)).loc[pairs]

xticks = np.arange(0, 5)
xlabs = ['DC', '0.1 - 4 Hz', '4 - 10 Hz', '10 - 25 Hz', '25 - 50 Hz']

all_raw = [nc1['all'].mean(), nc2['all'].mean(), nc3['all'].mean(), nc4['all'].mean(), nc5['all'].mean()]
all_raw_err = [nc1['all'].sem(), nc2['all'].sem(), nc3['all'].sem(), nc4['all'].sem(), nc5['all'].sem()]
all_pr = [nc1pr['all'].mean(), nc2pr['all'].mean(), nc3pr['all'].mean(), nc4pr['all'].mean(), nc5pr['all'].mean()]
all_pr_err = [nc1pr['all'].sem(), nc2pr['all'].sem(), nc3pr['all'].sem(), nc4pr['all'].sem(), nc5pr['all'].sem()]

bp = [nc1pr['bp'].mean(), nc2pr['bp'].mean(), nc3pr['bp'].mean(), nc4pr['bp'].mean(), nc5pr['bp'].mean()]
bp_err = [nc1pr['bp'].sem(), nc2pr['bp'].sem(), nc3pr['bp'].sem(), nc4pr['bp'].sem(), nc5pr['bp'].sem()]
sp = [nc1pr['sp'].mean(), nc2pr['sp'].mean(), nc3pr['sp'].mean(), nc4pr['sp'].mean(), nc5pr['sp'].mean()]
sp_err = [nc1pr['sp'].sem(), nc2pr['sp'].sem(), nc3pr['sp'].sem(), nc4pr['sp'].sem(), nc5pr['sp'].sem()]

bp = [nc1['bp'].mean(), nc2['bp'].mean(), nc3['bp'].mean(), nc4['bp'].mean(), nc5['bp'].mean()]
bp_err = [nc1['bp'].sem(), nc2['bp'].sem(), nc3['bp'].sem(), nc4['bp'].sem(), nc5['bp'].sem()]
sp = [nc1['sp'].mean(), nc2['sp'].mean(), nc3['sp'].mean(), nc4['sp'].mean(), nc5['sp'].mean()]
sp_err = [nc1['sp'].sem(), nc2['sp'].sem(), nc3['sp'].sem(), nc4['sp'].sem(), nc5['sp'].sem()]

f, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

ax[0].errorbar(xticks, all_raw, yerr=all_raw_err, color='forestgreen', marker='.')
ax[0].errorbar(xticks, all_pr, yerr=all_pr_err, color='purple', marker='.')
ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xlabs, rotation=45, fontsize=10)
ax[0].set_ylabel('rsc')
ax[0].set_xlabel('Frequency band')

ax[1].errorbar(xticks, bp, yerr=bp_err, color='firebrick', marker='.')
ax[1].errorbar(xticks, sp, yerr=sp_err, color='navy', marker='.')
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xlabs, rotation=45, fontsize=10)
ax[1].set_xlabel('Frequency band')

f.tight_layout()

plt.show()