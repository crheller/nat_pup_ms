"""
Illustrate that first and second order effects are independent by regressing out
pupil-explained variance. Show that delta noise correlations and decoding improvement
persist. 

Show that overall noise correlations can be predicted from first order effects, while
delta noise correlations cannot.
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
mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig6_pupil_regression.svg'
loader = decoding.DecodingResults()
raw_mn = 'dprime_jk10_zscore'
modelname = 'dprime_pr_rm2_jk10_zscore'
sim1 = 'dprime_sim1_pr_rm2_jk10_zscore'
sim2 = 'dprime_sim2_pr_rm2_jk10_zscore'
estval = '_train'
nbins = 20
cmap = 'PRGn'
high_var_only = True

# where to crop the data
if estval == '_train':
    x_cut = (2, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 9)
    y_cut = (0.35, 1) 

# set up subplots
f = plt.figure(figsize=(6, 6))

pcorax = plt.subplot2grid((2, 2), (0, 0))
dbax = plt.subplot2grid((2, 2), (0, 1))
ncax = plt.subplot2grid((2, 2), (1, 0))
bsax = plt.subplot2grid((2, 2), (1, 1))

sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
        'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
        'DRX007a.e1:64', 'DRX007a.e65:128', 
        'DRX008b.e1:64', 'DRX008b.e65:128']


# ======================================== Dprime results ================================================
df_raw = []
df = []
df_sim1 = []
df_sim2 = []
for site in sites:

    fn = os.path.join(path, site, raw_mn+'_TDR.pickle')
    results = loader.load_results(fn)
    _df_raw = results.numeric_results

    fn = os.path.join(path, site, modelname+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

    fn = os.path.join(path, site, sim1+'_TDR.pickle')
    results_sim1 = loader.load_results(fn)
    _df_sim1 = results_sim1.numeric_results

    fn = os.path.join(path, site, sim2+'_TDR.pickle')
    results_sim2 = loader.load_results(fn)
    _df_sim2 = results_sim2.numeric_results

    stim = results.evoked_stimulus_pairs
    high_var_pairs = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/high_pvar_stim_combos.csv', index_col=0)
    high_var_pairs = high_var_pairs[high_var_pairs.site==site].index.get_level_values('combo')
    if high_var_only:
        stim = [s for s in stim if s in high_var_pairs]
    if len(stim) == 0:
        pass
    else:
        _df_raw = _df_raw.loc[pd.IndexSlice[stim, 2], :]
        _df_raw['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
        _df_raw['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
        _df_raw['state_diff'] = (_df_raw['bp_dp'] - _df_raw['sp_dp']) / _df_raw['dp_opt_test']
        _df_raw['site'] = site
        df_raw.append(_df_raw)

        _df = _df.loc[pd.IndexSlice[stim, 2], :]
        _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
        _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
        _df['state_diff'] = (_df['bp_dp'] - _df['sp_dp']) / _df['dp_opt_test']
        _df['site'] = site
        df.append(_df)

        _df_sim1 = _df_sim1.loc[pd.IndexSlice[stim, 2], :]
        _df_sim1['state_diff'] = (_df_sim1['bp_dp'] - _df_sim1['sp_dp']) / _df_raw['dp_opt_test']
        _df_sim1['site'] = site
        df_sim1.append(_df_sim1)

        _df_sim2 = _df_sim2.loc[pd.IndexSlice[stim, 2], :]
        _df_sim2['state_diff'] = (_df_sim2['bp_dp'] - _df_sim2['sp_dp']) / _df_raw['dp_opt_test']
        _df_sim2['site'] = site
        df_sim2.append(_df_sim2)

df = pd.concat(df)
df_raw = pd.concat(df_raw)
df_sim1 = pd.concat(df_sim1)
df_sim2 = pd.concat(df_sim2)

# filter based on x_cut / y_cut
#mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
#mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
mask1 = (df_raw['dU_mag'+estval] < x_cut[1]) & (df_raw['dU_mag'+estval] > x_cut[0])
mask2 = (df_raw['cos_dU_evec'+estval] < y_cut[1]) & (df_raw['cos_dU_evec'+estval] > y_cut[0])
df = df[mask1 & mask2]
df_sim1 = df_sim1[mask1 & mask2]
df_sim2 = df_sim2[mask1 & mask2]

# append the simulation results as columns in the raw dataframe
df['sim1'] = df_sim1['state_diff']
df['sim2'] = df_sim2['state_diff']

# bar plot of delta dprime for raw data, 1st order, and 2nd order simulation
dbax.bar([0, 1, 2], 
        [df['state_diff'].mean(), df['sim1'].mean(), df['sim2'].mean()],
        yerr=[df['state_diff'].sem(), df['sim1'].sem(), df['sim2'].sem()],
        edgecolor='k', color=['lightgrey'], lw=2, width=0.5)
dbax.set_xticks([0, 1, 2])
dbax.set_xticklabels(['Raw', '1st order', '2nd order'])
dbax.set_ylabel(r"$\Delta d'^{2}$")
dbax.set_ylim((-.15, 1.25))
dbax.set_title('Pupil-Corrected \n Decoding Improvement')

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
ncax.set_ylabel('rsc')
ncax.set_ylim((-0.01, 0.08))
ncax.set_xlabel('Frequency Band (Hz)')

bsax.errorbar(xvals, bp_nc, yerr=bp_sem, marker='.', color=color.LARGE, label='Large')
bsax.errorbar(xvals, sp_nc, yerr=sp_sem, marker='.', color=color.SMALL, label='Small')
bsax.axhline(0, linestyle='--', lw=2, color='grey')
bsax.legend(frameon=False)
bsax.set_xticks(xvals)
bsax.set_xticklabels(f_band, rotation=45)
bsax.set_ylabel('rsc')
bsax.set_ylim((-0.01, 0.08))
bsax.set_xlabel('Frequency Band (Hz)')

# ========================================== Pupil/residual correlation =======================================
raw_corr = []
pr_corr = []
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
    raw_corr.extend(rc)
    pr_corr.extend(prc)

bins = np.arange(-0.45, 0.45, 0.05)
pcorax.hist([raw_corr, pr_corr], rwidth=0.8, edgecolor='k', 
                color=[color.RAW, color.CORRECTED], bins=bins)
pcorax.legend(['Raw', 'Pupil-corrected'], frameon=False)
pcorax.set_xlabel(r"$cor(r_{i}(t), p(t))$")
pcorax.set_ylabel(r"Neurons ($N$)")
pcorax.set_title('Residual spike count \n correlation with pupil')

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()