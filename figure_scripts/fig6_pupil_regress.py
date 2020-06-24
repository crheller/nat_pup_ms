"""
Illustrate that first and second order effects are independent by regressing out
pupil-explained variance. Show that delta noise correlations and decoding improvement
persist. 
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
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig6_pupil_regression.svg'
loader = decoding.DecodingResults()
raw_mn = 'dprime_jk10_zscore_nclvz_fixtdr2'
modelname = 'dprime_pr_rm2_jk10_zscore_nclvz_fixtdr2'
sim1 = 'dprime_simInTDR_sim1_pr_rm2_jk10_zscore_nclvz_fixtdr2'
sim2 = 'dprime_simInTDR_sim2_pr_rm2_jk10_zscore_nclvz_fixtdr2'
sim1_raw = 'dprime_simInTDR_sim1_jk10_zscore_nclvz_fixtdr2'
sim2_raw = 'dprime_simInTDR_sim2_jk10_zscore_nclvz_fixtdr2'
estval = '_test'
high_var_only = True
persite = False

# where to crop the data
if estval == '_train':
    x_cut = (2, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 8)
    y_cut = (0.2, 1) 

# set up subplots
f = plt.figure(figsize=(6, 3))

pcorax = plt.subplot2grid((1, 2), (0, 0))
dbax = plt.subplot2grid((1, 2), (0, 1))
#ncax = plt.subplot2grid((2, 2), (1, 0))
#bsax = plt.subplot2grid((2, 2), (1, 1))

#'bbl086b'
sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
        'DRX006b.e1:64', 'DRX006b.e65:128', 
        'DRX007a.e1:64', 'DRX007a.e65:128', 
        'DRX008b.e1:64', 'DRX008b.e65:128']

# ======================================== Dprime results ================================================
df_raw = []
df = []
df_sim1 = []
df_sim2 = []
df_sim1_raw = []
df_sim2_raw = []
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

    fn = os.path.join(path, site, sim1_raw+'_TDR.pickle')
    results_sim1_raw = loader.load_results(fn)
    _df_sim1_raw = results_sim1_raw.numeric_results

    fn = os.path.join(path, site, sim2_raw+'_TDR.pickle')
    results_sim2_raw = loader.load_results(fn)
    _df_sim2_raw = results_sim2_raw.numeric_results

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

        _df_sim1_raw = _df_sim1_raw.loc[pd.IndexSlice[stim, 2], :]
        _df_sim1_raw['state_diff'] = (_df_sim1_raw['bp_dp'] - _df_sim1_raw['sp_dp']) / _df_raw['dp_opt_test']
        _df_sim1_raw['site'] = site
        df_sim1_raw.append(_df_sim1_raw)

        _df_sim2_raw = _df_sim2_raw.loc[pd.IndexSlice[stim, 2], :]
        _df_sim2_raw['state_diff'] = (_df_sim2_raw['bp_dp'] - _df_sim2_raw['sp_dp']) / _df_raw['dp_opt_test']
        _df_sim2_raw['site'] = site
        df_sim2_raw.append(_df_sim2_raw)

df = pd.concat(df)
df_raw = pd.concat(df_raw)
df_sim1 = pd.concat(df_sim1)
df_sim2 = pd.concat(df_sim2)
df_sim1_raw = pd.concat(df_sim1_raw)
df_sim2_raw = pd.concat(df_sim2_raw)

# filter based on x_cut / y_cut
#mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
#mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
mask1 = (df_raw['dU_mag'+estval] < x_cut[1]) & (df_raw['dU_mag'+estval] > x_cut[0])
mask2 = (df_raw['cos_dU_evec'+estval] < y_cut[1]) & (df_raw['cos_dU_evec'+estval] > y_cut[0])
df = df[mask1 & mask2]
df_raw = df_raw[mask1 & mask2]
df_sim1 = df_sim1[mask1 & mask2]
df_sim2 = df_sim2[mask1 & mask2]
df_sim1_raw = df_sim1_raw[mask1 & mask2]
df_sim2_raw = df_sim2_raw[mask1 & mask2]

# append the simulation results as columns in the raw dataframe
df['sim1'] = df_sim1['state_diff']
df['sim2'] = df_sim2['state_diff']
df['raw'] = df_raw['state_diff']
df['sim1_raw'] = df_sim1_raw['state_diff']
df['sim2_raw'] = df_sim2_raw['state_diff']


# bar plot of delta dprime for raw data, 1st order, and 2nd order simulation
if not persite:
    dbax.bar([-0.25, 0.75, 1.75], 
            [df['raw'].mean(), df['sim1_raw'].mean(), df['sim2_raw'].mean()],
            yerr=[df['state_diff'].sem(), df['sim1'].sem(), df['sim2'].sem()],
            edgecolor='k', color=['tab:blue'], lw=1, width=0.4, label='Raw')
    dbax.bar([0.25, 1.25, 2.25], 
            [df['state_diff'].mean(), df['sim1'].mean(), df['sim2'].mean()],
            yerr=[df['state_diff'].sem(), df['sim1'].sem(), df['sim2'].sem()],
            edgecolor='k', color=['tab:orange'], lw=1, width=0.4, label='Pupil-corrected')

else:
    dbax.bar([0, 1, 2], df.groupby(by='site').mean()[['state_diff', 'sim1', 'sim2']].mean(), 
                        yerr=df.groupby(by='site').mean()[['state_diff', 'sim1', 'sim2']].sem(),
                        color='lightgrey', edgecolor='k', lw=1)
    dbax.axhline(0, linestyle='--', color='k')   

dbax.set_xticks([0, 1, 2])
dbax.set_xticklabels(['Raw', '1st order', '2nd order'], rotation=45)
dbax.set_ylabel(r"$\Delta d'^{2}$")
dbax.set_ylim((0, 1.5))
dbax.legend(frameon=False)
dbax.set_title('Discriminability Improvement')

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
pcorax.set_xlabel(r"Pearson's $r$")
pcorax.set_ylabel(r"Neurons ($N$)")
pcorax.set_title('Residual spike count \n correlation with pupil')


f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()