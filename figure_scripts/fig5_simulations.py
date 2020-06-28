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
from path_settings import DPRIME_DIR, PY_FIGURES_DIR
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
import colors as color
import ax_labels as alab

from nems_lbhb.baphy import parse_cellid

import charlieTools.preprocessing as preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = True

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

all_sites = True
barplot = True

# where to crop the data
if estval == '_train':
    x_cut = (2.5, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    #x_cut = (1, 8)
    #y_cut = (0.2, 1) 
    x_cut = (1.5, 6)
    y_cut = (0, 1)

if all_sites:
    sites = ALL_SITES
else:
    sites = HIGHR_SITES

# ========================================= Load results ====================================================
df = []
df_sim1 = []
df_sim12 = []
df_pr = []
df_sim1_pr = []
df_sim12_pr = []
for site in sites:
    if site in LOWR_SITES: mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

    if site in LOWR_SITES: mn = sim1.replace('_jk10', '_jk1_eev') 
    else: mn = sim1
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim1 = loader.load_results(fn)
    _df_sim1 = results_sim1.numeric_results

    if site in LOWR_SITES: mn = sim12.replace('_jk10', '_jk1_eev') 
    else: mn = sim12
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim12 = loader.load_results(fn)
    _df_sim12 = results_sim12.numeric_results

    # pr results
    if site in LOWR_SITES: mn = modelname_pr.replace('_jk10', '_jk1_eev') 
    else: mn = modelname_pr
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn)
    _df_pr = results.numeric_results

    if site in LOWR_SITES: mn = sim1_pr.replace('_jk10', '_jk1_eev') 
    else: mn = sim1_pr
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim1 = loader.load_results(fn)
    _df_sim1_pr = results_sim1.numeric_results

    if site in LOWR_SITES: mn = sim12_pr.replace('_jk10', '_jk1_eev') 
    else: mn = sim12_pr
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim12 = loader.load_results(fn)
    _df_sim12_pr = results_sim12.numeric_results

    stim = results.evoked_stimulus_pairs

    _df = _df.loc[pd.IndexSlice[stim, 2], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
    _df['state_diff'] = (_df['bp_dp'] - _df['sp_dp']) / _df['dp_opt_test']
    _df['site'] = site
    df.append(_df)

    _df_sim1 = _df_sim1.loc[pd.IndexSlice[stim, 2], :]
    _df_sim1['state_diff'] = (_df_sim1['bp_dp'] - _df_sim1['sp_dp']) / _df['dp_opt_test']
    _df_sim1['site'] = site
    df_sim1.append(_df_sim1)

    _df_sim12 = _df_sim12.loc[pd.IndexSlice[stim, 2], :]
    _df_sim12['state_diff'] = (_df_sim12['bp_dp'] - _df_sim12['sp_dp']) / _df['dp_opt_test']
    _df_sim12['site'] = site
    df_sim12.append(_df_sim12)

    # pr results
    _df_pr = _df_pr.loc[pd.IndexSlice[stim, 2], :]
    _df_pr['state_diff'] = (_df_pr['bp_dp'] - _df_pr['sp_dp']) / _df['dp_opt_test']
    _df_pr['site'] = site
    df_pr.append(_df_pr)

    _df_sim1_pr = _df_sim1_pr.loc[pd.IndexSlice[stim, 2], :]
    _df_sim1_pr['state_diff'] = (_df_sim1_pr['bp_dp'] - _df_sim1_pr['sp_dp']) / _df['dp_opt_test']
    _df_sim1_pr['site'] = site
    df_sim1_pr.append(_df_sim1_pr)

    _df_sim12_pr = _df_sim12_pr.loc[pd.IndexSlice[stim, 2], :]
    _df_sim12_pr['state_diff'] = (_df_sim12_pr['bp_dp'] - _df_sim12_pr['sp_dp']) / _df['dp_opt_test']
    _df_sim12_pr['site'] = site
    df_sim12_pr.append(_df_sim12_pr)

df_all = pd.concat(df)
df_sim1_all = pd.concat(df_sim1)
df_sim12_all = pd.concat(df_sim12)
df_pr_all = pd.concat(df_pr)
df_sim1_pr_all = pd.concat(df_sim1_pr)
df_sim12_pr_all = pd.concat(df_sim12_pr)

# filter based on x_cut / y_cut
mask1 = (df_all['dU_mag'+estval] < x_cut[1]) & (df_all['dU_mag'+estval] > x_cut[0])
mask2 = (df_all['cos_dU_evec'+estval] < y_cut[1]) & (df_all['cos_dU_evec'+estval] > y_cut[0])
df = df_all[mask1 & mask2]
df_sim1 = df_sim1_all[mask1 & mask2]
df_sim12 = df_sim12_all[mask1 & mask2]
df_pr = df_pr_all[mask1 & mask2]
df_sim1_pr = df_sim1_pr_all[mask1 & mask2]
df_sim12_pr = df_sim12_pr_all[mask1 & mask2]

df['sim1'] = df_sim1['state_diff']
df['sim12'] = df_sim12['state_diff']
df['pr'] = df_pr['state_diff']
df['sim1_pr'] = df_sim1_pr['state_diff']
df['sim12_pr'] = df_sim12_pr['state_diff']

# ========================================= Plot data =====================================================
# set up subplots
f = plt.figure(figsize=(9, 3.2))

dax = plt.subplot2grid((1, 3), (0, 0))
dprax = plt.subplot2grid((1, 3), (0, 1))
prax = plt.subplot2grid((1, 3), (0, 2))


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
            dax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.1, len(vals)),
                        vals, color='grey', marker='D', edgecolor='white', s=30, zorder=2)
        except:
            pass
        vals = df.loc[df.site.isin(HIGHR_SITES)].groupby(by='site').mean()[s]
        dax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.1, len(vals)),
                    vals, color='k', marker='o', edgecolor='white', s=50, zorder=3)
    dax.axhline(0, linestyle='--', color='grey', lw=2)     
dax.set_xticks([0, 1, 2])
dax.set_xticklabels(['None', '1st order', '1st + 2nd'], rotation=45)
dax.set_xlabel('Simulation')
dax.set_ylabel(r"$\Delta d'^{2}$")
dax.set_title('Discriminability Improvement \n Raw Data', color=color.RAW)
dax.set_ylim((-1, 2))

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
            dprax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.1, len(vals)),
                        vals, color='grey', marker='D', edgecolor='white', s=30, zorder=2)
        except:
            pass
        vals = df.loc[df.site.isin(HIGHR_SITES)].groupby(by='site').mean()[s]
        dprax.scatter(i*np.ones(len(vals))+np.random.normal(0, 0.1, len(vals)),
                    vals, color='k', marker='o', edgecolor='white', s=50, zorder=3)
    dprax.axhline(0, linestyle='--', color='grey', lw=2)     
dprax.set_xticks([0.5, 1.5])
dprax.set_xticklabels(['1st order', '1st + 2nd'], rotation=45)
dprax.set_xlabel('Simulation')
dprax.set_ylabel(r"$\Delta d'^{2}$")
dprax.set_title('Discriminability Improvement \n Pupil Corrected Data', color=color.CORRECTED)
dprax.set_ylim(dax.get_ylim())
dprax.set_xlim(dax.get_xlim())


# plot the residual correlation with pupil
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

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()