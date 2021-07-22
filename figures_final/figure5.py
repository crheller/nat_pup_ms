"""
Show decoding results for the latent variable model.
Scatter plots for all three models. Error bar plot showing that
1-D LV model works best.
"""
from global_settings import CPN_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR3, CACHE_PATH
import charlieTools.nat_sounds_ms.decoding as decoding
import figures_final.helpers as fhelp

from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob
import nems.db as nd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 6

from nems.xform_helper import load_model_xform

savefig = False
fig_fn = PY_FIGURES_DIR3 + 'fig5.svg'
np.random.seed(123)

batch = 331

# options / models for figure 5 in disseration
#decoder = 'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-6'
#rlv = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
#indep = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss1"
#plv = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
#plv2 = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"

decoder = 'dprime_mvm-25-1_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-5'
rlv = "psth.fs4.pup-loadpred.cpnmvm,t25,w1-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
indep = "psth.fs4.pup-loadpred.cpnmvm,t25,w1-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss1"
plv = "psth.fs4.pup-loadpred.cpndmvm,t25,w1-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
plv2 = "psth.fs4.pup-loadpred.cpnmvm,t25,w1-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"


# LOAD DECODING RESULTS FOR MODEL / RAW DATA
recache = False
sites = CPN_SITES
batches = [331]*len(CPN_SITES)
results = {
    'fit': {
        'pup_indep': [],
        'indep_noise': [],
        'lv': [],
        'lv2': [],
        'raw': []
    },
    'val': {
        'pup_indep': [],
        'indep_noise': [],
        'lv': [],
        'lv2': [],
        'raw': []
    }
}
for batch, site in zip(batches, sites): #[s for s in HIGHR_SITES if s not in ['CRD017c', 'CRD016d']]:
    if site in ['BOL006b', 'BOL005c']:
        batch2 = 294
    else:
        batch2 = batch

    if batch in [289, 294]:
        _rlv = rlv.replace('.cpn', '')
        _ind = indep.replace('.cpn', '')
        _plv = plv.replace('.cpn', '')
        _plv2 = plv2.replace('.cpn', '')
    else:
        _rlv = rlv
        _ind = indep
        _plv = plv
        _plv2 = plv2

    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+'_TDR.pickle')
    raw = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+f'_model-LV-{_rlv}_TDR.pickle')
    lv0 = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+f'_model-LV-{_ind}_TDR.pickle')
    indep_noise = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+f'_model-LV-{_plv}_TDR.pickle')
    lv = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+f'_model-LV-{_plv2}_TDR.pickle')
    lv2 = loader.load_results(fn, cache_path=None, recache=recache)

    # get the epochs of interest (fit epochs)
    mask_bins = lv.meta['mask_bins']
    fit_combos = [k for k, v in lv.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                        (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
    all_combos = lv.evoked_stimulus_pairs
    val_combos = [c for c in all_combos if c not in fit_combos]

    # save results for each model and for divide by fit / not fit stimuli
    
    # fit stims first
    for k, res in zip(['pup_indep', 'indep_noise', 'lv', 'lv2', 'raw'], [lv0, indep_noise, lv, lv2, raw]):
        df = res.numeric_results
        df['delta_dprime'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
        df['site'] = site
        results['fit'][k].append(df.loc[fit_combos])
        results['val'][k].append(df.loc[val_combos])

# concatenate data frames
for k in results['fit'].keys():
    results['fit'][k] = pd.concat(results['fit'][k])
    results['val'][k] = pd.concat(results['val'][k])

############################# MAKE FIGURE #############################
s = 5
edgecolor='none'
cmap = {
    'pup_indep': 'tab:blue',
    'indep_noise': 'tab:orange',
    'lv': 'tab:green',
    'lv2': 'tab:purple'
}
axlim = (-1, 1)
f, ax = plt.subplots(1, 4, figsize=(6.5, 1.9))

# Pupil-independent noise
ax[0].scatter(results['fit']['raw']['delta_dprime'], results['fit']['pup_indep']['delta_dprime'], s=s, edgecolor=edgecolor, color=cmap['pup_indep'])
ax[0].set_ylabel(r"$\Delta d'^2$ Model Prediction")
ax[0].set_xlabel(r"$\Delta d'^2$ Raw data")
ax[0].set_title("Pupil-independent\nvariance")
ax[0].axhline(0, linestyle='--', color='grey', zorder=-1); ax[0].axvline(0, linestyle='--', color='grey', zorder=-1)
ax[0].plot(axlim, axlim, '--', color='grey', zorder=-1)
ax[0].set_xlim(axlim); ax[0].set_ylim(axlim)
# Pupil-dependent single neuron variability
ax[1].scatter(results['fit']['raw']['delta_dprime'], results['fit']['indep_noise']['delta_dprime'], s=s, edgecolor=edgecolor, color=cmap['indep_noise'])
ax[1].set_xlabel(r"$\Delta d'^2$ Raw data")
ax[1].set_title("Pupil-dependent\nsingle neuron variance")
ax[1].axhline(0, linestyle='--', color='grey', zorder=-1); ax[1].axvline(0, linestyle='--', color='grey', zorder=-1)
ax[1].plot(axlim, axlim, '--', color='grey', zorder=-1)
ax[1].set_xlim(axlim); ax[1].set_ylim(axlim)

# Pupil-depdendent LV
ax[2].scatter(results['fit']['raw']['delta_dprime'], results['fit']['lv']['delta_dprime'], s=s, edgecolor=edgecolor, color=cmap['lv'])
ax[2].set_xlabel(r"$\Delta d'^2$ Raw data")
ax[2].set_title("Pupil-dependent\n"+r"shared modultator ($k=1$)")
ax[2].axhline(0, linestyle='--', color='grey', zorder=-1); ax[2].axvline(0, linestyle='--', color='grey', zorder=-1)
ax[2].plot(axlim, axlim, '--', color='grey', zorder=-1)
ax[2].set_xlim(axlim); ax[2].set_ylim(axlim)

# Model summary
raw = results['fit']['raw']
err = {}
for i, k in enumerate(results['fit'].keys()):
    if k != 'raw':
        pred = results['fit'][k]
        # get bootstrapped confidence interval
        d = {s: np.abs(raw[raw.site==s]['delta_dprime']-pred[pred.site==s]['delta_dprime']).values for s in raw.site.unique()}
        err[k] = d
        bootsamp = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000)
        low = np.quantile(bootsamp, .025)
        high = np.quantile(bootsamp, .975)
        ax[3].scatter(i, np.mean(bootsamp), color='white', edgecolor=cmap[k], s=25)
        ax[3].plot([i, i], [low, high], zorder=-1, color=cmap[k])

# get paired pvalue using bootstrapping
for i, pair in enumerate([['pup_indep', 'indep_noise'], ['indep_noise', 'lv'], ['lv', 'lv2']]):
    d = {s: err[pair[0]][s]-err[pair[1]][s] for s in err[pair[1]].keys()}
    bootsamp = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000)
    p = get_direct_prob(bootsamp, np.zeros(len(bootsamp)))[0]
    ax[3].text(i, ax[3].get_ylim()[-1], r"p=$%s$"%round(p, 3), fontsize=5)
ax[3].set_ylabel(r"$\Delta d'^2$ Prediction Error")
f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()
