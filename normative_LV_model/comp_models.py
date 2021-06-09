"""
Compare delta dprime between models
"""
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT, CPN_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION
import charlieTools.nat_sounds_ms.decoding as decoding
import figures_final.helpers as fhelp
from charlieTools.nat_sounds_ms.decoding import plot_stimulus_pair

import seaborn as sns
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 6

recache = False
sites = CPN_SITES
batches = [331]*len(CPN_SITES)

decoder = 'dprime_jk10_zscore_nclvz_fixtdr2-fa'
ind = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.so-inoise.3xR_ccnorm.t5.ss1"
rlv = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.so-inoise.2xR_ccnorm.t6.ss1"
plv = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.so-inoise.2xR_ccnorm.t6.ss1"

results = {
    'fit': {
        'pup_indep': [],
        'indep_noise': [],
        'lv': [],
        'raw': []
    },
    'val': {
        'pup_indep': [],
        'indep_noise': [],
        'lv': [],
        'raw': []
    }
}
for batch, site in zip(batches, sites): #[s for s in HIGHR_SITES if s not in ['CRD017c', 'CRD016d']]:
    if site in ['BOL006b', 'BOL005c']:
        batch = batch2 = 294

    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+'_TDR.pickle')
    raw = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{rlv}_TDR.pickle')
    lv0 = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{ind}_TDR.pickle')
    indep = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{plv}_TDR.pickle')
    lv = loader.load_results(fn, cache_path=None, recache=recache)

    # get the epochs of interest (fit epochs)
    mask_bins = lv.meta['mask_bins']
    fit_combos = [k for k, v in lv.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                        (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
    all_combos = lv.evoked_stimulus_pairs
    val_combos = [c for c in all_combos if c not in fit_combos]

    # save results for each model and for divide by fit / not fit stimuli
    
    # fit stims first
    for k, res in zip(['pup_indep', 'indep_noise', 'lv', 'raw'], [lv0, indep, lv, raw]):
        df = res.numeric_results
        df['delta_dprime'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
        df['site'] = site
        results['fit'][k].append(df.loc[fit_combos])
        results['val'][k].append(df.loc[val_combos])

# concatenate data frames
for k in results['fit'].keys():
    results['fit'][k] = pd.concat(results['fit'][k])
    results['val'][k] = pd.concat(results['val'][k])


# scatter plot of model vs. true for each model
f, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

# pupil independent model
ax[0].set_title('Pupil indep. model')
cc = np.round(np.corrcoef(results['fit']['raw']['delta_dprime'], results['fit']['pup_indep']['delta_dprime'])[0, 1], 3)
ax[0].scatter(results['fit']['raw']['delta_dprime'],
                results['fit']['pup_indep']['delta_dprime'], s=25, edgecolor='white', color='tab:orange', label=f"fit: {cc}")
cc = np.round(np.corrcoef(results['val']['raw']['delta_dprime'], results['val']['pup_indep']['delta_dprime'])[0, 1], 3)
ax[0].scatter(results['val']['raw']['delta_dprime'],
                results['val']['pup_indep']['delta_dprime'], s=25, edgecolor='white', color='tab:blue', label=f"val: {cc}")

# independent noise model
ax[1].set_title('Indep noise')
cc = np.round(np.corrcoef(results['fit']['raw']['delta_dprime'], results['fit']['indep_noise']['delta_dprime'])[0, 1], 3)
ax[1].scatter(results['fit']['raw']['delta_dprime'],
                results['fit']['indep_noise']['delta_dprime'], s=25, edgecolor='white', color='tab:orange', label=f"fit: {cc}")
cc = np.round(np.corrcoef(results['val']['raw']['delta_dprime'], results['val']['indep_noise']['delta_dprime'])[0, 1], 3)
ax[1].scatter(results['val']['raw']['delta_dprime'],
                results['val']['indep_noise']['delta_dprime'], s=25, edgecolor='white', color='tab:blue', label=f"val: {cc}")

# full model
ax[2].set_title("Full LV model")
cc = np.round(np.corrcoef(results['fit']['raw']['delta_dprime'], results['fit']['lv']['delta_dprime'])[0, 1], 3)
ax[2].scatter(results['fit']['raw']['delta_dprime'],
                results['fit']['lv']['delta_dprime'], s=25, edgecolor='white', color='tab:orange', label=f"fit: {cc}")
cc = np.round(np.corrcoef(results['val']['raw']['delta_dprime'], results['val']['lv']['delta_dprime'])[0, 1], 3)
ax[2].scatter(results['val']['raw']['delta_dprime'],
                results['val']['lv']['delta_dprime'], s=25, edgecolor='white', color='tab:blue', label=f"val: {cc}")

for a in ax:
    a.axhline(0, linestyle='--', color='k')
    a.axvline(0, linestyle='--', color='k')
    a.legend(frameon=False)

f.tight_layout()

# ID stim pairs where lv model outperforms indep. noise
cat_fit = pd.concat((results['fit']['pup_indep']['delta_dprime'], 
                    results['fit']['indep_noise']['delta_dprime'],
                    results['fit']['lv']['delta_dprime'],
                    results['fit']['raw']['delta_dprime']), axis=1)
cat_val = pd.concat((results['val']['pup_indep']['delta_dprime'], 
                    results['val']['indep_noise']['delta_dprime'],
                    results['val']['lv']['delta_dprime'],
                    results['val']['raw']['delta_dprime']), axis=1)
cat_fit.columns = ['pup_indep', 'indep_noise', 'lv', 'raw']
cat_val.columns = ['pup_indep', 'indep_noise', 'lv', 'raw']

# first, just look at error for both models
f, ax = plt.subplots(1, 2, figsize=(8, 4))

stat, pval = ss.wilcoxon(np.abs(cat_fit['indep_noise']-cat_fit['raw']), np.abs(cat_fit['lv']-cat_fit['raw']))
sns.scatterplot(x=cat_fit['indep_noise']-cat_fit['raw'], 
                y=cat_fit['lv']-cat_fit['raw'], hue=results['fit']['raw'].site, ax=ax[0], **{'s': 15})
ax[0].plot([-1, 1], [-1, 1], 'k--')
ax[0].axhline(0, linestyle='--', color='k'); ax[0].axvline(0, linestyle='--', color='k')
ax[0].set_xlabel('Indep. Error')
ax[0].set_ylabel("LV Error")
ax[0].set_title("Fit stimuli")

stat, pval = ss.wilcoxon(np.abs(cat_val['indep_noise']-cat_val['raw']), np.abs(cat_val['lv']-cat_val['raw']))
sns.scatterplot(x=cat_val['indep_noise']-cat_val['raw'], 
                y=cat_val['lv']-cat_val['raw'], hue=results['val']['raw'].site, ax=ax[1], **{'s': 15})
ax[1].plot([-1, 1], [-1, 1], 'k--')
ax[1].axhline(0, linestyle='--', color='k'); ax[1].axvline(0, linestyle='--', color='k')
ax[1].set_xlabel("Indep. Error")
ax[1].set_ylabel("LV Error")
ax[1].set_title("Validation Stimuli")

f.tight_layout()

# plot absolute value of error
f, ax = plt.subplots(1, 2, figsize=(8, 4))

stat, pval = ss.wilcoxon(np.abs(cat_fit['indep_noise']-cat_fit['raw']), np.abs(cat_fit['lv']-cat_fit['raw']))
sns.scatterplot(x=np.abs(cat_fit['indep_noise']-cat_fit['raw']), 
                y=np.abs(cat_fit['lv']-cat_fit['raw']), hue=results['fit']['raw'].site, ax=ax[0], **{'s': 15})
ax[0].plot([0, 0.65], [0, 0.65], 'k--')
ax[0].axhline(0, linestyle='--', color='k'); ax[0].axvline(0, linestyle='--', color='k')
ax[0].set_xlabel('Indep. Error')
ax[0].set_ylabel("LV Error")
ax[0].set_title("Fit stimuli")

stat, pval = ss.wilcoxon(np.abs(cat_val['indep_noise']-cat_val['raw']), np.abs(cat_val['lv']-cat_val['raw']))
sns.scatterplot(x=np.abs(cat_val['indep_noise']-cat_val['raw']), 
                y=np.abs(cat_val['lv']-cat_val['raw']), hue=results['val']['raw'].site, ax=ax[1], **{'s': 15})
ax[1].plot([0, 0.65], [0, 0.65], 'k--')
ax[1].axhline(0, linestyle='--', color='k'); ax[1].axvline(0, linestyle='--', color='k')
ax[1].set_xlabel("Indep. Error")
ax[1].set_ylabel("LV Error")
ax[1].set_title("Validation Stimuli")

f.tight_layout()

# find stimulus pairs where LV model outperforms Indep. Noise model
cat_val[np.abs(cat_val.lv-cat_val.raw)<np.abs(cat_val.indep_noise-cat_val.raw)]

plt.show()