"""
Compare delta dprime between models
"""
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT, CPN_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION
import charlieTools.nat_sounds_ms.decoding as decoding
import figures_final.helpers as fhelp
from charlieTools.nat_sounds_ms.decoding import plot_stimulus_pair
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob

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

plot_example = False
decoder = 'dprime_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-5' #'dprime_mvm-25-1_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-5'
rlv = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
ind = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss1"
plv = "psth.fs4.pup-loadpred.cpn-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
plv2 = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"

decoder = 'dprime_mvm-25-1_jk1_eev_fixtdr2-fa_noiseDim-2'
rlv = 'psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.er2-aev_stategain.SxR-spred-lvnorm.1xR.so-inoise.1xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3'
ind = 'psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.er2-aev_stategain.SxR-spred-lvnorm.1xR.so-inoise.SxR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3'
plv = 'psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.er2-aev_stategain.SxR-spred-lvnorm.SxR.so-inoise.SxR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3'
plv2 = 'psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.er2-aev_stategain.SxR-spred-lvnorm.SxR.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3'

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

    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+'_TDR.pickle')
    raw = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{rlv}_TDR.pickle')
    lv0 = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{ind}_TDR.pickle')
    indep = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{plv}_TDR.pickle')
    lv = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{plv2}_TDR.pickle')
    lv2 = loader.load_results(fn, cache_path=None, recache=recache)

    # get the epochs of interest (fit epochs)
    mask_bins = lv.meta['mask_bins']
    fit_combos = [k for k, v in lv.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                        (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
    all_combos = lv.evoked_stimulus_pairs
    val_combos = [c for c in all_combos if c not in fit_combos]

    # save results for each model and divide by fit / not fit stimuli
    for k, res in zip(['pup_indep', 'indep_noise', 'lv', 'lv2', 'raw'], [lv0, indep, lv, lv2, raw]):
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
f, ax = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)

for i, (k, tit) in enumerate(zip(['pup_indep', 'indep_noise', 'lv', 'lv2'], ['Pup. Shuff.', 'Private Noise', 'Shared Noise', 'Shared Noise (2)'])):
    ax[i].set_title(tit)
    cc = np.round(np.corrcoef(results['val']['raw']['delta_dprime'], results['val'][k]['delta_dprime'])[0, 1], 3)
    ax[i].scatter(results['val']['raw']['delta_dprime'],
                    results['val'][k]['delta_dprime'], s=15, edgecolor='white', color='tab:blue', label=f"val: {cc}")
    cc = np.round(np.corrcoef(results['fit']['raw']['delta_dprime'], results['fit'][k]['delta_dprime'])[0, 1], 3)
    ax[i].scatter(results['fit']['raw']['delta_dprime'],
                    results['fit'][k]['delta_dprime'], s=15, edgecolor='white', color='tab:orange', label=f"fit: {cc}")

# add unity line / best fit lines
mm = np.min(ax[i].get_xlim()+ax[i].get_ylim())
ma = np.max(ax[i].get_xlim()+ax[i].get_ylim())
for i, k in enumerate(['pup_indep', 'indep_noise', 'lv', 'lv2']):
    xran = np.arange(mm, ma, 0.01)
    m, b = np.polyfit(results['fit']['raw']['delta_dprime'], results['fit'][k]['delta_dprime'], 1)
    ax[i].plot(xran, m*xran + b, color='tab:orange')
    m, b = np.polyfit(results['val']['raw']['delta_dprime'], results['val'][k]['delta_dprime'], 1)
    ax[i].plot(xran, m*xran + b, color='tab:blue')
    ax[i].plot([mm, ma], [mm, ma], 'k--')
    ax[i].axhline(0, linestyle='--', color='k')
    ax[i].axvline(0, linestyle='--', color='k')
    ax[i].legend(frameon=False)

f.tight_layout()

# plot summary of error per model for validation / fit stimuli
f, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.errorbar(range(4), y = [
                        np.abs(results['val']['pup_indep']['delta_dprime']-results['val']['raw']['delta_dprime']).mean(),
                        np.abs(results['val']['indep_noise']['delta_dprime']-results['val']['raw']['delta_dprime']).mean(),
                        np.abs(results['val']['lv']['delta_dprime']-results['val']['raw']['delta_dprime']).mean(),
                        np.abs(results['val']['lv2']['delta_dprime']-results['val']['raw']['delta_dprime']).mean()
                        ],
                    yerr=[
                        np.abs(results['val']['pup_indep']['delta_dprime']-results['val']['raw']['delta_dprime']).sem(),
                        np.abs(results['val']['indep_noise']['delta_dprime']-results['val']['raw']['delta_dprime']).sem(),
                        np.abs(results['val']['lv']['delta_dprime']-results['val']['raw']['delta_dprime']).sem(),
                        np.abs(results['val']['lv2']['delta_dprime']-results['val']['raw']['delta_dprime']).sem()
                    ],
                    capsize=2, marker='o', color='k', label='Validation Stim.')
ax.errorbar(np.arange(4)+0.05, y = [
                        np.abs(results['fit']['pup_indep']['delta_dprime']-results['fit']['raw']['delta_dprime']).mean(),
                        np.abs(results['fit']['indep_noise']['delta_dprime']-results['fit']['raw']['delta_dprime']).mean(),
                        np.abs(results['fit']['lv']['delta_dprime']-results['fit']['raw']['delta_dprime']).mean(),
                        np.abs(results['fit']['lv2']['delta_dprime']-results['fit']['raw']['delta_dprime']).mean()
                        ],
                    yerr=[
                        np.abs(results['fit']['pup_indep']['delta_dprime']-results['fit']['raw']['delta_dprime']).sem(),
                        np.abs(results['fit']['indep_noise']['delta_dprime']-results['fit']['raw']['delta_dprime']).sem(),
                        np.abs(results['fit']['lv']['delta_dprime']-results['fit']['raw']['delta_dprime']).sem(),
                        np.abs(results['fit']['lv2']['delta_dprime']-results['fit']['raw']['delta_dprime']).sem()
                    ],
                    capsize=2, marker='o', color='orange', label='Fit Stim.')
ax.set_ylabel(r"$\Delta d'^2$ error")
ax.set_xlabel('Model')
ax.set_xticks(range(4))
ax.set_xticklabels(['pup indep.', 'private noise', 'shared noise', 'shared noise (2)'], rotation=45)
ax.legend(frameon=False)

f.tight_layout()


plt.show()
