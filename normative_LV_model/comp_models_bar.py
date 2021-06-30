"""
Very digested way to compare performance of many of models
    two "categories" -- which LV model and which decoding 'model'
    diff decoding models on "y axis" and diff LV models on "x axis"
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
#sites = HIGHR_SITES
#batches = [289]*len(CPN_SITES)
fit_val = 'fit'
bar = False # if false, do single points w/ error
aligned = False
sig_delta = False

decoders = [
    'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-dU',
    'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa',
    'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-1',
    'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-2',
    'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-3', 
    'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-4', 
    'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-5'
]
# then load each of these for each cov. rank
ranks = [1, 2, 3]
ind = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss"
rlv = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp0-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss"
plv = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss"
plv = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp0-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss"

#indep = 'psth.fs4.pup-ld-st.pup0.pvp-epcpn-mvm.25.2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.2xR.d.so-inoise.3xR_ccnorm.r.t5.ss'
#rlv = 'psth.fs4.pup-ld-st.pup0.pvp-epcpn-mvm.25.2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.2xR.d.so-inoise.2xR_ccnorm.r.t5.ss'
#plv = 'psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.25.2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_ccnorm.r.t5.ss'

nrows = len(decoders)
ncols = len(ranks) * 3

f, ax = plt.subplots(nrows, 1, figsize=(8, 8), sharey=True)

for row, decoder in enumerate(decoders):
    nNoiseDims = decoder.split('_')[-1]
    if nNoiseDims.startswith('noiseDim'):
        try:
            nNoiseDims = f"Noise Dims: {int(nNoiseDims.split('-')[1])+1}"
        except:
            nNoiseDims = 'Noise Dims: 0'
    else:
        nNoiseDims = 'Noise Dims: 1'
    ax[row].set_ylabel(nNoiseDims)
    for col, rank in enumerate(ranks):
        col = col * 3
        i = ind.replace('ss', f'ss{rank}')
        r = rlv.replace('ss', f'ss{rank}')
        p = plv.replace('ss', f'ss{rank}')

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

        for batch, site in zip(batches, sites):

            if site in ['BOL005c', 'BOL006b']:
                _batch = 294
            else:
                _batch = batch
            
            if batch in [289, 294]:
                _r = r.replace('.cpn', '')
                _i = i.replace('.cpn', '')
                _p = p.replace('.cpn', '')
            else:
                _r = r
                _i = i
                _p = p

            loader = decoding.DecodingResults()
            fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+'_TDR.pickle')
            raw = loader.load_results(fn, cache_path=None, recache=recache)
            fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+f'_model-LV-{_r}_TDR.pickle')
            lv0 = loader.load_results(fn, cache_path=None, recache=recache)
            fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+f'_model-LV-{_i}_TDR.pickle')
            indep = loader.load_results(fn, cache_path=None, recache=recache)
            fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+f'_model-LV-{_p}_TDR.pickle')
            lv = loader.load_results(fn, cache_path=None, recache=recache)

            # get the epochs of interest (fit epochs)
            mask_bins = lv.meta['mask_bins']
            fit_combos = [k for k, v in lv.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                                (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
            all_combos = lv.evoked_stimulus_pairs
            val_combos = [c for c in all_combos if c not in fit_combos]

            if sig_delta:
                # filter for stim pairs with sig change in delta dprime
                mask = np.abs(raw.numeric_results['bp_dp'] - raw.numeric_results['sp_dp']) > ((raw.numeric_results['bp_dp_sem'] * np.sqrt(10)) + (raw.numeric_results['sp_dp_sem'] * np.sqrt(10)))
                sig_combos = raw.numeric_results[mask].index.get_level_values(0)
                val_combos = [v for v in val_combos if v in sig_combos]
                fit_combos = [f for f in fit_combos if f in sig_combos]

            # save results for each model and for divide by fit / not fit stimuli
            
            # fit stims first
            for k, res in zip(['pup_indep', 'indep_noise', 'lv', 'raw'], [lv0, indep, lv, raw]):
                df = res.numeric_results
                df['delta_dprime'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp']) #(raw.numeric_results['bp_dp'] + raw.numeric_results['sp_dp'])
                df['site'] = site
                # filter based on noise alignment
                try:
                    if aligned:
                        df = df[raw.numeric_results.beta2_mag>0.7]
                except:
                    # beta2 alignement is not calculated for dU decoding
                    pass
                results['fit'][k].append(df.loc[fit_combos])
                results['val'][k].append(df.loc[val_combos])

        # concatenate data frames
        for k in results['fit'].keys():
            results['fit'][k] = pd.concat(results['fit'][k])
            results['val'][k] = pd.concat(results['val'][k])

        # plot results for these 3 models 
        if bar:
            ax[row].bar(x=[col-0.5], 
                        height=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['pup_indep']['delta_dprime']).mean()],
                        yerr=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['pup_indep']['delta_dprime']).sem()],
                        width=0.5, edgecolor='k', lw=1, color='tab:blue', label='Shuff. Pupil')
            ax[row].bar(x=[col], 
                        height=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['indep_noise']['delta_dprime']).mean()],
                        yerr=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['indep_noise']['delta_dprime']).sem()],
                        width=0.5, edgecolor='k', lw=1, color='tab:orange', label='Indep. Noise')
            ax[row].bar(x=[col+0.5], 
                        height=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['lv']['delta_dprime']).mean()],
                        yerr=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['lv']['delta_dprime']).sem()],
                        width=0.5, edgecolor='k', lw=1, color='tab:green', label='LV Model')
        else:
            ax[row].errorbar(x=[col-0.5], 
                        y=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['pup_indep']['delta_dprime']).mean()],
                        yerr=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['pup_indep']['delta_dprime']).sem()],
                        marker='o', capsize=2, lw=1, color='tab:blue', label='Shuff. Pupil')
            ax[row].errorbar(x=[col], 
                        y=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['indep_noise']['delta_dprime']).mean()],
                        yerr=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['indep_noise']['delta_dprime']).sem()],
                        marker='o', capsize=2, lw=1, color='tab:orange', label='Indep. Noise')
            ax[row].errorbar(x=[col+0.5], 
                        y=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['lv']['delta_dprime']).mean()],
                        yerr=[np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['lv']['delta_dprime']).sem()],
                        marker='o', capsize=2, lw=1, color='tab:green', label='LV Model')    

            
            # add stats

            rlv_err = np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['pup_indep']['delta_dprime'])
            ind_err = np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['indep_noise']['delta_dprime'])
            lv_err = np.abs(results[fit_val]['raw']['delta_dprime']-results[fit_val]['lv']['delta_dprime'])
            _sites = results[fit_val]['raw']['site']
            
            d = {s: rlv_err[_sites==s].values - ind_err[_sites==s].values for s in _sites.unique()} 
            bootsample = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000) 
            p = get_direct_prob(bootsample, np.zeros(len(bootsample)))[0]
            ax[row].text(col-0.5, ax[row].get_ylim()[1], f'p={round(p, 3)}', fontsize=6)  

            d = {s: ind_err[_sites==s].values - lv_err[_sites==s].values for s in _sites.unique()} 
            bootsample = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000) 
            p = get_direct_prob(bootsample, np.zeros(len(bootsample)))[0]
            ax[row].text(col+0.25, ax[row].get_ylim()[1], f'p={round(p, 3)}', fontsize=6)   

        ax[row].set_xticks([])
        if (row==0) & (col==0):
            ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
for a in ax:
    a.grid(axis='y')
ax[-1].set_xticks([0, 3, 6])
ax[-1].set_xticklabels(['ss1', 'ss2', 'ss3'])
ax[-1].set_xlabel('Covariance rank')
ax[0].set_title(r"$\Delta d'^2$ Error for all %s stim" % fit_val)

f.tight_layout()

plt.show()