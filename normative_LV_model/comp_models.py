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

plot_example = True
decoder = 'dprime_jk10_zscore_nclvz_fixtdr2-fa'
rlv = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss3"
ind = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss3"
plv = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss3"

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
for i, (k, tit) in enumerate(zip(['pup_indep', 'indep_noise', 'lv'], ['Pup. Shuff.', 'Private Noise', 'Shared Noise'])):
    ax[i].set_title(tit)
    cc = np.round(np.corrcoef(results['fit']['raw']['delta_dprime'], results['fit'][k]['delta_dprime'])[0, 1], 3)
    ax[i].scatter(results['fit']['raw']['delta_dprime'],
                    results['fit'][k]['delta_dprime'], s=25, edgecolor='white', color='tab:orange', label=f"fit: {cc}")
    cc = np.round(np.corrcoef(results['val']['raw']['delta_dprime'], results['val'][k]['delta_dprime'])[0, 1], 3)
    ax[i].scatter(results['val']['raw']['delta_dprime'],
                    results['val'][k]['delta_dprime'], s=25, edgecolor='white', color='tab:blue', label=f"val: {cc}")
# add unity line / best fit lines
mm = np.min(ax[i].get_xlim()+ax[i].get_ylim())
ma = np.max(ax[i].get_xlim()+ax[i].get_ylim())
for i, k in enumerate(['pup_indep', 'indep_noise', 'lv']):
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

# plot NMSE
f, ax = plt.subplots(1, 2, figsize=(8, 4))

er1 = (np.abs(cat_fit['indep_noise']-cat_fit['raw'])) / np.abs(cat_fit['raw']) #(cat_fit['indep_noise']*cat_fit['raw'])
er2 = (np.abs(cat_fit['lv']-cat_fit['raw'])) / np.abs(cat_fit['raw']) #(cat_fit['lv']*cat_fit['raw'])
stat, pval = ss.wilcoxon(er1, er2)
sns.scatterplot(x=er1, 
                y=er2, hue=results['fit']['raw'].site, ax=ax[0], **{'s': 15})
ax[0].plot([0, 25], [0, 25], 'k--')
ax[0].axhline(0, linestyle='--', color='k'); ax[0].axvline(0, linestyle='--', color='k')
ax[0].set_xlabel('Indep. Error')
ax[0].set_ylabel("LV Error")
ax[0].set_title("Fit stimuli")

er1 = (np.abs(cat_val['indep_noise']-cat_val['raw'])) / np.abs(cat_val['raw']) #(cat_val['indep_noise']*cat_val['raw'])
er2 = (np.abs(cat_val['lv']-cat_val['raw'])) / np.abs(cat_val['raw']) #(cat_val['lv']*cat_val['raw']) 
stat, pval = ss.wilcoxon(er1, er2)
sns.scatterplot(x=er1, 
                y=er2, hue=results['val']['raw'].site, ax=ax[1], **{'s': 15})
ax[1].plot([0, 25], [0, 25], 'k--')
ax[1].axhline(0, linestyle='--', color='k'); ax[1].axvline(0, linestyle='--', color='k')
ax[1].set_xlabel("Indep. Error")
ax[1].set_ylabel("LV Error")
ax[1].set_title("Validation Stimuli")

f.tight_layout()

# plot summary of error per model for validation / fit stimuli
f, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.errorbar(range(3), y = [
                        np.abs(cat_val.pup_indep-cat_val.raw).mean(),
                        np.abs(cat_val.indep_noise-cat_val.raw).mean(),
                        np.abs(cat_val.lv-cat_val.raw).mean()
                        ],
                    yerr=[
                        np.abs(cat_val.pup_indep-cat_val.raw).sem(),
                        np.abs(cat_val.indep_noise-cat_val.raw).sem(),
                        np.abs(cat_val.lv-cat_val.raw).sem()
                    ],
                    capsize=2, marker='o', color='k', label='Validation Stim.')
ax.errorbar(np.arange(3)+0.05, y = [
                        np.abs(cat_fit.pup_indep-cat_fit.raw).mean(),
                        np.abs(cat_fit.indep_noise-cat_fit.raw).mean(),
                        np.abs(cat_fit.lv-cat_fit.raw).mean()
                        ],
                    yerr=[
                        np.abs(cat_fit.pup_indep-cat_fit.raw).sem(),
                        np.abs(cat_fit.indep_noise-cat_fit.raw).sem(),
                        np.abs(cat_fit.lv-cat_fit.raw).sem()
                    ],
                    capsize=2, marker='o', color='orange', label='Fit Stim.')
ax.set_ylabel(r"$\Delta d'^2$ error")
ax.set_xlabel('Model')
ax.set_xticks(range(3))
ax.set_xticklabels(['pup indep.', 'private noise', 'shared noise'], rotation=45)
ax.legend(frameon=False)

f.tight_layout()


plt.show()

if plot_example:
    # find stimulus pairs where LV model outperforms Indep. Noise model and 
    # show an example in the decoding space (dDR)
    exdf = cat_fit.copy()
    #exdf = cat_val.copy()
    exdf['site'] = results['fit']['raw']['site']
    #exdf = exdf[np.abs(cat_val.lv-cat_val.raw)<np.abs(cat_val.indep_noise-cat_val.raw)]
    exdf['indep_err'] = np.abs(exdf['indep_noise'] - exdf['raw'])
    exdf['lv_err'] = np.abs(exdf['lv'] - exdf['raw'])
    exdf['rlv_err'] = np.abs(exdf['pup_indep'] - exdf['raw'])
    #exdf['diff'] = exdf['rlv_err']-exdf['lv_err']
    exdf['diff'] = exdf['indep_err']-exdf['lv_err']
    #pair = (5, 11)
    #site = 'ARM029a'
    pair = (0, 9)
    site = 'AMT026a'
    # pair=(2, 10), site='AMT020a' -- Great example of where LV helps and delta-dprime is positive
    #pair = (2, 7)
    #site = 'ARM032a'
    figpath = '/auto/users/hellerc/temp4/'
    exdf = exdf.sort_values(by='diff', ascending=False)
    reshuf = True
    for i in range(exdf.shape[0]):
        temp = figpath+f'_{i}.png'
        site = exdf['site'].iloc[i]
        pair = exdf.index.get_level_values(0)[i]
        pair = tuple([int(x) for x in pair.split('_')])
        f, ax = plt.subplots(1, 5, figsize=(15, 3), sharex=True)
        rand_ax = None #np.random.normal(0, 1, 19)
        plot_stimulus_pair(site, batch, pair, axlabs=[r'$\Delta \mu$', 'Noise Dim. 1'], 
                                    ellipse=True, pup_split=True, ax=ax[0],
                                    xforms_modelname=rlv,
                                    xforms_signal='pred0',
                                    reshuf=reshuf,
                                    title_string=f"First order, delta dprime: NA",
                                    lv_axis=rand_ax, s=5)
        plot_stimulus_pair(site, batch, pair, axlabs=[r'$\Delta \mu$', 'Noise Dim. 1'], 
                                    ellipse=True, pup_split=True, ax=ax[1],
                                    xforms_modelname=rlv,
                                    xforms_signal='pred',
                                    reshuf=reshuf,
                                    title_string=f"Pupil. indep. LV, delta dprime: {round(float(exdf['pup_indep'].iloc[i]), 3)}",
                                    lv_axis=rand_ax, s=5)
        plot_stimulus_pair(site, batch, pair, axlabs=[r'$\Delta \mu$', 'Noise Dim. 1'], 
                                    ellipse=True, pup_split=True, ax=ax[2],
                                    xforms_modelname=ind,
                                    reshuf=reshuf,
                                    xforms_signal='pred',
                                    title_string=f"Indp. Noise, delta dprime: {round(float(exdf['indep_noise'].iloc[i]), 3)}",
                                    lv_axis=rand_ax, s=5)
        plot_stimulus_pair(site, batch, pair, axlabs=[r'$\Delta \mu$', 'Noise Dim. 1'], 
                                    ellipse=True, pup_split=True, ax=ax[3],
                                    xforms_modelname=plv,
                                    reshuf=reshuf,
                                    xforms_signal='pred',
                                    title_string=f"LV, delta dprime: {round(float(exdf['lv'].iloc[i]), 3)}",
                                    lv_axis=rand_ax, s=5)
        plot_stimulus_pair(site, batch, pair, axlabs=[r'$\Delta \mu$', 'Noise Dim. 1'], 
                                    ellipse=True, pup_split=True, ax=ax[4],
                                    xforms_modelname=None,
                                    title_string=f"Raw data -- pair: {pair}, site: {site}, delta dprime: {round(float(exdf['raw'].iloc[i]), 3)}",
                                    lv_axis=rand_ax, s=15)
        
        # set all axes lims the same
        for a in ax:
            a.axis('equal')

        f.tight_layout()

        f.savefig(temp)

        plt.close('all')

