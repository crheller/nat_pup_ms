"""
Show model performance (in terms of delta dprime estimate)
for ss1 vs. ss2 vs. ss3 models. Justification for choosing low-rank
approximation / control that says single LV is not consquence of low-rank cc.
"""
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT, CPN_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR3, CACHE_PATH, REGRESSION
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
fit_val = 'fit'

savefig = True
fig_fn = PY_FIGURES_DIR3 + 'S5_lvModelSelection.svg'
np.random.seed(123)

s = 5
edgecolor='none'
cmap = {
    'pup_indep': 'tab:blue',
    'indep_noise': 'tab:orange',
    'lv': 'tab:green',
    'lv2': 'tab:purple'
}

decoder = 'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-6'
# then load each of these for each cov. rank
ranks = [1]
ind = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss"
rlv = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss"
plv = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss"
plv2 = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss"
ranks = [1, 2, 3]

f, ax = plt.subplots(1, 1, figsize=(6, 2))


for col, rank in enumerate(ranks):
    col = col * 2
    i = ind.replace('ss', f'ss{rank}')
    r = rlv.replace('ss', f'ss{rank}')
    p = plv.replace('ss', f'ss{rank}')
    p2 = plv2.replace('ss', f'ss{rank}')

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

    for batch, site in zip(batches, sites):

        if site in ['BOL005c', 'BOL006b']:
            _batch = 294
        else:
            _batch = batch
        
        if batch in [289, 294]:
            _r = r.replace('.cpn', '')
            _i = i.replace('.cpn', '')
            _p = p.replace('.cpn', '')
            _p2 = p2.replace('.cpn', '')
        else:
            _r = r
            _i = i
            _p = p
            _p2 = p2

        loader = decoding.DecodingResults()
        fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+'_TDR.pickle')
        raw = loader.load_results(fn, cache_path=None, recache=recache)
        fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+f'_model-LV-{_r}_TDR.pickle')
        lv0 = loader.load_results(fn, cache_path=None, recache=recache)
        fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+f'_model-LV-{_i}_TDR.pickle')
        indep = loader.load_results(fn, cache_path=None, recache=recache)
        fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+f'_model-LV-{_p}_TDR.pickle')
        lv = loader.load_results(fn, cache_path=None, recache=recache)
        fn = os.path.join(DPRIME_DIR, str(_batch), site, decoder+f'_model-LV-{_p2}_TDR.pickle')
        lv2 = loader.load_results(fn, cache_path=None, recache=recache)

        # get the epochs of interest (fit epochs)
        mask_bins = lv.meta['mask_bins']
        fit_combos = [k for k, v in lv.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                            (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
        all_combos = lv.evoked_stimulus_pairs
        val_combos = [c for c in all_combos if c not in fit_combos]

        # save results for each model and for divide by fit / not fit stimuli
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

    # plot results for these 3 models using 95% CI
    raw = results['fit']['raw']
    err = {}
    for i, (k, x) in enumerate(zip(results['fit'].keys(), [col-0.3, col-0.15, col+0.15, col+0.3])):
        if k != 'raw':
            pred = results['fit'][k]
            # get bootstrapped confidence interval
            d = {s: np.abs(raw[raw.site==s]['delta_dprime']-pred[pred.site==s]['delta_dprime']).values for s in raw.site.unique()}
            err[k] = d
            bootsamp = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000)
            low = np.quantile(bootsamp, .025)
            high = np.quantile(bootsamp, .975)
            ax.scatter(x, np.mean(bootsamp), color='white', edgecolor=cmap[k], s=25)
            ax.plot([x, x], [low, high], zorder=-1, color=cmap[k])

        
    # add stats
    for i, (pair, loc) in enumerate(zip([['pup_indep', 'indep_noise'], ['indep_noise', 'lv'], ['lv', 'lv2']], [col-0.3, col, col+0.3])):
        d = {s: err[pair[0]][s]-err[pair[1]][s] for s in err[pair[1]].keys()}
        bootsamp = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000)
        p = get_direct_prob(bootsamp, np.zeros(len(bootsamp)))[0]
        ax.text(loc, ax.get_ylim()[-1], r"p=$%s$"%round(p, 3), fontsize=5)

#ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
ax.grid(axis='y')
ax.set_xticks([0, 2, 4])
ax.set_xticklabels(['ss1', 'ss2', 'ss3'])
ax.set_xlabel('Covariance rank')
ax.set_title(r"$\Delta d'^2$ Error for all %s stim" % fit_val)

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()