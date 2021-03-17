"""
compare true decoding results to model results
"""
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION

import charlieTools.nat_sounds_ms.decoding as decoding

import seaborn as sns
from scipy.stats import gaussian_kde
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

figpath = '/auto/users/hellerc/results/nat_pupil_ms/tmp_figures/lv_models/'
ext = '.e.sp' # '', '.e' '.e.sp' (different cost functions)
for site in HIGHR_SITES:
    lvstr = ['indep', 'dc11', 'dc10', 'dc00', 'gn11', 'gn10', 'gn00']
    lvstr = [lv+ext for lv in lvstr]
    modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
    recache = True

    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    lv_results = {}
    for k in lvstr:
        fn = os.path.join(DPRIME_DIR, site, modelname+f'_model-LV-{k}_TDR.pickle')
        lv_results[k] = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)

    # plot crude comparison -- big / small pupil mean /sem for each model
    f, ax = plt.subplots(1, 2, figsize=(8, 3))

    for i, m in enumerate(['raw']+lvstr):
        
        if m=='raw':
            df = results.numeric_results.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]
        else:
            df = lv_results[m].numeric_results.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]

        if i == 0:
            lab = ('large pupil', 'small pupil')
        else:
            lab = (None, None)
        ax[0].errorbar(i, df['bp_dp'].mean(), yerr=df['bp_dp'].sem(), color='red', capsize=2, label=lab[0])
        ax[0].errorbar(i, df['sp_dp'].mean(), yerr=df['sp_dp'].sem(), color='blue', capsize=2, label=lab[1])

        delta = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
        ax[1].errorbar(i, delta.mean(), yerr=delta.sem(), capsize=2, marker='o', color='k')

    ax[0].set_xticks(np.arange(len(lvstr)+1))
    ax[0].set_xticklabels(['raw']+lvstr)
    ax[0].set_xlabel("Model")
    ax[0].set_ylabel(r"$d'^2$")

    ax[1].set_xticks(np.arange(len(lvstr)+1))
    ax[1].set_xticklabels(['raw']+lvstr)
    ax[1].set_xlabel("Model")
    ax[1].set_ylabel(r"$\Delta d'^2$")

    f.tight_layout()

    f.savefig(figpath+f'lvplot1_{site}_{ext}.png')

    # for each model, scatter plot of raw vs. lv model results
    f, ax = plt.subplots(1, len(lv_results.keys()), figsize=(14, 3), sharex=True, sharey=True)
    df = results.numeric_results.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]
    delta = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
    for i, k in enumerate(lv_results.keys()):
        df2 = lv_results[k].numeric_results.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]
        delta2 = (df2['bp_dp'] - df2['sp_dp']) / (df2['bp_dp'] + df2['sp_dp'])
        xy = np.vstack([delta, delta2])
        z = gaussian_kde(xy)(xy)
        #ax[i].scatter(delta, delta2, c=z, s=5)
        #ax[i].scatter(delta, delta2, s=2)
        sns.regplot(delta, delta2, ax=ax[i], scatter_kws = {'s':2, 'alpha': 0.3})
        ax[i].axhline(0, linestyle='--', color='k')
        ax[i].axvline(0, linestyle='--', color='k')
        ax[i].set_xlabel('raw')
        ax[i].set_ylabel(k)
        ax[i].set_title(r"$\Delta d'^2$")

    ll = np.min(ax[i].get_ylim() + ax[i].get_xlim())
    hl = np.max(ax[i].get_ylim() + ax[i].get_xlim())
    for a in ax:
        a.plot([ll, hl], [ll, hl], 'k--')

    f.tight_layout()

    f.savefig(figpath+f'lvplot2_{site}_{ext}.png')

plt.show()