"""
compare true decoding results to model results
"""
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION

import charlieTools.nat_sounds_ms.decoding as decoding

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

site = 'TAR010c'
lvstr = ['indep', 'dc11', 'dc10', 'dc00', 'gn11', 'gn10', 'gn00']
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
recache = False

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

plt.show()