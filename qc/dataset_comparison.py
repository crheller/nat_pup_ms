"""
Compare results between different "datasets" -- NAT vs. small CPN vs. big CPN

Quesitons:
    pupil-dependent...
        decoding the same?
        noise correlations the same?
        first order changes?
    overall pupil variance?
    number of cells?
    mean firing rates?
    MUA / instability?
"""
from global_settings import CPN_SITES, HIGHR_SITES
import charlieTools.nat_sounds_ms.decoding as decoding
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

nat_data = HIGHR_SITES
cpns_data = [s for s in CPN_SITES if 'TNC' not in s]
cpnb_data = [s for s in CPN_SITES if 'TNC' in s]

#################################### PUPIL VARIANCE / RAW DATA ##################################
rdata = {}
batches = [289]*len(nat_data) + [331]*len(CPN_SITES)
for site, batch in zip(nat_data+CPN_SITES, batches):
    if site in ['BOL005c', 'BOL006b']:
        b = 294
    else:
        b = batch
    X, sp_bins, X_pup, pup_mask = decoding.load_site(site=site, batch=b)
    rdata[site] = (X, sp_bins, X_pup, pup_mask)


######################################## DELTA DPRIME ###########################################
f, ax = plt.subplots(1, 3, figsize=(8, 4))

loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_fixtdr2-fa'
n_components = 2

for i, (dataset, batch, sites) in enumerate(zip(['NAT', 'CPN-small', 'CPN-big'], [289, 331, 331], [nat_data, cpns_data, cpnb_data])):
    for j, site in enumerate(sites):
        if site in ['BOL005c', 'BOL006b']:
            _batch = 294
        else:
            _batch = batch
        fn = os.path.join(DPRIME_DIR, str(_batch), site, modelname+'_TDR.pickle')
        results = loader.load_results(fn)
        df = results.numeric_results; df['delta_dprime'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp']); df['site'] = site; df['batch'] = batch
        ax[i].scatter(df['delta_dprime'].mean(), j, marker='o', facecolor='white', edgecolor='k')
        ax[i].plot([np.quantile(df['delta_dprime'], 0.025), np.quantile(df['delta_dprime'], 0.975)],
                    [j, j], color='k', zorder=-1)
    ax[i].axvline(0, linestyle='--', color='grey', zorder=-1)
    ax[i].set_xlim((-1, 1))
    ax[i].set_yticks(np.arange(len(sites)))
    ax[i].set_yticklabels(sites, fontsize=6)
    ax[i].set_title(dataset)

f.tight_layout()
plt.show()
#################################################################################################