"""
Compare raw delta dprime with delta dprime after shuffling trials within pupil state.
Idea is that delta dprime magnitude should be smaller (one hypothesis) after shuffling because 
you no longer have correlation changes between states.
"""

import charlieTools.nat_sounds_ms.decoding as decoding
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT, CPN_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


loader = decoding.DecodingResults()
modelname = 'dprime_shuffle_jk1_eev_zscore_fixtdr2-fa_noiseDim-5'
n_components = 2

# list of sites with > 10 reps of each stimulus
sites = CPN_SITES
batch = 331

df = []
dfs = []
for site in sites:
    fn = os.path.join(DPRIME_DIR, str(batch), site, modelname.replace('_shuffle', '')+'_TDR.pickle')
    results = loader.load_results(fn)
    fn = os.path.join(DPRIME_DIR, str(batch), site, modelname+'_TDR.pickle')
    shuf_results = loader.load_results(fn)

    _df = results.numeric_results; _df['delta_dprime'] = (_df['bp_dp'] - _df['sp_dp']) / (_df['bp_dp'] + _df['sp_dp']); _df['site'] = site
    _dfs = shuf_results.numeric_results; _dfs['delta_dprime'] = (_dfs['bp_dp'] - _dfs['sp_dp']) / (_dfs['bp_dp'] + _dfs['sp_dp']); _dfs['site'] = site

    df.append(_df)
    dfs.append(_dfs)

df = pd.concat(df)
dfs = pd.concat(dfs)

f, ax = plt.subplots(1, 1, figsize=(5, 5))

#ax.scatter(df['bp_dp']-df['sp_dp'],
#            dfs['bp_dp']-dfs['sp_dp'], s=5)
sns.scatterplot(x=df['bp_dp']-df['sp_dp'],
            y=dfs['bp_dp']-dfs['sp_dp'],
            hue=df['site'], ax=ax, **{'s': 5})
ax.plot([-60, 60], [-60, 60], 'k--')
ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')
ax.set_xlabel("Actual delta-dprime")
ax.set_ylabel("Shuffled delta-dprime")

f.tight_layout()

plt.show()