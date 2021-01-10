from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
import colors as color
import ax_labels as alab

import charlieTools.nat_sounds_ms.decoding as decoding

from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# subsample results
nSamples = 500
np.random.seed(123)

density = True  # density scatter plot
ALL_TRAIN_DATA = False
sites = HIGHR_SITES
path = DPRIME_DIR

df = []
for site in sites:
    if (site in LOWR_SITES) | (ALL_TRAIN_DATA): mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]
    _df['site'] = site
    df.append(_df)

df = pd.concat(df)


idx = np.random.choice(range(df.shape[0]), nSamples, replace=False)
bp = df.iloc[idx]['bp_dp']
sp = df.iloc[idx]['sp_dp']

s = 15
m = pd.concat([bp, sp]).max()
f, ax = plt.subplots(1, 1, figsize=(4, 4))

if density:
    xy = np.vstack([bp.values, sp.values])
    z = gaussian_kde(xy)(xy)
    ax.scatter(sp, bp, s=s, c=z)
else:
    ax.scatter(sp, bp, s=s, color='k', edgecolor='white')
ax.plot([0, m], [0, m], color='grey', linestyle='--')

ax.set_xlabel('Small Pupil')
ax.set_ylabel('Large Pupil')
ax.set_title(r"$d'^2$")

f.tight_layout()

plt.show()