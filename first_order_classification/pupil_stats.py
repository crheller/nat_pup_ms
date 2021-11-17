"""
quantify / cache a summary of pupil variability / mean for each site
"""
import charlieTools.nat_sounds_ms.decoding as decoding
from global_settings import CPN_SITES, HIGHR_SITES

import numpy as np
import pandas as pd

saveto = "/auto/users/hellerc/code/projects/nat_pupil_ms/first_order_classification/pupil_stats.pickle"

sites = CPN_SITES + HIGHR_SITES
batches = [331] * len(CPN_SITES) + [322] * len(HIGHR_SITES)

df = pd.DataFrame(index=np.arange(len(sites)), columns=["site", "batch", "p_std", "p_median", "p_mean", "p_norm_std", "p_norm_median", "p_norm_mean"])

for i, (site, batch) in enumerate(zip(sites, batches)):
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    X, sp_bins, X_pup, pup_mask = decoding.load_site(site=site, batch=batch)
    p = X_pup.flatten()
    max_pupil = decoding.get_max_pupil(site[:7], rasterfs=4)
    df.loc[i, :] = [
        site,
        batch,
        p.std(),
        np.median(p),
        p.mean(),
        (p / max_pupil).std(),
        np.median(p / max_pupil),
        (p / max_pupil).mean()
    ]

df.to_pickle(saveto)