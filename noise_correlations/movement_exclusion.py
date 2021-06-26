"""
Evaluate impact of different criteria for artifact removal on 
noise correlations
"""
from global_settings import CPN_SITES
import load_results as ld

import matplotlib.pyplot as plt
import seaborn as sns

params = [
    'rsc_ev_perstim',
    'rsc_ev_perstim_mvm-25-1',
    'rsc_ev_perstim_mvm-25-2',
    'rsc_ev_perstim_mvm-25-3',
    'rsc_ev_perstim_mvm-25-4',
    'rsc_ev_perstim_mvm-25-5'
]

f, ax = plt.subplots(2, 3, figsize=(9, 6))
for i, (p, a) in enumerate(zip(params, ax.flatten())):
    df = ld.load_noise_correlation(p)
    df = df[df.site.isin(CPN_SITES)]
    df = df[(df.gm_bp>1) & (df.gm_sp>1)]
    bp = df.groupby(by=['site', 'stim']).mean()['bp']
    sp = df.groupby(by=['site', 'stim']).mean()['sp']
    site = df.groupby(by=['site', 'stim']).mean().index.get_level_values(0)
    sns.scatterplot(sp, bp, hue=site, ax=a, **{'s': 25})
    a.set_ylabel(f"BIG: {round(bp.mean(), 3)}")
    a.set_xlabel(f"SMALL: {round(sp.mean(), 3)}")
    a.plot([-0.05, 0.2], [-0.05, 0.2], 'k--')
    a.axhline(0, linestyle='--', color='k')
    a.axvline(0, linestyle='--', color='k')
    a.set_title(p)

f.tight_layout()

plt.show()