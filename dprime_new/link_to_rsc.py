"""
Link decoding results / dDR space statistics to noise correlation changes per stimulus.

Idea is to show that there are cases where *mean* noise correlations decrease, but coding does not improve
and/or changes in noise corr. do/don't predict dDR noise statistics.
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, CPN_SITES
import charlieTools.nat_sounds_ms.decoding as decoding
import load_results as ld

import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

path = DPRIME_DIR
loader = decoding.DecodingResults()
recache = False
df_all = []
sites = CPN_SITES
batches = [331]*len(CPN_SITES)
#sites = HIGHR_SITES
#batches = [289] * len(HIGHR_SITES)
modelname = 'dprime_pr_rm2_jk10_zscore_nclvz_fixtdr2'
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
ndim = 2
modelname = modelname.replace('loadpred', 'loadpred.cpn')

rsc = ld.load_noise_correlation('rsc_ev_perstim')
rsc = rsc[rsc.site.isin(sites)]
na = (~rsc.bp.isna()) & (~rsc.sp.isna())
rsc = rsc[na]
rsc = rsc[(rsc.gm_bp>1) & (rsc.gm_sp>1)]
rsc['delta_gm'] = rsc.gm_bp - rsc.gm_sp
rsc['delta_rsc'] = rsc.sp - rsc.bp

sim = []
f, ax = plt.subplots(len(sites), 1, figsize=(6, 8))
for i, (batch, site) in enumerate(zip(batches, sites)):
    if (site in LOWR_SITES) & (batch != 331):
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    try:
        fn = os.path.join(path, str(batch), site, mn+'_TDR.pickle')
        results = loader.load_results(fn, cache_path=None, recache=recache)
        _df = results.numeric_results
    except:
        raise ValueError(f"WARNING!! NOT LOADING SITE {site}")

    stim = results.evoked_stimulus_pairs
    _df = _df.loc[pd.IndexSlice[stim, ndim], :]
    _df['site'] = site
    _df['sp_noise_mag'] = results.array_results['sp_evals'].loc[pd.IndexSlice[stim, ndim], 'mean'].apply(lambda x: x.sum())
    _df['bp_noise_mag'] = results.array_results['bp_evals'].loc[pd.IndexSlice[stim, ndim], 'mean'].apply(lambda x: x.sum())
    _df['noise_alignment'] = results.slice_array_results('cos_dU_evec_test', stim, ndim, idx=(0,0))[0]
    _df['delta_dprime'] = (_df['bp_dp'] - _df['sp_dp']) / (_df['bp_dp'] + _df['sp_dp'])

    # add epoch names back to dataframe
    _df['epoch1'] = [results.mapping[k][0] for k in _df.index.get_level_values(0)]
    _df['epoch2'] = [results.mapping[k][1] for k in _df.index.get_level_values(0)]

    # add mean big/small pupil noise correlation for each epoch
    _df['rsc_epoch1_big'] = [rsc[(rsc.site==site) & (rsc.stim==e1)].bp.mean() for e1 in _df['epoch1']]
    _df['rsc_epoch1_small'] = [rsc[(rsc.site==site) & (rsc.stim==e1)].sp.mean() for e1 in _df['epoch1']]
    _df['rsc_epoch2_big'] = [rsc[(rsc.site==site) & (rsc.stim==e2)].bp.mean() for e2 in _df['epoch2']]
    _df['rsc_epoch2_small'] = [rsc[(rsc.site==site) & (rsc.stim==e2)].sp.mean() for e2 in _df['epoch2']]
    _df['rsc_epoch1_delta'] = _df['rsc_epoch1_small'] - _df['rsc_epoch1_big']
    _df['rsc_epoch2_delta'] = _df['rsc_epoch2_small'] - _df['rsc_epoch2_big']
    _df['rsc_mean_delta'] = (_df['rsc_epoch1_delta'] + _df['rsc_epoch2_delta']) / 2
    df_all.append(_df)
    
    du=np.stack(results.array_results['dU_all_test'].loc[pd.IndexSlice[stim, ndim], 'mean'].apply(lambda x: x/np.linalg.norm(x)).values).squeeze()
    print(du.shape)
    _sim = np.abs(du.dot(du.T)[np.tril_indices(du.shape[0], -1)])
    ax[i].hist(_sim, bins=np.arange(-0.2, 1, 0.05), histtype='step', density=False, label=site+f" std: {round(np.std(_sim), 3)}", lw=1)
    ax[i].legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    
    #plt.imshow(du.dot(du.T), vmin=-1, vmax=1, cmap='bwr')
    #plt.colorbar()
    #plt.title(site)
    #plt.show()

f.tight_layout()
plt.show()

df = pd.concat(df_all)
df['delta_noise'] = (df['sp_noise_mag'] - df['bp_noise_mag']) / (df['sp_noise_mag'] + df['bp_noise_mag'])

# dDR noise change vs. mean noise correlation change
# only for stim pairs where both stim's noise corr. change in same direction
f, ax = plt.subplots(1, 2, figsize=(8, 4))
mask = ((df['rsc_epoch1_delta']<0) & (df['rsc_epoch2_delta']<0)) | ((df['rsc_epoch1_delta']>0) & (df['rsc_epoch2_delta']>0))
sns.scatterplot(x='rsc_mean_delta', y='delta_noise', data=df[mask], hue='site', ax=ax[0])

ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0, linestyle='--', color='k')

ax[0].set_xlabel('Mean noise corr. change')
ax[0].set_ylabel('dDR noise change')

df[mask].plot.hexbin(
    x='rsc_mean_delta',
    y='delta_noise',
    C='delta_dprime',
    cmap='bwr',
    vmin=-0.5,
    vmax=0.5,
    ax=ax[1]
)
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0, linestyle='--', color='k')
ax[1].set_xlabel('Mean noise corr. change')
ax[1].set_ylabel('dDR noise change')
ax[1].set_title(r"$\Delta d'^2$")
f.tight_layout()

plt.show()