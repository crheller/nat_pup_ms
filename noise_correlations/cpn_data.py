"""
Noticed that for the cpn data, there are some sites that seem to be "backwards" -- bigger noise correlations in big pupil vs. small pupil.

Investigate the cause here. Hypothesis is that this is due to very low responses during small pupil leading to small noise correlations.
If that's the case, show that under "normal" conditions, noise correlation does in fact decrease with pupil, as expected.
"""
from global_settings import CPN_SITES, HIGHR_SITES
import colors
import load_results as ld

import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np

# load perstim noise correlations and keep only batch 331
#df = ld.load_noise_correlation('rsc_ev_pr_rm2_perstim')
df = ld.load_noise_correlation('rsc_ev_perstim')
df = df[df.site.isin(CPN_SITES+HIGHR_SITES)]
celltypes = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/celltypes.csv', index_col=0)
celltypes.index = celltypes.cellid


# remove cases where noise corr is nan in big/small
na = (~df.bp.isna()) & (~df.sp.isna())
df = df[na]
df['delta_gm'] = df.gm_bp-df.gm_sp
df['delta_rsc'] = df.sp - df.bp

for k in ['gm_all', 'gm_bp', 'gm_sp']:
    m = df[k].isna()
    df[k][m] = 0

'''
df['type'] = np.nan
for pair in df.index:
    try:
        if (celltypes.loc[pair.split('_')[0], 'type']==1) & (celltypes.loc[pair.split('_')[1], 'type']==1):
            df.loc[pair, 'type'] = 'RS'
        elif (celltypes.loc[pair.split('_')[0], 'type']==0) & (celltypes.loc[pair.split('_')[1], 'type']==0):
            df.loc[pair, 'type'] = 'FS'
        else:
            df.loc[pair, 'type'] = 'FS_RS'
    except:
        df.loc[pair, 'type'] = 'MUA'
'''
# plot big / small pupil noise correlation as function of bp/sp/delta geo mean
df = df[(df.gm_bp<3) & (df.gm_sp<3)]
f, ax = plt.subplots(1, 2, figsize=(6, 3))

df.plot.hexbin(
    x='gm_bp',
    y='gm_sp',
    C=None, 
    cmap='Reds',
    scale='log',
    vmin=0,
    vmax=None,
    gridsize=20,
    ax=ax[0]
)
ax[0].set_title(r"$\Delta$ Noise Corr.")
ax[0].set_xlabel(r"Large Pupil Geo. Mean")
ax[0].set_ylabel(r"Small Pupil Geo. Mean")
mi, ma = (np.min(ax[0].get_ylim()+ax[0].get_xlim()), np.max(ax[0].get_ylim()+ax[0].get_xlim()))
ax[0].plot([mi, ma], [mi, ma], 'k--')

# heatmap of delta noise corr as function of delta geo mean and small pupil geo mean
df.plot.hexbin(
    x='gm_bp',
    y='gm_sp',
    C='delta_rsc', 
    cmap='bwr',
    vmin=-0.3,
    vmax=0.3,
    gridsize=20,
    ax=ax[1]
)
ax[1].set_title(r"$\Delta$ Noise Corr.")
ax[1].set_xlabel(r"Large Pupil Geo. Mean")
ax[1].set_ylabel(r"Small Pupil Geo. Mean")
ax[1].plot([mi, ma], [mi, ma], 'k--')

f.tight_layout()

# plot above for each site alone
for site in df.site.unique():
    _df = df[df.site==site]
    #_df = _df[(_df.delta_gm < 1) & (_df.delta_gm > -0.5) & (_df.gm_sp<5)]
    f, ax = plt.subplots(1, 2, figsize=(6, 3))

    _df.plot.hexbin(
    x='gm_bp',
    y='gm_sp',
    C=None, 
    cmap='Reds',
    vmin=0,
    vmax=None,
    gridsize=20,
    ax=ax[0]
    )
    ax[0].set_title(r"$\Delta$ Noise Corr.")
    ax[0].set_xlabel(r"Large Pupil Geo. Mean")
    ax[0].set_ylabel(r"Small Pupil Geo. Mean")
    mi, ma = (np.min(ax[0].get_ylim()+ax[0].get_xlim()), np.max(ax[0].get_ylim()+ax[0].get_xlim()))
    ax[0].plot([mi, ma], [mi, ma], 'k--')

    _df.plot.hexbin(
    x='gm_bp',
    y='gm_sp',
    C='delta_rsc', 
    cmap='bwr',
    vmin=-0.3,
    vmax=0.3,
    gridsize=20,
    ax=ax[1]
    )
    ax[1].set_title(r"$\Delta$ Noise Corr.")
    ax[1].set_xlabel(r"Large Pupil Geo. Mean")
    ax[1].set_ylabel(r"Small Pupil Geo. Mean")
    ax[1].plot([mi, ma], [mi, ma], 'k--')

    f.tight_layout()
    f.canvas.set_window_title(site)

plt.show()

# plot delta noise corr. as a function of site / number neuron pairs
f, ax = plt.subplots(1, 1, figsize=(4, 4))
g = df.groupby(by=['site', 'stim']).mean()
sem = df.groupby(by=['site', 'stim']).sem()
count = df.groupby(by=['site', 'stim']).count()['all'].values
ax.errorbar(count, g['delta_rsc']-sem['delta_rsc'], g['delta_rsc']+sem['delta_rsc'], marker='.', lw=0)
ax.axhline(0, linestyle='--', color='k')
plt.show()