'''
Discriminability (and changes) are not predicted by response magnitude

dDR schematic (to define population factors)
Overall dprime heatmap / delta dprime heatmap
Regression results - both overall / delta

Layout:
    2 rows
        top row: schematic, dprime, delta
        bottom row: regression results / coefficients
'''

import colors as color
import ax_labels as alab
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH, REGRESSION

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.plotting as cplt
from charlieTools.statistics import get_direct_prob
import pickle
import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True
fig_fn = PY_FIGURES_DIR2+'fig4.svg'

recache = False # recache dprime results locally
ALL_TRAIN_DATA = False  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = HIGHR_SITES
path = DPRIME_DIR
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
val = 'dp_opt_test'
estval = '_test'
nbins = 12
cmap = 'coolwarm'
vmax = 50
cmap_delta = 'coolwarm'

# only crop the dprime value. Show count for everything
if estval == '_train':
    x_cut = (3, 8.5)
    y_cut = (0.1, .45) 
elif estval == '_test':
    x_cut = DU_MAG_CUT
    y_cut = NOISE_INTERFERENCE_CUT

df = []
for site in sites:
    if (site in LOWR_SITES) | (ALL_TRAIN_DATA):
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results

    stim = results.evoked_stimulus_pairs
    _df = _df.loc[pd.IndexSlice[stim, 2], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=(0,0))[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=(0,0))[0]
    _df['site'] = site
    df.append(_df)

df = pd.concat(df)
df['delta'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])

# filter based on x_cut / y_cut
if (x_cut is not None) & (y_cut is not None):
    mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
    mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
else:
    mask1 = (True * np.ones(df.shape[0])).astype(bool)
    mask2 = (True * np.ones(df.shape[0])).astype(bool)
df_dp = df[mask1 & mask2]

# generate cartoon data for schematic
np.random.seed(123)
u1 = [-1, .1]
u2 = [1, -.2]
cov = np.array([[1, 0.5], [0.5, 1]])
A = np.random.multivariate_normal(u1, cov, (200,))
B = np.random.multivariate_normal(u2, cov, (200,))
Ael = cplt.compute_ellipse(A[:, 0], A[:, 1])
Bel = cplt.compute_ellipse(B[:, 0], B[:, 1])

# Load regression results
r2_all = pickle.load(open(REGRESSION+'dpall_r2.pickle', 'rb'))
r2_delta = pickle.load(open(REGRESSION+'delta_r2.pickle', 'rb'))
coef_all = pickle.load(open(REGRESSION+'dpall_coef.pickle', 'rb'))
coef_delta = pickle.load(open(REGRESSION+'delta_coef.pickle', 'rb'))

# ================================ MAKE FIGURE ==================================
f = plt.figure(figsize=(7.1, 6))

gs = mpl.gridspec.GridSpec(3, 6)
sch = f.add_subplot(gs[0, 0:2])
dpall = f.add_subplot(gs[0, 2:4])
delta = f.add_subplot(gs[0, 4:])
reg1 = f.add_subplot(gs[1, :3])
reg2 = f.add_subplot(gs[2, :3])
coef1 = f.add_subplot(gs[1, 3:])
coef2 = f.add_subplot(gs[2, 3:])
# plot heatmaps
df_dp.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C=val, 
                  gridsize=nbins, ax=dpall, cmap=cmap, vmax=vmax, norm=cplt.MidpointNormalize(midpoint=0)) 
dpall.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
dpall.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
dpall.spines['bottom'].set_color(color.SIGNAL)
dpall.xaxis.label.set_color(color.SIGNAL)
dpall.tick_params(axis='x', colors=color.SIGNAL)
dpall.spines['left'].set_color(color.COSTHETA)
dpall.yaxis.label.set_color(color.COSTHETA)
dpall.tick_params(axis='y', colors=color.COSTHETA)
dpall.set_title(r"$d'^2$")

df_dp.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C='delta', 
                  gridsize=nbins, ax=delta, cmap=cmap_delta, norm=cplt.MidpointNormalize(midpoint=0))
delta.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
delta.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
delta.spines['bottom'].set_color(color.SIGNAL)
delta.xaxis.label.set_color(color.SIGNAL)
delta.tick_params(axis='x', colors=color.SIGNAL)
delta.spines['left'].set_color(color.COSTHETA)
delta.yaxis.label.set_color(color.COSTHETA)
delta.tick_params(axis='y', colors=color.COSTHETA)
delta.set_title(r"$\Delta d'^2$")

# plot schematic
sch.plot(Ael[0], Ael[1], color='tab:blue', lw=2)
sch.plot(Bel[0], Bel[1], color='tab:orange', lw=2)
sch.axvline(0, linestyle='--', color='lightgrey', zorder=-1)
sch.axhline(0, linestyle='--', color='lightgrey', zorder=-1)
sch.set_xlabel(r"$dDR_1 (\Delta \mathbf{\mu})$")
sch.set_ylabel(r"$dDR_2$")

# plot regression
palette = {'full': 'lightgrey', 'upc1_mean': 'r', 'upc1_diff': 'r', 'udU_mag_test': color.SIGNAL, 'unoiseAlign': color.COSTHETA}
r2a = r2_all[[k for k in r2_all if (k=='full') | (k.startswith('u'))]]
sns.boxplot(x='variable', y='value', data=r2a.melt(), 
                        palette=palette, width=0.3, showfliers=False, linewidth=1, ax=reg1)
reg1.axhline(0, linestyle='--', color='k')
reg1.set_xlabel("Regressor")
reg1.set_ylabel(r"$R^2$ (unique)")
reg1.set_xticks(range(r2a.shape[1]))
reg1.set_xticklabels(['']*r2a.shape[1])
# add pvalue for each regressor
ym = reg1.get_ylim()[-1]
for i, r in enumerate(r2a.keys()):
    p = get_direct_prob(r2a[r], np.zeros(r2a.shape[0]))[0]
    reg1.text(i-0.3, ym, f"p={p:.4f}", fontsize=6)

r2d = r2_delta[[k for k in r2_delta if (k=='full') | (k.startswith('u'))]]
sns.boxplot(x='variable', y='value', data=r2d.melt(), 
                        palette=palette, width=0.3, showfliers=False, linewidth=1, ax=reg2)
reg2.axhline(0, linestyle='--', color='k')
reg2.set_xlabel("Regressor")
reg2.set_ylabel(r"$R^2$ (unique)")
reg2.set_xticks(range(r2d.shape[1]))
reg2.set_xticklabels(['']*r2d.shape[1])
# add pvalue for each regressor
ym = reg2.get_ylim()[-1]
for i, r in enumerate(r2d.keys()):
    p = get_direct_prob(r2d[r], np.zeros(r2d.shape[0]))[0]
    if p>0.5:
        p = 1 - p
    reg2.text(i-0.3, ym, f"p={p:.4f}", fontsize=6)

# plot coefficients
palette = {'pc1_mean': 'r', 'pc1_diff': 'r', 'dU_mag_test': color.SIGNAL, 'noiseAlign': color.COSTHETA}
sns.boxplot(x='variable', y='value', data=coef_all.melt(), 
                        palette=palette, width=0.3, showfliers=False, linewidth=1, ax=coef1)
coef1.axhline(0, linestyle='--', color='k')
coef1.set_xlabel("Regressor")
coef1.set_ylabel("Coefficient (norm.)")
coef1.set_xticks(range(coef_all.shape[1]))
coef1.set_xticklabels(['']*coef_all.shape[1])
# add pvalue for each regressor
ym = coef1.get_ylim()[-1]
for i, r in enumerate(coef_all.keys()):
    p = get_direct_prob(coef_all[r], np.zeros(coef_all.shape[0]))[0]
    if p > 0.5:
        p = 1 - p
    coef1.text(i-0.3, ym, f"p={p:.4f}", fontsize=6)

sns.boxplot(x='variable', y='value', data=coef_delta.melt(), 
                        palette=palette, width=0.3, showfliers=False, linewidth=1, ax=coef2)
coef2.axhline(0, linestyle='--', color='k')
coef2.set_xlabel("Regressor")
coef2.set_ylabel("Coefficient (norm.)")
coef2.set_xticks(range(coef_delta.shape[1]))
coef2.set_xticklabels(['']*coef_delta.shape[1])
# add pvalue for each regressor
ym = coef2.get_ylim()[-1]
for i, r in enumerate(coef_delta.keys()):
    p = get_direct_prob(coef_delta[r], np.zeros(coef_delta.shape[0]))[0]
    coef2.text(i-0.3, ym, f"p={p:.4f}", fontsize=6)

f.tight_layout()

if savefig: 
    f.savefig(fig_fn)

plt.show()