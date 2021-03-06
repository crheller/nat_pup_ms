"""
Illustration of decoding procedure (I think this can be done in inkscape)
         * example stimulus pairs, heatmap of dprime. 

- 2 rows x 3 columns
    (0, 0) = d'^2 vs. axes (axes = cos_dU_ev1, mag(dU))
            - draw quadrant lines on this plot to map to examples
    (1, 0) = count histogram (or kde plot?)
    (0, 1) = example quadrant II
    (0, 2) = example quadrant I
    (1, 1) = example quadrant III
    (1, 2) = example quadrant IV

Also, label the heatmap axis values for each example pair, and illustrate
what the respective angles / vectors are (done inside plot fn)
"""

import colors as color
import ax_labels as alab
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = False
recache = False # recache dprime results locally
ALL_TRAIN_DATA = False  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = HIGHR_SITES
path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR+'fig3_overall_dprime.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk1_loocv_zscore_nclvz_fixtdr2'
val = 'dp_opt_test'
estval = '_test'
nbins = 20
cmap = 'Greens'
vmax = 50
hexscale = 'log' # or 'log'

# only crop the dprime value. Show count for everything
if estval == '_train':
    x_cut = (3, 8.5)
    y_cut = (0.1, .45) 
elif estval == '_test':
    x_cut = DU_MAG_CUT
    y_cut = NOISE_INTERFERENCE_CUT

f = plt.figure(figsize=(9, 6))

hax = plt.subplot2grid((2, 3), (0, 0))
cax = plt.subplot2grid((2, 3), (1, 0))
q1ax = plt.subplot2grid((2, 3), (0, 2))
q2ax = plt.subplot2grid((2, 3), (0, 1))
q3ax = plt.subplot2grid((2, 3), (1, 1))
q4ax = plt.subplot2grid((2, 3), (1, 2))


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
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0])[0]
    _df['site'] = site
    df.append(_df)

df = pd.concat(df)

# filter based on x_cut / y_cut
if (x_cut is not None) & (y_cut is not None):
    mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
    mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
else:
    mask1 = (True * np.ones(df.shape[0])).astype(bool)
    mask2 = (True * np.ones(df.shape[0])).astype(bool)
df_dp = df[mask1 & mask2]

# plot dprime
df_dp.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C=val, 
                  gridsize=nbins, ax=hax, cmap=cmap, vmax=vmax) 
hax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
hax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
hax.spines['bottom'].set_color(color.SIGNAL)
hax.xaxis.label.set_color(color.SIGNAL)
hax.tick_params(axis='x', colors=color.SIGNAL)
hax.spines['left'].set_color(color.COSTHETA)
hax.yaxis.label.set_color(color.COSTHETA)
hax.tick_params(axis='y', colors=color.COSTHETA)
hax.set_title(r"$d'^2$")

# plot count histogram
df.plot.hexbin(x='dU_mag'+estval, 
               y='cos_dU_evec'+estval, 
               C=None, 
               gridsize=nbins, ax=cax, cmap='Reds', bins=hexscale) 
# overlay box for data extracted
line = np.array([[x_cut[0], y_cut[0]], 
                 [x_cut[0], y_cut[1]], 
                 [x_cut[1], y_cut[1]], 
                 [x_cut[1], y_cut[0]], 
                 [x_cut[0], y_cut[0]]])
cax.plot(line[:, 0], line[:, 1], linestyle='--', lw=2, color='k')

cax.set_xlabel(alab.SIGNAL, color=color.SIGNAL)
cax.set_ylabel(alab.COSTHETA, color=color.COSTHETA)
cax.spines['bottom'].set_color(color.SIGNAL)
cax.xaxis.label.set_color(color.SIGNAL)
cax.tick_params(axis='x', colors=color.SIGNAL)
cax.spines['left'].set_color(color.COSTHETA)
cax.yaxis.label.set_color(color.COSTHETA)
cax.tick_params(axis='y', colors=color.COSTHETA)
cax.set_title('Count')

xlim = (-7, 7)
ylim = (-7, 7)
# plot quadrant 1 example
mask1 = (df_dp['cos_dU_evec'+estval] > 0.7) & (df_dp['cos_dU_evec'+estval] < 1)
mask2 = (df_dp['dU_mag'+estval] > 7) & (df_dp['dU_mag'+estval] < 8)
mask3 = df_dp['site'] == 'TAR010c'
#mask3 = (abs(df_dp['dp_opt_test']-df_dp['dp_opt_train']) < 1) & (df_dp['dp_opt_test']<6)
out = df_dp[mask1 & mask2 & mask3][['dp_opt_test', 'dp_opt_train', 'site']]

# 51_54 TAR010c **
# 10_33
pair, site, batch = (9, 49), 'TAR010c', 289
decoding.plot_stimulus_pair(site,
                            batch, 
                            pair,
                            colors=[color.STIMA, color.STIMB],
                            axlabs=[alab.DU + r" ($TDR_1$)", r'$TDR_2$'],
                            ylim=ylim,
                            xlim=xlim,
                            ellipse=True,
                            ax_length=5,
                            ax=q1ax)

# plot quadrant 2 example
# 17_74 DRX006b.e1:64
# 7_78 DRX006b.e1:64
# 19_76 DRX007a.e1:64
# 4_46 DRX007a.e1:64
pair, site, batch = (9, 12), 'TAR010c', 289
decoding.plot_stimulus_pair(site,
                            batch, 
                            pair,
                            colors=[color.STIMA, color.STIMB],
                            axlabs=[alab.DU + r" ($TDR_1$)", r'$TDR_2$'],
                            ylim=ylim,
                            xlim=xlim,
                            ellipse=True,
                            ax_length=5,
                            ax=q2ax)

# plot quadrant 3 example
#14_18 TAR010c
#36_38 TAR010c
#48_49 TAR010c
#50_54 TAR010c *
pair, site, batch = (29, 37), 'TAR010c', 289
decoding.plot_stimulus_pair(site,
                            batch, 
                            pair,
                            colors=[color.STIMA, color.STIMB],
                            axlabs=[alab.DU + r" ($TDR_1$)", r'$TDR_2$'],
                            ylim=ylim,
                            xlim=xlim,
                            ellipse=True,
                            ax_length=5,
                            ax=q3ax)

# plot quadrant 4 example
# 11_50 TAR010c
pair, site, batch = (14, 59), 'TAR010c', 289
decoding.plot_stimulus_pair(site,
                            batch, 
                            pair,
                            colors=[color.STIMA, color.STIMB],
                            axlabs=[alab.DU + r" ($TDR_1$)", r'$TDR_2$'],
                            ylim=ylim,
                            xlim=xlim,
                            ellipse=True,
                            ax_length=5,
                            ax=q4ax)

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()
