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

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore'
val = 'dp_opt_test'
estval = '_train'
nbins = 20
cmap = 'Greens'
high_var_only = False

# only crop the dprime value. Show count for everything
if estval == '_train':
    x_cut = (2, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 9)
    y_cut = (0.35, 1) 

f = plt.figure(figsize=(12, 8))

hax = plt.subplot2grid((2, 3), (0, 0))
cax = plt.subplot2grid((2, 3), (1, 0))
q1ax = plt.subplot2grid((2, 3), (0, 2))
q2ax = plt.subplot2grid((2, 3), (0, 1))
q3ax = plt.subplot2grid((2, 3), (1, 1))
q4ax = plt.subplot2grid((2, 3), (1, 2))

sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']
df = []
for site in sites:
    fn = os.path.join(path, site, modelname+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

    stim = results.evoked_stimulus_pairs
    high_var_pairs = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/high_pvar_stim_combos.csv', index_col=0)
    high_var_pairs = high_var_pairs[high_var_pairs.site==site].index.get_level_values('combo')
    if high_var_only:
        stim = [s for s in stim if s in high_var_pairs]

    if len(stim) == 0:
        pass
    else:
        _df = _df.loc[pd.IndexSlice[stim, 2], :]
        _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
        _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
        _df['site'] = site
        df.append(_df)

df = pd.concat(df)

# filter based on x_cut / y_cut
mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
df_dp = df[mask1 & mask2]

# plot dprime
df_dp.plot.hexbin(x='dU_mag'+estval, 
                  y='cos_dU_evec'+estval, 
                  C=val, 
                  gridsize=nbins, ax=hax, cmap=cmap) 
hax.set_xlabel(r'$|\Delta \mathbf{\mu}|$', color='orange')
hax.set_ylabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$', color='purple')
hax.spines['bottom'].set_color('orange')
hax.spines['bottom'].set_lw(2)
hax.xaxis.label.set_color('orange')
hax.tick_params(axis='x', colors='orange')
hax.spines['left'].set_color('purple')
hax.spines['left'].set_lw(2)
hax.yaxis.label.set_color('purple')
hax.tick_params(axis='y', colors='purple')
hax.set_title(r"$d'^2$")

# plot count histogram
df.plot.hexbin(x='dU_mag'+estval, 
               y='cos_dU_evec'+estval, 
               C=None, 
               gridsize=nbins, ax=cax, cmap='Reds') 
# overlay box for data extracted
line = np.array([[x_cut[0], y_cut[0]], 
                 [x_cut[0], y_cut[1]], 
                 [x_cut[1], y_cut[1]], 
                 [x_cut[1], y_cut[0]], 
                 [x_cut[0], y_cut[0]]])
cax.plot(line[:, 0], line[:, 1], linestyle='--', lw=2, color='k')

cax.set_xlabel(r'$|\Delta \mathbf{\mu}|$', color='orange')
cax.set_ylabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$', color='purple')
cax.spines['bottom'].set_color('orange')
cax.spines['bottom'].set_lw(2)
cax.xaxis.label.set_color('orange')
cax.tick_params(axis='x', colors='orange')
cax.spines['left'].set_color('purple')
cax.spines['left'].set_lw(2)
cax.yaxis.label.set_color('purple')
cax.tick_params(axis='y', colors='purple')
cax.set_title('Count')

ylim = (-6, 10)
xlim = (-14, 14)
# plot quadrant 1 example
mask1 = (df_dp['cos_dU_evec'+estval] >0.1) & (df_dp['cos_dU_evec'+estval] < 0.2)
mask2 = (df_dp['dU_mag'+estval] > 3) & (df_dp['dU_mag'+estval] < 4)
mask3 = df_dp['site'] == 'TAR010c'
df_dp[mask1 & mask2 & mask3]

# 51_54 TAR010c
pair, site, batch = (51, 54), 'TAR010c', 289
decoding.plot_stimulus_pair(site,
                            batch, 
                            pair,
                            colors=['red', 'blue'],
                            axlabs=['TDR 1', 'TDR 2'],
                            ylim=ylim,
                            xlim=xlim,
                            ellipse=True,
                            ax=q1ax)

# plot quadrant 2 example
# 17_74 DRX006b.e1:64
# 7_74 DRX006b.e1:64
# 7_78 DRX006b.e1:64
# 19_76 DRX007a.e1:64
# 4_46 DRX007a.e1:64
pair, site, batch = (76, 77), 'DRX007a.e1:64', 289
decoding.plot_stimulus_pair(site,
                            batch, 
                            pair,
                            colors=['red', 'blue'],
                            axlabs=['TDR 1', 'TDR 2'],
                            ylim=ylim,
                            xlim=xlim,
                            ellipse=True,
                            ax=q2ax)

# plot quadrant 3 example
#14_18 TAR010c
#36_38 TAR010c
#48_49 TAR010c
#50_54 TAR010c *
pair, site, batch = (50, 54), 'TAR010c', 289
decoding.plot_stimulus_pair(site,
                            batch, 
                            pair,
                            colors=['red', 'blue'],
                            axlabs=['TDR 1', 'TDR 2'],
                            ylim=ylim,
                            xlim=xlim,
                            ellipse=True,
                            ax=q3ax)

# plot quadrant 4 example
# 11_50 TAR010c
pair, site, batch = (11, 50), 'TAR010c', 289
decoding.plot_stimulus_pair(site,
                            batch, 
                            pair,
                            colors=['red', 'blue'],
                            axlabs=['TDR 1', 'TDR 2'],
                            ylim=ylim,
                            xlim=xlim,
                            ellipse=True,
                            ax=q4ax)

f.tight_layout()

plt.show()