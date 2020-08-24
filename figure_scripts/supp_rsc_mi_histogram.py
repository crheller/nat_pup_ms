"""
Plot noise correlation (and delta noise correaltion) as function 
of each cell's MI (supp. to Figure 7)
"""
import load_results as ld
import colors as color
from path_settings import PY_FIGURES_DIR

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.preprocessing as preproc
from nems_lbhb.baphy import parse_cellid
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = True
fig_fn = PY_FIGURES_DIR+ 'supp_rsc_mi_histogram.svg'
mi_max = 0.3
mi_min = -0.2
# set up subplots
f = plt.figure(figsize=(3.5, 3))

cax = plt.subplot2grid((1, 1), (0, 0))
#rscax = plt.subplot2grid((1, 3), (0, 1), colspan=1)
#drscax = plt.subplot2grid((1, 3), (0, 2), colspan=1)

path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'
MI_pred = True
gain_pred = False
df = pd.concat([pd.read_csv(os.path.join(path,'d_289_pup_sdexp.csv'), index_col=0),
                pd.read_csv(os.path.join(path,'d_294_pup_sdexp.csv'), index_col=0)])
try:
    df['r'] = [np.float(r.strip('[]')) for r in df['r'].values]
    df['r_se'] = [np.float(r.strip('[]')) for r in df['r_se'].values]
except:
    pass

df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se'])
MI = df.loc[:, pd.IndexSlice['MI', 'st.pup']]

rsc_path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'
rsc_df = ld.load_noise_correlation('rsc_ev', xforms_model='NULL', path=rsc_path)

# add column for the gain of each neuron
m1 = [MI.loc[p.split('_')[0]] for p in rsc_df.index]
m2 = [MI.loc[p.split('_')[1]] for p in rsc_df.index]
rsc_df['m1'] = m1
rsc_df['m2'] = m2
rsc_df['diff'] = rsc_df['sp'] - rsc_df['bp']
mask = (rsc_df['m1'] < mi_max) & (rsc_df['m1'] > -mi_min) & (rsc_df['m2'] < mi_max) & (rsc_df['m2'] > mi_min)


# bin and plot

# plot count
xbins = np.linspace(-0.5, 0.5, 20)
ybins = np.linspace(-0.5, 0.5, 20)
count = ss.binned_statistic_2d(x=rsc_df['m1'], 
                            y=rsc_df['m2'],
                            values=rsc_df['all'],
                            statistic='count',
                            bins=[xbins, ybins])

im = cax.imshow(count[0], cmap='Reds', aspect='auto', origin='lower',
                            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
divider = make_axes_locatable(cax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
cax.set_title(r"Count")
cax.set_xlabel(r"$MI_j$")
cax.set_ylabel(r"$MI_i$")

line = np.array([[mi_min, mi_min], 
                 [mi_max, mi_min], 
                 [mi_max, mi_max], 
                 [mi_min, mi_max], 
                 [mi_min, mi_min]])
cax.plot(line[:, 0], line[:, 1], linestyle='--', lw=2, color='k')

'''
# plot overall noise correlation
xbins = np.linspace(mi_min, mi_max, 10)
ybins = np.linspace(mi_min, mi_max, 10)
heatmap_rsc = ss.binned_statistic_2d(x=rsc_df['m1'], 
                            y=rsc_df['m2'],
                            values=rsc_df['all'],
                            statistic='mean',
                            bins=[xbins, ybins])

im = rscax.imshow(heatmap_rsc[0], cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax,
                            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                            origin='lower', interpolation='gaussian')
divider = make_axes_locatable(rscax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
rscax.set_title(r"Noise Correlation")
rscax.set_xlabel(r"$MI_j$")
rscax.set_ylabel(r"$MI_i$")

# plot diff
heatmap_drsc = ss.binned_statistic_2d(x=rsc_df['m1'], 
                            y=rsc_df['m2'],
                            values=rsc_df['diff'],
                            statistic='mean',
                            bins=[xbins, ybins])

im = drscax.imshow(heatmap_drsc[0], cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax,
                            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                            origin='lower', interpolation='gaussian')
divider = make_axes_locatable(drscax)
cbarax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cbarax, orientation='vertical')
drscax.set_title(r"$\Delta$ Noise Correlation")
drscax.set_xlabel(r"$MI_j$")
drscax.set_ylabel(r"$MI_i$")
'''

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()
