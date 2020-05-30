"""
Plot an example pair of stimuli where beta1 is orthogonal to dU, but gain changes help.
e.g. a case where the noise "fans" out due to gain.
"""

import load_results as ld
import colors as color
import ax_labels as alab

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.preprocessing as preproc
from nems_lbhb.baphy import parse_cellid
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/supp3_beta1_caveat.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'

site = 'TAR010c'

fn = os.path.join(path, site, modelname+'_TDR.pickle')
results = loader.load_results(fn)
df = results.numeric_results.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]
cos_dU_evec = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, 2, idx=[0,0])[0]

# find a case where e1 is ortho to dU, and evec_sim is low
mask = df['evec_sim_test']<0.3
#mask = mask & (cos_dU_evec<0.3)
mask = mask & (df['beta1_dot_dU']<0.2)
mask = mask & ((df['bp_dU_mag']-df['sp_dU_mag'])>2)
temp = df[mask]
pairs = [(int(p[0].split('_')[0]), int(p[0].split('_')[1])) for p in temp.index]

# get beta1
fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/nc_zscore_lvs.pickle'
with open(fn, 'rb') as handle:
    lv_dict = pickle.load(handle)
beta1 = lv_dict[site]['beta1']

pairs = [(9, 39), (36, 57), (8, 57), (9, 57)]

for p in pairs:
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    decoding.plot_stimulus_pair(site,
                                289,
                                p,
                                colors=[color.STIMB, color.STIMA],
                                axlabs=[alab.SIGNAL_short, 'TDR2'],
                                ellipse=True,
                                pup_cmap=True,
                                lv_axis=beta1,
                                ax_length=5,
                                ax=ax)
    print(p)
    f.tight_layout()
    plt.show()

pair = (9, 57)
f, ax = plt.subplots(1, 1, figsize=(4, 4))
decoding.plot_stimulus_pair(site,
                            289,
                            pair,
                            colors=[color.STIMB, color.STIMA],
                            axlabs=[alab.SIGNAL_short, 'TDR2'],
                            ellipse=True,
                            pup_cmap=True,
                            lv_axis=beta1,
                            lv_ax_name=r'$\beta_1$',
                            ax_length=5,
                            ax=ax)

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()