"""
Look at pariwsie noise correlations as a function of the modulation
index of each neuron in the pair. Idea is that if change in 
correlations is due to a change in a gain factor, there will be a smaller
(or oppsite) effect of pupil on correlations for large MI pairs
"""

import load_results as ld
import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.baphy as nb
from nems.recording import Recording
from nems_lbhb.preprocessing import create_pupil_mask
import charlieTools.noise_correlations as nc
import pandas as pd

site = 'BRT026c'
spont = False

if not spont:
    path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/noise_correlations/results_xforms/'
    nc = ld.load_noise_correlation('rsc_bal', path=path)
    nc = nc[nc['site']==site]

else:
    batch = 289
    ops = {'batch': batch, 'siteid': site, 'rasterfs': 4, 'pupil': 1, 
                'rem': 1, 'stim': 0}
    uri = nb.baphy_load_recording_uri(**ops)

    rec = Recording.load(uri)
    rec = rec.and_mask(['PreStimSilence'])
    rec = rec.apply_mask(reset_epochs=True)

    pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
    bp_mask = create_pupil_mask(rec, **pup_ops)['mask']
    sp_mask = bp_mask._modified_copy(~bp_mask._data)
    rec_sp = rec.copy()
    rec_sp['mask'] = sp_mask
    rec_sp = rec_sp.apply_mask(reset_epochs=True)
    rec_bp = rec.copy()
    rec_bp['mask'] = bp_mask
    rec_bp = rec_bp.apply_mask(reset_epochs=True)

    # Get response dictionaries
    epochs = 'PreStimSilence'
    big_r = rec['resp'].extract_epochs(epochs, mask=bp_mask)
    small_r = rec['resp'].extract_epochs(epochs, mask=sp_mask)
    cells = big_r['PreStimSilence'].shape[1]
    big_r['PreStimSilence'] = big_r['PreStimSilence'].transpose([0, 2, 1]).reshape(-1, cells, 1)
    small_r['PreStimSilence'] = small_r['PreStimSilence'].transpose([0, 2, 1]).reshape(-1, cells, 1)

    nc_big = nc.compute_rsc(big_r, chans=rec['resp'].chans)
    nc_small = nc.compute_rsc(small_r, chans=rec['resp'].chans)

    nc = pd.DataFrame(columns=['bp', 'sp'], index=nc_big.index, data=np.vstack((nc_big['rsc'].values, nc_small['rsc'].values)).T)

mi = ld.load_mi()
mi = mi[mi['site']==site]

idx_tup = [(i.split('_')[0], i.split('_')[1]) for i in nc.index]
bins = np.round(np.arange(-.2, .3, 0.1), 2)
val = []
err = []
n = []
ran = []
for i, b in enumerate(bins):
    if i == 0:
        tf1 = np.array([True if ((mi.loc[c1]['MI']<b) & (mi.loc[c2]['MI']<b)) else False for c1, c2 in idx_tup])

        m = (nc[tf1]['sp'] - nc[tf1]['bp']).mean()
        sem = (nc[tf1]['sp'] - nc[tf1]['bp']).sem()
        ran.append((b,))
        n.append(tf1.sum())

    elif i == len(bins)-1:
        tf1 = np.array([True if ((mi.loc[c1]['MI']>b) & (mi.loc[c2]['MI']>b)) else False for c1, c2 in idx_tup])

        m = (nc[tf1]['sp'] - nc[tf1]['bp']).mean()
        sem = (nc[tf1]['sp'] - nc[tf1]['bp']).sem()
        ran.append((b,))
        n.append(tf1.sum())

    else:
        b1 = bins[i+1]
        tf1 = np.array([True if ((mi.loc[c1]['MI']>b) & (mi.loc[c2]['MI']>b)) else False for c1, c2 in idx_tup])
        tf2 = np.array([True if ((mi.loc[c1]['MI']<b1) & (mi.loc[c2]['MI']<b1)) else False for c1, c2 in idx_tup])

        m = (nc[tf1 & tf2]['sp'] - nc[tf1 & tf2]['bp']).mean()
        sem = (nc[tf1 & tf2]['sp'] - nc[tf1 & tf2]['bp']).sem()
        ran.append((b, b1))
        n.append((tf1&tf2).sum())

    val.append(m)
    err.append(sem)

f, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(np.arange(0, len(bins)), val, color='k', marker='o')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].set_ylabel('Delta N.C. (small minus big)', fontsize=8)

ax[1].plot(n, color='k', marker='o')
ax[1].set_ylabel('n_pairs', fontsize=8)
ax[1].set_yscale('log')
ax[1].set_xlabel('MI boundary', fontsize=8)
ax[1].set_xticks(range(0, len(bins)))
ax[1].set_xticklabels(ran, rotation=45)

f.tight_layout()

plt.show()