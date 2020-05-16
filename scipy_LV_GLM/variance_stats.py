"""
Plot mean variance relationship over all neurons / stimuli (unique time bins)
Determine if noise is super poisson and if it's additive or multiplicative
"""

from nems.recording import Recording
import nems_lbhb.baphy as nb
from nems_lbhb.preprocessing import mask_high_repetion_stims
import nems.db as nd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b']
site = 'TAR010c'
batch = 289
fs = 4
ops = {'batch': batch, 'siteid': site, 'rasterfs': fs, 'pupil': 1, 'rem': 1,
    'stim': 1}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri)
rec['resp'] = rec['resp'].rasterize()
rec['stim'] = rec['stim'].rasterize()
rec = mask_high_repetion_stims(rec)
rec = rec.apply_mask(reset_epochs=True)

epochs = np.unique([e for e in rec.epochs.name if 'STIM' in e]).tolist()
resp = rec['resp'].extract_epochs(epochs)
nbins = resp[epochs[0]].shape[-1]

u = []
var = []

for e in epochs:
    for b in range(nbins):
        m = resp[e][:, :, b].mean(axis=-1)
        v = resp[e][:, :, b].var(axis=-1)
        u.extend(m)
        var.extend(v)

#def func(x, a, b):
#    return a * x ** 2 + b
    
#popt, pcov = curve_fit(func, u, var)

f, ax = plt.subplots(1, 1)

ax.plot(u, var, '.', color='lightgrey')
#m = np.max(u)
#ax.plot(np.arange(-0.2, m, 0.1), func(np.arange(-0.2, m, 0.1), *popt), 'k-')
m = 20
ax.plot([0, m], [0, m],'k--')
ax.set_xlabel('Mean')
ax.set_ylabel('Variance')
ax.axhline(0, color='k')
ax.axvline(0, color='k')

plt.show()
    