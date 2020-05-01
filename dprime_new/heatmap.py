"""
Plot decoding per site on a heatmap. Axis are |dU| and overlap of eigenvector n with signal (dU).
Plot test deocding performance.
"""

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore'
xaxis_data = 'dprime_jk10_zscore'
site = 'TAR010c'
nbins = 15
val = 'dp_opt_test'  #'state_diff'  #'dp_opt_test'
xaxis = 'cos_dU_evec'  #'dU_dot_evec_sq'  # 'cos_dU_evec'
vmin = 0
vmax = None
cmap = 'Greens'

fn = os.path.join(path, site, modelname+'_PCA.pickle')
results = loader.load_results(fn)

fn = os.path.join(path, site, xaxis_data+'_PCA.pickle')
results_raw = loader.load_results(fn)

n_components = 2

df = results.numeric_results
df['state_diff'] = (df['bp_dp'] - df['sp_dp']) / df['bp_dp']

stim = results.evoked_stimulus_pairs

# plot cos(dU, evec) as fn of alpha
du_evec = np.stack(results.slice_array_results('cos_dU_evec_test', stim, n_components, idx=[0, None])[0].values.tolist())
f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(du_evec.mean(axis=0), 'o-')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$')

f.tight_layout()

for alpha in range(0, n_components):
    df[xaxis] = results_raw.slice_array_results(xaxis+'_test', stim, n_components, idx=[0, alpha])[0]

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title(r"$d'^{2}$")
    df.loc[pd.IndexSlice[stim, n_components], :].plot.hexbin(x=xaxis, 
                                                y='dU_mag_test', 
                                                C=val, 
                                                gridsize=nbins, ax=ax[0], cmap=cmap, vmin=vmin, vmax=vmax) 
    ax[0].set_ylabel(r'$|\Delta \mathbf{\mu}|$')
    ax[0].set_xlabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$')

    ax[1].set_title("Count")
    df.loc[pd.IndexSlice[stim, n_components],:].plot.hexbin(x=xaxis, 
                                                y='dU_mag_test', 
                                                C=None, 
                                                gridsize=nbins, ax=ax[1], cmap='Reds', vmin=0) 
    ax[1].set_ylabel(r'$|\Delta \mathbf{\mu}|$')
    ax[1].set_xlabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$')

    f.canvas.set_window_title('alpha = {0}'.format(alpha))

    f.tight_layout()

plt.show()
