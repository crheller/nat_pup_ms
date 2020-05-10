"""
Pull out decoding weights, eigenvector weights, and dU weights. See if there are
any interesting patterns, for instance, if the SUs map onto FS/RS cell types in an 
interesting way.
"""

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

import nems_lbhb.baphy as nb

site = 'DRX008b.e1:64'
modelname = 'dprime_jk10_zscore'
batch = 289

# get list of cells at this site
options = {'batch': batch, 'cellid': site}
cells, _ = nb.parse_cellid(options)
cells = np.array(cells)

celltypes = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/celltypes.csv')
# get list of SU BS at this site
bs_cells = np.sort([c for c in celltypes[celltypes.type==0].cellid if c in cells])
bs_idx = [np.argwhere(cells==c)[0][0] for c in bs_cells]

# get list of SU NS at this site
ns_cells = np.sort([c for c in celltypes[celltypes.type==1].cellid if c in cells])
ns_idx = [np.argwhere(cells==c)[0][0] for c in ns_cells]

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
loader = decoding.DecodingResults()

fn = os.path.join(path, site, modelname+'_TDR.pickle')
results = loader.load_results(fn)
pairs = results.evoked_stimulus_pairs 
df = results.numeric_results.loc[pd.IndexSlice[pairs, :], :]

dU_mean, dU_sem = results.get_result('dU_all', pairs, 2)
wopt_mean, wopt_sem = results.get_result('wopt_all', pairs, 2)
ev1_mean, ev1_sem = results.slice_array_results('evecs_all', pairs, 2, idx=[None, 1])

bs_dU = results.slice_array_results('dU_all', pairs, 2, idx=[0, bs_idx])[0]
ns_dU = results.slice_array_results('dU_all', pairs, 2, idx=[0, ns_idx])[0]
bs_ev1 = results.slice_array_results('evecs_all', pairs, 2, idx=[bs_idx, 0])[0]
ns_ev1 = results.slice_array_results('evecs_all', pairs, 2, idx=[ns_idx, 0])[0]

f, ax = plt.subplots(1, 2, figsize=(8, 4))

# plot distibutions of dU for NS and BS cells over all stimulus pairs
dubins = np.arange(-5, 5, 0.5)
ax[0].hist([np.stack(bs_dU.values).flatten(),
            np.stack(ns_dU.values).flatten()], bins=dubins, stacked=False, 
            rwidth=0.7, edgecolor='none', density=True)
ax[0].legend(['BS', 'NS'])
ax[0].set_xlabel(r"$\Delta \mathbf{\mu}_i$")

# plot distributions of noise eigenvector weights for BS / NS cells over all stimulus pairs
evbins = np.arange(-1, 1, 0.1)
ax[1].hist([np.stack(bs_ev1.values).flatten(),
            np.stack(ns_ev1.values).flatten()], bins=evbins, stacked=False, 
            rwidth=0.7, edgecolor='none', density=True)
ax[1].legend(['BS', 'NS'])
ax[1].set_xlabel(r"$\mathbf{e}_{1, i}$")

f.tight_layout()

plt.show()