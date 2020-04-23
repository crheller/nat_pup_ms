"""
Plot decoding per site on a heatmap. Axis are |dU| and overlap of eigenvector n with wopt (or wdiag).
Plot test deocding performance.
"""

import charlieTools.nat_sounds_ms.decoding as decoding
import os

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10'
site = 'TAR010c'
nbins = 20

fn = os.path.join(path, site, modelname+'_PLS.pickle')
results = loader.load_results(fn)

df = results.numeric_results
stim = results.evoked_stimulus_pairs


f, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title(r"$d'^{2}$")
df.loc[pd.IndexSlice[stim,2],:].plot.hexbin(x='cos_dU_wopt_test', 
                                            y='dU_mag_test', 
                                            C='dp_opt_test', 
                                            gridsize=nbins, ax=ax[0], vmin=0) 
ax[0].set_ylabel(r'$|\Delta \mathbf{\mu}|$')

ax[1].set_title("Count")
df.loc[pd.IndexSlice[stim,2],:].plot.hexbin(x='cos_dU_wopt_test', 
                                            y='dU_mag_test', 
                                            C=None, 
                                            gridsize=nbins, ax=ax[1], cmap='Reds', vmin=0) 
ax[1].set_ylabel(r'$|\Delta \mathbf{\mu}|$')

f.tight_layout()

plt.show()
