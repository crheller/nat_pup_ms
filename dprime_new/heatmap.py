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

fn = os.path.join(path, site, modelname+'_PLS.pickle')
results = loader.load_results(fn)