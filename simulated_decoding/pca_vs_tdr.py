"""
Illustrate projections into TDR space (done on per stimulus pair basis)
and projection into PCA space (done over all the data).

Do this for 3 stimuli, so 3 pairwise combinations. Show that PCA distorts space (dU) 
while TDR preserves both PC1 of noise and dU perfectly.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.plotting as cplt

np.random.seed(123)

Ndim = 100
Ntrials= 5000
var_ratio = 3 # pc1 has X times the variance as pc2

# simulated data
u1 = 4
u2 = 4
u3 = 4
u = np.stack((np.random.poisson(u1, Ndim), np.random.poisson(u2, Ndim), np.random.poisson(u3, Ndim)))

# make two dimensional noise:
# one large dim ~orthogonal to dU and one smaller dim ~ parallel to dU
dU = u[[1], :] - u[[0], :]
dU = dU / np.linalg.norm(dU)

diff_cor = dU + np.random.normal(0, 0.001, dU.shape)
diff_cor = diff_cor / np.linalg.norm(diff_cor) * 5 

dU2 = u[[2], :] - u[[0], :]
dU2 = dU2 / np.linalg.norm(dU2)
diff_cor2 = dU2 + np.random.normal(0, 0.001, dU.shape)
diff_cor2 = diff_cor2 / np.linalg.norm(0, 0.001, dU2.shape)

pc1 = np.random.normal(0, 1, dU.shape)
pc1 = (pc1 / np.linalg.norm(pc1)) * 5 * var_ratio

evecs = np.concatenate((diff_cor, diff_cor2, pc1), axis=0)
cov = evecs.T.dot(evecs)

# simulate full data matrix
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
X1 = _X + u[0, :]
X2 = _X + u[1, :]
X3 = _X + u[2, :]
X_raw = np.stack((X1, X2, X3)).transpose([-1, 1, 0])

# add random noise to data matrix
X_raw += np.random.normal(0, 0.5, X_raw.shape)

# preprocess data (zscore)
X, _ = nat_preproc.scale_est_val([X_raw], [X_raw])
X = X[0]
