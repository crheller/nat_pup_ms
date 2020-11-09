"""
Simulate changes in independent vs. shared noise for n Neurons
"""
import numpy as np
import matplotlib.pyplot as plt
from charlieTools.dim_reduction import TDR
from charlieTools.plotting import compute_ellipse

np.random.seed(123)

Ndim = 100
Ntrials = 1000
var_ratio = 1.2 # pc1 has X times the variance as pc2

# simulated data
u1 = 4
u2 = 4
u = np.stack((np.random.poisson(u1, Ndim), np.random.poisson(u2, Ndim)))

# make two dimensional noise:
# one large dim ~orthogonal to dU and one smaller dim ~ parallel to dU
dU = u[[1], :] - u[[0], :]
dU = dU / np.linalg.norm(dU)

diff_cor = dU + np.random.normal(0, 0.001, dU.shape)
diff_cor = diff_cor / np.linalg.norm(diff_cor) * 2
pc1 = np.random.normal(0, 1, dU.shape)
pc1 = (pc1 / np.linalg.norm(pc1)) * 2  * var_ratio

noise_axis = pc1
evecs = np.concatenate((diff_cor, pc1), axis=0)
cov = evecs.T.dot(evecs)

# simulate full data matrix
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
X1 = _X + u[0, :]
X2 = _X + u[1, :]
X_raw = np.stack((X1, X2)).transpose([-1, 1, 0])

# add random noise to data matrix to make things behave well
X_raw += np.random.normal(0, 0.5, X_raw.shape)

# simulate 2 datasets. First has low ind / shared variance, second has high ind / shared variance.

# subsample from full population for various numbers of neurons.
# fix ind. variance or shared variance between conditions
for nunits in range(nNeurons):
