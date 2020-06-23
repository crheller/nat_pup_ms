"""
Three clouds. Illustrate delta u for each pair, and illustrate pooled noise data (e.g. subtract stim means) and
show first PC of that. Then, project one example comparison down into TDR space
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.plotting as cplt

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

np.random.seed(123)

savefig = True
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/TDR_illustration.svg'

f = plt.figure(figsize=(9, 3))

rawax = plt.subplot2grid((1, 3), (0, 0))
noiseax = plt.subplot2grid((1, 3), (0, 1))
tdrax = plt.subplot2grid((1, 3), (0, 2))

# Define 3 stimulus clouds, A, B, C, all with identical covariacne matrices
k = 100
cov = np.array([[1, 0.4], [0.4, 1]])
u1 = [2.5, -2.5]
u2 = [2.5, 2.5]
u3 = [-2.5, 2.5]

X1 = np.random.multivariate_normal(u1, cov, k)
X2 = np.random.multivariate_normal(u2, cov, k)
X3 = np.random.multivariate_normal(u3, cov, k)

# plot raw data
el1 = cplt.compute_ellipse(X1[:, 0], X1[:, 1])
el2 = cplt.compute_ellipse(X2[:, 0], X2[:, 1])
el3 = cplt.compute_ellipse(X3[:, 0], X3[:, 1])

rawax.scatter(X1[:, 0], X1[:, 1], s=15, alpha=0.3, color='c', edgecolor='none')
rawax.scatter(X2[:, 0], X2[:, 1], s=15, alpha=0.3, color='m', edgecolor='none')
rawax.scatter(X3[:, 0], X3[:, 1], s=15, alpha=0.3, color='y', edgecolor='none')

rawax.plot(el1[0], el1[1], lw=2, color='c', label='A')
rawax.plot(el2[0], el2[1], lw=2, color='m', label='B')
rawax.plot(el3[0], el3[1], lw=2, color='y', label='C')

rawax.set_xlim((-6, 6))
rawax.set_ylim((-6, 6))

rawax.legend(frameon=False)
rawax.set_xlabel('Neuron 1 response')
rawax.set_ylabel('Neuron 2 response')
rawax.set_title('Raw population data')

# plot pooled noise data
X1c = X1 - X1.mean(axis=0, keepdims=True)
X2c = X2 - X2.mean(axis=0, keepdims=True)
X3c = X3 - X3.mean(axis=0, keepdims=True)

el1 = cplt.compute_ellipse(X1c[:, 0], X1c[:, 1])
el2 = cplt.compute_ellipse(X2c[:, 0], X2c[:, 1])
el3 = cplt.compute_ellipse(X3c[:, 0], X3c[:, 1])

noiseax.scatter(X1c[:, 0], X1c[:, 1], s=15, alpha=0.3, color='c', edgecolor='none')
noiseax.scatter(X2c[:, 0], X2c[:, 1], s=15, alpha=0.3, color='m', edgecolor='none')
noiseax.scatter(X3c[:, 0], X3c[:, 1], s=15, alpha=0.3, color='y', edgecolor='none')

noiseax.plot(el1[0], el1[1], lw=2, color='c')
noiseax.plot(el2[0], el2[1], lw=2, color='m')
noiseax.plot(el3[0], el3[1], lw=2, color='y')

# compute noise axis
pca = PCA(n_components=1)
pca.fit(np.concatenate((X1c, X2c, X3c), axis=0))
noise_axis = pca.components_[0] * 3
noiseax.plot([0, noise_axis[0]], [0, noise_axis[1]], 'k-', lw=2, label='Noise Axis')
noiseax.plot([0, -noise_axis[0]], [0, -noise_axis[1]], 'k-', lw=2)

noiseax.set_xlim((-6, 6))
noiseax.set_ylim((-6, 6))

noiseax.legend(frameon=False)
noiseax.set_xlabel('Neuron 1 response')
noiseax.set_ylabel('Neuron 2 response')
noiseax.set_title('Pooled noise data')

# rotate data onto TDR axes
du = X1.mean(axis=0) - X2.mean(axis=0)
du /= np.linalg.norm(du)
noise_axis /= np.linalg.norm(noise_axis)
noise_on_dec = (np.dot(noise_axis, du)) * du
orth_ax = noise_axis - noise_on_dec
orth_ax /= np.linalg.norm(orth_ax)
tdr = np.stack((du, orth_ax))

X1_tdr = np.matmul(X1 - np.concatenate((X1, X2), axis=0).mean(axis=0, keepdims=True), tdr.T)
X2_tdr = np.matmul(X2 - np.concatenate((X1, X2), axis=0).mean(axis=0, keepdims=True), tdr.T)

el1 = cplt.compute_ellipse(X1_tdr[:, 0], X1_tdr[:, 1])
el2 = cplt.compute_ellipse(X2_tdr[:, 0], X2_tdr[:, 1])

tdrax.scatter(X1_tdr[:, 0], X1_tdr[:, 1], s=15, alpha=0.3, color='c', edgecolor='none')
tdrax.scatter(X2_tdr[:, 0], X2_tdr[:, 1], s=15, alpha=0.3, color='m', edgecolor='none')

tdrax.plot(el1[0], el1[1], lw=2, color='c')
tdrax.plot(el2[0], el2[1], lw=2, color='m')

# plot relevant axes in tdr space
tdr_noise = noise_axis.dot(tdr)
tdrax.plot([0, 1], [0, 0], 'grey', lw=2, label=r"$\Delta \mu$")
tdrax.plot([0, 0], [0, 1], linestyle='dotted', lw=2, color='k')
tdrax.plot([0, -tdr_noise[0]], [0, -tdr_noise[1]], 'k-', lw=2, label='Noise Axis')

tdrax.legend(frameon=False)
tdrax.set_xlabel(r"$TDR_1 (\Delta \mu_{A, B})$")
tdrax.set_ylabel(r"$TDR_2$")
tdrax.set_title('Roation of A and B \n into TDR space')

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()
