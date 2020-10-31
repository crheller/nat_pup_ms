"""
Simulate changes in single cell (independent) variance. Determine how this looks
in the dDR space. i.e. are changes in variance in TDR space due to correlations 
(second order) or reduced indpendent variance (first order)
"""

import numpy as np
import matplotlib.pyplot as plt
from charlieTools.dim_reduction import TDR
from charlieTools.plotting import compute_ellipse

np.random.seed(123)

Ndim = 10
Ntrials= 5000
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

# simulate four different datasets:
    # one vs. two differ in strength of INDEPENDENT NOISE
    # three vs. four differ in strength of SHARED NOISE (pc1 / pc2)


# ==================== modulate independent variance ======================
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ind_noise = np.random.normal(0, 0.5, X_raw.shape)
x1_noise = ind_noise * 0.1
x2_noise = ind_noise * 2

# dataset one
X1 = X_raw + x1_noise
# dataset two
X2 = X_raw + x2_noise

# fit dDR to all data
xall = np.concatenate((X1, X2), axis=1)
tdr = TDR(tdr2_init=noise_axis)
tdr.fit(xall[:, :, 0].T, xall[:, :, 1].T)

x11 = X1[:,:,0].T.dot(tdr.weights.T)
x12 = X1[:,:,1].T.dot(tdr.weights.T)
x21 = X2[:,:,0].T.dot(tdr.weights.T)
x22 = X2[:,:,1].T.dot(tdr.weights.T)

ax[0].scatter(x11[:, 0], x11[:, 1], color='tab:blue', s=10, edgecolor='white')
el = compute_ellipse(x11[:, 0], x11[:, 1])
ax[0].plot(el[0], el[1], color='tab:blue', lw=2)
ax[0].scatter(x12[:, 0], x12[:, 1], color='tab:orange', s=10, edgecolor='white')
el = compute_ellipse(x12[:, 0], x12[:, 1])
ax[0].plot(el[0], el[1], color='tab:orange', lw=2)
ax[0].set_xlabel(r"$dDR_1 (\Delta \mu)$")
ax[0].set_ylabel(r"$dDR_2$")
tot_var = round(np.var(X1, axis=(1,2)).sum(), 2)
tdr_var = round(np.var(np.stack([x11, x12]), axis=(0, 1)).sum(), 2)
ax[0].set_title(f"Low\nfull dataset var: {tot_var}, dDR var: {tdr_var}")

ax[1].scatter(x21[:, 0], x21[:, 1], color='tab:blue', s=10, edgecolor='white')
el = compute_ellipse(x21[:, 0], x21[:, 1])
ax[1].plot(el[0], el[1], color='tab:blue', lw=2)
ax[1].scatter(x22[:, 0], x22[:, 1], color='tab:orange', s=10, edgecolor='white')
el = compute_ellipse(x22[:, 0], x22[:, 1])
ax[1].plot(el[0], el[1], color='tab:orange', lw=2)
ax[1].set_xlabel(r"$dDR_1 (\Delta \mu)$")
ax[1].set_ylabel(r"$dDR_2$")
tot_var = round(np.var(X2, axis=(1,2)).sum(), 2)
tdr_var = round(np.var(np.stack([x21, x22]), axis=(0, 1)).sum(), 2)
ax[1].set_title(f"High\nfull dataset var: {tot_var}, dDR var: {tdr_var}")

f.canvas.set_window_title("Independent variance")


f.tight_layout()


# ===================== modulate shared variance ===========================
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

evecs = np.concatenate((diff_cor, pc1), axis=0)
cov = evecs.T.dot(evecs)
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
_X1 = _X + u[0, :]
_X2 = _X + u[1, :]
X1 = np.stack((_X1, _X2)).transpose([-1, 1, 0])
# add random noise to data matrix to make things behave well
X1 += np.random.normal(0, 0.5, X1.shape)

evecs = np.concatenate((diff_cor*2, pc1*2), axis=0)
cov = evecs.T.dot(evecs)
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
_X1 = _X + u[0, :]
_X2 = _X + u[1, :]
X2 = np.stack((_X1, _X2)).transpose([-1, 1, 0])
# add random noise to data matrix to make things behave well
X2 += np.random.normal(0, 0.5, X2.shape)

# fit dDR to all data
xall = np.concatenate((X1, X2), axis=1)
tdr = TDR(tdr2_init=noise_axis)
tdr.fit(xall[:, :, 0].T, xall[:, :, 1].T)

x11 = X1[:,:,0].T.dot(tdr.weights.T)
x12 = X1[:,:,1].T.dot(tdr.weights.T)
x21 = X2[:,:,0].T.dot(tdr.weights.T)
x22 = X2[:,:,1].T.dot(tdr.weights.T)

ax[0].scatter(x11[:, 0], x11[:, 1], color='tab:blue', s=10, edgecolor='white')
el = compute_ellipse(x11[:, 0], x11[:, 1])
ax[0].plot(el[0], el[1], color='tab:blue', lw=2)
ax[0].scatter(x12[:, 0], x12[:, 1], color='tab:orange', s=10, edgecolor='white')
el = compute_ellipse(x12[:, 0], x12[:, 1])
ax[0].plot(el[0], el[1], color='tab:orange', lw=2)
ax[0].set_xlabel(r"$dDR_1 (\Delta \mu)$")
ax[0].set_ylabel(r"$dDR_2$")
tot_var = round(np.var(X1, axis=(1,2)).sum(), 2)
tdr_var = round(np.var(np.stack([x11, x12]), axis=(0, 1)).sum(), 2)
ax[0].set_title(f"Low\nfull dataset var: {tot_var}, dDR var: {tdr_var}")

ax[1].scatter(x21[:, 0], x21[:, 1], color='tab:blue', s=10, edgecolor='white')
el = compute_ellipse(x21[:, 0], x21[:, 1])
ax[1].plot(el[0], el[1], color='tab:blue', lw=2)
ax[1].scatter(x22[:, 0], x22[:, 1], color='tab:orange', s=10, edgecolor='white')
el = compute_ellipse(x22[:, 0], x22[:, 1])
ax[1].plot(el[0], el[1], color='tab:orange', lw=2)
ax[1].set_xlabel(r"$dDR_1 (\Delta \mu)$")
ax[1].set_ylabel(r"$dDR_2$")
tot_var = round(np.var(X2, axis=(1,2)).sum(), 2)
tdr_var = round(np.var(np.stack([x21, x22]), axis=(0, 1)).sum(), 2)
ax[1].set_title(f"High\nfull dataset var: {tot_var}, dDR var: {tdr_var}")

f.canvas.set_window_title("Shared variance")

f.tight_layout()


plt.show()