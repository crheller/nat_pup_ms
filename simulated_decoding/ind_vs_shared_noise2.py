"""
Simulate changes in variance in the dDR space for a variety of different conditions
    modulating indpendent noise only
    modulating shared noise only
    each for a variety of different numbers of neurons (always keep trial n high)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from charlieTools.dim_reduction import TDR
from charlieTools.plotting import compute_ellipse

np.random.seed(123)

nNeurons = np.arange(1, 100, 2)
Ntrials = 1000

cols = ['low_tot_var', 'low_tdr_var', 'high_tot_var', 'high_tdr_var']
indRes = pd.DataFrame(index=nNeurons, columns=cols)
sharRes = pd.DataFrame(index=nNeurons, columns=cols)

for Ndim in nNeurons:
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
    ind_noise = np.random.normal(0, 0.5, X_raw.shape)
    x1_noise = ind_noise * 0.1
    x2_noise = ind_noise * 5

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

    # low var
    low_tot_var = round(np.var(X1, axis=(1,2)).sum(), 2)
    low_tdr_var = round(np.var(np.stack([x11, x12]), axis=(0, 1)).sum(), 2)

    # high var
    high_tot_var = round(np.var(X2, axis=(1,2)).sum(), 2)
    high_tdr_var = round(np.var(np.stack([x21, x22]), axis=(0, 1)).sum(), 2)

    indRes.at[Ndim, 'low_tot_var'] = low_tot_var
    indRes.at[Ndim, 'low_tdr_var'] = low_tdr_var
    indRes.at[Ndim, 'high_tot_var'] = high_tot_var
    indRes.at[Ndim, 'high_tdr_var'] = high_tdr_var

    # ===================== modulate shared variance ===========================
    evecs = np.concatenate((diff_cor, pc1), axis=0)
    cov = evecs.T.dot(evecs)
    _X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
    _X1 = _X + u[0, :]
    _X2 = _X + u[1, :]
    X1 = np.stack((_X1, _X2)).transpose([-1, 1, 0])
    # add random noise to data matrix to make things behave well
    X1 += np.random.normal(0, 0.5, X1.shape)

    # twice as much shared variance
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

    # low var
    low_tot_var = round(np.var(X1, axis=(1,2)).sum(), 2)
    low_tdr_var = round(np.var(np.stack([x11, x12]), axis=(0, 1)).sum(), 2)

    # high var
    high_tot_var = round(np.var(X2, axis=(1,2)).sum(), 2)
    high_tdr_var = round(np.var(np.stack([x21, x22]), axis=(0, 1)).sum(), 2)

    sharRes.at[Ndim, 'low_tot_var'] = low_tot_var
    sharRes.at[Ndim, 'low_tdr_var'] = low_tdr_var
    sharRes.at[Ndim, 'high_tot_var'] = high_tot_var
    sharRes.at[Ndim, 'high_tdr_var'] = high_tdr_var



f, ax = plt.subplots(2, 1, figsize=(6, 8))

ax[0].set_title(r"$\Delta$ Total population variance")
ax[0].plot(nNeurons, indRes['high_tot_var'] - indRes['low_tot_var'], label='Mod. indpendent noise')
ax[0].plot(nNeurons, sharRes['high_tot_var'] - sharRes['low_tot_var'], label='Mod. shared noise')
ax[0].legend()

ax[1].set_title(r"$\Delta dDR$ space variance")
ax[1].plot(nNeurons, indRes['high_tdr_var'] - indRes['low_tdr_var'], label='Mod. indpendent noise')
ax[1].plot(nNeurons, sharRes['high_tdr_var'] - sharRes['low_tdr_var'], label='Mod. shared noise')
ax[1].legend()

ax[1].set_xlabel('Number of neurons')

f.tight_layout()

plt.show()