"""
simulate two neurons, one stimulus, big pupil small pupil.
Noise corr. depend on pupil. 
Find the axis along which noise corr. change, remove it, replot 
data.
"""

import numpy as np
import charlieTools.plotting as cplt
import matplotlib.pyplot as plt

bp_cov = np.array([[1, 0.01], [0.01, 1]])
sp_cov = np.array([[1, 2], [2, 1]])

bp = np.random.multivariate_normal([0, 0], bp_cov, (1000,))
sp = np.random.multivariate_normal([0, 0], sp_cov, (1000,))

f, ax = plt.subplots(2, 2)

ax[0, 0].plot(bp[:,0], bp[:,1], '.', color='firebrick', alpha=0.2)
ax[0, 0].plot(sp[:,0], sp[:,1], '.', color='navy', alpha=0.2)
bp_el = cplt.compute_ellipse(bp[:,0], bp[:,1])
sp_el = cplt.compute_ellipse(sp[:,0], sp[:,1])
ax[0, 0].plot(bp_el[0, :], bp_el[1, :], color='firebrick', lw=2)
ax[0, 0].plot(sp_el[0, :], sp_el[1, :], color='navy', lw=2)

ax[0, 0].set_aspect(cplt.get_square_asp(ax[0, 0]))

# diff of cov matrices
cov = np.cov(bp.T) - np.cov(sp.T)
ax[0, 1].imshow(cov, cmap='seismic')
ax[0, 1].set_aspect(cplt.get_square_asp(ax[0, 0]))
ax[0, 1].set_title('difference of cov matrices', fontsize=8)

# find the top eigenvector of covariance matrix
ev, eg = np.linalg.eig(cov)

# sort eg and ev by eigen eigenvalues (in descending order
# -- want the biggest negative eigenvalue)
sort_args = np.argsort(abs(ev))[::-1]
eg = eg[:, sort_args]

# plot first eigenvector of the diff on the plot (want the eigenvector
# corresponding to the largest decrease in variance)
ax[0, 0].plot([0, eg[0, 0]], [0, eg[1, 0]], 'k-', lw=3, label='delta cov axis')
ax[0, 0].legend(fontsize=8)

# regress out variance along this dimension by subtracting the projection
# onto the eigenvector and then replot
proj_ax = eg[:,0] / np.linalg.norm(eg[:,0])
bp_proj = np.matmul(bp, proj_ax)
sp_proj = np.matmul(sp, proj_ax)
bp_proj_out = np.matmul(bp_proj[:, np.newaxis], proj_ax[np.newaxis, :])
sp_proj_out = np.matmul(sp_proj[:, np.newaxis], proj_ax[np.newaxis, :])
bp_new = bp - bp_proj_out
sp_new = sp - sp_proj_out

ax[1, 0].plot(bp_new[:,0], bp_new[:,1], '.', color='firebrick', alpha=0.2)
ax[1, 0].plot(sp_new[:,0], sp_new[:,1], '.', color='navy', alpha=0.2)
bp_el = cplt.compute_ellipse(bp_new[:,0], bp_new[:,1])
sp_el = cplt.compute_ellipse(sp_new[:,0], sp_new[:,1])
ax[1, 0].plot(bp_el[0, :], bp_el[1, :], color='firebrick', lw=2)
ax[1, 0].plot(sp_el[0, :], sp_el[1, :], color='navy', lw=2)
ax[1, 0].set_xlim(ax[0, 0].get_xlim())
ax[1, 0].set_ylim(ax[0, 0].get_ylim())
ax[1, 0].set_title('Remove variability along \n covariance axis', fontsize=8)
ax[1, 0].set_aspect(cplt.get_square_asp(ax[1, 0]))

# project large / small pupil data onto this axis,
# concatenate, and plot "timecourse"
ax[1, 1].plot(range(0, bp.shape[0]), bp_proj, '.', color='firebrick')
ax[1, 1].plot(range(bp.shape[0], bp.shape[0]+sp.shape[0]), sp_proj, '.', color='navy')
ax[1, 1].set_ylabel('Covariance axis', fontsize=8)

f.tight_layout()

plt.show()