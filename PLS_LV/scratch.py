"""
Idea is to subtract first order model prediction and then use method based on PLS regression / NIPALS algorithm
to find the dimension(s) whose *power* most strongly covaries with pupil. 
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale

from charlieTools.preprocessing import generate_state_corrected_psth
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr

from nems_lbhb.baphy import parse_cellid

site = 'DRX006b.e1:64'
batch = 289
ops = {'batch': batch, 'cellid': site}
xmodel = 'ns.fs100.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'
path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'
pls_correction = True
n_components = 8
low = 0.1
high = 25

cells, _ = parse_cellid(ops)
rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                    cache_path=path, recache=False)
rec = rec.apply_mask()

pupil = rec['pupil']._data
residual = rec['resp']._data - rec['psth']._data
raw_residual = rec['resp']._data - rec['psth_sp']._data

if pls_correction:
    X = raw_residual
    y = pupil
    pls = PLSRegression(n_components=n_components)
    pls.fit(X.T, y.T)

    # plot correlation of xscores with pupil as fn of component
    # goal is to see how many dims have power correlating with pupil
    cor = []
    for d in range(pls.x_weights_.shape[-1]):
        _c = abs(np.corrcoef(y, pls.x_scores_[:, d])[0, 1])
        cor.append(_c)

    f, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(cor, 'o-')
    ax[0].set_xlabel('PLS dimension')
    ax[0].set_ylabel(r"$corr(p(t), PLS_{i}(t))$")

    # figure out fraction of raw_residual variance explained by 1st order pupil, and each PLS dimension
    total_var = np.var(raw_residual)
    pls_var = []
    for p in range(pls.x_weights_.shape[-1]):
        v = (np.var(residual.T.dot(pls.x_weights_[:,p])[:, np.newaxis] @ pls.x_weights_[:,[p]].T) / total_var) * 100
        pls_var.append(v)

    # percent variance as function of PLS dimensions
    variance = pls_var
    xticks = np.arange(0, len(variance))
    ax[1].axhline(0, linestyle='--', color='grey')
    ax[1].bar(xticks, variance, edgecolor='k', width=0.7)
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(np.arange(1, len(variance)+1).tolist())

    ax[1].set_ylabel('% variance explained')
    ax[1].set_xlabel('PLS Dimension')

    f.canvas.set_window_title('1st order PLS')

    f.tight_layout()

    # subtract out 1st order dimensions
    x1 = (raw_residual.T.dot(pls.x_weights_[:,:3]) @ pls.x_weights_[:,:3].T).T
    residual = raw_residual - x1

    first_order_weights = pls.x_weights_


# Calculate power in X
f, t, s = ss.spectrogram(raw_residual, fs=100, axis=-1)
f_idx = np.argwhere((f > low) & (f < high))
X = s[:, f_idx, :].sum(axis=1).squeeze()
X = scale(X, with_mean=True, with_std=True, axis=-1)

# downsample pupil
y = ss.resample(pupil, X.shape[-1], axis=-1)
y = scale(y, with_mean=True, with_std=True, axis=-1)

pls = PLSRegression(n_components=n_components)
pls.fit(X.T, y.T)

second_order_weights = pls.x_weights_

# plot results

# plot correlation of xscores with pupil as fn of component
# goal is to see how many dims have power correlating with pupil
cor = []
rcor = []
rt = scale(raw_residual, with_mean=True, with_std=True)
rt = raw_residual
for d in range(pls.x_weights_.shape[-1]):
    _c = abs(np.corrcoef(y, pls.x_scores_[:, d])[0, 1])
    # get correlation of raw data projected onto weights and transformed to power
    proj = rt.T.dot(pls.x_weights_[:, d])
    _, _, st = ss.spectrogram(proj, fs=100, axis=-1)
    st = st[f_idx, :].mean(axis=0)
    _cr = abs(np.corrcoef(y, st)[0, 1])

    cor.append(_c)
    rcor.append(_cr)

f, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(cor, 'o-', label='PLS score')
ax[0].plot(rcor, 'o-', label='raw data projection')
ax[0].set_xlabel('PLS dimension')
ax[0].set_ylabel(r"$corr(p(t), PLS_{i}(t))$")

# figure out fraction of raw_residual variance explained by 1st order pupil, and each PLS dimension
total_var = np.var(raw_residual)
first_order_var = (np.var(raw_residual - residual) / total_var) * 100
pls_var = []
for p in range(pls.x_weights_.shape[-1]):
    v = (np.var(raw_residual.T.dot(pls.x_weights_[:,p])[:, np.newaxis] @ pls.x_weights_[:,[p]].T) / total_var) * 100
    pls_var.append(v)

# percent variance as function of PLS dimensions
variance = [first_order_var] + pls_var
xticks = np.arange(0, len(variance))
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].bar(xticks, variance, edgecolor='k', width=0.7)
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(['1st order']+np.arange(1, len(variance)).tolist())

ax[1].set_ylabel('% variance explained')
ax[1].set_xlabel('PLS Dimension')

f.canvas.set_window_title('Second order PLS')

f.tight_layout()

if pls_correction:
    # compare first / second order dimensions
    # by plotting cosine similarity for all pairwise dims
    cos_matrix = abs(first_order_weights.T.dot(second_order_weights))
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    im = ax.imshow(cos_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=1)

    ax.set_xlabel('First order PLS dim')
    ax.set_ylabel('Second order PLS dim')
    ax.set_title(r"$cos(PLS_{i, 1st}, PLS_{j, 2nd})$")
    
    f.colorbar(im, ax=ax)

    f.tight_layout()


# plot the first PLS 2nd order dimenion scores, and projection
f, ax = plt.subplots(3, 1, figsize=(8, 6))

ax[0].plot(pupil.T, label='pupil')
ax[0].legend(frameon=False)

ax[1].plot(pls.x_scores_[:, 0], label='PLS1 scores (power)')
ax[1].legend(frameon=False)

ax[2].plot(residual.T.dot(pls.x_weights_[:, [0]]).squeeze(), label='PLS1 projection')
ax[2].legend(frameon=False)

f.tight_layout()

# plot correlation of each neuron with raw_residual and corrected residual
pr = []
raw = []
for n in range(residual.shape[0]):
    pr.append(np.corrcoef(pupil, residual[n, :])[0, 1])
    raw.append(np.corrcoef(pupil, raw_residual[n, :])[0, 1])
bins = np.arange(-0.1, 0.1, 0.01)
f, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.hist([raw, pr], rwidth=0.7, edgecolor='none', bins=bins)
ax.legend(['raw', 'pupil-corrected'])
ax.set_xlabel(r"$corr(p(t), r_{i}(t))$")
ax.set_ylabel(r"$N$ Cells")
f.tight_layout()

plt.show()