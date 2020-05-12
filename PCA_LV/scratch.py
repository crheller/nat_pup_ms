"""
PCs over all residuals. How much is each 
    1) each PC correlated with pupil
    2) power of each PC correlated with pupil

Also try on bandpass filtered residuals...
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp
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

shuff_residual = raw_residual.copy().T
np.random.shuffle(shuff_residual)
shuff_residual = shuff_residual.T

# do PCA on residuals
pca = PCA()
pca.fit(raw_residual.T)

f, ax = plt.subplots(1, 3, figsize=(12, 4))

# scree plot
ax[0].plot(pca.explained_variance_ratio_, 'o-', label='raw')
ax[0].set_ylabel('% Variance explained')
ax[0].set_xlabel('PC')
ax[0].legend(frameon=False)

# correlation with pupil
transform = raw_residual.T.dot(pca.components_.T)
shuf_transform = shuff_residual.T.dot(pca.components_.T)
corr = [abs(np.corrcoef(transform[:, i], pupil)[0, 1]) for i in range(residual.shape[0])]
corr_noise_floor = [abs(np.corrcoef(shuf_transform[:, i], pupil)[0, 1]) for i in range(residual.shape[0])]
ax[1].plot(corr, 'o-', label='raw')
ax[1].plot(corr_noise_floor, 'o-', label='shuffled')
ax[1].set_ylabel('Corr. Coef w/ pupil')
ax[1].set_xlabel('PC')
ax[1].legend(frameon=False)

# correlation of power with pupil
transform_filter = bandpass_filter_resp(transform.T, low, high, fs=100)
# TODO convert to power over sliding window...

f.tight_layout()

plt.show()

