"""
First step, do PCA on residual to get components that reflect correlated activity.

Next, do 1st order PLS to find pupil dims.

Finally, tranform PCs to power and do PLS again to determine which dims 
have power that correlates with pupil.

Idea behind this is to avoid the complications of trying to do some sort of PLS
with the power of the spikes.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from scipy.optimize import nnls
import scipy.ndimage.filters as sf

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr

from nems_lbhb.baphy import parse_cellid

site = 'TAR010c' #'DRX006b.e1:64'
batch = 289
ops = {'batch': batch, 'cellid': site}
xmodel = 'ns.fs100.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'
path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'
n_components = 30
low = 0.5
high = 10

cells, _ = parse_cellid(ops)
rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                    cache_path=path, recache=False)
ff_residuals = rec['resp']._data - rec['psth_sp']._data
ff_residuals = bandpass_filter_resp(ff_residuals, low, high, fs=100, boxcar=True)
rec['ff_residuals'] = rec['resp']._modified_copy(ff_residuals)
rec = rec.apply_mask()

pupil = rec['pupil']._data
raw_residual = rec['resp']._data - rec['psth_sp']._data

# first, do full PCA on residuals
pca = PCA()
pca.fit(raw_residual.T)
pca_transform = raw_residual.T.dot(pca.components_.T).T

# do first order regression
X = pca_transform
X = scale(X, with_mean=True, with_std=True, axis=-1)
y = scale(pupil, with_mean=True, with_std=True, axis=-1)
lr = LinearRegression(fit_intercept=True)
lr.fit(X.T, y.squeeze())
first_order_weights = lr.coef_
fow_norm = lr.coef_ / np.linalg.norm(lr.coef_)

# do second order regression
ff = rec['ff_residuals']._data
# get power of each neuron
power = []
for n in range(ff.shape[0]):
    _ff = ff[[n], :]
    t, _ffsw = sliding_window(_ff, fs=100, window_size=4, step_size=2)
    _ffpower = np.sum(_ffsw**2, axis=-1) / _ffsw.shape[-1]
    power.append(_ffpower)
power = np.stack(power)
t, _p = sliding_window(pupil, fs=100, window_size=4, step_size=2)
pds = np.mean(_p, axis=-1)[np.newaxis, :]

power = scale(power, with_mean=True, with_std=True, axis=-1)
pds = scale(pds, with_mean=True, with_std=True, axis=-1)

# do nnls regression to avoid to sign ambiguity due to power conversion
x, r = nnls(-power.T, pds.squeeze())
second_order_weights = x
sow_norm = x / np.linalg.norm(x)

# now, compare the two dimensions, 
# the amount of variance explained by each
# and the overall PCA spcae




