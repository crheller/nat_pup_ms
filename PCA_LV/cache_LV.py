"""
CRH 05/13/2020

For each site, cache:
        - first order pupil weights
        - second order pupil weights
        - residual variance explained for the two axes above, and for each PC
        - cos similarity between:
            - 1st / 2nd order weights
            - 1st order weights and PC1
            - 2nd order weights and PC1

Method for finding first / second order weights:
    - do PCA on raw residuals (resp minus psth)
    - perform MLR between PCs and pupil to get first order weights
    - convert PCs to power and perform (rectified) MLR with pupil for second order weights
    - Finally, project weights back into neuron space for comparisons with PCs and with 
            decoding axis etc.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from scipy.optimize import nnls
import pickle

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.preprocessing as cpreproc

from nems_lbhb.baphy import parse_cellid

fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/pca_regression_lvs.pickle'

sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
        'DRX006b.e1:64', 'DRX006b.e65:128',
        'DRX007a.e1:64', 'DRX007a.e65:128',
        'DRX008b.e1:64', 'DRX008b.e65:128',
        'BOL005c', 'BOL006b']

# set params for filtering and for power calcuation
window_size = 4  # s
step_size = 2    # s
low = 0.5         # Hz
high = 2         # HZ
filt_params = {
    'low_cut': low, 
    'high_cut': high,
    'window_size': window_size,
    'step_size': step_size
}

results_dict = dict.fromkeys(sites)

for site in sites:
    print('Analyzing site {}'.format(site))
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    else:
        batch = 289
    fs = 4
    ops = {'batch': batch, 'cellid': site}
    xmodel = 'ns.fs{}.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'.format(fs)
    path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'
    low = 0.5
    high = 2

    cells, _ = parse_cellid(ops)
    rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                        cache_path=path, recache=False)
    epochs = [e for e in rec.epochs.name.unique() if 'STIM' in e]
    dresp = rec['resp'].extract_epochs(epochs)
    zresp = cpreproc.zscore_per_stim(dresp, d2=None)
    zresp = rec['resp'].replace_epochs(zresp, mask=rec['mask'])
    rec['zresp'] = zresp

    ff_residuals = rec['zresp']._data.copy()
    nan_idx = np.isnan(ff_residuals[0, :])
    ff_residuals[:, nan_idx] = 0
    ff_residuals = bandpass_filter_resp(ff_residuals, low, high, fs=fs, boxcar=True)
    rec['ff_residuals'] = rec['resp']._modified_copy(ff_residuals)
    rec = rec.apply_mask()

    raw_residual = rec['zresp']._data
    pupil = rec['pupil']._data

    # first, do full PCA on residuals
    raw_residual = scale(raw_residual, with_mean=True, with_std=True)
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
        t, _ffsw = sliding_window(_ff, fs=fs, window_size=window_size, step_size=step_size)
        _ffpower = np.sum(_ffsw**2, axis=-1) / _ffsw.shape[-1]
        power.append(_ffpower)
    power = np.stack(power)
    t, _p = sliding_window(pupil, fs=fs, window_size=4, step_size=2)
    pds = np.mean(_p, axis=-1)[np.newaxis, :]

    power = scale(power, with_mean=True, with_std=True, axis=-1)
    pds = scale(pds, with_mean=True, with_std=True, axis=-1)

    # do nnls regression to avoid to sign ambiguity due to power conversion
    x, r = nnls(power.T, -pds.squeeze())
    second_order_weights = x

    if np.linalg.norm(x)==0:
        sow_norm = x
    else:
        sow_norm = x / np.linalg.norm(x)

    # project weights back into neuron space (then the can be compared with PC weights too)
    fow_nspace = pca.components_.T.dot(fow_norm)
    sow_nspace = pca.components_.T.dot(sow_norm)

    # compute cosine similarities
    cos_fow_sow = fow_nspace.dot(sow_nspace)
    cos_fow_PC1 = fow_nspace.dot(pca.components_[0])
    cos_sow_PC1 = sow_nspace.dot(pca.components_[0])

    # compute variance explained (in PC space, but exact same as if in neuron space)
    var_1st_order = np.var(pca_transform.T.dot(fow_norm)[:, np.newaxis] @ fow_norm[np.newaxis,:]) / np.var(pca_transform)
    var_2nd_order = np.var(pca_transform.T.dot(sow_norm)[:, np.newaxis] @ sow_norm[np.newaxis,:]) / np.var(pca_transform)
    pc_variance = pca.explained_variance_ratio_

    # pack results into a dict and append to list
    results = {
        'fow': fow_nspace,
        'sow': sow_nspace,
        'cos_fow_sow': cos_fow_sow,
        'cos_fow_PC1': cos_fow_PC1,
        'cos_sow_PC1': cos_sow_PC1,
        'var_1st_order': var_1st_order,
        'var_2nd_order': var_2nd_order,
        'pc_variance': pc_variance,
        'filt_params': filt_params,
        'beta1': fow_nspace[:, np.newaxis],
        'beta2': sow_nspace[:, np.newaxis],
        'site': site
    }

    results_dict[site] = results

# pickle results
with open(fn, 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
