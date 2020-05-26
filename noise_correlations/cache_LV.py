"""
LV defined as first (positive) PC of the difference between small pupil
covariance and large pupil covariance. e.g. the dimension that explains
decreased noise correlations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.preprocessing as cpreproc

from nems_lbhb.baphy import parse_cellid
from nems_lbhb.preprocessing import create_pupil_mask

sites = ['TAR010c', 'TAR017b', 
        'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
        'DRX007a.e1:64', 'DRX007a.e65:128', 
        'DRX008b.e1:64', 'DRX008b.e65:128',
        'BOL005c', 'BOL006b']
zscore = False

lv_dict = {}
for site in sites:
    print('Analyzing site {}'.format(site))
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    else:
        batch = 289

    lv_dict[site] = {}

    fs = 4
    ops = {'batch': batch, 'cellid': site}
    xmodel = 'ns.fs{}.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'.format(fs)
    if batch == 294:
        xmodel = xmodel.replace('ns.fs4.pup', 'ns.fs4.pup.voc')
    path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'
    low = 0.5
    high = 2  # for filtering the projection

    cells, _ = parse_cellid(ops)
    rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                        cache_path=path, gain_only=True, recache=True)
    rec = rec.apply_mask(reset_epochs=True)
    pupil = rec['pupil']._data.squeeze()
    epochs = [e for e in rec.epochs.name.unique() if 'STIM' in e]

    rec['resp2'] = rec['resp']._modified_copy(rec['resp']._data)
    rec['pupil2'] = rec['pupil']._modified_copy(rec['pupil']._data)

    # ===================================== perform analysis on raw data =======================================
    rec_bp = rec.copy()
    ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_bp = create_pupil_mask(rec_bp, **ops)
    ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_sp = rec.copy()
    rec_sp = create_pupil_mask(rec_sp, **ops)

    real_dict_small = rec_sp['resp'].extract_epochs(epochs, mask=rec_sp['mask'])
    real_dict_big = rec_bp['resp'].extract_epochs(epochs, mask=rec_bp['mask'])
    real_dict_all = rec['resp'].extract_epochs(epochs)
    pred_dict_all = rec['psth'].extract_epochs(epochs)

    real_dict_small = cpreproc.zscore_per_stim(real_dict_small, d2=real_dict_small, with_std=zscore)
    real_dict_big = cpreproc.zscore_per_stim(real_dict_big, d2=real_dict_big, with_std=zscore)
    real_dict_all = cpreproc.zscore_per_stim(real_dict_all, d2=real_dict_all, with_std=zscore)
    pred_dict_all = cpreproc.zscore_per_stim(pred_dict_all, d2=pred_dict_all, with_std=zscore)

    eps = list(real_dict_big.keys())
    nCells = real_dict_big[eps[0]].shape[1]
    for i, k in enumerate(real_dict_big.keys()):
        if i == 0:
            resp_matrix = np.transpose(real_dict_all[k], [1, 0, -1]).reshape(nCells, -1)
            resp_matrix_small = np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)
            resp_matrix_big = np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)
            pred_matrix = np.transpose(pred_dict_all[k], [1, 0, -1]).reshape(nCells, -1)
        else:
            resp_matrix = np.concatenate((resp_matrix, np.transpose(real_dict_all[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
            resp_matrix_small = np.concatenate((resp_matrix_small, np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
            resp_matrix_big = np.concatenate((resp_matrix_big, np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
            pred_matrix = np.concatenate((pred_matrix, np.transpose(pred_dict_all[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)

    nc_resp_small = resp_matrix_small
    nc_resp_big = resp_matrix_big 
    small = np.cov(nc_resp_small)
    np.fill_diagonal(small, 0)
    big = np.cov(nc_resp_big)
    np.fill_diagonal(big, 0)
    diff = small - big
    evals, evecs = np.linalg.eig(diff)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    beta2 = evecs[:, [0]]

    lv_dict[site]['beta2'] = beta2


    # use model pred to get beta1
    residual = rec['psth']._data - rec['psth_sp']._data
    # get first PC of residual
    pca2 = PCA()
    pca2.fit(residual.T)
    beta1 = pca2.components_[0, :]

    '''
    residual = rec['resp2']._data - rec['psth_sp']._data  # get rid of stimulus information
    # zscore residual
    residual = residual - residual.mean(axis=-1, keepdims=True)
    #residual = residual / residual.std(axis=-1, keepdims=True)
    rec['residual'] = rec['resp']._modified_copy(residual)
    rec['pupil'] = rec['pupil']._modified_copy(rec['pupil2']._data)

    # get large and small pupil means
    rec = rec.create_mask(True)
    ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
    rec = create_pupil_mask(rec, **ops)

    large = rec.apply_mask()['residual']._data

    rec = rec.create_mask(True)
    ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
    rec = create_pupil_mask(rec, **ops)
    small = rec.apply_mask()['residual']._data

    beta1 = large.mean(axis=-1) - small.mean(axis=-1)
    beta1 = beta1 / np.linalg.norm(beta1)
    '''

    lv_dict[site]['beta1'] = beta1[:, np.newaxis]

# pickle the results
fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/nc_based_lvs.pickle'

if zscore:
    fn = fn.replace('nc_based_lvs.pickle', 'nc_zscore_lvs.pickle')
else:
    # just mean centered data
    pass

# pickle results
with open(fn, 'wb') as handle:
    pickle.dump(lv_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Success!")