"""
LV defined as first (positive) PC of the difference between small pupil
covariance and large pupil covariance. e.g. the dimension that explains
decreased noise correlations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import pandas as pd 
import os 

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.preprocessing as cpreproc

from nems_lbhb.baphy import parse_cellid
from nems_lbhb.preprocessing import create_pupil_mask

sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
        'DRX006b.e1:64', 'DRX006b.e65:128',
        'DRX007a.e1:64', 'DRX007a.e65:128',
        'DRX008b.e1:64', 'DRX008b.e65:128',
        'BOL005c', 'BOL006b']
zscore = True

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

    cells, _ = parse_cellid(ops)
    rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                        cache_path=path, gain_only=False, recache=False)
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
    beta2_lambda = evals[0]

    lv_dict[site]['beta2'] = beta2
    lv_dict[site]['beta2_lambda'] = evals[0]

    # project rank1 data onto first eval of diff
    r1_resp = (resp_matrix.T.dot(evecs[:, 0])[:, np.newaxis] @ evecs[:, [0]].T).T

    # compute PCs over all the data
    pca = PCA()
    pca.fit(resp_matrix.T)

    # compute variance of rank1 matrix along each PC
    var = np.zeros(resp_matrix.shape[0])
    fo_var = np.zeros(pred_matrix.shape[0])
    for pc in range(0, resp_matrix.shape[0]):
        var[pc] = np.var(r1_resp.T.dot(pca.components_[pc])) / np.sum(pca.explained_variance_)
        fo_var[pc] = np.var(pred_matrix.T.dot(pca.components_[pc])) / np.sum(pca.explained_variance_)

    lv_dict[site]['max_var_pc'] = np.argmax(var)
    lv_dict[site]['b2_dot_pc1'] = pca.components_[0].dot(evecs[:,0])
    lv_dict[site]['b2_tot_var_ratio'] = np.sum(var)
    lv_dict[site]['b2_var_pc1_ratio'] = np.sum(var) / pca.explained_variance_ratio_[0]

    # ===================================== perform analysis on shuff data =======================================
    # do beta2 analysis 20 times on shuffled pupil to determine if first eval is significant pup dimension
    np.random.seed(123)
    shuffled_eval1 = []
    niters = 20
    for k in range(niters):
        pupil = rec['pupil']._data.copy().squeeze()
        np.random.shuffle(pupil)
        rec['pupil'] = rec['pupil']._modified_copy(pupil[np.newaxis, :])

        rec_bp = rec.copy()
        ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
        rec_bp = create_pupil_mask(rec_bp, **ops)
        ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
        rec_sp = rec.copy()
        rec_sp = create_pupil_mask(rec_sp, **ops)

        shuf_dict_small = rec_sp['resp'].extract_epochs(epochs, mask=rec_sp['mask'])
        shuf_dict_big = rec_bp['resp'].extract_epochs(epochs, mask=rec_bp['mask'])

        shuf_dict_small = cpreproc.zscore_per_stim(shuf_dict_small, d2=shuf_dict_small, with_std=True)
        shuf_dict_big = cpreproc.zscore_per_stim(shuf_dict_big, d2=shuf_dict_big, with_std=True)

        eps = list(shuf_dict_big.keys())
        nCells = shuf_dict_big[eps[0]].shape[1]
        eps = [e for e in eps if e in shuf_dict_small.keys()]
        for i, k in enumerate(eps):
            if i == 0:
                shuf_matrix_small = np.transpose(shuf_dict_small[k], [1, 0, -1]).reshape(nCells, -1)
                shuf_matrix_big = np.transpose(shuf_dict_big[k], [1, 0, -1]).reshape(nCells, -1)
            else:
                shuf_matrix_small = np.concatenate((shuf_matrix_small, np.transpose(shuf_dict_small[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
                shuf_matrix_big = np.concatenate((shuf_matrix_big, np.transpose(shuf_dict_big[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)

        shuf_small = np.corrcoef(shuf_matrix_small)
        shuf_big = np.corrcoef(shuf_matrix_big)
        shuf_diff = shuf_small - shuf_big
        shuf_evals, shuf_evecs = np.linalg.eig(shuf_diff)
        shuf_evals = shuf_evals[np.argsort(shuf_evals)[::-1]]

        shuffled_eval1.append(shuf_evals[0])

    mean_shuf_beta2_lambda = np.mean(shuffled_eval1)
    sem_shuf_beta2_lambda = np.std(shuffled_eval1) / np.sqrt(niters)

    lv_dict[site]['shuf_beta2_lambda'] = mean_shuf_beta2_lambda
    lv_dict[site]['shuf_beta2_lambda_sem'] = sem_shuf_beta2_lambda

    # figure out if dim is significant
    if (lv_dict[site]['beta2_lambda'] - lv_dict[site]['shuf_beta2_lambda']) > lv_dict[site]['shuf_beta2_lambda_sem']: lv_dict[site]['beta2_sig'] = True
    else: lv_dict[site]['beta2_sig'] = False

    # use model pred to get beta1
    residual = rec['psth']._data - rec['psth_sp']._data
    if zscore:
        residual = residual - residual.mean(axis=-1, keepdims=True)
        residual = residual / residual.std(axis=-1, keepdims=True)
    # get first PC of residual
    pca2 = PCA()
    pca2.fit(residual.T)
    beta1 = pca2.components_[0, :]

    lv_dict[site]['beta1'] = beta1[:, np.newaxis]

    lv_dict[site]['b1_dot_b2'] = beta1.dot(beta2)

    path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'
    df = pd.concat([pd.read_csv(os.path.join(path,'d_289_pup_sdexp.csv'), index_col=0),
                    pd.read_csv(os.path.join(path,'d_294_pup_sdexp.csv'), index_col=0)])
    df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se']) 
    gain = pd.DataFrame(df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']])
    gain.loc[:, 'site'] = [i.split('-')[0] for i in gain.index]
    gain = gain.loc[[c for c in rec['resp'].chans]]
    g = gain['gain_mod']['st.pup'].values
    g = [g for g in g]

    lv_dict[site]['b2_corr_gain'] = np.corrcoef(g, evecs[:, 0])[0, 1]

    # get largest PC 

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