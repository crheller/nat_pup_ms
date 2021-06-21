"""
LV defined as first (positive) PC of the difference between small pupil
covariance and large pupil covariance. e.g. the dimension that explains
decreased noise correlations.

These get loaded by the decoding analysis (if specified in the modelname)
"""

from global_settings import ALL_SITES, PEG_SITES, CPN_SITES, HIGHR_SITES

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import pandas as pd 
import os 
import scipy.stats as ss
import scipy.ndimage.filters as sf

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.preprocessing as cpreproc

from nems_lbhb.baphy_io import parse_cellid
from nems_lbhb.preprocessing import create_pupil_mask
import nems.db as nd

# A1 data
sites = CPN_SITES #+ PEG_SITES + CPN_SITES HIGHR_SITES + 
batches = [331]*len(CPN_SITES) #[289]*len(HIGHR_SITES) + [331]*len(CPN_SITES) #+ [323]*len(PEG_SITES) + [331]*len(CPN_SITES)
zscore = True
move_mask = True

lv_dict = {}
for batch, site in zip(batches, sites):
    print('Analyzing site {}'.format(site))
    if site in ['BOL005c', 'BOL006b']:
        batch = 294

    lv_dict[site+str(batch)] = {}

    fs = 4
    ops = {'batch': batch, 'cellid': site}
    xmodel = 'ns.fs{}.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'.format(fs)
    recache=False
    if batch == 294:
        xmodel = xmodel.replace('ns.fs4.pup', 'ns.fs4.pup.voc')
    elif batch == 331:
        if move_mask:
            xmodel = xmodel.replace('-hrc', '-epcpn-mvm-hrc')
        else:
            xmodel = xmodel.replace('-hrc', '-epcpn-hrc')
        recache = False
    path = f'/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/{batch}/'

    if not os.path.isdir(path):
        os.mkdir(path)

    cells, _ = parse_cellid(ops)
    rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                        cache_path=path, gain_only=False, recache=recache)
    rec = rec.apply_mask(reset_epochs=True)
    pupil = rec['pupil']._data.squeeze()
    epochs = [e for e in rec.epochs.name.unique() if 'STIM' in e]

    rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

    # ===================================== perform analysis on raw data =======================================
    rec_bp = rec.copy()
    ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_bp = create_pupil_mask(rec_bp, **ops)
    ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_sp = rec.copy()
    rec_sp = create_pupil_mask(rec_sp, **ops)

    real_dict_small = rec_sp['resp'].extract_epochs(epochs, mask=rec_sp['mask'], allow_incomplete=True)
    real_dict_big = rec_bp['resp'].extract_epochs(epochs, mask=rec_bp['mask'], allow_incomplete=True)
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

    # ==============  save all the "raw" results (new 6.2.2021) for post hoc significance testing etc. ==========
    lv_dict[site+str(batch)]['raw'] = {}
    lv_dict[site+str(batch)]['raw']['evecs'] = evecs
    lv_dict[site+str(batch)]['raw']['evals'] = evals

    lv_dict[site+str(batch)]['beta2'] = beta2
    lv_dict[site+str(batch)]['beta2_lambda'] = evals[0]

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

    lv_dict[site+str(batch)]['max_var_pc'] = np.argmax(var)
    lv_dict[site+str(batch)]['b2_dot_pc1'] = pca.components_[0].dot(evecs[:,0])
    lv_dict[site+str(batch)]['b2_tot_var_ratio'] = np.sum(var)
    lv_dict[site+str(batch)]['b2_var_pc1_ratio'] = np.sum(var) / pca.explained_variance_ratio_[0]

    # ===================================== perform analysis on shuff data =======================================
    # do beta2 analysis 20 times on shuffled pupil to determine if first eval is significant pup dimension
    np.random.seed(123)
    shuffled_eval1 = []
    shuffled_evals_all = []
    shuffled_evecs_all = []
    niters = 100
    for k in range(niters):
        #pupil = rec['pupil']._data.copy().squeeze()
        #np.random.shuffle(pupil)
        # generate random smooth process
        data = np.random.normal(0, 1, rec['pupil'].shape[-1])
        pupil = sf.gaussian_filter1d(data, sigma=16)

        rec['pupil'] = rec['pupil']._modified_copy(pupil[np.newaxis, :])

        rec_bp = rec.copy()
        ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
        rec_bp = create_pupil_mask(rec_bp, **ops)
        ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
        rec_sp = rec.copy()
        rec_sp = create_pupil_mask(rec_sp, **ops)

        shuf_dict_small = rec_sp['resp'].extract_epochs(epochs, mask=rec_sp['mask'], allow_incomplete=True)
        shuf_dict_big = rec_bp['resp'].extract_epochs(epochs, mask=rec_bp['mask'], allow_incomplete=True)

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

        shuf_small = np.cov(shuf_matrix_small)
        np.fill_diagonal(shuf_small, 0)
        shuf_big = np.cov(shuf_matrix_big)
        np.fill_diagonal(shuf_big, 0)
        shuf_diff = shuf_small - shuf_big
        shuf_evals, shuf_evecs = np.linalg.eig(shuf_diff)

        sortidx = np.argsort(shuf_evals)[::-1]

        shuffled_evecs_all.append(shuf_evecs[:, sortidx])
        shuffled_evals_all.append(shuf_evals[sortidx])
        shuffled_eval1.append(shuf_evals[0])

    # ==============  save all the "raw" results (new 6.2.2021) for post hoc significance testing etc. ==========
    lv_dict[site+str(batch)]['raw']['evecs_shuff'] = np.stack(shuffled_evecs_all)
    lv_dict[site+str(batch)]['raw']['evals_shuff'] = np.stack(shuffled_evals_all)

    mean_shuf_beta2_lambda = np.mean(shuffled_eval1)
    mean_all = np.mean(np.stack(shuffled_evals_all), axis=0)
    sem_all = np.std(np.stack(shuffled_evals_all), axis=0) #/ np.sqrt(niters)
    sem_shuf_beta2_lambda = np.std(shuffled_eval1) #/ np.sqrt(niters)

    lv_dict[site+str(batch)]['shuf_beta2_lambda'] = mean_shuf_beta2_lambda
    lv_dict[site+str(batch)]['shuf_beta2_lambda_sd'] = sem_shuf_beta2_lambda

    # plot and save for quick inspection of sites
    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(evals, '.-', label='True diff')
    ax.fill_between(range(len(mean_all)), mean_all+sem_all, mean_all-sem_all, color='tab:orange', alpha=0.8, label='Shuff. Pupil')
    ax.set_ylabel(r"$\lambda$")
    ax.set_xlabel(r"component (PC)")
    ax.set_title(f"Eigenspectrum of diff covariance \n site: {site}, batch: {batch}")
    ax.legend(frameon=False)
    f.tight_layout()

    figpath = f'/auto/users/hellerc/results/nat_pupil_ms/LV/figures/{batch}_{site}.png'
    f.savefig(figpath)

    # figure out if dim is significant
    if (lv_dict[site+str(batch)]['beta2_lambda'] - lv_dict[site+str(batch)]['shuf_beta2_lambda']) > lv_dict[site+str(batch)]['shuf_beta2_lambda_sd']: lv_dict[site+str(batch)]['beta2_sig'] = True
    else: lv_dict[site+str(batch)]['beta2_sig'] = False

    # get significance with sign test rather than using standard deviation
    if ss.wilcoxon(lv_dict[site+str(batch)]['beta2_lambda']-np.stack(shuffled_evals_all)[:, 0]).pvalue<0.05:
        lv_dict[site+str(batch)]['beta2_sig_wilcox'] = True
    else:
        lv_dict[site+str(batch)]['beta2_sig_wilcox'] = False

    if (sum(lv_dict[site+str(batch)]['beta2_lambda']>np.stack(shuffled_evals_all)[:,0]) / niters) > 0.95:
        lv_dict[site+str(batch)]['beta_sig_shufTest'] = True
    else:
        lv_dict[site+str(batch)]['beta_sig_shufTest'] = False

    # use model pred to get beta1
    residual = rec['psth']._data - rec['psth_sp']._data
    if zscore:
        residual = residual - residual.mean(axis=-1, keepdims=True)
        sd = residual.mean(axis=-1)
        sd[sd==0] = 1
        residual = (residual.T / sd).T
    # get first PC of residual
    pca2 = PCA()
    pca2.fit(residual.T)
    beta1 = pca2.components_[0, :]

    lv_dict[site+str(batch)]['beta1'] = beta1[:, np.newaxis]

    lv_dict[site+str(batch)]['b1_dot_b2'] = beta1.dot(beta2)

    path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'
    df = pd.concat([pd.read_csv(os.path.join(path,'d_289_pup_sdexp.csv'), index_col=0),
                    pd.read_csv(os.path.join(path,'d_294_pup_sdexp.csv'), index_col=0),
                    pd.read_csv(os.path.join(path,'d_323_pup_sdexp.csv'), index_col=0),
                    pd.read_csv(os.path.join(path,'d_331_pup_sdexp.csv'), index_col=0)])
    df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se']) 
    gain = pd.DataFrame(df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']])
    gain.loc[:, 'site'] = [i.split('-')[0] for i in gain.index]
    gain = gain.loc[[c for c in rec['resp'].chans]]
    g = gain['gain_mod']['st.pup'].values
    try: 
        g = [float(g.strip('[]')) for g in g]
    except: 
        g = [g for g in g]

    lv_dict[site+str(batch)]['b2_corr_gain'] = np.corrcoef(g, evecs[:, 0])[0, 1]

#plt.show()   
plt.close('all')
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
