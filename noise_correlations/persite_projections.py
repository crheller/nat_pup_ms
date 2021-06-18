"""
Project data from each site onto 3 different axes:
    First PC
    First delta PC
    Last delta PC
Compare variance over time with pupil
"""
from global_settings import CPN_SITES, HIGHR_SITES
from charlieTools.nat_sounds_ms import decoding
import charlieTools.preprocessing as cpreproc

from sklearn.decomposition import FactorAnalysis, PCA
import matplotlib.pyplot as plt
import numpy as np

import nems.preprocessing as preproc
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.preprocessing import fix_cpn_epochs, create_pupil_mask

sites = CPN_SITES + HIGHR_SITES
batches = [331]*len(CPN_SITES) + [289]*len(HIGHR_SITES)
zscore = True

for b, site in zip(batches, sites):
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    else:
        batch = b

    manager = BAPHYExperiment(cellid=site, batch=batch)
    options = {'rasterfs': 4, 'resp': True, 'stim': False, 'pupil': True, 'pupil_artifacts': False}
    rec = manager.get_recording(recache=True, **options)
    rec['resp'] = rec['resp'].rasterize()
    if batch==331:
        rec = fix_cpn_epochs(rec)
    
    # remove epochs
    if batch in [294, 331]:
        epochs = [epoch for epoch in rec['resp'].epochs.name.unique() if ('STIM_' in epoch) & ('Pips' not in epoch)]
    else:
        epochs = [epoch for epoch in rec['resp'].epochs.name.unique() if 'STIM_00' in epoch]
    rec = rec.and_mask(epochs)
    # if artifacts, make them perectly tile REFs
    try:
        art = rec['artifacts'].extract_epochs(epochs)
        for epoch in epochs:
            for r in range(art[epoch].shape[0]):
                if np.any(art[epoch][r]>0):
                    art[epoch][r] = 1
        rec['artifacts'] = rec['artifacts'].replace_epochs(art)
        rec['mask'] = rec['mask']._modified_copy(rec['mask']._data & (rec['artifacts']._data==0))
    except:
        pass

    all_dict = rec['resp'].extract_epochs(epochs, mask=rec['mask'])
    all_dict = cpreproc.zscore_per_stim(all_dict, d2=all_dict, with_std=zscore)
    rec['resp'] = rec['resp'].replace_epochs(all_dict, mask=rec['mask'])

    # remove spont
    rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

    # delta noise corr. axes
    rec_bp = rec.copy()
    ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_bp = create_pupil_mask(rec_bp, **ops)
    ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_sp = rec.copy()
    rec_sp = create_pupil_mask(rec_sp, **ops)

    real_dict_small = rec_sp['resp'].extract_epochs(epochs, mask=rec_sp['mask'], allow_incomplete=True)
    real_dict_big = rec_bp['resp'].extract_epochs(epochs, mask=rec_bp['mask'], allow_incomplete=True)

    real_dict_small = cpreproc.zscore_per_stim(real_dict_small, d2=real_dict_small, with_std=zscore)
    real_dict_big = cpreproc.zscore_per_stim(real_dict_big, d2=real_dict_big, with_std=zscore)

    eps = list(real_dict_big.keys())
    nCells = real_dict_big[eps[0]].shape[1]
    for i, k in enumerate(real_dict_big.keys()):
        if i == 0:
            resp_matrix_small = np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)
            resp_matrix_big = np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)
        else:
            resp_matrix_small = np.concatenate((resp_matrix_small, np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
            resp_matrix_big = np.concatenate((resp_matrix_big, np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)

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

    og_mask = rec['mask']._data.squeeze()
    rec = rec.apply_mask() 
    X = rec['resp']._data

    dproj0 = X.T.dot(evecs[:, 0])
    dproj1 = X.T.dot(evecs[:, -1])

    # get axes to project onto
    # first PC
    pca = PCA(n_components=1)
    pc1_proj = pca.fit_transform(X.T).squeeze()


    # plot projections
    f = plt.figure(figsize=(8, 6))

    pup = plt.subplot2grid((4, 5), (0, 0), colspan=4)
    p1 = plt.subplot2grid((4, 5), (1, 0), colspan=4, sharex=pup)
    p1h = plt.subplot2grid((4, 5), (1, 4), colspan=1)
    d0 = plt.subplot2grid((4, 5), (2, 0), colspan=4, sharex=pup)
    d0h = plt.subplot2grid((4, 5), (2, 4), colspan=1)
    d1 = plt.subplot2grid((4, 5), (3, 0), colspan=4, sharex=pup)
    d1h = plt.subplot2grid((4, 5), (3, 4), colspan=1)

    pup.set_title(site)
    pup.plot(rec['pupil']._data.T)

    p1.set_title("PC1 projection")
    p1.plot(np.argwhere(rec_bp['mask']._data.squeeze()[og_mask]).squeeze(), pc1_proj[rec_bp['mask']._data.squeeze()[og_mask]], '.', color='red')
    p1.plot(np.argwhere(rec_sp['mask']._data.squeeze()[og_mask]).squeeze(), pc1_proj[rec_sp['mask']._data.squeeze()[og_mask]], '.', color='blue')
    p1h.hist(pc1_proj[rec_bp['mask']._data.squeeze()[og_mask]], color='red', histtype='step', lw=2, orientation='horizontal')
    p1h.hist(pc1_proj[rec_sp['mask']._data.squeeze()[og_mask]], color='blue', histtype='step', lw=2, orientation='horizontal')

    d0.set_title("Delta1 projection")
    d0.plot(np.argwhere(rec_bp['mask']._data.squeeze()[og_mask]).squeeze(), dproj0[rec_bp['mask']._data.squeeze()[og_mask]], '.', color='red')
    d0.plot(np.argwhere(rec_sp['mask']._data.squeeze()[og_mask]).squeeze(), dproj0[rec_sp['mask']._data.squeeze()[og_mask]], '.', color='blue')
    d0h.hist(dproj0[rec_bp['mask']._data.squeeze()[og_mask]], color='red', histtype='step', lw=2, orientation='horizontal')
    d0h.hist(dproj0[rec_sp['mask']._data.squeeze()[og_mask]], color='blue', histtype='step', lw=2, orientation='horizontal')

    '''
    d1.set_title("Delta -1 projection")
    d1.plot(np.argwhere(rec_bp['mask']._data.squeeze()[og_mask]).squeeze(), dproj1[rec_bp['mask']._data.squeeze()[og_mask]], '.', color='red')
    d1.plot(np.argwhere(rec_sp['mask']._data.squeeze()[og_mask]).squeeze(), dproj1[rec_sp['mask']._data.squeeze()[og_mask]], '.', color='blue')
    '''
    d1h.hist(dproj1[rec_bp['mask']._data.squeeze()[og_mask]], color='red', histtype='step', lw=2, orientation='horizontal')
    d1h.hist(dproj1[rec_sp['mask']._data.squeeze()[og_mask]], color='blue', histtype='step', lw=2, orientation='horizontal')
    d1.imshow(rec['resp']._data, cmap='Greys', aspect='auto')

    f.tight_layout()

plt.show()