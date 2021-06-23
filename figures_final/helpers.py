""" 
Helper functions specifically for figures in directory: figures_final
"""
import numpy as np
def get_cov_matrices(rec, sig='resp', sub='pred0', stims=None, ss=0):
    """
    Return small / large pupil noise covariance matrices
        1) z-score data across all stims
        2) extract stims
        3) stack / compute cov
    """
    r = rec.copy()
    r = r.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    r = r.apply_mask()
    all_big = [sig for sig in r.signals.keys() if sig.startswith('mask_') & sig.endswith('_lg')]
    all_small = [sig for sig in r.signals.keys() if sig.startswith('mask_') & sig.endswith('_sm')]
    if stims is not None:
        big_stims = np.array(all_big)[np.array(stims)]
        small_stims = np.array(all_small)[np.array(stims)]
    else:
        big_stims = all_big
        small_stims = all_small
    if len(big_stims)>1:
        lgmask = np.concatenate([r[sig]._data for sig in all_big if sig in big_stims], axis=0).sum(axis=0).astype(bool)
        smmask = np.concatenate([r[sig]._data for sig in all_small if sig in small_stims], axis=0).sum(axis=0).astype(bool)
    else:
        lgmask = r[big_stims[0]]._data.squeeze()
        smmask = r[small_stims[0]]._data.squeeze()
    #lgmask = r['mask_large']._data.squeeze()
    #smmask = r['mask_small']._data.squeeze()

    lg_residual = r[sig]._data[:, lgmask] - r[sub]._data[:, lgmask]
    sm_residual = r[sig]._data[:, smmask] - r[sub]._data[:, smmask]

    if ss > 0:
        u,s,vh = np.linalg.svd(np.cov(lg_residual))
        Ulg = u[:,:ss] @ u[:,:ss].T
        u,s,vh = np.linalg.svd(np.cov(sm_residual))
        Usm = u[:,:ss] @ u[:,:ss].T
        return Ulg, Usm
    else:
        return np.cov(lg_residual), np.cov(sm_residual)