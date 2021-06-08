""" 
Helper functions specifically for figures in directory: figures_final
"""
import numpy as np
def get_cov_matrices(rec, sig='resp'):
    """
    Return small / large pupil noise covariance matrices
        1) z-score data across all stims
        2) extract stims
        3) stack / compute cov
    """
    r = rec.copy()
    r = r.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    r = r.apply_mask()

    lgmask = r['mask_large']._data.squeeze()
    smmask = r['mask_small']._data.squeeze()

    lg_residual = r[sig]._data[:, lgmask] - r['psth_sp']._data[:, lgmask]
    sm_residual = r[sig]._data[:, smmask] - r['psth_sp']._data[:, smmask]

    return np.cov(lg_residual), np.cov(sm_residual)