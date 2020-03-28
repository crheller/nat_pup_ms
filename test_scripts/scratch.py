import numpy as np

def cov_cost_fn(lv_weights, bp, sp):
    
    if len(lv_weights.shape) > 1:
        pass
    else:
        lv_weights = lv_weights[np.newaxis, :]

    lv_weights = lv_weights / np.linalg.norm(lv_weights)

    # project data onto weights. compute cov.
    # compute diff in cov matrices
    bp_proj = np.matmul(np.matmul(bp, lv_weights.T), lv_weights)
    sp_proj = np.matmul(np.matmul(sp, lv_weights.T), lv_weights)

    bp = bp - bp_proj
    sp = sp - sp_proj

    bp_cov = np.cov(bp.T)
    sp_cov = np.cov(sp.T)

    diff = np.sum(abs(bp_cov - sp_cov))

    return diff

def 