"""
Tools to invert GLM fit. i.e. "regress out" a using given weights
"""

import numpy as np

def regress_gain_factor(r, r0, factor, weights):
    """
    rec['resp'] is shape N x T
    
    n is the number of factors

    Given a time varying factor with dim n x T and a set of weights of dim n x N,
    "invert" the factor to remove it from the response.

    model: r = r0 * exp(weights * factor) + E(t)

    So, invert by: r_inverted = r - (r0 * np.exp(weights * factor)) + r0
    """

    # outer product of weights / factor
    if len(factor.shape) < 2:
        factor = factor[np.newaxis, :]
    if len(weights.shape) < 2:
        weights = weights[np.newaxis, :]
    
    for i, n in enumerate(range(0, weights.shape[0])):
        if i == 0:
            op = np.matmul(factor[n, :][np.newaxis, :].T, weights[n, :][np.newaxis, :])
        else:
            op += np.matmul(factor[n, :][np.newaxis, :].T, weights[n, :][np.newaxis, :])

    r_inverted = r / np.exp(op).T

    return r_inverted

def regress_dc_factor(r, factor, weights):
    # outer product of weights / factor
    if len(factor.shape) < 2:
        factor = factor[np.newaxis, :]
    if len(weights.shape) < 2:
        weights = weights[np.newaxis, :]
    
    for i, n in enumerate(range(0, weights.shape[0])):
        if i == 0:
            op = np.matmul(factor[n, :][np.newaxis, :].T, weights[n, :][np.newaxis, :])
        else:
            op += np.matmul(factor[n, :][np.newaxis, :].T, weights[n, :][np.newaxis, :])

    r_inverted = r - op.T

    return r_inverted