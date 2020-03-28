import numpy as np
import copy

def pupil_only_objective(w, rec, verbose=False):
    """
    Same general idea as the `gain_only_objective` function, but this one
    has no latent variable / additional pupil constraint. Plus, it has a
    DC term. Really just for testing purposes at this point. See if can get
    similar performance to nems models.
    Just fits a set of weights for pupil alone and tries to minimize mse
    between pred and resp.
    """

    newrec = copy.deepcopy(rec)

    nCells = len(newrec['resp'].chans)

    w = w.reshape(3, nCells)
    g1 = w[0, :]
    d1 = w[1, :]

    # baseline
    b = w[-1, :]

    r = newrec['resp']._data
    r0 = newrec['psth']._data
    p = newrec['pupil']._data
    p = p - p.mean()
    p = p / p.std()

    # compute the full model prediction
    r_pred = ((r0 * np.exp(np.matmul(p.T, g1[np.newaxis, :])).T).T + b).T + np.matmul(p.T, d1[np.newaxis, :]).T

    # compute cost function terms:

    # mean NMSE over all neurons
    nmse = mean_square_error(r, r_pred).mean()

    cost = nmse

    if verbose:
        return cost, r_pred
    else:
        return cost

def gain_only_objective(w, rec, big_mask, small_mask, b1=1, nLV=1, verbose=False):
    """
    Minimize MSE between prediction and true response, with an added
    term to the cost function to also minimize the variance between small and
    large pupil in the latent variable(s). The weight of each term in the cost
    function can be tuned with b1 and b2

    model:
        r(t) = r0(t) * exp(w1 * p(t) + w2 * lv(t)) + b

    scipy is annoying and only takes a vectors of params to optimize, so the
    param `w` holds both w1 and w2: w1 = w[:nCells] and w2 = w[nCells:]
        and by default the baseline term (b) is w[-1, :]

    big_mask and small_mask are the big / small pupil nems signal masks
    """
    newrec = copy.deepcopy(rec)

    nCells = newrec['resp'].shape[0]

    # right now, hardcoded to expect only one latent variable, and pupil
    w = w.reshape(nLV + 2, nCells)
    # first order pupil weights
    w1 = w[0, :]
    # will iterate over LV weights below

    # baseline
    b = w[-1, :]

    r = newrec['resp']._data
    r0 = newrec['psth']._data
    p = newrec['pupil']._data
    p = p - p.mean()
    p = p / p.std()

    # compute 1st order prediction, in order to calc the second order latent variable
    gain = np.exp(np.matmul(p.T, w1[np.newaxis, :]))
    gain[~np.isfinite(gain)] = 100 # force to some high ceiling value
    first_order_pred = r0 * gain.T

    if nLV == 0:
        r_pred = first_order_pred
        den = (np.mean(r) * np.mean(r_pred))
        nmse = np.mean( ((r - r_pred) ** 2) / den )

        if verbose:
            # return an array of 0's just for backwards compatibility
            return nmse, r_pred, np.zeros((1, r_pred.shape[-1]))
        else:
            return nmse

    # if nLVs is greater than 0:
    # compute latent variable(s) by projecting residuals onto w[1:-1, :]
    z_score_residuals = r - first_order_pred
    u = z_score_residuals.mean(axis=-1)
    std = z_score_residuals.std(axis=-1)
    z_score_residuals = z_score_residuals.T - u
    z_score_residuals = z_score_residuals / std
    lv = np.zeros((nLV, first_order_pred.shape[-1]))
    for n in range(nLV):
        lv[n, :] = np.matmul(z_score_residuals, w[n+1, :])[np.newaxis, :]

    # compute the full model prediction
    gain = np.exp(np.matmul(p.T, w1[np.newaxis, :]))
    for n in range(nLV):
        gain += np.matmul(lv[n, :][:, np.newaxis], w[n+1, :][np.newaxis, :])
    gain[~np.isfinite(gain)] = 100   # force to some high ceiling value
    r_pred = ((r0 * gain.T).T + b).T

    # compute cost function terms:

    # mean NMSE over all neurons
    #nmse = mean_square_error(r, r_pred).mean()
    den = (np.mean(r) * np.mean(r_pred))
    nmse = np.mean( ((r - r_pred) ** 2) / den )

    # maximize diff in variance in latent variable between big / small pupil
    lv_big = lv[:, big_mask._data.squeeze()]
    lv_small = lv[:, small_mask._data.squeeze()]
    var_diff = -abs(np.var(lv_big) - np.var(lv_small))

    if np.var(lv) != 0:
        var_diff /= np.var(lv)

    cost = (b1 * var_diff) + (1 * nmse)

    if verbose:
        return cost, r_pred, lv
    else:
        return cost

def corrcoef_by_neuron(r, pred):
    cells = r.shape[0]
    cc = np.zeros(cells)
    for c in range(cells):
        cc[c] = np.corrcoef(r[c, :], pred[c, :])[0, 1]
    return cc

def mean_square_error(r, pred):
    cells = r.shape[0]
    nmse = np.zeros(cells)
    for c in range(cells):
        nmse[c] = np.mean( ((r[c, :] - pred[c, :])**2) / (np.mean(r[c, :]) * np.mean(pred[c, :])))
    return nmse
