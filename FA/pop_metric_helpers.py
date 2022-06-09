import numpy as np
from scipy.signal import argrelextrema

# measure change in dimensionality, %sv, loading sim, across jackknifes
def get_dim(LL):
    try:
        return argrelextrema(LL, np.greater)[0][0]+1
    except:
        print("No relative LL max, choosing overall maximum")
        return np.argmax(LL)+1

def sigma_shared(model):
    return (model.components_.T @ model.components_)

def sigma_ind(model):
    return np.diag(model.noise_variance_)

def pred_cov(model):
    return sigma_shared(model) + sigma_ind(model)

def get_dim95(model):
    """
    number of dims to explain 95% of shared var
    """
    ss = sigma_shared(model)
    evals, _ = np.linalg.eig(ss)
    evals = evals / sum(evals)
    return np.argwhere(np.cumsum(evals)>=0.95)[0][0]+1

def get_sv(model):
    sig_shared = sigma_shared(model) # rank n_components cov matrix
    full_cov_pred = pred_cov(model)
    # % shared variance
    # per neuron
    pn = np.diag(sig_shared) / np.diag(full_cov_pred)
    # average
    sv = np.mean(pn)
    return sv

def get_loading_similarity(model, dim=0):
    # loading similarity
    loading = model.components_[dim, :]
    loading /= np.linalg.norm(loading)
    load_sim = 1 - (np.var(loading) / (1 / len(loading)))
    return load_sim