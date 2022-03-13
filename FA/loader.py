import pickle

def load_pop_metrics(site, batch, modelname):
    """
    Load population metrics (%sv, dimensionality, loading similarity)
    for given site / batch.
    """
    path = f"/auto/users/hellerc/results/nat_pupil_ms/factor_analysis/{batch}/{site}/"
    filename = modelname+".pickle"
    with open(path + filename, 'rb') as handle:
        b = pickle.load(handle)
    return b