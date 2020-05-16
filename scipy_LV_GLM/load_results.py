import os
import pickle
import logging

log = logging.getLogger(__name__)

path = '/auto/users/hellerc/code/projects/pupil_ms/GLM/results/'

def load_fit(modelname, site=None):

    if site == None:
        sites = os.listdir(path)
    else:
        sites = [site]
    results = {}
    for s in sites:
        try:
            with open(path+s+'/{}.pickle'.format(modelname), 'rb') as handle:
                d = pickle.load(handle)

            results[s] = d
        except:
            log.info("No fit exists yet for {}".format(s))

    return results
