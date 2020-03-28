"""
Fit GLM over a range of hyperparameters for a given site
"""
import sys
sys.path.append('/auto/users/hellerc/code/crh_tools/')
import preprocessing as preproc
import noise_correlations as nc
from nems.recording import Recording
import nems_lbhb.baphy as nb
from nems_lbhb.preprocessing import mask_high_repetion_stims
import nems
import nems.db as nd
import plotting as cplt
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms_final/')
import GLM.glm_fitter_tools as glm
import GLM.glm_regression_tools as glmr
import nems.db as nd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.optimize as opt
from sklearn.decomposition import PCA
import scipy.ndimage.filters as sf
import copy
from tqdm import tqdm
import os
import logging

log = logging.getLogger(__name__)

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick

else:
    queueid = 0

if queueid:
    log.info("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

# first system argument is the cellid
site = sys.argv[1]  # very first (index 0) is the script to be run
# second systems argument is the batch
batch = sys.argv[2]
# third system argument in the modelname
modelname = sys.argv[3]

# set filenames to save
path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/GLM/results/{0}/'.format(site)
dictsave = path+'{}.pickle'.format(modelname)
figsave =  path+'{}.png'.format(modelname)

# parse modelname options
evoked_only = '_ev' in modelname
balance_stims = '_bal' in modelname
nLVs = int([s for s in modelname.split('_') if s[:3] == 'nLV'][0][3:])
a_step = np.float([s for s in modelname.split('_') if s[:5] == 'astep'][0][5:])
a_max = np.float([s for s in modelname.split('_') if s[:4] == 'amax'][0][4:])

# set hyperparameters for cost function
alpha1 = 1
alpha2 = np.round(np.arange(0, a_max, a_step), 2)
log.info("Fitting over range of alpha values: {}".format(alpha2))
batch = 289
fs = 4
ops = {'batch': batch, 'siteid': site, 'rasterfs': fs, 'pupil': 1, 'rem': 1,
    'stim': 1}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri)
rec['resp'] = rec['resp'].rasterize()
rec['stim'] = rec['stim'].rasterize()
rec = mask_high_repetion_stims(rec)
rec = rec.apply_mask(reset_epochs=True)

rec = preproc.generate_psth(rec)

if balance_stims:
    epochs = preproc.get_pupil_balanced_epochs(rec)
    rec = rec.and_mask(epochs)
    rec = rec.apply_mask(reset_epochs=True)
if evoked_only:
    rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    rec = rec.apply_mask(reset_epochs=True)

pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
rec_bp = preproc.create_pupil_mask(rec.copy(), **pup_ops)
pup_ops['state'] = 'small'
rec_sp = preproc.create_pupil_mask(rec.copy(), **pup_ops)
big_mask = rec_bp['mask']
small_mask = rec_sp['mask']

# Now, fit the model for the range of alpha2 values
results = dict.fromkeys(alpha2)
for i, a2 in tqdm(enumerate(alpha2)):
    log.info("alpha: {}".format(a2))
    nCells = len(rec['resp'].chans)
    if i == 0:
        # intialize fit randomly. On other iterations, previous fit will be new x0
        x0 = np.zeros(nCells * (2+nLVs))

    model_output = opt.minimize(glm.gain_only_objective, x0, (rec, big_mask, small_mask, a2, nLVs), options={'gtol':1e-6, 'disp': True})
    weights = model_output.x
    w1 = weights.reshape((2+nLVs), nCells)[0, :]
    w2 = weights.reshape((2+nLVs), nCells)[1:-1, :]
    b = weights.reshape((2+nLVs), nCells)[-1, :]

    # compute model prediction
    mse, pred, lv = glm.gain_only_objective(weights, rec, big_mask, small_mask, b1=a2, nLV=nLVs, verbose=True)

    # compute prediction correlation, save model prediction, weights, and latent variable
    cc = glm.corrcoef_by_neuron(rec['resp']._data, pred)
    mse = glm.mean_square_error(rec['resp']._data, pred)
    results[a2] = {}
    results[a2]['cc'] = cc
    results[a2]['mse'] = mse
    results[a2]['pred'] = pred
    results[a2]['lv'] = lv
    results[a2]['beta1'] = w1
    results[a2]['beta2'] = w2
    results[a2]['baseline'] = b

    # finally, udpate initial conditions
    x0 = model_output.x

# save results dictionary
log.info('pickling results dictionary: {}'.format(dictsave))
with open(dictsave, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save plot of cost function over different hyperparameters to quickly
# evaluate if should re-fit with diff params

f, ax = plt.subplots(1, 2)
best = 0
for k in results.keys():
    if results[k] is not None:
        if results[k]['cc'].mean() > best:
            best = results[k]['cc'].mean()
            best_alpha = k
        ax[0].plot(k, results[k]['cc'].mean(), 'ko')
        ax[1].plot(k, results[k]['mse'].mean(), 'ro')
if best_alpha == 0:
    best_alpha = alpha2[1]
ax[0].set_ylabel('prediction correlation')
ax[0].set_xlabel('pupil constraint')
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].set_ylabel('NMSE')
ax[1].set_xlabel('pupil constraint')
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

log.info('saving model performance fig: {}'.format(figsave))
f.savefig(figsave)

plt.close('all')

if queueid:
    nd.update_job_complete(queueid)
