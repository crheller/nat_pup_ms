"""
Fit pupil only GLM for a given site
"""

import preprocessing as preproc
import noise_correlations as nc
import nems.db as nd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn.decomposition import PCA
import scipy.ndimage.filters as sf
import plotting as cplt
from nems.recording import Recording
import nems_lbhb.baphy as nb
from nems_lbhb.preprocessing import mask_high_repetion_stims
import GLM.glm_fitter_tools as glm
import GLM.glm_regression_tools as glmr
import copy
from tqdm import tqdm

evoked_only = True # only fit on sound evoked activity
balance_stims = False # only fit on epochs balanced across pupil conditions

# set hyperparameters for cost function
site = 'TAR010c'
batch = 289
cellids = [c for c in nd.get_batch_cells(289).cellid if site in c]
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

nCells = len(rec['resp'].chans)
# intialize fit randomly. On other iterations, previous fit will be new x0
x0 = np.zeros(nCells*3)  # because of exp, this effectively initializes gain to 1 and baseline terms to 0

model_output = opt.minimize(glm.pupil_only_objective, x0, (rec), options={'gtol':1e-6, 'disp': True})
weights = model_output.x
g1 = weights.reshape(3, nCells)[0, :]
#w1 = w1 / np.linalg.norm(w1)
d1 = weights.reshape(3, nCells)[1, :]
b = weights.reshape(3, nCells)[-1, :]

# compute model prediction
mse, pred = glm.pupil_only_objective(weights, rec, verbose=True)

# compute prediction correlation, save model prediction, weights, and latent variable
cc = glm.corrcoef_by_neuron(rec['resp']._data, pred)
mse = glm.mean_square_error(rec['resp']._data, pred)
results = {}
results['cc'] = cc
results['mse'] = mse
results['pred'] = pred
results['gain'] = g1
results['dc'] = d1
results['baseline'] = b

# create plots to evaluate outcome of fits

plt.figure(figsize=(6, 4))
ax2 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)

# plot the model weights
ax2.scatter(results['gain'], results['dc'], s=25, color='k', edgecolor='white')
ax2.set_xlabel('gain weights', fontsize=8)
ax2.set_ylabel('DC weights', fontsize=8)
ax2.axhline(0, color='k', linestyle='--')
ax2.axvline(0, color='k', linestyle='--')
ax2.set_aspect(cplt.get_square_asp(ax2))

# plot the model performance
null_cc = glm.corrcoef_by_neuron(rec['resp']._data, rec['psth']._data)
ax3.scatter(null_cc, results['cc'], s=25, color='k', edgecolor='white')
ax3.plot([0, 1], [0, 1], 'k--')
ax3.axhline(0, color='k', linestyle='--')
ax3.axvline(0, color='k', linestyle='--')
ax3.set_xlabel('Null model (r0)', fontsize=8)
ax3.set_ylabel('Full model', fontsize=8)
ax3.set_aspect(cplt.get_square_asp(ax3))

plt.tight_layout()

# perform regression of different factors and determine how noise correlations change
r_true = rec['resp']._data
p = rec['pupil']._data.copy()
p -= p.mean()
p /= p.std()

r_invert1 = glmr.regress_gain_factor(rec['resp']._data, rec['psth']._data, p, g1)
r_invert1 = glmr.regress_dc_factor(r_invert1, p, d1)


# plot a big first order effect neuron
n = 48
#n = 39  # looks like unit with shit isolation
#n = 2
#n = 22  # neuron 22 could be getting overfit with the lv ?

f, ax = plt.subplots(3, 1)

ax[0].plot(p.T, color='green')
ax[1].plot(r_true[n, :], label='true response')
ax[1].plot(r_invert1[n, :], label='first order invert')
ax[1].plot(pred[n, :], label='full prediction')

ax[1].legend()

cc = np.round(np.corrcoef(r_true[n, :] - pred[n, :], p)[0, 1], 2)
ax[2].plot(r_true[n, :] - pred[n, :], label='full model residual, corr w/ pupil: {}'.format(cc))
cc = np.round(np.corrcoef(r_true[n, :] - r_invert1[n, :], p)[0, 1], 2)
ax[2].plot(r_true[n, :] - r_invert1[n, :], label='first order residual, corr w/ pupil: {}'.format(cc))
ax[2].legend(fontsize=8)


# finally look at how noise correlations change

# compute raw noise correlations and after regressing out pupil effects
epochs = np.unique([e for e in rec.epochs.name if 'STIM' in e]).tolist()
all_raw_rsc = nc.compute_rsc(rec['resp'].extract_epochs(epochs)) 
bp_raw_rsc = nc.compute_rsc(rec['resp'].extract_epochs(epochs, mask=big_mask))
sp_raw_rsc = nc.compute_rsc(rec['resp'].extract_epochs(epochs, mask=small_mask))

r1_rec = copy.deepcopy(rec)

r1_rec['resp'] = r1_rec['resp']._modified_copy(r_invert1)

all_r1_rsc = nc.compute_rsc(r1_rec['resp'].extract_epochs(epochs)) 
bp_r1_rsc = nc.compute_rsc(r1_rec['resp'].extract_epochs(epochs, mask=big_mask))
sp_r1_rsc = nc.compute_rsc(r1_rec['resp'].extract_epochs(epochs, mask=small_mask))

f, ax = plt.subplots(1, 2)

raw_mean = all_raw_rsc['rsc'].mean()
r1_mean = all_r1_rsc['rsc'].mean()

ax[0].plot([0, 1], [raw_mean, r1_mean], 'o-', color='k')
ax[0].set_ylabel('mean noise correlation')
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['raw', 'remove 1st order'], rotation=90)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

raw_delta = (sp_raw_rsc['rsc'].mean() - bp_raw_rsc['rsc'].mean()) / raw_mean
r1_delta = (sp_r1_rsc['rsc'].mean() - bp_r1_rsc['rsc'].mean()) / r1_mean


ax[1].plot([0, 1], [raw_delta, r1_delta], 'o-', color='k')
ax[1].set_ylabel('normalized delta noise correlation')
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['raw', 'remove 1st order'], rotation=90)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

plt.show()