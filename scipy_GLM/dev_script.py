"""
Develop GLM to model both first- and second-order pupil effects.
In essence, fitting first / second order weights and learning a latent
variable.

Initialize lv weigths with PC of residuals?
"""

import preprocessing as preproc
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

evoked_only = True # only fit on sound evoked activity
balance_stims = True # only fit on epochs balanced across pupil conditions
p2_weight = 0.2  # second order constaint weight in the cost function
p1_weight = 1  # first order constaint weight in the cost function

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


nCells = len(rec['resp'].chans)
x0 = np.random.normal(0, 1, nCells*2)

model_output = opt.minimize(glm.objective, x0, (rec, big_mask, small_mask, p2_weight, p1_weight), options={'gtol':1e-6, 'disp': True})
weights = model_output.x
w1 = weights[:nCells]
w2 = weights[nCells:]
w1 = w1 / np.linalg.norm(w1)
w2 = w2 / np.linalg.norm(w2)

# compute model prediction
c, pred, lv = glm.objective(weights, rec, big_mask, small_mask, b1=p2_weight, b2=p1_weight, verbose=True)

# fit first order only model
model_output2 = opt.minimize(glm.objective, x0, (rec, big_mask, small_mask, p2_weight, p1_weight, True), options={'gtol':1e-6, 'disp': True})
weights = model_output2.x

c, pred2, lv2 = glm.objective(weights, rec, big_mask, small_mask, b1=p2_weight, b2=p1_weight, first_only=True, verbose=True)


# plot model predictions and pupil and latent variable
f, ax = plt.subplots(4, 1)

ax[0].imshow(rec['resp']._data, aspect='auto')
ax[0].set_title('Real response')

ax[1].imshow(pred, aspect='auto')
ax[1].set_title('Prediction')

ax[2].plot(rec['pupil']._data.T)
ax[2].set_title('pupil trace')

ax[3].scatter(range(0, lv.shape[-1]), lv.T, cmap='Purples', c=rec['pupil']._data.squeeze())
ax[3].set_title('second-order pupil lv')

f.tight_layout()

# Look at model weights and prediction coef for each neuron
# and noise correlations before / after
f, ax = plt.subplots(2, 2)

ax[0, 0].scatter(w1, w2, s=25, color='k', edgecolor='white')
ax[0, 0].axhline(0, linestyle='--', color='k')
ax[0, 0].axvline(0, linestyle='--', color='k')
ax[0, 0].set_xlabel('first-order weights')
ax[0, 0].set_ylabel('second-order weights')
ax[0, 0].set_aspect(cplt.get_square_asp(ax[0, 0]))

null_cc = glm.corrcoef_by_neuron(rec['resp']._data, rec['psth']._data)
pred_cc = glm.corrcoef_by_neuron(rec['resp']._data, pred)
first_cc = glm.corrcoef_by_neuron(rec['resp']._data, pred2)

ax[0, 1].scatter(null_cc, pred_cc, s=25, color='k', edgecolor='white')
ax[0, 1].plot([0, 1], [0, 1], 'k--')
ax[0, 1].set_xlabel('null model (r0)')
ax[0, 1].set_ylabel('full model')
ax[0, 1].set_aspect(cplt.get_square_asp(ax[0, 1]))

# full model vs. first order only model
ax[1, 1].scatter(null_cc, first_cc, s=25, color='k', edgecolor='white')
ax[1, 1].plot([0, 1], [0, 1], 'k--')
ax[1, 1].set_xlabel('null model model')
ax[1, 1].set_ylabel('first order model')
ax[1, 1].set_aspect(cplt.get_square_asp(ax[1, 1]))

epochs = np.unique([e for e in rec['resp'].epochs.name if 'STIM' in e]).tolist()
r_z = preproc.zscore_per_stim(rec['resp'].extract_epochs(epochs), rec['resp'].extract_epochs(epochs))
pred_rec = rec.copy()
pred_rec['resp'] = rec['resp']._modified_copy(pred)
r_z_pred = preproc.zscore_per_stim(pred_rec['resp'].extract_epochs(epochs), pred_rec['resp'].extract_epochs(epochs))

for i, k in enumerate(epochs):
    if i == 0:
        real_z = np.transpose(r_z[k], [1, 0, -1]).reshape(nCells, -1)
        pred_z = np.transpose(r_z_pred[k], [1, 0, -1]).reshape(nCells, -1)
    else:
        real_z = np.concatenate((real_z, np.transpose(r_z[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
        pred_z = np.concatenate((pred_z, np.transpose(r_z_pred[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)

noise_corr = np.corrcoef(real_z)[np.triu_indices(nCells, 1)]
pred_noise_corr = np.corrcoef(pred_z)[np.triu_indices(nCells, 1)]

ax[1, 0].scatter(noise_corr, pred_noise_corr, s=10, color='grey')
ax[1, 0].set_xlabel('rsc real')
ax[1, 0].set_ylabel('rsc model prediction')
ax[1, 0].set_aspect(cplt.get_square_asp(ax[1, 0]))


f.tight_layout()

plt.show()
