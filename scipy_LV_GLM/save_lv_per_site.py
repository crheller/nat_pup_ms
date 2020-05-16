"""
Inspect the model fits over various hyperparamteres and decide which LV
to cache for final n.c. / decoding analysis
"""

import matplotlib.pyplot as plt
import GLM.load_results as ld
import GLM.glm_fitter_tools as glm
import preprocessing as preproc
import noise_correlations as nc
import numpy as np
import plotting as cplt
from nems.recording import Recording
import nems_lbhb.baphy as nb
from nems_lbhb.preprocessing import mask_high_repetion_stims
import copy

save = True
save_path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/GLM/latent_variables/'

site = 'TAR010c'
nLV = 1
balance_stims = False
evoked_only = True
alpha = 0.02
lv_regress = 0
modelname = 'glm_ev_nLV{0}_astep0.02_amax1'.format(nLV)
results = ld.load_fit(modelname, site)[site]

alpha2 = list(results.keys())

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

f, ax = plt.subplots(1, 2)
vals = []
best = 0
for i, k in enumerate(results.keys()):
    if results[k] is not None:
        vals.append(results[k]['mse'].mean())
        if results[k]['cc'].mean() > best:
            best = results[k]['cc'].mean()
            best_alpha = k
        ax[0].plot(k, results[k]['cc'].mean(), 'o', color='lightgrey')
        ax[1].plot(k, results[k]['mse'].mean(), 'o', color='lightgrey')

best_alpha = list(results.keys())[np.argmax(np.diff(vals))]

if best_alpha == 0:
    best_alpha = alpha2[1]

if alpha is None:
    pass
else:
    best_alpha = alpha

ax[0].plot(best_alpha, results[best_alpha]['cc'].mean(), 'ko')
ax[0].set_ylabel('prediction correlation')
ax[0].set_xlabel('pupil constraint')
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(best_alpha, results[best_alpha]['mse'].mean(), 'ro')
ax[1].set_ylabel('NMSE')
ax[1].set_xlabel('pupil constraint')
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

# load prediction
pred = results[best_alpha]['pred']

for n in range(nLV):
    plt.figure(figsize=(8, 6))
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

    # plot pupil, and the latent variable, for the best fit
    p = rec['pupil']._data.T / rec['pupil']._data.max()

    lv = results[best_alpha]['lv'][n, :].T
    lv = lv + abs(np.min(lv))
    lv = lv / lv.max()

    ax1.scatter(range(results[best_alpha]['lv'].shape[-1]), 
                lv, 
                cmap='Purples', c=p.squeeze(), label='LV{}'.format(n+1), vmin=p.min()-0.2)
    ax1.plot(p, color='green', label='pupil', lw=3)
    ax1.set_xlabel('Time')
    ax1.legend()

    # plot the model weights
    ax2.scatter(results[best_alpha]['beta1'], results[best_alpha]['beta2'][n, :], s=25, color='k', edgecolor='white')
    ax2.set_xlabel('Pupil weights', fontsize=8)
    ax2.set_ylabel('LV weights', fontsize=8)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.axvline(0, color='k', linestyle='--')
    ax2.set_aspect(cplt.get_square_asp(ax2))

    # plot the model performance
    null_cc = glm.corrcoef_by_neuron(rec['resp']._data, rec['psth']._data)
    ax3.scatter(null_cc, results[best_alpha]['cc'], s=25, color='k', edgecolor='white')
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.axhline(0, color='k', linestyle='--')
    ax3.axvline(0, color='k', linestyle='--')
    ax3.set_xlabel('Null model (r0)', fontsize=8)
    ax3.set_ylabel('Full model', fontsize=8)
    ax3.set_aspect(cplt.get_square_asp(ax3))

    f.tight_layout()

# perform regression of different factors and determine how noise correlations change
r_true = rec['resp']._data
p = rec['pupil']._data.copy()
p -= p.mean()
p /= p.std()

# choose the latent variable that is shared amongst most neurons
if lv_regress is None:
    lv_regress = np.argmax(abs(np.mean(np.sign(results[best_alpha]['beta2']), axis=-1)))
lv = results[best_alpha]['lv'][lv_regress, :][np.newaxis, :].copy()
lv -= lv.mean()
lv /= lv.std()
'''
r_invert1 = glmr.regress_gain_factor(rec['resp']._data, rec['psth']._data, p, w1)
r_invert2 = glmr.regress_gain_factor(rec['resp']._data, rec['psth']._data, lv, w2)
r_invert12 = glmr.regress_gain_factor(rec['resp']._data, rec['psth']._data, np.concatenate((p, lv), axis=0), 
                                np.concatenate((w1[np.newaxis, :], w2[np.newaxis, :]), axis=0))
'''
rec['lv'] = rec['resp']._modified_copy(lv)
rec['pupil'] = rec['pupil']._modified_copy(p)
r_invert1 = preproc.regress_state(rec, state_sigs=['pupil'], regress=['pupil'])['resp']._data
r_invert2 = preproc.regress_state(rec, state_sigs=['lv'], regress=['lv'])['resp']._data
r_invert12 = preproc.regress_state(rec, state_sigs=['pupil', 'lv'], regress=['pupil', 'lv'])['resp']._data
# plot a big first order effect neuron
n = 2
#n=22
#n=39
f, ax = plt.subplots(3, 1)

ax[0].plot(p.T, color='green')
ax[0].plot(lv.T, color='purple')

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
r2_rec = copy.deepcopy(rec)
r12_rec = copy.deepcopy(rec)

r1_rec['resp'] = r1_rec['resp']._modified_copy(r_invert1)
r2_rec['resp'] = r2_rec['resp']._modified_copy(r_invert2)
r12_rec['resp'] = r12_rec['resp']._modified_copy(r_invert12)

all_r1_rsc = nc.compute_rsc(r1_rec['resp'].extract_epochs(epochs)) 
bp_r1_rsc = nc.compute_rsc(r1_rec['resp'].extract_epochs(epochs, mask=big_mask))
sp_r1_rsc = nc.compute_rsc(r1_rec['resp'].extract_epochs(epochs, mask=small_mask))

all_r2_rsc = nc.compute_rsc(r2_rec['resp'].extract_epochs(epochs)) 
bp_r2_rsc = nc.compute_rsc(r2_rec['resp'].extract_epochs(epochs, mask=big_mask))
sp_r2_rsc = nc.compute_rsc(r2_rec['resp'].extract_epochs(epochs, mask=small_mask))

all_r12_rsc = nc.compute_rsc(r12_rec['resp'].extract_epochs(epochs)) 
bp_r12_rsc = nc.compute_rsc(r12_rec['resp'].extract_epochs(epochs, mask=big_mask))
sp_r12_rsc = nc.compute_rsc(r12_rec['resp'].extract_epochs(epochs, mask=small_mask))

f, ax = plt.subplots(1, 2)

f.suptitle("Regressing out LV {}".format(lv_regress+1))

raw_mean = all_raw_rsc['rsc'].mean()
r1_mean = all_r1_rsc['rsc'].mean()
r2_mean = all_r2_rsc['rsc'].mean()
r12_mean = all_r12_rsc['rsc'].mean()

raw_sem = all_raw_rsc['rsc'].sem()
r1_sem = all_r1_rsc['rsc'].sem()
r2_sem = all_r2_rsc['rsc'].sem()
r12_sem = all_r12_rsc['rsc'].sem()

ax[0].errorbar([0, 1, 2, 3], [raw_mean, r1_mean, r2_mean, r12_mean], 
            yerr=[raw_sem, r1_sem, r2_sem, r12_sem], marker='o', color='k')
ax[0].set_ylabel('mean noise correlation')
ax[0].set_xticks([0, 1, 2, 3])
ax[0].set_xticklabels(['raw', 'remove 1st order', 'remove 2nd order', 'remove all'], rotation=90)
ax[0].axhline(0, linestyle='--', color='k')
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

raw_delta = (sp_raw_rsc['rsc'] - bp_raw_rsc['rsc']).mean()
r1_delta = (sp_r1_rsc['rsc'] - bp_r1_rsc['rsc']).mean()
r2_delta = (sp_r2_rsc['rsc'] - bp_r2_rsc['rsc']).mean()
r12_delta = (sp_r12_rsc['rsc'] - bp_r12_rsc['rsc']).mean()

raw_delta_sem = (sp_raw_rsc['rsc'] - bp_raw_rsc['rsc']).sem()
r1_delta_sem = (sp_r1_rsc['rsc'] - bp_r1_rsc['rsc']).sem()
r2_delta_sem = (sp_r2_rsc['rsc'] - bp_r2_rsc['rsc']).sem()
r12_delta_sem = (sp_r12_rsc['rsc'] - bp_r12_rsc['rsc']).sem()

ax[1].errorbar([0, 1, 2, 3], [raw_delta, r1_delta, r2_delta, r12_delta], 
        yerr=[raw_delta_sem, r1_delta_sem, r2_delta_sem, r12_delta_sem], marker='o', color='k')
ax[1].set_ylabel('small minus big')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_xticks([0, 1, 2, 3])
ax[1].set_xticklabels(['raw', 'remove 1st order', 'remove 2nd order', 'remove all'], rotation=90)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()


if save:
    # save weights
    fn = site+'_latent_variable.npy'
    np.save(save_path+fn, results[best_alpha]['beta2'][lv_regress, :])

plt.show()