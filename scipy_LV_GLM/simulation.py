import numpy as np
import matplotlib.pyplot as plt
from nems.recording import Recording
from nems.signal import RasterizedSignal
import GLM.glm_fitter_tools as glm
import scipy.optimize as opt
from tqdm import tqdm
import plotting as cplt

# simlulate data set. 

nCells = 50
T = 2000

# constant mean firing rate
u0 = np.random.poisson(4, nCells)
u = np.random.poisson(u0, (T, nCells)).T

# binary first-order gain applied (enhanced during "big pupil", decreased during "small pupil")
pupil = np.ones((1, T)) + np.random.normal(0.05, 0.01, (1, T))
pupil[0, :int(T/2)] = pupil[0, :int(T/2)] - 0.7
pupil -= pupil.mean()
pupil /= pupil.std()
g1 = np.random.normal(0.5, 0.1, (nCells, 1))

gain1 = np.matmul(pupil.T, g1.T).T

# second order gain (stronger when pupil is small (first half of recording))
lv1 = 0.3 * np.sin(.2 * np.arange(0, T))[np.newaxis, :]
lv1[0, int(T/2):] = lv1[0, int(T/2):] * 0.2
lv1 -= lv1.mean() 
lv1 /= lv1.std()
g2 = np.random.normal(0.5, 0.1, (nCells, 1))

gain2 = np.matmul(lv1.T, g2.T).T

resp = u * np.exp(gain1 + gain2)
psth = np.tile(resp.mean(axis=-1), (T, 1)).T

# get rid of any "cells" that never fired
idx = (resp==0).sum(axis=-1) == T
resp = resp[~idx, :]
psth = psth[~idx, :]
nCells = resp.shape[0]

# pack into nems recording
resp_sig = RasterizedSignal(fs=4, data=resp, name='resp', recording='simulation')
psth_sig = RasterizedSignal(fs=4, data=psth, name='psth', recording='simulation')
pupil_sig = RasterizedSignal(fs=4, data=pupil, name='pupil', recording='simulation')
bm = pupil > pupil.mean()
big_mask = RasterizedSignal(fs=4, data=bm, name='big_mask', recording='simulation')
sm = pupil < pupil.mean()
small_mask = RasterizedSignal(fs=4, data=sm, name='small_mask', recording='simulation')

rec = Recording({'resp': resp_sig, 'psth': psth_sig, 'pupil': pupil_sig})

# fit the GLM for different hyperparameters
x0 = np.zeros(3 * nCells)
nLV = 1
alpha1 = np.arange(0, 0.5, 0.05)
results = dict.fromkeys(alpha1)
for i, a2 in tqdm(enumerate(alpha1)):
    model_output = opt.minimize(glm.gain_only_objective, x0, (rec, big_mask, small_mask, a2, nLV), options={'gtol':1e-6, 'disp': True})
    weights = model_output.x

    # get model output
    w1 = weights.reshape((2+nLV), nCells)[0, :]
    w2 = weights.reshape((2+nLV), nCells)[1:-1, :]
    b = weights.reshape((2+nLV), nCells)[-1, :]

    # compute model prediction
    mse, pred, lv = glm.gain_only_objective(weights, rec, big_mask, small_mask, b1=a2, nLV=nLV, verbose=True)

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

    x0 = model_output.x

f2, ax2 = plt.subplots(1, 2)
best = 0
for k in results.keys():
    if results[k] is not None:
        if results[k]['cc'].mean() > best:
            best = results[k]['cc'].mean()
            best_alpha = k
        ax2[0].plot(k, results[k]['cc'].mean(), 'ko')
        ax2[1].plot(k, results[k]['mse'].mean(), 'ro')
if best_alpha == 0:
    best_alpha = alpha1[1]
ax2[0].set_ylabel('prediction correlation')
ax2[0].set_xlabel('pupil constraint')
ax2[0].set_aspect(cplt.get_square_asp(ax2[0]))

ax2[1].set_ylabel('NMSE')
ax2[1].set_xlabel('pupil constraint')
ax2[1].set_aspect(cplt.get_square_asp(ax2[1]))

f, ax = plt.subplots(4, 1)

ax[0].set_title('Simulated data')
ax[0].imshow(resp, aspect='auto')

ax[2].set_title('True state-variables')
ax[2].plot(lv1.T, color='purple', label='LV')
ax[2].plot(pupil.T, color='green', label='pupil')
ax[2].legend()

# finish the data plot with the predictions
ax[1].set_title('Model prediction')
ax[1].imshow(results[best_alpha]['pred'], aspect='auto')

ax[3].set_title('Predicted state')
ax[3].plot(results[best_alpha]['lv'].T, color='purple', label='Pred LV')
ax[3].legend()

f.tight_layout()

# plot a single cell true and predicted
f, ax = plt.subplots(1, 1)

ax.plot(resp[0, :], label='Response')
ax.plot(pred[0, :], label='Prediction')



# plot the model weights

f, ax = plt.subplots(1, 1)
ax.scatter(results[best_alpha]['beta1'], results[best_alpha]['beta2'], s=35, color='k', edgecolor='white', label='fit weights')
m = np.max([results[best_alpha]['beta1'].max(), results[best_alpha]['beta2'].max()])
mi = 0 #np.min([results[best_alpha]['beta1'].min(), results[best_alpha]['beta2'].min()])
ax.plot([mi, m], [mi, m], 'k--')
ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')
ax.legend()
ax.set_aspect(cplt.get_square_asp(ax))
ax.set_xlabel('First-order weights')
ax.set_ylabel('LV weights')

# mean rate for each neuron vs. variance. Show super-poisson, multiplicative relationship
f, ax = plt.subplots(1, 1)

mean = resp.mean(axis=-1)
var = resp.var(axis=-1)

ax.plot(mean, var, '.', color='grey')
ax.plot([0, var.max()], [0, var.max()], 'k--')
ax.set_xlabel('mean')
ax.set_ylabel('variance')

plt.show()
