"""
Load single model, extract weights / LVs etc.
Compare with true rec
Compare decoding
"""
from nems.xform_helper import load_model_xform
from nems_lbhb.preprocessing import create_pupil_mask
import charlieTools.preprocessing as cpreproc

import matplotlib.pyplot as plt
import numpy as np

site = 'AMT020a'
#site = 'AMT026a'
#site = 'ARM029a'
#site = 'ARM031a'
#site = 'ARM032a'
#site = 'ARM033a'
#site = 'CRD018d'
batch = 331
#site = 'TAR010c'
#batch = 322

modelname = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t6.ss1"
modelname = 'psth.fs4.pup-loadpred.cpn-st.drf.pup-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR-inoise.2xR_ccnorm.t6.ss1'
if batch == 322:
    modelname = modelname.replace('.cpn', '')

xf, ctx = load_model_xform(site, batch, modelname=modelname)
rec = ctx['rec'].copy()

# measure both predicted and raw delta noise correlation axes
zscore = True

if batch == 322:
    epochs = [e for e in rec['resp'].epochs.name.unique() if e.startswith('STIM_00')]
else:
    epochs = [e for e in rec['resp'].epochs.name.unique() if e.startswith('STIM_')]

# raw data
real_dict_small = rec['resp'].extract_epochs(epochs, mask=rec['mask_small'], allow_incomplete=True)
real_dict_big = rec['resp'].extract_epochs(epochs, mask=rec['mask_large'], allow_incomplete=True)
real_dict_small = cpreproc.zscore_per_stim(real_dict_small, d2=real_dict_small, with_std=zscore)
real_dict_big = cpreproc.zscore_per_stim(real_dict_big, d2=real_dict_big, with_std=zscore)
eps = list(real_dict_big.keys())
nCells = real_dict_big[eps[0]].shape[1]
for i, k in enumerate(real_dict_big.keys()):
    if i == 0:
        resp_matrix_small = np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)
        resp_matrix_big = np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)
    else:
        resp_matrix_small = np.concatenate((resp_matrix_small, np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
        resp_matrix_big = np.concatenate((resp_matrix_big, np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
nc_resp_small = resp_matrix_small
nc_resp_big = resp_matrix_big 
small = np.cov(nc_resp_small)
np.fill_diagonal(small, 0)
big = np.cov(nc_resp_big)
np.fill_diagonal(big, 0)
diff = small - big
evals, evecs = np.linalg.eig(diff)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

real_evecs = evecs
real_evals = evals

# pred data
pred_dict_small = rec['pred'].extract_epochs(epochs, mask=rec['mask_small'], allow_incomplete=True)
pred_dict_big = rec['pred'].extract_epochs(epochs, mask=rec['mask_large'], allow_incomplete=True)
pred_dict_small = cpreproc.zscore_per_stim(pred_dict_small, d2=pred_dict_small, with_std=zscore)
pred_dict_big = cpreproc.zscore_per_stim(pred_dict_big, d2=pred_dict_big, with_std=zscore)
eps = list(pred_dict_big.keys())
nCells = pred_dict_big[eps[0]].shape[1]
for i, k in enumerate(pred_dict_big.keys()):
    if i == 0:
        resp_matrix_small = np.transpose(pred_dict_small[k], [1, 0, -1]).reshape(nCells, -1)
        resp_matrix_big = np.transpose(pred_dict_big[k], [1, 0, -1]).reshape(nCells, -1)
    else:
        resp_matrix_small = np.concatenate((resp_matrix_small, np.transpose(pred_dict_small[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
        resp_matrix_big = np.concatenate((resp_matrix_big, np.transpose(pred_dict_big[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
nc_resp_small = resp_matrix_small
nc_resp_big = resp_matrix_big 
small = np.cov(nc_resp_small)
np.fill_diagonal(small, 0)
big = np.cov(nc_resp_big)
np.fill_diagonal(big, 0)
diff = small - big
evals, evecs = np.linalg.eig(diff)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

pred_evals = evals
pred_evecs = evecs

f, ax = plt.subplots(2, 1, figsize=(6, 6))

ax[0].plot(pred_evals / np.abs(pred_evals).max(), 'o-', label='pred')
ax[0].plot(real_evals / np.abs(real_evals).max(), 'o-', label='real')
ax[0].set_ylabel(r"$\lambda$")
ax[0].legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
ax[0].set_title(site)

alignment = []
for e in range(pred_evecs.shape[-1]): 
    alignment.append(np.abs(pred_evecs[:,e].dot(real_evecs[:,e])))
ax[1].plot(alignment, 'o-')
ax[1].set_ylabel("Alignment of real vs. pred axes")
ax[1].set_xlabel("Eigenvector")

f.tight_layout()

plt.show()