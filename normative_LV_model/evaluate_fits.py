"""
LV model seems slightly unstable for various possible reasons.
Figure out useful intermediate visualization of the model fit / results
19.09.2021

Notes: 
    adding psth.z stablizes pop model fits for stategain models
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 6

import figures_final.helpers as fhelp

import nems.db as db
from nems.xform_helper import load_model_xform

batch = 331
site = 'ARM032a'
modelname = 'psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.er2-aev_stategain.SxR-spred-lvnorm.SxR.so-inoise.SxR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3'
modelname0 = 'psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.er1-aev_stategain.SxR-spred-lvnorm.SxR.so-inoise.SxR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3'
cellids = [cellid for cellid in db.get_batch_cells(batch).cellid if site in cellid]

xf, ctx = load_model_xform(batch=batch, cellid=cellids[0], modelname=modelname)
xf0, ctx0 = load_model_xform(batch=batch, cellid=cellids[0], modelname=modelname0)
rec = ctx['val'].copy()
rec0 = ctx0['val'].copy() 
first_rep = np.ones((1, rec['resp'].shape[-1]))
first_rep[0, int(rec['resp'].shape[-1]/8):] = 0
rec['mask'] = rec['mask']._modified_copy(rec['mask']._data & first_rep.astype(bool))
rec0['mask'] = rec0['mask']._modified_copy(rec0['mask']._data & first_rep.astype(bool))

# first, compare pred0 performance to psth performance for each cell. If pred0 < psth, then there's definitely a problem
for r, tit in zip([rec, rec0], ['rec', 'rec0']):
    psth_r = []
    pred0_r = []
    resp = r.apply_mask()['resp']._data
    psth = r.apply_mask()['psth']._data
    pred0 = r.apply_mask()['pred0']._data
    for i in range(resp.shape[0]):
        psth_r.append(np.corrcoef(resp[i], psth[i])[0, 1])
        pred0_r.append(np.corrcoef(resp[i], pred0[i])[0, 1])

    # get covariance matrices big / small
    stim = 0 # or can be index of STIM_XX epoch
    pbg, psm = fhelp.get_cov_matrices(r.copy(), sig='pred', sub='pred0', stims=stim, ss=0)
    bg, sm = fhelp.get_cov_matrices(r.copy(), sig='resp', sub='pred0', stims=stim, ss=3)

    # layout / plot overall figure
    f = plt.figure(figsize=(9, 6))
    axs = plt.subplot2grid((4, 3), (1, 0), colspan=1, rowspan=2)
    axbg = plt.subplot2grid((4, 3), (0, 1), colspan=1, rowspan=2)
    axsm = plt.subplot2grid((4, 3), (0, 2), colspan=1, rowspan=2)
    axbgp = plt.subplot2grid((4, 3), (2, 1), colspan=1, rowspan=2)
    axsmp = plt.subplot2grid((4, 3), (2, 2), colspan=1, rowspan=2)

    vmin = -np.concatenate((bg, sm)).std()*2
    vmax = np.concatenate((bg, sm)).std()*2

    axs.scatter(pred0_r, psth_r, edgecolor='white', s=25, color='grey')
    axs.plot([0, 1], [0, 1], 'k--')
    axs.set_title(site)
    axs.set_xlabel('pred0 pred corr')
    axs.set_ylabel('psth pred corr.')

    axbg.imshow(bg, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    axbg.set_title("big, raw")
    axsm.imshow(sm, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    axsm.set_title("small, raw")

    axbgp.imshow(pbg, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    axbgp.set_title("big, pred")
    axsmp.imshow(psm, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    axsmp.set_title("small, pred")

    f.tight_layout()

    f.canvas.set_window_title(tit)

    # investigate predictions of covariance more closely
    diff = sm - bg
    np.fill_diagonal(diff, 0)
    idx = np.unravel_index(np.argmax(diff), sm.shape) 

    f, ax = plt.subplots(1, 3, figsize=(9,3))

    ax[0].scatter(pbg.flatten(), bg.flatten(), s=5)
    ll, ul = (np.min(ax[0].get_xlim()+ax[0].get_ylim()), np.max(ax[0].get_xlim()+ax[0].get_ylim())) 
    ax[0].plot([ll, ul], [ll, ul], 'k--')
    ax[0].set_ylabel('Raw value')
    ax[0].set_xlabel('Prediction')
    ax[0].set_title("Big pupil covariance")

    ax[1].scatter(pbg.flatten(), bg.flatten(), s=5)
    ll, ul = (np.min(ax[1].get_xlim()+ax[1].get_ylim()), np.max(ax[1].get_xlim()+ax[1].get_ylim())) 
    ax[1].plot([ll, ul], [ll, ul], 'k--')
    ax[1].set_ylabel('Raw value')
    ax[1].set_xlabel('Prediction')
    ax[1].set_title("Small pupil covariance")

    diff = bg - sm
    np.fill_diagonal(diff, 0)
    diffp = pbg - psm
    np.fill_diagonal(diffp, 0)
    ax[2].scatter(diffp.flatten(), diff.flatten(), s=5)
    ll, ul = (np.min(ax[2].get_xlim()+ax[2].get_ylim()), np.max(ax[2].get_xlim()+ax[2].get_ylim())) 
    ax[2].plot([ll, ul], [ll, ul], 'k--')
    ax[2].axvline(0, linestyle='--', color='k')
    ax[2].axhline(0, linestyle='--', color='k')
    ax[2].set_ylabel('Raw value')
    ax[2].set_xlabel('Prediction')
    ax[2].set_title("Diff. covariance")

    print(np.corrcoef(diff.flatten(), diffp.flatten())[0, 1])
    print(np.sum(np.abs(diff-diffp)))

    f.tight_layout()

    f.canvas.set_window_title(tit)

plt.show()