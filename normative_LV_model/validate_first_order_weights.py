"""
Some question of whether we're actually hitting global minimum for first order stage of the model when fitting the
population simultaneously. Evaluate first order model predictions for each cell, compare model weights from population
fit to single cell fits.
"""
import matplotlib.pyplot as plt
import numpy as np

from nems.xform_helper import load_model_xform
import nems.db as db

site = 'TNC013a'
batch = 331

cellids = [c for c in db.get_batch_cells(batch).cellid if site in c]

popmodel = 'psth.fs4.pup-ld-st.pup.pvp0-epcpn-mvm.t25.w1-hrc-psthfr-plgsm.e10.sp-aev_stategain.SxR-spred-lvnorm.SxR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.f0.ss1'
fpopmodel = 'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr.z-aev_stategain.SxR_tfinit.n.lr1e4.cont.et5.i50000'
jkpopmodel = 'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr_stategain.SxR_jk.nf10-tfinit.n.lr1e4.cont.et5.i50000'
singlecellmodel = 'ns.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr_stategain.SxR.bound_jk.nf10-basic'

# load pop model
xfp, ctxp = load_model_xform(modelname=popmodel, batch=batch, cellid=cellids[0])

# find "bad cells"
r = ctxp['val'].apply_mask()['resp']._data
pr = ctxp['val'].apply_mask()['pred0']._data
ps = ctxp['val'].apply_mask()['psth_sp']._data
pred0_r = []
psth_r = []
for i in range(r.shape[0]):
    print(f"{i}, r: {np.corrcoef(r[i], pr[i])[0,1]}")
    pred0_r.append(np.corrcoef(r[i], pr[i])[0,1])
    psth_r.append(np.corrcoef(r[i], ps[i])[0,1])

# load single cells / first order only predictions
# load pop model for first order fit only
xfpf, ctxpf = load_model_xform(modelname=fpopmodel, batch=batch, cellid=cellids[0])
# load pop model for first order fit only with jackknifing
xfjk, ctxjk = load_model_xform(modelname=jkpopmodel, batch=batch, cellid=cellids[0])
# load single cell model
xfs, ctxs = load_model_xform(modelname=singlecellmodel, batch=batch, cellid=cellids[24])


# compare single cell pred to pop pred for a single cell
cellid = ctxs['val']['resp'].chans[0]
resps = ctxs['val'].apply_mask()['resp']._data[0, :]
psths = ctxs['val'].apply_mask()['psth']._data[0, :]
preds = ctxs['val'].apply_mask()['pred']._data[0, :]

lim = resps.shape[-1]

respp = ctxp['val'].apply_mask()['resp'].extract_channels([cellid])._data[0, :lim]
psthp = ctxp['val'].apply_mask()['psth'].extract_channels([cellid])._data[0, :lim]
predp = ctxp['val'].apply_mask()['pred0'].extract_channels([cellid])._data[0, :lim]

resppf = ctxpf['val'].apply_mask()['resp'].extract_channels([cellid])._data[0, :lim]
psthpf = ctxpf['val'].apply_mask()['psth'].extract_channels([cellid])._data[0, :lim]
predpf = ctxpf['val'].apply_mask()['pred'].extract_channels([cellid])._data[0, :lim]

respjk = ctxjk['val'].apply_mask()['resp'].extract_channels([cellid])._data[0, :lim]
psthjk = ctxjk['val'].apply_mask()['psth'].extract_channels([cellid])._data[0, :lim]
predjk = ctxjk['val'].apply_mask()['pred'].extract_channels([cellid])._data[0, :lim]

f, ax = plt.subplots(3, 1, figsize=(8, 6))

ax[0].set_title(cellid)
ax[0].plot(respp, label='pop model, resp')
ax[0].plot(resps, label='single cell model resp')

ax[1].set_title('psth')
ax[1].plot(psthp, label='pop model, resp')
ax[1].plot(psths, label='single cell model resp')

ax[2].set_title(f'pred / pred0')
ax[2].plot(predp, label='pop model, resp')
ax[2].plot(preds, label='single cell model resp')

ax[2].legend()

f.tight_layout()

plt.show()