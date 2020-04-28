import nems.xform_helper as xhelp
import matplotlib.pyplot as plt
import numpy as np

cellid = 'bbl099g-31-1'
batch = 289
modelname = 'ns.fs4.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'

xf, ctx = xhelp.load_model_xform(cellid=cellid, batch=batch, modelname=modelname)
r = ctx['val'].apply_mask()

f, ax = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

ax[0].plot(r['resp']._data.T, color='grey', label='resp')
ax[0].plot(r['pred']._data.T, color='k', label='pred')
ax[0].legend()

ax[1].plot(r['gain']._data.T, label='gain')
ax[1].plot(r['dc']._data.T, label='DC')
ax[1].legend()

pupil_corrected_resp1 = r['resp']._data.T / r['gain']._data.T - r['dc']._data.T
ax[2].plot(pupil_corrected_resp1, label='pupil correction')
ax[2].plot(r['psth_sp']._data.T, label='psth')
ax[2].legend()

pupil_corrected_resp2 = r['resp']._data.T - r['pred']._data.T + r['psth_sp']._data.T
ax[3].plot(pupil_corrected_resp2, label='pupil correction')
ax[3].plot(r['psth_sp']._data.T, label='psth')
ax[3].legend()


f.tight_layout()

# plot correlation of residuals with pupil before / after
resp_resid = r['resp']._data.T - r['psth_sp']._data.T
corr_resid1 = pupil_corrected_resp1 - r['psth_sp']._data.T
corr_resid2 = pupil_corrected_resp2 - r['psth_sp']._data.T
pupil = r['pupil']._data.T

f, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].scatter(pupil, resp_resid, s=20)
ax[0].set_title(r"$r_r = {0}$".format(np.round(np.corrcoef(pupil.T, resp_resid.T)[0, 1], 3)))

ax[1].scatter(pupil, corr_resid1, s=20)
ax[1].set_title(r"$r_c = {0}$".format(np.round(np.corrcoef(pupil.T, corr_resid1.T)[0, 1], 3)))

ax[2].scatter(pupil, corr_resid2, s=20)
ax[2].set_title(r"$r_c = {0}$".format(np.round(np.corrcoef(pupil.T, corr_resid2.T)[0, 1], 3)))

plt.show()