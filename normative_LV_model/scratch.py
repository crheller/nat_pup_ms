"""
testing some new model fitting pipelines where we fit first / second order in the same model

Confirm that the first order fit is converging to what the loadpred models were doing
"""
from nems.xform_helper import load_model_xform
import nems.db as nd
import matplotlib.pyplot as plt

# loadpred first order model
lp_model = 'psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w2-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000'
# load pred lv model
lp_model = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"

# new model (noticing now this could be weird because two pupil state channels for sdexp)
model = "psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.xx1.it5000"
model = "psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w2-hrc-psthfr-aev_sdexp2.SxR_basic"
model = "psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w2-hrc-psthfr-plgsm.e10.sp-aev_sdexp2.2xR-lvnorm.SxR.d.so-inoise.2xR_init.xx1.it50000-lvnoise.r8-aev-ccnorm.f0.ss1"                          
#"psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w1-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.xx1.it5000-ccnorm.f0.ss1",
#"psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w1-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.xx1.it5000",
#"psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w1-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.xx1.it5000-ccnorm.f0.ss1-ccnorm.r.ss1"

site = 'CRD018d'
batch = 331

cellid = [c for c in nd.get_batch_cells(batch).cellid if site in c][0]
xf1, ctx1 = load_model_xform(cellid=site, batch=batch, modelname=lp_model)
xf2, ctx2 = load_model_xform(cellid=cellid, batch=batch, modelname=model)

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.scatter(ctx2['modelspec'][0]['meta']['state_mod'][1, :], ctx1['modelspec'][0]['meta']['state_mod'][1, :], edgecolor='white')

ax.set_xlabel(model, fontsize=6)
ax.set_ylabel(lp_model, fontsize=6)
ax.plot([-1, 1], [-1, 1], 'k--', zorder=-1)
ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')
ax.axis('equal')
f.tight_layout()

plt.show()

# compare second order (LV weights) for the old (loadpred) model and the new model
f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.scatter(ctx2['modelspec'][1]['phi']['g'][:, 1], ctx1['modelspec'][0]['phi']['g'][:, 1], edgecolor='white')

ax.set_xlabel(model, fontsize=6)
ax.set_ylabel(lp_model, fontsize=6)
ax.plot([-1, 1], [-1, 1], 'k--', zorder=-1)
ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')
ax.axis('equal')
ax.set_title("gain weights for LV")
f.tight_layout()

plt.show()
