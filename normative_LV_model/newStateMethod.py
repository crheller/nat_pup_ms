"""
Confirm that new state variable schema (st.pup+r1) does the same thing as (st.pup.pvp)

Confirmed, they give the exact same results.
"""
from nems.xform_helper import load_model_xform
from nems_lbhb.baphy_io import parse_cellid
import numpy as np

old = 'psth.fs4.pup-ld-st.pupGP.pvp-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.SxR-spred-lvnorm.2xR.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1'
new = 'psth.fs4.pup-ld-st.pup+r1+gp0-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.SxR-spred-lvnorm.2xR.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1'

site = "AMT020a"
batch = 331
cells, _ = parse_cellid({"batch": batch, "cellid": site})

xf, ctx = load_model_xform(cellid=cells[0], batch=batch, modelname=old)
xfn, ctxn = load_model_xform(cellid=cells[0], batch=batch, modelname=new)

# state signals should be identical
sdiff = np.sum(ctx["val"]["state"]._data - ctxn["val"]["state"]._data)
print(f"sum(diff of state signals): {sdiff}")
# phi should be identical
phidiff = np.sum(ctx["modelspec"].phi[0]["g"] - ctxn["modelspec"].phi[0]["g"])
print(f"sum(diff of phi): {phidiff}")
# pred should be identical
preddiff = np.nansum(ctx["val"]["pred"]._data - ctxn["val"]["pred"]._data)
print(f"sum(diff of state signals): {preddiff}")