"""
Attempt to compare first and second order model weights.
"""
import matplotlib.pyplot as plt
import numpy as np

from nems.xform_helper import load_model_xform
import nems.db as nd

batch = 331
site = 'AMT020a'
model0 = 'psth.fs4.pup-ld-st.pup-epcpn-mvm.25.2-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000'
model = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"

xf0, ctx0 = load_model_xform(cellid=[c for c in nd.get_batch_cells(batch).cellid if site in c][0], modelname=model0, batch=331)
xf, ctx = load_model_xform(cellid=site, modelname=model, batch=331)