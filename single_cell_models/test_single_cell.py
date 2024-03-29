
import matplotlib.pyplot as plt
import numpy as np
import charlieTools.xforms_fit as xfit
import nems.xform_helper as xhelp

cellid = 'TAR017b-33-3'  # weird cell that failed with pupil model. Does random init fix?
batch = 289
cellid = 'ARM033a-02-1'
batch = 331

# test single cell models
modelname = 'ns.fs4.pup-ld-st.pup-mvm-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'
modelname2 = 'ns.fs4.pup-ld-st.pup0-mvm-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'

if batch==331:
    modelname = modelname.replace('-st.pup', '-st.pup-epcpn')

xf, ctx = xhelp.fit_model_xform(batch=batch, cellid=cellid, modelname=modelname, saveInDB=False, returnModel=True)
xf, ctx2 = xfit.fit_xforms_model(batch, cellid, modelname2, save_analysis=False)

