
import matplotlib.pyplot as plt
import numpy as np
import charlieTools.xforms_fit as xfit

cellid = 'TAR010c-27-3'
batch = 289

# test single cell models
modelname = 'ns.fs4.pup-ld-st.pup-hrc-psthfr-ev-aev_slogsig.SxR.d_basic'
modelname2 = 'ns.fs4.pup-ld-st.pup-hrc-psthfr-ev-aev_sdexp.SxR_basic'

ctx = xfit.fit_xforms_model(batch, cellid, modelname, save_analysis=False)
ctx2 = xfit.fit_xforms_model(batch, cellid, modelname2, save_analysis=False)

