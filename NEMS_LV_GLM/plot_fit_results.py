import nems.db as nd
import nems.xforms as xforms
import nems_lbhb.plots as plots
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss

import charlieTools.plotting as cplt

batch = 294
site = 'BOL005c'

modelname = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-ev-aev_stategain.SxR-lv.1xR.f.pred.hp0,1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7'
modelname = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3'
if batch == 294:
    modelname = modelname.replace('fs4.pup', 'fs4.pup.voc')

cellid = [c for c in nd.get_batch_cells(batch).cellid if site in c][0]
mp = nd.get_results_file(batch, [modelname], [cellid]).modelpath[0]

xfspec, ctx = xforms.load_analysis(mp)

plots.lv_quickplot(ctx['val'], ctx['modelspec']) 
#plots.lv_logsig_plot(ctx['val'], ctx['modelspec'])

# plot decoding weights for slow vs. fast
f, ax = plt.subplots(1, 1)

ax.plot(ctx['modelspec'].phi[0]['g'][:, 1], ctx['modelspec'].phi[2]['g'][:, 1], 'k.')
ax.set_xlabel('Pupil loading')
ax.set_ylabel('LV loading')
ax.axhline(0, linestyle='--', color='grey')
ax.axvline(0, linestyle='--', color='grey')
ax.set_aspect(cplt.get_square_asp(ax))

plt.show()
