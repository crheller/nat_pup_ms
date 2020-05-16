#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:26:49 2018

@author: hellerc
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import load_results as ld
import nems
import nems.db as nd
import nems_lbhb.xform_wrappers as xfw
import nems.xform_helper as xhelp
import nems.xforms as xforms
from nems import get_setting
from nems.plugins import (default_keywords, default_loaders, default_fitters,
                          default_initializers)
from nems.registry import KeywordRegistry
import nems.gui.editors as gui
import charlieTools.xforms_fit as xfit
import io
import nems_lbhb.plots as plots
import charlieTools.plotting as cplt
import logging
log = logging.getLogger(__name__)

cellid = 'DRX006b.e65:128'
batch = 289


modelname = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0'
modelname0 = 'ns.fs4.pup-ld.pop-st.pup-hrc-psthfr-aev_stategain.SxR_basic.t7'

#modelname = 'ns.fs20.pup-ld-st.pup.beh-psthfr.tar-tar_stategain.S_jk.nf20-basic' 
#cellid = 'DRX005c-04-1'
#batch = 302

xfspec, ctx = xfit.fit_xforms_model(batch, cellid, modelname, save_analysis=False)
_, ctx2 = xfit.fit_xforms_model(batch, cellid, modelname0, save_analysis=False)

plots.lv_quickplot(ctx['val'], ctx['modelspec'])
plots.lv_logsig_plot(ctx['val'], ctx['modelspec'])

# compare first order model weights between two fits
f, ax = plt.subplots(1, 1)

ax.plot(ctx['modelspec'].phi[0]['g'][:, 1], ctx2['modelspec'].phi[0]['g'][:, 1], 'k.')
ma = np.concatenate((ctx['modelspec'].phi[0]['g'][:, 1], ctx2['modelspec'].phi[0]['g'][:, 1])).max() 
mi = np.concatenate((ctx['modelspec'].phi[0]['g'][:, 1], ctx2['modelspec'].phi[0]['g'][:, 1])).min()
ax.plot([mi, ma], [mi, ma], '--', color='grey')
ax.set_xlabel('lv model first order gain', fontsize=8)
ax.set_ylabel('pupil model first order gain', fontsize=8)
ax.set_aspect(cplt.get_square_asp(ax))

# compare first order gain weights to second order gain
f, ax = plt.subplots(1, 1)

ax.plot(ctx2['modelspec'].phi[0]['g'][:, 1], ctx['modelspec'].phi[2]['g'][:, 1], 'k.')
ma = np.concatenate((ctx2['modelspec'].phi[0]['g'][:, 1], ctx['modelspec'].phi[2]['g'][:, 1])).max() 
mi = np.concatenate((ctx2['modelspec'].phi[0]['g'][:, 1], ctx['modelspec'].phi[2]['g'][:, 1])).min()
ax.plot([mi, ma], [mi, ma], '--', color='grey')
ax.axhline(0, linestyle='--', color='grey')
ax.axvline(0, linestyle='--', color='grey')
ax.set_xlabel('first order gain', fontsize=8)
ax.set_ylabel('second order gain', fontsize=8)
ax.set_aspect(cplt.get_square_asp(ax))


plt.show()