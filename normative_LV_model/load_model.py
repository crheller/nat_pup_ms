"""
Load single model, extract weights / LVs etc.
Compare with true rec
Compare decoding
"""
from nems.xform_helper import load_model_xform

import matplotlib.pyplot as plt
import numpy as np

site = 'AMT020a'
batch = 331

modelname = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2"

xf, ctx = load_model_xform(site, batch, modelname=modelname)