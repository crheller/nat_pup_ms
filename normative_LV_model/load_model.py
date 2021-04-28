"""
Load single model, extract weights / LVs etc.
Compare with true rec
Compare decoding
"""
from nems.xform_helper import load_model_xform

import matplotlib.pyplot as plt
import numpy as np

site = 'TAR010c'
batch = 322

modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5.ss2"

xf, ctx = load_model_xform(site, batch, modelname=modelname)