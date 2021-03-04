all_siteids = {322: ['AMT005c','AMT018a','AMT019a','AMT020a',
   'AMT021b','AMT023d','AMT024b','bbl086b','bbl099g','bbl104h',
   'BRT026c','BRT034f','BRT036b','BRT038b','BRT039c',
   'DRX006b.e1:64','DRX006b.e65:128','DRX007a.e1:64','DRX007a.e65:128',
   'DRX008b.e1:64','DRX008b.e65:128','TAR010c','TAR017b'],
  323: ['ARM021b','AMT032a','AMT028b','ARM018a',
   'AMT029a','ARM019a','ARM022b','AMT031a','ARM017a']}
batch=322
cellid=all_siteids[batch][21]
modelnames={
    'indep': "psth.fs4.pup-loadpred-st.pup-plgsm-lvnoise.r4-aev_lvnorm.1xR.d-inoise.2xR_ccnorm.t5",
    'dc11': "psth.fs4.pup-loadpred-st.pup.pvp-plgsm-lvnoise.r4-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5",
    'dc10': "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm-lvnoise.r4-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5",
    'dc00': "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm-lvnoise.r4-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5",
    'gn11': "psth.fs4.pup-loadpred-st.pup.pvp-plgsm-lvnoise.r4-aev_lvnorm.SxR-inoise.2xR_ccnorm.t5",
    'gn10': "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm-lvnoise.r4-aev_lvnorm.SxR-inoise.2xR_ccnorm.t5",
    'gn00': "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm-lvnoise.r4-aev_lvnorm.SxR-inoise.2xR_ccnorm.t5",
}
from nems.xform_helper import load_model_xform
xf,ctx = load_model_xform(cellid,batch,modelname=modelnames['indep'])
import nems.plots.state as sp
d=sp.cc_comp(ctx['val'],ctx['modelspec']);

'''
dc00 - first order (baseline/gain) only, no noise or correlation
indep - independent (per neuron?) noise, no correlation
dc10 - full model, single pupil dependent LV ("rank 1"). LV is additive
Then if you want more complex:
dc11 - rank 2 LV
gn10 - multiplicative LV (instead of additive) - makes very little difference predicting total change in corr matrix between large and small pupil. Maybe more profound differences between stimuli?
'''