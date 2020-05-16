import nems.db as nd
import numpy as np
from itertools import permutations

batch = 294
force_rerun = True

# LV model using stategain for first order effects. stategain for single cell only agrees with stategain for single cells 
# in pop model. for others (sdexp, slogsig etc.), single cell on its own does better than when in population for some reason.
# to be consistent / keep things interepretable, sticking with stategain.
modelnames = [
               'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
               'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.2xR.f2.pred-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
               'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.3xR.f3.pred-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0'
]
# 0.01 Hz
modelnames += [
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred.hp0,01-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.2xR.f2.pred.hp0,01-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.3xR.f3.pred.hp0,01-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0'
]
# add highpass filter models (0.1 Hz)
modelnames += [
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred.hp0,1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.2xR.f2.pred.hp0,1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.3xR.f3.pred.hp0,1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0'
]
# 0.5 Hz
modelnames += [
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred.hp0,5-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.2xR.f2.pred.hp0,5-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.3xR.f3.pred.hp0,5-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0'
]
# 1.0 Hz
modelnames += [
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred.hp1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.2xR.f2.pred.hp1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
              'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.3xR.f3.pred.hp1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
]

# add alpha constraints
m1 = [m.replace('.pLV0', '.pLV0,1') for m in modelnames]
m2 = [m.replace('.pLV0', '.pLV0,2') for m in modelnames]
m3 = [m.replace('.pLV0', '.pLV0,3') for m in modelnames]

modelnames += m1
modelnames += m2
modelnames += m3 

# Finally, for pLV fitter you need to add meta data (for reference length)
modelnames = [m.replace('-aev', '-addmeta-aev') for m in modelnames]

if batch == 294:
    modelnames = [m.replace('fs4.pup', 'fs4.pup.voc') for m in modelnames]

# ====================== For VOC data =======================
# slow only model
#lv = 'ns.fs4.pup.voc-ld-hrc-apm-psthfr-ev-residual-addmeta_lv.1xR.s-lvlogsig.2xR.ipsth_jk.nf5.p-pupLVbasic.constrLVonly.a{}'

# full lv model
#lv = 'ns.fs4.pup.voc-ld-hrc-apm-psthfr-ev-residual-addmeta_lv.2xR.f.s-lvlogsig.3xR.ipsth_jk.nf5.p-pupLVbasic.constrLVonly.af{0}.as{1}.sc.rb10'

if batch == 289:
    #sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
    #        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
    #        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b']
    # 
    #
    sites = ['DRX006b.e1:64', 'DRX006b.e65:128',
             'DRX007a.e1:64', 'DRX007a.e65:128',
             'DRX008b.e1:64', 'DRX008b.e65:128']
elif batch == 294:
    sites = ['BOL005c', 'BOL006b']

script = '/auto/users/hellerc/code/projects/nat_pupil_ms/NEMS_GLM/fit_nems_glm.py'
python_path = '/auto/users/hellerc/anaconda3/envs/crh_nems/bin/python'

nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modelnames,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=4)