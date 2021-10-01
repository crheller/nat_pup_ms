"""
Pop model for single cell
"""
import nems.db as nd
from global_settings import CPN_SITES

batch = 294 # 294, 289 || 323 (PEG) | 331 (CPN)
batch = 331
force_rerun = True

cellids = CPN_SITES

script = '/auto/users/hellerc/code/NEMS/scripts/fit_single.py'
python_path = '/auto/users/hellerc/anaconda3/envs/tf/bin/python'

# 04/26/2020 - Queue sdexp models again. New sdexp architecture allows easy extraction 
# gain params. Should be easy to invert these models to get rid of first order pupil
modelnames = ['psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr-aev_stategain.SxR_tfinit.n.lr1e4.cont.et5.i50000',
              'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr-aev_sdexp2.SxR_tfinit.n.lr1e4.cont.et5.i50000',
              'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr_stategain.SxR_jk.nf10-tfinit.n.lr1e4.cont.et5.i50000',
              'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr_sdexp2.SxR_jk.nf10-tfinit.n.lr1e4.cont.et5.i50000'
                ]

nd.enqueue_models(celllist=cellids,
                  batch=batch,
                  modellist=modelnames,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  priority=2,
                  reserve_gb=2)
