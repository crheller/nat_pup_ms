import nems.db as nd

batch = 289
force_rerun = True

script = '/auto/users/hellerc/code/projects/nat_pupil_ms/single_cell_models/fit_script.py'
python_path = '/auto/users/hellerc/anaconda3/envs/crh_nems/bin/python'

# single cell model architectures, w and w/o evoked periods
modelnames = ['ns.fs4.pup-ld-st.pup-hrc-psthfr-aev_slogsig.SxR_basic',
              'ns.fs4.pup-ld-st.pup-hrc-psthfr-aev_sdexp.SxR_basic',
              'ns.fs4.pup-ld-st.pup-hrc-psthfr-aev_slogsig.SxR.d_basic',
              'ns.fs4.pup-ld-st.pup-hrc-psthfr-aev_stategain.SxR_basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr-aev_slogsig.SxR_basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr-aev_sdexp.SxR_basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr-aev_slogsig.SxR.d_basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr-aev_stategain.SxR_basic',
              'ns.fs4.pup-ld-st.pup-hrc-psthfr-ev-aev_slogsig.SxR_basic',
              'ns.fs4.pup-ld-st.pup-hrc-psthfr-ev-aev_sdexp.SxR_basic',
              'ns.fs4.pup-ld-st.pup-hrc-psthfr-ev-aev_slogsig.SxR.d_basic',
              'ns.fs4.pup-ld-st.pup-hrc-psthfr-ev-aev_stategain.SxR_basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr-ev-aev_slogsig.SxR_basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr-ev-aev_sdexp.SxR_basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr-ev-aev_slogsig.SxR.d_basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr-ev-aev_stategain.SxR_basic']

# 04/26/2020 - Queue sdexp models again. New sdexp architecture allows easy extraction 
# gain params. Should be easy to invert these models to get rid of first order pupil
modelnames = ['ns.fs4.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic',
              'ns.fs4.pup-ld-st.pup0-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic']

if batch == 294:
    modelnames = [m.replace('fs4.pup', 'fs4.pup.voc') for m in modelnames]

cellids = nd.get_batch_cells(batch).cellid.tolist()

if batch == 294:
    cellids = [c for c in cellids if c.split('-')[0] in ['BOL005c', 'BOL006b']]

nd.enqueue_models(celllist=cellids,
                  batch=batch,
                  modellist=modelnames,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=1)