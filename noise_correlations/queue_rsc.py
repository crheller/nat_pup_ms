import nems.db as nd

batch = 289
update = False
force_rerun = True

'''
modelnames = ['rsc', 'rsc_bal', 'rsc_pr_rm1', 'rsc_pr_bal_rm1', 'rsc_pr_rm2', 'rsc_pr_bal_rm2', 
              'rsc_pr_lvr_rm1', 'rsc_pr_lvr_bal_rm1', 'rsc_pr_lvr_rm2', 'rsc_pr_lvr_bal_rm2']
#modelnames = ['rsc_fft0-1', 'rsc_fft0.25-1', 'rsc_fft0.5-3', 'rsc_fft2-10', 'rsc_fft10-50']
#modelnames = ['rsc_pr_fft0-1', 'rsc_pr_fft0.25-1', 'rsc_pr_fft0.5-3', 'rsc_pr_fft2-10', 'rsc_pr_fft10-50']
#modelnames = ['rsc_pr_lvr_fft0-1', 'rsc_pr_lvr_fft0.25-1', 'rsc_pr_lvr_fft0.5-3', 'rsc_pr_lvr_fft2-10', 'rsc_pr_lvr_fft10-50']
'''

modelnames = ['rsc', 'rsc_pr_rm2']
modelnames = ['rsc_fft0-1', 'rsc_fft0.25-1', 'rsc_fft0.5-3', 'rsc_fft2-10', 'rsc_fft10-50',
                'rsc_pr_rm2_fft0-1', 'rsc_pr_rm2_fft0.25-1', 'rsc_pr_rm2_fft0.5-3', 'rsc_pr_rm2_fft2-10', 'rsc_pr_rm2_fft10-50']
modelnames = ['rsc_fft0-0.1', 'rsc_pr_rm2_fft0-0.1']
modelnames = ['rsc_fft0.25-4', 'rsc_pr_rm2_fft0.25-4',
              'rsc_fft5-10', 'rsc_pr_rm2_fft5-10',
              'rsc_fft25-50', 'rsc_pr_rm2_fft25-50']
modelnames = ['rsc_fft10-25', 'rsc_pr_rm2_fft10-25']
modelnames = ['rsc_fft0-0.05', 'rsc_pr_rm2_fft0-0.05',
               'rsc_fft0-0.5', 'rsc_pr_rm2_fft0-0.5']
modelnames = ['rsc_fft4-10', 'rsc_pr_rm2_fft4-10',
             'rsc_fft0.1-4', 'rsc_pr_rm2_fft0.1-4']

modelnames = ['rsc', 'rsc_pr_rm2', 
              'rsc_fft0-0.05', 'rsc_pr_rm2_fft0-0.05',
              'rsc_fft0.1-4', 'rsc_pr_rm2_fft0.1-4',
              'rsc_fft4-10', 'rsc_pr_rm2_fft4-10',
              'rsc_fft10-25', 'rsc_pr_rm2_fft10-25',
              'rsc_fft25-50', 'rsc_pr_rm2_fft25-50']
if boxcar:
    modelnames = [m.replace('fft', 'boxcar_fft') for m in modelnames]
    
if batch == 289:
    sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
            'DRX006b.e1:64', 'DRX006b.e65:128',
            'DRX007a.e1:64', 'DRX007a.e65:128',
            'DRX008b.e1:64', 'DRX008b.e65:128']
            
if batch == 294:
    sites = ['BOL005c', 'BOL006b']

script = '/auto/users/hellerc/code/projects/nat_pupil_ms/noise_correlations/cache_rsc.py'
#script = '/auto/users/hellerc/code/projects/nat_pupil_ms/noise_correlations/cache_rsc_xforms.py'
python_path = '/auto/users/hellerc/anaconda3/envs/crh_nems/bin/python'

nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modelnames,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=2)
