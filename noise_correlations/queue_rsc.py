import nems.db as nd

batches = [289, 294]
force_rerun = True
boxcar = True
evoked = True
slow = False
custom = True  # run a subset of jobs
for batch in batches:

    if custom:
        modelnames = ['rsc_pr_rm1', 
                    'rsc_pr_rm1_fft0-0.05',
                    'rsc_pr_rm1_fft0.05-0.1',
                    'rsc_pr_rm1_fft0.1-0.25',
                    'rsc_pr_rm1_fft0.25-0.5',
                    'rsc_pr_rm1_fft0-0.1',
                    'rsc_pr_rm1_fft0-0.25',
                    'rsc_pr_rm1_fft0-0.5',
                    'rsc_pr_rm1_fft0.5-2',
                    'rsc_pr_rm1_fft2-4',
                    'rsc_pr_rm1_fft0.1-4',
                    'rsc_pr_rm1_fft0.25-4',
                    'rsc_pr_rm1_fft0.5-4',
                    'rsc_pr_rm1_fft4-10',
                    'rsc_pr_rm1_fft10-25',
                    'rsc_pr_rm1_fft25-50']
    else:

        modelnames = ['rsc', 'rsc_pr_rm2', 
                    'rsc_fft0-0.05', 'rsc_pr_rm2_fft0-0.05',
                    'rsc_fft0.05-0.1', 'rsc_pr_rm2_fft0.05-0.1',
                    'rsc_fft0.1-0.25', 'rsc_pr_rm2_fft0.1-0.25',
                    'rsc_fft0.25-0.5', 'rsc_pr_rm2_fft0.25-0.5',
                    'rsc_fft0-0.1', 'rsc_pr_rm2_fft0-0.1',
                    'rsc_fft0-0.25', 'rsc_pr_rm2_fft0-0.25',
                    'rsc_fft0-0.5', 'rsc_pr_rm2_fft0-0.5',
                    'rsc_fft0.5-2', 'rsc_pr_rm2_fft0.5-2',
                    'rsc_fft2-4', 'rsc_pr_rm2_fft2-4',
                    'rsc_fft0.1-4', 'rsc_pr_rm2_fft0.1-4',
                    'rsc_fft0.25-4', 'rsc_pr_rm2_fft0.25-4',
                    'rsc_fft0.5-4', 'rsc_pr_rm2_fft0.5-4',
                    'rsc_fft4-10', 'rsc_pr_rm2_fft4-10',
                    'rsc_fft10-25', 'rsc_pr_rm2_fft10-25',
                    'rsc_fft25-50', 'rsc_pr_rm2_fft25-50']

    if slow:
        modelnames = ['rsc', 'rsc_pr_rm2', 
                'rsc_fft0-0.05', 'rsc_pr_rm2_fft0-0.05',
                'rsc_fft0.05-0.1', 'rsc_pr_rm2_fft0.05-0.1',
                'rsc_fft0.1-0.25', 'rsc_pr_rm2_fft0.1-0.25',
                'rsc_fft0.25-0.5', 'rsc_pr_rm2_fft0.25-0.5',
                'rsc_fft0-0.1', 'rsc_pr_rm2_fft0-0.1',
                'rsc_fft0-0.25', 'rsc_pr_rm2_fft0-0.25',
                'rsc_fft0-0.5', 'rsc_pr_rm2_fft0-0.5',
                'rsc_fft0.5-2', 'rsc_pr_rm2_fft0.5-2']
                
    if boxcar:
        modelnames = [m.replace('fft', 'boxcar_fft') for m in modelnames]

    if evoked:
        modelnames = [m.replace('rsc', 'rsc_ev') for m in modelnames]

    if slow:
        modelnames = [m.replace('fft', 'fs4_fft') for m in modelnames]

    if batch == 289:
        sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
                'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
                'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
                'DRX006b.e1:64', 'DRX006b.e65:128',
                'DRX007a.e1:64', 'DRX007a.e65:128',
                'DRX008b.e1:64', 'DRX008b.e65:128']
    # BRT032, BRT033, and BRT037 and TAR009d??            
    if batch == 294:
        sites = ['BOL005c', 'BOL006b']

    script = '/auto/users/hellerc/code/projects/nat_pupil_ms/noise_correlations/cache_rsc.py'
    #script = '/auto/users/hellerc/code/projects/nat_pupil_ms/noise_correlations/cache_rsc_xforms.py'
    python_path = '/auto/users/hellerc/anaconda3/envs/crh_nems/bin/python'
    python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'
    nd.enqueue_models(celllist=sites,
                    batch=batch,
                    modellist=modelnames,
                    executable_path=python_path,
                    script_path=script,
                    user='hellerc',
                    force_rerun=force_rerun,
                    reserve_gb=2)
