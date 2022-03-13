import nems.db as nd
from global_settings import HIGHR_SITES, CPN_SITES

batches = [294, 322, 331] #[289, 294, 331]
force_rerun = True
boxcar = True
evoked = True
slow = False
perstim = False
custom = False  # run a subset of jobs
movement_mask = False #(25, 2) #True

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

if perstim:
    modelnames = [m+'_perstim' for m in modelnames]

if movement_mask != False:
    if type(movement_mask)==tuple:
        modelnames = [m+f'_mvm-{movement_mask[0]}-{movement_mask[1]}' for m in modelnames]
    else:
        modelnames = [m+'_mvm' for m in modelnames]

modelnames = [m for m in modelnames if ('fft' not in m) & ('pr' not in m)]

for batch in batches:

    if batch == 322:
        sites = [s for s in HIGHR_SITES if s not in ['BOL005c', 'BOL006b']]  
    if batch == 294:
        sites = ['BOL005c', 'BOL006b']
    if batch == 331:
        sites = CPN_SITES

    script = '/auto/users/hellerc/code/projects/nat_pupil_ms/noise_correlations/cache_rsc.py'
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
