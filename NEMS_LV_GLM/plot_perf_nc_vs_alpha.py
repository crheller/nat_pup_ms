"""
Plot overall model performance for each site as a function of the pupil constraint weight (alpha)
"""

import nems.db as nd
import matplotlib.pyplot as plt
import numpy as np
import noise_correlations as nc 
from nems_lbhb.preprocessing import create_pupil_mask
import nems.xforms as xforms
import charlieTools.preprocessing as preproc
import charlieTools.plotting as cplt
import json
import sys
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime/')
import dprime_helpers as helpers

sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b']
#sites = ['AMT020a']
batch = 289
correction_method = 2  # 1 means brute force regress out variable, 2 means subtract model prediction, add back psth
metric = 'mse_test'
#modelname = 'ns.fs4.pup-ld-st.pup-hrc-psthfr-ev_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.a{}'

# gain only model, evoked only
modelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-psthfr-ev_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a{}'
modelname0 = 'ns.fs4.pup-ld-st.pup-hrc-apm-psthfr-ev_slogsig.SxR-lv.1xR.shuf-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a{}'

# gain + DC model with spont
modelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr_slogsig.SxR.d-lv.1xR-lvlogsig.2xR.d_jk.nf5.p-pupLVbasic.constrNC.a{}'
modelname0 = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr_slogsig.SxR.d-lv.1xR.shuf-lvlogsig.2xR.d_jk.nf5.p-pupLVbasic.constrNC.a{}'

# gain + DC model evoked only
modelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev_slogsig.SxR.d-lv.1xR-lvlogsig.2xR.d_jk.nf5.p-pupLVbasic.constrNC.a{}'
modelname0 = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev_slogsig.SxR.d-lv.1xR.shuf-lvlogsig.2xR.d_jk.nf5.p-pupLVbasic.constrNC.a{}'

# new method for residual calc, gain only
modelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-psthfr-ev-residual_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a{}'
modelname0 = 'ns.fs4.pup-ld-st.pup-hrc-apm-psthfr-ev-residual0_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a{}'
pmodelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-psthfr-ev_slogsig.SxR_jk.nf5.p-basic'

# new method for residual calc, gain only, balanced
modelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev-residual_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a{}'
modelname0 = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev-residual0_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a{}'
pmodelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev_slogsig.SxR_jk.nf5.p-basic'

# new method for residual calc, gain only, balanced, perstim constaint
modelname = 'ns.fs4.pup-ld-st.pup-hrc-epsig-apm-pbal-psthfr-ev-residual_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a{}'
modelname0 = 'ns.fs4.pup-ld-st.pup-hrc-epsig-apm-pbal-psthfr-ev-residual0_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a{}'
pmodelname = 'ns.fs4.pup-ld-st.pup-hrc-epsig-apm-pbal-psthfr-ev_slogsig.SxR_jk.nf5.p-basic'


alpha = np.round(np.arange(0, 1, 0.05), 2)
for site in sites:

    mn = '.'.join(modelname.split('.')[:-1])
    best_lv_model = helpers.choose_best_model(site, batch, mn, pmodelname, corr_method=correction_method)

    query = "SELECT {0}, {1}, {2} FROM NarfResults WHERE modelname = %s AND cellid like %s".format(metric, 'extra_results', 'cellid')
    m = np.zeros(alpha.shape)
    sem = np.zeros(alpha.shape)
    m0 = np.zeros(alpha.shape)
    sem0 = np.zeros(alpha.shape)
    mp = np.zeros(alpha.shape)
    semp = np.zeros(alpha.shape)
    delta_nc =  np.zeros(alpha.shape)
    delta_nc_sem = np.zeros(alpha.shape)
    delta_nc0 = np.zeros(alpha.shape)
    delta_nc_sem0 = np.zeros(alpha.shape)
    delta_ncp = np.zeros(alpha.shape)
    delta_nc_semp = np.zeros(alpha.shape)
    for i, a in enumerate(alpha):
        # save performance

        # pupil only first
        mn = pmodelname
        params = (mn, site+'%')
        results = nd.pd_query(sql=query, params=params)
        perf_m = results[metric].mean()
        perf_sem = results[metric].sem()
        extra = json.loads(results['extra_results'][0])
        if correction_method==1:
            dnc = (extra['rsc_perstim_small'] - extra['rsc_perstim_big'])
            dnc_sem = np.sqrt(extra['rsc_perstim_small_sem']**2 + extra['rsc_perstim_big_sem']**2)
        elif correction_method==2:
            dnc = (extra['rsc_perstim_small2'] - extra['rsc_perstim_big2'])
            dnc_sem = np.sqrt(extra['rsc_perstim_small_sem2']**2 + extra['rsc_perstim_big_sem2']**2)

        mp[i] = perf_m
        semp[i] = perf_sem
        delta_ncp[i] = dnc
        delta_nc_semp[i] = dnc_sem

        # LV Model
        astring = str(a).replace('.', ':')
        mn = modelname.format(astring)
        if mn == best_lv_model:
             alpha_best = a

        params = (mn, site+'%')
        results = nd.pd_query(sql=query, params=params)
        perf_m = results[metric].mean()
        perf_sem = results[metric].sem()
        extra = json.loads(results['extra_results'][0])
        if correction_method==1:
            dnc = (extra['rsc_perstim_small'] - extra['rsc_perstim_big'])
            dnc_sem = np.sqrt(extra['rsc_perstim_small_sem']**2 + extra['rsc_perstim_big_sem']**2)
        elif correction_method==2:
            dnc = (extra['rsc_perstim_small2'] - extra['rsc_perstim_big2'])
            dnc_sem = np.sqrt(extra['rsc_perstim_small_sem2']**2 + extra['rsc_perstim_big_sem2']**2)

        mn0 = modelname0.format(astring)
        params = (mn0, site+'%')
        results = nd.pd_query(sql=query, params=params)
        try:
            perf_m0 = results[metric].mean()
            perf_sem0 = results[metric].sem()
            extra = json.loads(results['extra_results'][0])
            if correction_method==1:
                dnc0 = extra['rsc_perstim_small'] - extra['rsc_perstim_big']
                dnc_sem0 = np.sqrt(extra['rsc_perstim_small_sem']**2 + extra['rsc_perstim_big_sem']**2)
            elif correction_method==2:
                dnc0 = extra['rsc_perstim_small2'] - extra['rsc_perstim_big2']
                dnc_sem0 = np.sqrt(extra['rsc_perstim_small_sem2']**2 + extra['rsc_perstim_big_sem2']**2)

            m0[i] = perf_m0
            sem0[i] = perf_sem0

            delta_nc0[i] = dnc0
            delta_nc_sem0[i] = dnc_sem0
        except: 
            "shuff fit doesn't exists"
            m0[i] = np.nan
            sem0[i] = np.nan

            delta_nc0[i] = np.nan
            delta_nc_sem0[i] = np.nan
        
        m[i] = perf_m
        sem[i] = perf_sem
        delta_nc[i] = dnc
        delta_nc_sem[i] = dnc_sem
        
    f, ax = plt.subplots(1, 2)

    f.suptitle(site)
    #temp = alpha[0]
    #alpha[0] = 0.9
    ax[0].plot(alpha, m, 'o-', color='k', label='full model')
    ax[0].axvline(alpha_best, color='k', linestyle='--')
    ax[0].fill_between(alpha, m+sem, m-sem, color='lightgrey', alpha=0.5)
    ax[0].plot(alpha, m0, '--', color='r', label='shuffle LV')
    ax[0].fill_between(alpha, m0+sem0, m0-sem0, color='coral', alpha=0.5)
    ax[0].plot(alpha, mp, '--', color='g', label='pupil only')
    ax[0].fill_between(alpha, mp+semp, mp-semp, color='lightgreen', alpha=0.5)
    ax[0].set_ylabel(metric)
    ax[0].set_xlabel('alpha')
    ax[0].legend(fontsize=8)
    ax[0].set_aspect(cplt.get_square_asp(ax[0]))
    
    ax[1].plot(alpha, delta_nc, 'o-', color='k', label='full model')
    ax[1].axvline(alpha_best, color='k', linestyle='--')
    ax[1].fill_between(alpha, delta_nc-delta_nc_sem, delta_nc+delta_nc_sem, color='lightgrey', alpha=0.5)
    ax[1].plot(alpha, delta_nc0, '--', color='r', label='shuffle LV')
    ax[1].fill_between(alpha, delta_nc0-delta_nc_sem0, delta_nc0+delta_nc_sem0, color='coral', alpha=0.5)
    ax[1].plot(alpha, delta_ncp, '--', color='g', label='pupil only')
    ax[1].fill_between(alpha, delta_ncp-delta_nc_semp, delta_ncp+delta_nc_semp, color='lightgreen', alpha=0.5)
    ax[1].axhline(0, linestyle='--', color='k', lw=3)
    ax[1].set_ylabel('Delta noise corr.')
    ax[1].set_xlabel('alpha')
    ax[1].set_aspect(cplt.get_square_asp(ax[1]))

    #alpha[0] = temp
    f.tight_layout()

    f.savefig('/auto/users/hellerc/code/projects/nat_pupil_ms_final/NEMS_GLM/figures/{}.png'.format(site))

plt.show()