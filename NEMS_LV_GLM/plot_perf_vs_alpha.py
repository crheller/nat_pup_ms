import nems.db as nd
import matplotlib.pyplot as plt
from itertools import permutations
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
batch = 289
metric = 'r_test'

# LV only models
modelname = 'ns.fs4.pup-ld-hrc-apm-pbal-psthfr-ev-residual_lv.2xR.f.s-lvlogsig.3xR.ipsth_jk.nf5.p-pupLVbasic.constrLVonly.a{}'
modelname0 = 'ns.fs4.pup-ld-hrc-apm-pbal-psthfr-ev-residual0_lv.2xR.f.s-lvlogsig.3xR.ipsth_jk.nf5.p-pupLVbasic.constrLVonly.a{}'
pmodelname = 'ns.fs4.pup-ld-st.pup-hrc-epsig-apm-pbal-psthfr-ev_slogsig.SxR_jk.nf5.p-basic'

modelname = 'ns.fs4.pup-ld-hrc-apm-pbal-psthfr-ev-residual-addmeta_lv.2xR.f.s-lvlogsig.3xR.ipsth_jk.nf5.p-pupLVbasic.constrLVonly.af{0}.as{1}.rb10'

alpha1 = np.round(np.arange(0, 1, 0.05), 2)
a_unique = alpha1
alpha2= np.round(np.arange(0, 1, 0.05), 2)
alpha = set(list(permutations(np.concatenate((alpha1, alpha2)), 2)))
alpha = [a for a in alpha if (a[0] + a[1]) < 1]
asort = np.argsort(np.array(alpha), axis=0)
alpha = np.array(alpha)[asort[:,0]]
for site in sites:

    query = "SELECT {0}, {1} FROM NarfResults WHERE modelname = %s AND cellid like %s".format(metric, 'cellid')
    m = np.zeros(len(alpha))
    sem = np.zeros(len(alpha))
    m0 = np.zeros(len(alpha))
    sem0 = np.zeros(len(alpha))
    mp = np.zeros(len(alpha))
    semp = np.zeros(len(alpha))
    grid = np.nan*np.ones((len(a_unique), len(a_unique)))

    for i, a in enumerate(alpha):
        # save performance

        # pupil only first
        mn = pmodelname
        params = (mn, site+'%')
        results = nd.pd_query(sql=query, params=params)
        perf_m = results[metric].mean()
        perf_sem = results[metric].sem()

        mp[i] = perf_m
        semp[i] = perf_sem

        # LV
        mn = modelname.format(str(a[0]).replace('.', ':'), str(a[1]).replace('.', ':'))
        params = (mn, site+'%')
        results = nd.pd_query(sql=query, params=params)
        perf_m = results[metric].mean()
        perf_sem = results[metric].sem()

        m[i] = perf_m
        sem[i] = perf_sem

        x = np.argwhere(a_unique==a[0])[0][0]
        y = np.argwhere(a_unique==a[1])[0][0]

        grid[x, y] = perf_m

        # LV shuf
        #mn = modelname0.format(str(a).replace('.', ':'))
        #params = (mn, site+'%')
        #results = nd.pd_query(sql=query, params=params)
        #perf_m = results[metric].mean()
        #perf_sem = results[metric].sem()

        #m0[i] = perf_m
        #sem0[i] = perf_sem

    f, ax = plt.subplots(1, 1)
    f.canvas.set_window_title(site)

    ax.plot(range(len(alpha)), m, color='k', label='full LV model')
    ax.fill_between(range(len(alpha)), m-sem, m+sem, alpha=0.3, color=ax.get_lines()[-1].get_color(), lw=0)
    #ax.plot(alpha, m0, color='r', linestyle='--', label='shuff. LV model')
    #ax.fill_between(alpha, m0-sem0, m0+sem0, alpha=0.3, color=ax.get_lines()[-1].get_color(), lw=0)
    ax.plot(range(len(alpha)), mp, color='green', linestyle='--', label='pupil model')
    ax.fill_between(range(len(alpha)), mp-semp, mp+semp, alpha=0.3, color=ax.get_lines()[-1].get_color(), lw=0)

    ax.set_xticks(range(len(alpha)))
    ax.set_xticklabels([str(a[0])+'_'+str(a[1]) for a in alpha], fontsize=6, rotation=90)

    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel('alpha')
    ax.set_ylabel(metric)
    f.tight_layout()

    f, ax = plt.subplots(1, 1)

    ax.imshow(grid, aspect='auto', cmap='Greys')
    ax.set_title(metric)
    ax.set_xlabel('fast alpha')
    ax.set_ylabel('slow alpha')

plt.show()