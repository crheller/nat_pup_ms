"""
Compare gain, DC changes.
    Probably makes the most sense to show just gain, if it's straightforward, because only gain should 
    help decoding.
"""
from single_cell_models.mod_per_state import get_model_results_per_state_model
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

sig_only = True
recache = False

path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'

if recache:
    state_list = ['st.pup0','st.pup']
    basemodel2 = "-hrc-psthfr_sdexp.SxR.bound"
    loader = "ns.fs4.pup-ld-"
    fitter = '_jk.nf10-basic'
    batches = [294, 289]
    for batch in batches:
        if batch == 294:
            _loader = loader.replace('fs4.pup', 'fs4.pup.voc')
        else:
            _loader = loader
        d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                            basemodel=basemodel2, loader=_loader, fitter=fitter)
        d.to_csv(os.path.join(path, 'd_{}_pup_sdexp.csv'.format(batch)))

df = pd.concat([pd.read_csv(os.path.join(path,'d_289_pup_sdexp.csv'), index_col=0),
                pd.read_csv(os.path.join(path,'d_294_pup_sdexp.csv'), index_col=0)])
try:
    df['r'] = [np.float(r.strip('[]')) for r in df['r'].values]
    df['r_se'] = [np.float(r.strip('[]')) for r in df['r_se'].values]
except:
    pass

df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se'])
dc = df.loc[:, pd.IndexSlice['dc_mod', 'st.pup']]
gain = df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']]
sig = (df.loc[:, pd.IndexSlice['r', 'st.pup']] - df.loc[:, pd.IndexSlice['r', 'st.pup0']]) > \
            (df.loc[:, pd.IndexSlice['r_se', 'st.pup']] + df.loc[:, pd.IndexSlice['r_se', 'st.pup0']])
nsig = sig.sum()
ntot = gain.shape[0]
# plot gain vs. dc
f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(dc, gain, color='k', edgecolor='white', s=25)

ax[0].axhline(0, linestyle='--', color='grey')
ax[0].axvline(0, linestyle='--', color='grey')

ax[0].set_xlabel('DC')
ax[0].set_ylabel('gain')
ax[0].set_title('All units')


celltypes = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/celltypes.csv')
bs_cells = celltypes[celltypes.type==0].cellid
ns_cells = celltypes[celltypes.type==1].cellid
if sig_only:
    bs_cells = [c for c in bs_cells if c in df[sig].index]
    ns_cells = [c for c in ns_cells if c in df[sig].index]

ax[1].scatter(dc.loc[bs_cells], gain.loc[bs_cells], color='b', edgecolor='white', s=25, label='BS')
ax[1].scatter(dc.loc[ns_cells], gain.loc[ns_cells], color='r', edgecolor='white', s=25, label='NS')

ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0, linestyle='--', color='grey')

ax[1].set_xlabel('DC')
ax[1].set_ylabel('gain')
ax[1].legend(frameon=False)

m1 = np.round(gain.loc[bs_cells].mean(), 3)
m2 = np.round(gain.loc[ns_cells].mean(), 3)
pval = np.round(ss.ranksums(gain.loc[bs_cells], gain.loc[ns_cells]).pvalue, 3)

m1d = np.round(dc.loc[bs_cells].mean(), 3)
m2d = np.round(dc.loc[ns_cells].mean(), 3)
pvald = np.round(ss.ranksums(dc.loc[bs_cells], dc.loc[ns_cells]).pvalue, 3)

ax[1].set_title('SU only \n'
                    r"$\mu_{BS, g} = %s, \mu_{NS, g} = %s, pval = %s$"
                    "\n"
                    r"$\mu_{BS, DC} = %s, \mu_{NS, DC} = %s, pval = %s$"  % \
                        (m1, m2, pval, m1d, m2d, pvald), fontsize=10)

f.tight_layout()

# histogram of gain modulation
bins = np.arange(-0.2, 1, 0.05)
f, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.hist(gain, bins=bins, color='lightgrey', edgecolor='k', lw=1, rwidth=0.7)

ax.set_title(r"$\mu_{group} = %s \pm %s$ "
                "\n"
              "%s / %s sig. pupil cells " % (round(gain.mean(), 3), round(gain.sem(), 3), nsig, ntot), fontsize=12)
ax.set_xlabel(r"$\bar g_{large} - \bar g_{small}$", fontsize=12)
ax.set_ylabel(r"$n_{cells}$", fontsize=12)

f.tight_layout()

# histogram of DC modulation
bins = np.arange(-4, 4, 0.2)
f, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.hist(dc, bins=bins, color='lightgrey', edgecolor='k', lw=1, rwidth=0.7)

ax.set_title(r"$\mu_{group} = %s \pm %s$ "
                "\n"
              "%s / %s sig. pupil cells " % (round(dc.mean(), 3), round(dc.sem(), 3), nsig, ntot), fontsize=12)
ax.set_xlabel(r"$\bar dc_{large} - \bar dc_{small}$", fontsize=12)
ax.set_ylabel(r"$n_{cells}$", fontsize=12)

f.tight_layout()

plt.show()