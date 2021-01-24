"""
Summary plots of r_test, gain vs. DC, and MI.
"""

import nems.db as nd

import matplotlib.pyplot as plt
import pandas as pd

batches = [289, 294, 323]
modelnames = ['ns.fs4.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic', 'ns.fs4.pup-ld-st.pup0-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic',
        'ns.fs4.pup.voc-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic', 'ns.fs4.pup.voc-ld-st.pup0-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic']
sql = "SELECT r_test, se_test, cellid, modelname, batch from Results WHERE modelname in {0} and batch in {1}".format(tuple(modelnames), tuple(batches))

results = nd.pd_query(sql)
results['state_mod'] = ['st.pup' if 'st.pup0' not in s else 'st.pup0' for s in results['modelname']]

results = results[results.batch==323]

r = results.pivot(columns='state_mod', index='cellid')
rdiff = r.loc[:, pd.IndexSlice['r_test', 'st.pup']] - r.loc[:, pd.IndexSlice['r_test', 'st.pup0']]
se = r.loc[:, pd.IndexSlice['se_test', 'st.pup']] + r.loc[:, pd.IndexSlice['se_test', 'st.pup0']]
sig_mask = rdiff > se

f, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(r.loc[:, pd.IndexSlice['r_test', 'st.pup0']] ** 2,
           r.loc[:, pd.IndexSlice['r_test', 'st.pup']] ** 2, 
           marker='o', edgecolor='white', color='grey')
ax.scatter(r.loc[sig_mask, pd.IndexSlice['r_test', 'st.pup0']] ** 2,
           r.loc[sig_mask, pd.IndexSlice['r_test', 'st.pup']] ** 2, 
           marker='o', edgecolor='white', color='k')

ax.plot([0, 1], [0, 1], 'k--')
ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')

ax.set_xlabel(r'$R^{2}$ st.pup0')
ax.set_ylabel(r"$R^{2}$ st.pup")
ax.set_title('{0} / {1} sig cells'.format(sum(sig_mask), r.shape[0]))

f.tight_layout()

plt.show()