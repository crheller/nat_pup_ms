"""
Load spike waveform params for all SUs in batch 289 and 294. Cluster based on this.
"""

import nems.db as nd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path = '/auto/users/hellerc/results/nat_pupil_ms/'
cellids_cache = path + 'celltypes.csv'

cellid_289 = pd.DataFrame(nd.get_batch_cells(289).cellid)
isolation_289 = [nd.get_isolation(cellid=c, batch=289).values[0][0] for c in cellid_289['cellid']]
cellid_289['isolation'] = isolation_289

cellid_294 = pd.DataFrame(nd.get_batch_cells(294, cellid='BOL').cellid)
isolation_294 = [nd.get_isolation(cellid=c, batch=294).values[0][0] for c in cellid_294['cellid']]
cellid_294['isolation'] = isolation_294

cellids = pd.concat([cellid_289, cellid_294])

# keep only SU
cellids = cellids[cellids.isolation>=95]
sw = [nd.get_gSingleCell_meta(cellid=c, fields='wft_spike_width') for c in cellids.cellid] 
cellids['spike_width'] = sw

# remove cellids that weren't sorted with KS (so don't have waveform stats)
cellids = cellids[cellids.spike_width!=-1]

# now save endslope and peak trough ratio
es = [nd.get_gSingleCell_meta(cellid=c, fields='wft_endslope') for c in cellids.cellid] 
pt = [nd.get_gSingleCell_meta(cellid=c, fields='wft_peak_trough_ratio') for c in cellids.cellid] 

cellids['end_slope'] = es
cellids['peak_trough'] = pt


g = sns.pairplot(cellids[['spike_width', 'end_slope', 'peak_trough']], diag_kind='kde')

plt.tight_layout()


# cluster using endsplope and spike width
km = KMeans(n_clusters=2).fit(cellids[['spike_width', 'end_slope']])
cellids['type'] = km.labels_

# looks like endslope and spike width most effect for clustering... focus in on them
f, ax = plt.subplots(1, 1, figsize=(5, 5))
g = sns.scatterplot(x='spike_width', y='end_slope', hue='type', data=cellids, s=25, ax=ax)

ax.set_xlabel('Spike Width (s)')
ax.set_ylabel('Spike endslope (dV / dt)')
ax.set_title(r"$n_{type=1} = %s / %s$" % (cellids['type'].sum(), cellids.shape[0]))

plt.tight_layout()

plt.show()

# cache cell type results dataframe
cellid.to_csv(cellids_cache)