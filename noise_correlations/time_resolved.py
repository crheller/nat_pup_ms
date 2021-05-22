"""
Play with some example sites and attempt to identify a time resolved measure of correlations (pop CV, sliding window noise corr etc.) to show that
changes in second order statistics really do correlate with pupil

Idea is that this would help control for spurious stuff due to the slow nature of pupil (think of Ken Harris paper)
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.preprocessing as nems_preproc
import load_results as ld
from nems.preprocessing import generate_psth_from_resp

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# load noise correlation results for comparison
df = ld.load_noise_correlation('rsc_ev')

win = 15 # total window size (non overlapping across data)
subwin = 0.25 # sub window size (mean rate across all / sd across all)
# CV = sd of spike counts across all subwindows divided by the mean across all sub windows
# If all neurons are Poisson and statistically independent, then the CV of the population rate will approach zero

site = 'TAR010c'
batch = 289

manager = BAPHYExperiment(cellid=site, batch=batch)
options = {'rasterfs': 4, 'resp': True, 'stim': False, 'pupil': True}
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()
if batch==331:
    rec = nems_preproc.fix_cpn_epochs(rec)
else:
    rec = nems_preproc.mask_high_repetion_stims(rec)
rec = generate_psth_from_resp(rec)

# extract continuous data (subtract psth?)
data = rec.apply_mask()['resp']._data #- rec.apply_mask()['psth_sp']._data
pupil = rec.apply_mask()['pupil']._data

# divide into bins
win_bin = int(rec['resp'].fs * win)
subwin_bin = int(rec['resp'].fs * subwin)
CV = []
bpupil = []
i = 0
while ((i * win_bin) <= data.shape[-1]):
    mean = data[:, int(i * win_bin):int((i+1) * win_bin)].mean()
    sd = data[:, int(i * win_bin):int((i+1) * win_bin)].std()
    CV.append(sd/mean)
    bpupil.append(pupil[:, int(i * win_bin):int((i+1) * win_bin)].mean(axis=-1))
    i+=1

CV = np.array(CV)
bpupil = np.concatenate(bpupil, axis=-1)

f = plt.figure(figsize=(12, 6))

pupilplot = plt.subplot2grid((2, 4), (0, 0), colspan=3)
cvplot = plt.subplot2grid((2, 4), (1, 0), colspan=3)
scat = plt.subplot2grid((2, 4), (0, 3))
rsc = plt.subplot2grid((2, 4), (1, 3))

pupilplot.plot(bpupil, label='pupil')
pupilplot.legend(frameon=False)
pupilplot.set_title(f"{site}, batch: {batch}")

cvplot.plot(CV, label='pop. CV')
cvplot.legend(frameon=False)

scat.scatter(bpupil, CV, edgecolor='white')
scat.set_xlabel('pupil'); scat.set_ylabel('CV')
scat.set_title(f"r: {round(np.corrcoef(CV, bpupil)[0, 1], 3)}")

df = df[(df.batch==batch) & (df.site==site)]
rsc.scatter(df['bp'], df['sp'], s=5, alpha=0.5, edgecolor='none')
rsc.plot([-0.5, 0.5], [-0.5, 0.5], 'k--')
rsc.axhline(0, linestyle='--', color='k')
rsc.axvline(0, linestyle='--', color='k')
rsc.set_xlabel("big pupil")
rsc.set_ylabel('small pupil')
rsc.set_title(f"noise corr.\nsmall: {round(df['sp'].mean(),3)}, large: {round(df['bp'].mean(), 3)}")

f.tight_layout()

plt.show()