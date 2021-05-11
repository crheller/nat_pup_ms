"""
Screen for sites with low(er) pupil variance. Might exclude these from analysis as pupil (arousal) effects can't be easily measured here.
"""
import matplotlib.pyplot as plt

from charlieTools.nat_sounds_ms.decoding import load_site
from global_settings import CPN_SITES

sites = CPN_SITES
batches = [331]*len(sites)

for site, batch, in zip(sites, batches):
    # Load data that gets used for decoding analysis (so, pupil isn't actually in order)
    X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)

    pupil = X_pup.flatten()
    pupil -= np.mean(pupil)
    pupil /= np.std(pupil)
    # plot pupil and pupil histogram
    f = plt.figure(figsize=(8, 2))
    ptrace = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    phist = plt.subplot2grid((1, 4), (0, 3))

    ptrace.plot(pupil)
    phist.hist(pupil, bins=50)

    f.canvas.set_window_title(site)

    f.tight_layout()

plt.show()