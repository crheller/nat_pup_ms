# %%
# goal is to determine if there is more motor activity during high arousal for the 331 data
# eventually, register this with noise correlation / populaiton metric results
# could be the reason for the increase in %sv for the latter

# %%
from loader import load_facePCs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 12

from nems_lbhb.baphy_experiment import BAPHYExperiment

import sys
sys.path.append("/auto/users/hellerc/code/projects/nat_pupil_ms/")
from global_settings import CPN_SITES, HIGHR_SITES

# %%
modelname="face_pca"
site = "TAR010c"
batch = 322
res = load_facePCs(site=site, batch=batch, modelname=modelname)
manager = BAPHYExperiment(cellid=site, batch=batch)
options = {'rasterfs': 4, 'resp': True, 'stim': False, 'pupil': True, 'pupil_variable_name': 'area', "verbose": False}
rec = manager.get_recording(**options)
pupil = rec["pupil"]._data[0, :]

# %%
f = plt.figure(figsize=(12, 6))

pcs = plt.subplot2grid((2, 4), (0, 0), colspan=3)
pupplot = plt.subplot2grid((2, 4), (1, 0), colspan=3)
varexp = plt.subplot2grid((2, 4), (0, 3))
comp1 = plt.subplot2grid((2, 4), (1, 3))

pcs.set_title("pc projections")
pcs.plot(res["proejction"][:, 0], label="pc1")
pcs.plot(res["proejction"][:, 1], label="pc2")
pcs.plot(res["proejction"][:, 2], label="pc3")
pcs.legend()

pupplot.plot(pupil)
pupplot.set_title("pupil")

varexp.plot(res["var_explained"][:10], "o-")
varexp.set_ylabel("variance exp.")

comp1.imshow(res["components"][0].reshape(224, 224), cmap="Greys")
comp1.set_title("PC1")


f.tight_layout()

# %%



