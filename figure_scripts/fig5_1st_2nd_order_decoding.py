"""
Demonstrate that both first and second order differences between large and small pupil 
contibute. 
Show that effects are in different areas of the heatmap.
    (with heatmap? Or with bar plots per quadrant? Or with linear regression model?)
"""

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = True

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig3_modeldprime.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore'
sim1 = 'dprime_sim1_jk10_zscore'
sim2 = 'dprime_sim2_jk10_zscore'
estval = '_train'
high_var_only = True

# where to crop the data
if estval == '_train':
    x_cut = (2, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 9)
    y_cut = (0.35, 1) 