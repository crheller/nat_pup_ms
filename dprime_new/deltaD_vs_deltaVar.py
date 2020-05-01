"""
Plot change in dprime as function of difference in normalized variance along each
eigenvector (lamba_alpha / sum(lambda_alpha_i))
"""
import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import copy
import scipy.stats as ss

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore'
modelname_raw = 'dprime_jk10_zscore'
n_components = 2
pc = 1

site = 'DRX006b.e1:64'

fn = os.path.join(path, site, modelname+'_PLS.pickle')
results = loader.load_results(fn)

if modelname != modelname_raw:
    fn = os.path.join(path, site, modelname_raw+'_PLS.pickle')
    results_raw = loader.load_results(fn)
else:
    results_raw = copy.deepcopy(results)

pairs = results.evoked_stimulus_pairs
bp_evals = np.stack(results_raw.slice_array_results('bp_evals', pairs, n_components)[0].values)
sp_evals = np.stack(results_raw.slice_array_results('sp_evals', pairs, n_components)[0].values)

bp_tot = bp_evals.sum(axis=-1)
bp_var = bp_evals[:, pc] / bp_tot

sp_tot = sp_evals.sum(axis=-1)
sp_var = sp_evals[:, pc] / sp_tot

bp_dp = results.get_result('bp_dp', pairs, n_components)[0]
sp_dp = results.get_result('sp_dp', pairs, n_components)[0]

df_hex = pd.DataFrame(index=bp_dp.index, columns=['bp_var', 'sp_var', 'dp_diff'])
df_hex['dp_diff'] = (bp_dp - sp_dp) #/ bp_dp
df_hex['bp_var'] = bp_var
df_hex['sp_var'] = sp_var
df_hex['var_diff'] = sp_var - bp_var

f, ax = plt.subplots(1, 2, figsize=(10, 5))
vmax = df_hex['dp_diff'].std()
nbins = 15
cmap = 'PRGn'
ax[0].set_title(r"$\Delta d'^{2}$")
df_hex.plot.hexbin(x='sp_var', 
                    y='bp_var', 
                    C='dp_diff', 
                    gridsize=nbins, ax=ax[0], cmap=cmap, vmin=-vmax, vmax=vmax) 
ma = np.max([np.max(bp_var), np.max(sp_var)])
ax[0].plot([0, ma], [0, ma], 'k--')
ax[0].set_ylabel(r'$\frac{\lambda_{alpha}}{\lambda_{1} + \lambda_{2}}$, large')
ax[0].set_xlabel(r'$\frac{\lambda_{alpha}}{\lambda_{1} + \lambda_{2}}$, small')

ax[1].set_title("Count")
g = df_hex.plot.hexbin(x='sp_var', 
                    y='bp_var', 
                    C=None, 
                    gridsize=nbins, ax=ax[1], cmap='Reds', vmin=0) 
ma = np.max([np.max(bp_var), np.max(sp_var)])
ax[1].plot([0, ma], [0, ma], 'k--')
ax[1].set_ylabel(r'$\frac{\lambda_{alpha}}{\lambda_{1} + \lambda_{2}}$, large')
ax[1].set_xlabel(r'$\frac{\lambda_{alpha}}{\lambda_{1} + \lambda_{2}}$, small')

f.canvas.set_window_title('alpha = {0}'.format(pc+1))

f.tight_layout()

# simply plot delta dprime as function of difference in bp / sp variance
g = sns.jointplot(x='var_diff', y='dp_diff', data=df_hex, kind='reg', 
                xlim=(-.25, .25), ylim=(-100, 100))
r, pval = np.round(ss.pearsonr(sp_var - bp_var, df_hex['dp_diff']), 3)
g.fig.suptitle(r"$r = %s, p = %s$" % (r, pval))
g.ax_joint.set_ylabel(r"$\Delta d'^{2}$", fontsize=12)
g.ax_joint.set_xlabel(r"$\frac{\lambda_{\alpha}}{\lambda_{1} + \lambda_{2}}$"
                        "\n"
                        "small minus big", fontsize=12)

g.fig.tight_layout()

#ax.set_title(r"r = %s, p = %s" % (r, pval))

plt.show()