"""
Plot decoding per site on a heatmap. Axis are |dU| and overlap of eigenvector n with signal (dU).
Plot test deocding performance.
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

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore'
xaxis_data = 'dprime_jk10_zscore'
nbins = 20
val = 'state_diff'  #'state_diff'  #'dp_opt_test'
xaxis = 'cos_dU_evec'  #'dU_dot_evec_sq'  # 'cos_dU_evec'
high_var_only = False
plot_sites = False

if val == 'state_diff':
    vmin = -1
    vmax = 1
    cmap = 'PRGn'
else:
    vmin = 0
    vmax = None
    cmap = 'Greens'


sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']
results_dict = {'a1': [], 'a2': []}
for site in sites:

    fn = os.path.join(path, site, modelname+'_TDR.pickle')
    results = loader.load_results(fn)

    fn = os.path.join(path, site, xaxis_data+'_TDR.pickle')
    results_raw = loader.load_results(fn)

    n_components = 2

    df = results.numeric_results
    df['dp_opt_test'] = df['dp_opt_test'] #.pow(1/2)
    df['state_diff'] = (df['bp_dp'].pow(1) - df['sp_dp'].pow(1)) / df['dp_opt_test'].pow(1)

    stim = results.evoked_stimulus_pairs
    high_var_pairs = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/high_pvar_stim_combos.csv', index_col=0)
    high_var_pairs = high_var_pairs[high_var_pairs.site==site].index.get_level_values('combo')
    if high_var_only:
        stim = [s for s in stim if s in high_var_pairs]
    
    if len(stim) == 0:
        pass
    else:
        df = df.loc[pd.IndexSlice[stim, 2], :]

        for alpha, key in zip(range(0, n_components), ['a1', 'a2']):
            df[xaxis] = results_raw.slice_array_results(xaxis+'_test', stim, n_components, idx=[0, alpha])[0]
            df['site'] = site
            results_dict[key].append(df.copy())
            if plot_sites:
                f, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].set_title(r"$d'^{2}$")
                df.loc[pd.IndexSlice[stim, n_components], :].plot.hexbin(x=xaxis, 
                                                            y='dU_mag_test', 
                                                            C=val, 
                                                            gridsize=nbins, ax=ax[0], cmap=cmap, vmin=vmin, vmax=vmax) 
                ax[0].set_ylabel(r'$|\Delta \mathbf{\mu}|$')
                ax[0].set_xlabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$')

                ax[1].set_title("Count")
                df.loc[pd.IndexSlice[stim, n_components],:].plot.hexbin(x=xaxis, 
                                                            y='dU_mag_test', 
                                                            C=None, 
                                                            gridsize=nbins, ax=ax[1], cmap='Reds', vmin=0) 
                ax[1].set_ylabel(r'$|\Delta \mathbf{\mu}|$')
                ax[1].set_xlabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$')

                f.canvas.set_window_title('alpha = {0}, site={1}'.format(alpha, site))

                f.tight_layout()

# plot results over all sites
nbins = 20
for alpha, key in zip(range(0, n_components), ['a1', 'a2']):
    df = pd.concat(results_dict[key])

    #df = df[(df[xaxis]>0.4) & (df['dU_mag_test']<8) & (df['dU_mag_test'] > 2)]

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    if val == 'state_diff':
        ax[0].set_title(r"Normalized $\Delta d'$")
    elif val == 'dp_opt_test':
        ax[0].set_title(r"$d'$")
    df.loc[pd.IndexSlice[stim, n_components], :].plot.hexbin(x='dU_mag_test', 
                                                y=xaxis, 
                                                C=val, 
                                                gridsize=nbins, ax=ax[0], cmap=cmap, vmin=vmin, vmax=vmax) 
    ax[0].set_xlabel(r'$|\Delta \mathbf{\mu}|$', color='orange')
    ax[0].set_ylabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$', color='purple')
    ax[0].spines['bottom'].set_color('orange')
    ax[0].spines['bottom'].set_lw(2)
    ax[0].xaxis.label.set_color('orange')
    ax[0].tick_params(axis='x', colors='orange')

    ax[0].spines['left'].set_color('purple')
    ax[0].spines['left'].set_lw(2)
    ax[0].yaxis.label.set_color('purple')
    ax[0].tick_params(axis='y', colors='purple')


    ax[1].set_title("Count")
    df.loc[pd.IndexSlice[stim, n_components],:].plot.hexbin(x='dU_mag_test', 
                                                y=xaxis, 
                                                C=None, 
                                                gridsize=nbins, ax=ax[1], cmap='Reds', vmin=0) 
    ax[1].set_xlabel(r'$|\Delta \mathbf{\mu}|$', color='orange')
    ax[1].set_ylabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$', color='purple')

    ax[1].spines['bottom'].set_color('orange')
    ax[1].spines['bottom'].set_lw(2)
    ax[1].xaxis.label.set_color('orange')
    ax[1].tick_params(axis='x', colors='orange')

    ax[1].spines['left'].set_color('purple')
    ax[1].spines['left'].set_lw(2)
    ax[1].yaxis.label.set_color('purple')
    ax[1].tick_params(axis='y', colors='purple')

    f.canvas.set_window_title('alpha = {0}, all sites'.format(alpha))

    f.tight_layout()

# use stats models to build linear model for predicting value
df_all = pd.concat(results_dict['a1'])
#df_all = df_all[(df_all[xaxis]>0.4) & (df_all['dU_mag_test']<8) & (df_all['dU_mag_test'] > 2)]
df_all['interaction'] = df_all['dU_mag_test'] * df_all[xaxis]
X = df_all[[xaxis, 'dU_mag_test', 'interaction']]
X = sm.add_constant(X)
y = df_all[val]
model = sm.OLS(y, X).fit()
beta = model.params.values
low_ci = model.conf_int().values[:,0]
high_ci = model.conf_int().values[:,1]

f, ax = plt.subplots(1, 2, figsize=(8, 4))
fd, a_dummy = plt.subplots(1,1)
ax[0].errorbar([0, 1, 2, 3], beta, yerr=(high_ci - low_ci) / 2,
               linestyle='none', marker='o', lw=2, color='k')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].set_xticks([0, 1, 2, 3])
ax[0].set_xticklabels([r'$\beta_{0}$', r'$\beta_{1}$', r'$\beta_{2}$', r'$\beta_{3}$'])
ax[0].set_ylabel(r"$\Delta d'$ per unit $\beta$")

b0 = []
b1 = []
b2 = []
b3 = []
x = np.linspace(0, 1, 100)
y = np.linspace(df_all['dU_mag_test'].min(), df_all['dU_mag_test'].max(), 100)
xx, yy = np.meshgrid(x, y)
positions = np.vstack([xx.ravel(), yy.ravel()])
colors = plt.cm.get_cmap('jet', len(df_all.site.unique()))
for i, site in enumerate(df_all.site.unique()):
    df = df_all[df_all.site==site]
    X = df[[xaxis, 'dU_mag_test', 'interaction']]
    X = sm.add_constant(X)
    y = df[val]
    model = sm.OLS(y, X).fit()
    beta = model.params.values
    b0.append(beta[0])
    b1.append(beta[1])
    b2.append(beta[2])
    b3.append(beta[3])

    # estimate kde and plot
    values = df[[xaxis, 'dU_mag_test']].values.T
    kde = ss.gaussian_kde(values)
    cont = np.reshape(kde(positions).T, xx.shape)
    cset = a_dummy.contour(xx, yy, cont, levels=1)
    seg = cset.allsegs[1][0]
    ax[1].plot(seg[:, 1], seg[:, 0], '-', color=colors(i), label=site, lw=3)

sites = df_all.site.unique()
ax[0].scatter(0 * np.ones(len(b0)) + np.random.normal(0, 0.1, len(b0)),
                b0, cmap=colors, c=range(0, len(sites)), edgecolor='white', s=100)
ax[0].scatter(1 * np.ones(len(b0)) + np.random.normal(0, 0.1, len(b0)),
                b1, cmap=colors, c=range(0, len(sites)), edgecolor='white', s=100)
ax[0].scatter(2 * np.ones(len(b0)) + np.random.normal(0, 0.1, len(b0)),
                b2, cmap=colors, c=range(0, len(sites)), edgecolor='white', s=100)
ax[0].scatter(3 * np.ones(len(b0)) + np.random.normal(0, 0.1, len(b0)),
                b3, cmap=colors, c=range(0, len(sites)), edgecolor='white', s=100)

ax[1].set_xlabel(r'$|\Delta \mathbf{\mu}|$')
ax[1].set_ylabel(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})|$')
ax[1].spines['bottom'].set_color('orange')
ax[1].spines['bottom'].set_lw(2)
ax[1].xaxis.label.set_color('orange')
ax[1].tick_params(axis='x', colors='orange')

ax[1].spines['left'].set_color('purple')
ax[1].spines['left'].set_lw(2)
ax[1].yaxis.label.set_color('purple')
ax[1].tick_params(axis='y', colors='purple')
ax[1].legend()

f.tight_layout()


plt.show()
