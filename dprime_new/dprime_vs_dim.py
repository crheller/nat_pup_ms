"""
Plot dprime vs. PLS dimensionality for all sites
"""
import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


loader = decoding.DecodingResults()
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
modelname = 'dprime_jk10'
plot_train = False
dprime2 = False
limit_ax = False

# list of sites with > 10 reps of each stimulus
sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']

for site in sites:

    fn = os.path.join(path, site, modelname+'_PLS.pickle')
    results = loader.load_results(fn)
    if not dprime2:
        dp_cols = [c for c in results.numeric_results.columns if 'dp' in c]
        results.numeric_results[dp_cols] = np.sqrt(results.numeric_results[dp_cols])
        ylab = r"$d'$"
    else:
        ylab = r"$d'^{2}$"
    idx = pd.IndexSlice

    f, ax = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

    # =================================== spont vs. spont ===========================================
    spont_results = results.numeric_results.loc[idx['spont_spont', :], :]
    components = spont_results.index.get_level_values('n_components').values

    ax[0, 0].plot(components, spont_results.dp_opt_test, 'o-', label='wopt, test')
    ax[0, 0].fill_between(components, spont_results.dp_opt_test - spont_results.dp_opt_test_sem,
                                spont_results.dp_opt_test + spont_results.dp_opt_test_sem, alpha=0.5)

    ax[0, 0].plot(components, spont_results.dp_diag_test, 'o-', label='wdiag, test')
    ax[0, 0].fill_between(components, spont_results.dp_diag_test - spont_results.dp_diag_test_sem,
                                spont_results.dp_diag_test + spont_results.dp_diag_test_sem, alpha=0.5)

    if plot_train:
        ax[0, 0].plot(components, spont_results.dp_opt_train, 'o-', label='wopt, train')
        ax[0, 0].fill_between(components, spont_results.dp_opt_train - spont_results.dp_opt_train_sem,
                                    spont_results.dp_opt_train + spont_results.dp_opt_train_sem, alpha=0.5)

        ax[0, 0].plot(components, spont_results.dp_diag_train, 'o-', label='wdiag, train')
        ax[0, 0].fill_between(components, spont_results.dp_diag_train - spont_results.dp_diag_train_sem,
                                    spont_results.dp_diag_train + spont_results.dp_diag_train_sem, alpha=0.5)

    ax[0, 0].axhline(0, linestyle='--', lw=3, color='grey')
    ax[0, 0].legend(frameon=False)
    ax[0, 0].set_xlabel('PLS Dimension')
    ax[0, 0].set_ylabel(ylab)
    ax[0, 0].set_title('Spont vs. spont')

    ax[1, 0].plot(components, spont_results.var_explained_test, 'o-', color='purple', label='test')
    ax[1, 0].fill_between(components, spont_results.var_explained_test - spont_results.var_explained_test_sem,
                                spont_results.var_explained_test + spont_results.var_explained_test_sem, alpha=0.5, 
                                lw=0, color=ax[1, 0].get_lines()[-1].get_color())
    ax[1, 0].plot(components, spont_results.var_explained_train, 'o-', color='goldenrod', label='train')
    ax[1, 0].fill_between(components, spont_results.var_explained_train - spont_results.var_explained_train_sem,
                                spont_results.var_explained_train + spont_results.var_explained_train_sem, alpha=0.5, 
                                lw=0, color=ax[1, 0].get_lines()[-1].get_color())
    ax[1, 0].set_xlabel('PLS Dimenion')
    ax[1, 0].set_ylabel('Single trial \n variance explained ratio')
    ax[1, 0].axhline(0, linestyle='--', lw=3, color='grey')
    ax[1, 0].set_ylim((-0.05, 1.05))
    ax[1, 0].legend(frameon=False)

    # ====================================== evoked vs. spont =========================================
    # (average over all sounds)
    evsp_idx = results.spont_evoked_stimulus_pairs
    evsp_results = results.numeric_results.loc[idx[evsp_idx, :], :]
    err_cols = [c for c in evsp_results.columns if '_sem' in c]
    cols = [c for c in evsp_results.columns if '_sem' not in c]
    evsp_collapse = evsp_results.groupby(by='n_components')[cols].mean()
    evsp_collapse_err = evsp_results.groupby(by='n_components')[err_cols].apply(decoding.error_prop)

    ax[0, 1].plot(components, evsp_collapse.dp_opt_test, 'o-', label='wopt, test')
    ax[0, 1].fill_between(components, evsp_collapse.dp_opt_test - evsp_collapse_err.dp_opt_test_sem,
                                evsp_collapse.dp_opt_test + evsp_collapse_err.dp_opt_test_sem, alpha=0.5)

    ax[0, 1].plot(components, evsp_collapse.dp_diag_test, 'o-', label='wdiag, test')
    ax[0, 1].fill_between(components, evsp_collapse.dp_diag_test - evsp_collapse_err.dp_diag_test_sem,
                                evsp_collapse.dp_diag_test + evsp_collapse_err.dp_diag_test_sem, alpha=0.5)

    if plot_train:
        ax[0, 1].plot(components, evsp_collapse.dp_opt_train, 'o-', label='wopt, train')
        ax[0, 1].fill_between(components, evsp_collapse.dp_opt_train - evsp_collapse_err.dp_opt_train_sem,
                                    evsp_collapse.dp_opt_train + evsp_collapse_err.dp_opt_train_sem, alpha=0.5)

        ax[0, 1].plot(components, evsp_collapse.dp_diag_train, 'o-', label='wdiag, train')
        ax[0, 1].fill_between(components, evsp_collapse.dp_diag_train - evsp_collapse_err.dp_diag_train_sem,
                                    evsp_collapse.dp_diag_train + evsp_collapse_err.dp_diag_train_sem, alpha=0.5)

    ax[0, 1].axhline(0, linestyle='--', lw=3, color='grey')
    ax[0, 1].legend(frameon=False)
    ax[0, 1].set_xlabel('PLS Dimension')
    ax[0, 1].set_ylabel(ylab)
    ax[0, 1].set_title('Spont vs. evoked')

    ax[1, 1].plot(components, evsp_collapse.var_explained_test, 'o-', color='purple', label='test')
    ax[1, 1].fill_between(components, evsp_collapse.var_explained_test - evsp_collapse_err.var_explained_test_sem,
                                evsp_collapse.var_explained_test + evsp_collapse_err.var_explained_test_sem, alpha=0.5, 
                                lw=0, color=ax[1, 1].get_lines()[-1].get_color())
    ax[1, 1].plot(components, evsp_collapse.var_explained_train, 'o-', color='goldenrod', label='train')
    ax[1, 1].fill_between(components, evsp_collapse.var_explained_train - evsp_collapse_err.var_explained_train_sem,
                                evsp_collapse.var_explained_train + evsp_collapse_err.var_explained_train_sem, alpha=0.5, 
                                lw=0, color=ax[1, 1].get_lines()[-1].get_color())
    ax[1, 1].set_xlabel('PLS Dimenion')
    ax[1, 1].set_ylabel('Single trial \n variance explained ratio')
    ax[1, 1].axhline(0, linestyle='--', lw=3, color='grey')
    ax[1, 1].set_ylim((-0.05, 1.05))
    ax[1, 1].legend(frameon=False)


    # ====================================== evoked vs. evoked =========================================
    # (average over all sounds)
    ev_idx = results.evoked_stimulus_pairs
    ev_results = results.numeric_results.loc[idx[ev_idx, :], :]
    err_cols = [c for c in ev_results.columns if '_sem' in c]
    cols = [c for c in ev_results.columns if '_sem' not in c]
    ev_collapse = ev_results.groupby(by='n_components')[cols].mean()
    ev_collapse_err = ev_results.groupby(by='n_components')[err_cols].apply(decoding.error_prop)

    ax[0, 2].plot(components, ev_collapse.dp_opt_test, 'o-', label='wopt, test')
    ax[0, 2].fill_between(components, ev_collapse.dp_opt_test - ev_collapse_err.dp_opt_test_sem,
                                ev_collapse.dp_opt_test + ev_collapse_err.dp_opt_test_sem, alpha=0.5)

    ax[0, 2].plot(components, ev_collapse.dp_diag_test, 'o-', label='wdiag, test')
    ax[0, 2].fill_between(components, ev_collapse.dp_diag_test - ev_collapse_err.dp_diag_test_sem,
                                ev_collapse.dp_diag_test + ev_collapse_err.dp_diag_test_sem, alpha=0.5)

    if plot_train:
        ax[0, 2].plot(components, ev_collapse.dp_opt_train, 'o-', label='wopt, train')
        ax[0, 2].fill_between(components, ev_collapse.dp_opt_train - ev_collapse_err.dp_opt_train_sem,
                                    ev_collapse.dp_opt_train + ev_collapse_err.dp_opt_train_sem, alpha=0.5)

        ax[0, 2].plot(components, ev_collapse.dp_diag_train, 'o-', label='wdiag, train')
        ax[0, 2].fill_between(components, ev_collapse.dp_diag_train - ev_collapse_err.dp_diag_train_sem,
                                    ev_collapse.dp_diag_train + ev_collapse_err.dp_diag_train_sem, alpha=0.5)

    ax[0, 2].axhline(0, linestyle='--', lw=3, color='grey')
    ax[0, 2].legend(frameon=False)
    ax[0, 2].set_xlabel('PLS Dimension')
    ax[0, 2].set_ylabel(ylab)
    ax[0, 2].set_title('Evoked vs. evoked')

    if limit_ax:
        ax[0, 2].set_ylim((-5, 100))

    ax[1, 2].plot(components, ev_collapse.var_explained_test, 'o-', color='purple', label='test')
    ax[1, 2].fill_between(components, ev_collapse.var_explained_test - ev_collapse_err.var_explained_test_sem,
                                ev_collapse.var_explained_test + ev_collapse_err.var_explained_test_sem, alpha=0.5, 
                                lw=0, color=ax[1, 2].get_lines()[-1].get_color())
    ax[1, 2].plot(components, ev_collapse.var_explained_train, 'o-', color='goldenrod', label='train')
    ax[1, 2].fill_between(components, ev_collapse.var_explained_train - ev_collapse_err.var_explained_train_sem,
                                ev_collapse.var_explained_train + ev_collapse_err.var_explained_train_sem, alpha=0.5, 
                                lw=0, color=ax[1, 2].get_lines()[-1].get_color())
    ax[1, 2].set_xlabel('PLS Dimenion')
    ax[1, 2].set_ylabel('Single trial \n variance explained ratio')
    ax[1, 2].axhline(0, linestyle='--', lw=3, color='grey')
    ax[1, 2].set_ylim((-0.05, 1.05))
    ax[1, 2].legend(frameon=False)

    f.canvas.set_window_title(site)

    f.tight_layout()

    f.savefig(os.path.join(path, 'Figures', site+'_{}'+'.png'.format(modelname)))


# ==================================== heatmap of results as funct of dU and eigvals ======================================


plt.show()

