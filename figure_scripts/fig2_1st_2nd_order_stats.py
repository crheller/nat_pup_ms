"""
1) First order model schematic / example cell?. r_test for pup vs. pup0. Show gain results (and DC?)
    - Consider showing breakdown of FS vs. RS (if we see an effect for decoding weights)

2) Delta noise correlations.

3) (?) Can't predict delta noise correlations from first order effect magnitude.
    - Or show this as supplementary figure
"""

from single_cell_models.mod_per_state import get_model_results_per_state_model
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as ss
import seaborn as sns
import load_results as ld
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = False
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig2_1st_2nd_order_stats.svg'

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

df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se', 'isolation'])
dc = df.loc[:, pd.IndexSlice['dc_mod', 'st.pup']]
gain = df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']]
sig = (df.loc[:, pd.IndexSlice['r', 'st.pup']] - df.loc[:, pd.IndexSlice['r', 'st.pup0']]) > \
            (df.loc[:, pd.IndexSlice['r_se', 'st.pup']] + df.loc[:, pd.IndexSlice['r_se', 'st.pup0']])
nsig = sig.sum()
ntot = gain.shape[0]

# load noise correlations
rsc_path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'
rsc_df = ld.load_noise_correlation('rsc_bal', path=rsc_path)

# set up figures
f = plt.figure(figsize=(6, 2))
mp = plt.subplot2grid((1, 3), (0, 0))
gdc = plt.subplot2grid((1, 3), (0, 1))
nc = plt.subplot2grid((1, 3), (0, 2))

# plot model performance
mp.scatter(df.loc[:, pd.IndexSlice['r', 'st.pup0']],
            df.loc[:, pd.IndexSlice['r', 'st.pup']],
            color='grey', edgecolor='white', s=20)

mp.scatter(df.loc[sig, pd.IndexSlice['r', 'st.pup0']],
            df.loc[sig, pd.IndexSlice['r', 'st.pup']],
            color='k', edgecolor='white', s=20)
mp.legend([r"$n = %s $" % ntot, r"$n = %s $" % nsig], frameon=False, fontsize=8)

mp.axhline(0, linestyle='--', color='k')
mp.axvline(0, linestyle='--', color='k')
mp.plot([0, 1], [0, 1], 'k--')

mp.set_xlabel('Shuffled Pupil')
mp.set_ylabel('Full Model')
mp.set_title('Prediction Correlation')

# plot gain vs. dc
gdc.scatter(gain, dc, color='grey', edgecolor='white', s=20)
gdc.scatter(gain[sig], dc[sig], color='k', edgecolor='white', s=20)

gdc.axhline(0, linestyle='--', color='k')
gdc.axvline(0, linestyle='--', color='k')

gdc.set_xlabel('Gain')
gdc.set_ylabel('Baseline')
gdc.set_title('Pupil-dependent \n response modulation')

# plot change in noise correlations
nc.bar([0, 1],
       [rsc_df['bp'].mean(), rsc_df['sp'].mean()],
       yerr=[rsc_df['bp'].sem(), rsc_df['sp'].sem()],
       color=['firebrick', 'navy'],
       edgecolor='k',
       width=0.5)
nc.legend([r"n = %s" % rsc_df.shape[0]], frameon=False, fontsize=8)
nc.set_xticks([0, 1])
nc.set_xticklabels(['Large', 'Small'])
nc.set_ylabel("Noise Correlation")
nc.set_xlabel('Pupil State')

f.tight_layout()

if savefig:
    f.savefig(fig_fn)


# print out relevant stats

# full model vs. shuffled model
# single cells
pval = ss.wilcoxon(df.loc[:, pd.IndexSlice['r', 'st.pup0']], df.loc[:, pd.IndexSlice['r', 'st.pup']]).pvalue
print("r vs. r0 for single cells: r: {0}, {1}\n r0: {2}, {3}\n pval: {2}".format(
                        df.loc[:, pd.IndexSlice['r', 'st.pup']].mean(),
                        df.loc[:, pd.IndexSlice['r', 'st.pup']].sem(),
                        df.loc[:, pd.IndexSlice['r', 'st.pup0']].mean(),
                        df.loc[:, pd.IndexSlice['r', 'st.pup0']].sem(),
                        pval
))
print("\n")

# gain modulation
pval = ss.wilcoxon(gain).pvalue
print("gain: mean: {0}, {1} \n pval: {2}".format(
                        gain.mean(),
                        gain.sem(),
                        pval
))
print("\n")

# DC modulation
pval = ss.wilcoxon(dc).pvalue
print("DC: mean: {0}, {1} \n pval: {2}".format(
                        dc.mean(),
                        dc.sem(),
                        pval
))
print("\n")

# noise correlations
pval = ss.wilcoxon(rsc_df['bp'], rsc_df['sp'])
print("rsc: large: {0}, {1} \n small: {2}, {3}, \n pval: {4}".format(
                        rsc_df['bp'].mean(),
                        rsc_df['bp'].sem(),
                        rsc_df['sp'].mean(),
                        rsc_df['sp'].sem(),
                        pval
))
print("\n")

plt.show()

