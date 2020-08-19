"""
1) First order model schematic / example cell?. r_test for pup vs. pup0. Show gain results (and DC?)
    - Consider showing breakdown of FS vs. RS (if we see an effect for decoding weights)

2) Delta noise correlations.

3) (?) Can't predict delta noise correlations from first order effect magnitude.
    - Or show this as supplementary figure
"""
import charlieTools.statistics as stats
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
rsc_df = ld.load_noise_correlation('rsc_ev', xforms_model='NULL', path=rsc_path)
mask = ~(rsc_df['bp'].isna() | rsc_df['sp'].isna())
rsc_df = rsc_df[mask]
# hierarchical bootstrapping, compare even / weighted sampling
rsc_df['diff'] = rsc_df['sp']-rsc_df['bp']
d = dict()
for s in rsc_df.site.unique(): 
     d[s] = rsc_df[rsc_df.site==s]['diff'].values.squeeze() 
bs_even = stats.get_bootstrapped_sample(d, even_sample=True, nboot=10000)
bs_weighted = stats.get_bootstrapped_sample(d, even_sample=False, nboot=10000)
bins = np.arange(-0.02, 0.02, 0.001)

# get pvalue for each method (and for standard Wilcoxon over the population, ignoring sites, one sided)
p_even = round(1 - stats.get_direct_prob(np.zeros(len(bs_even)), bs_even)[0], 5)
p_weighted = round(1 - stats.get_direct_prob(np.zeros(len(bs_even)), bs_weighted)[0], 5)
p_wilcox = round(ss.wilcoxon(rsc_df['bp'], rsc_df['sp'], alternative='less').pvalue, 5)
p_wilcox_g = round(ss.wilcoxon(rsc_df.groupby(by='site').mean()['bp'], 
                        rsc_df.groupby(by='site').mean()['sp'], alternative='less').pvalue, 5)

# plot results
f, ax = plt.subplots(1, 1)

ax.hist(bs_even, bins=bins, alpha=0.5, label='Even re-sampling, pvalue: {}'.format(p_even))
ax.hist(bs_weighted, bins=bins, alpha=0.5, label='Weighted re-sampling, pvalue: {}'.format(p_weighted))
ax.axvline(rsc_df['diff'].mean(), color='k', linestyle='--', lw=2, label='True pop. mean, Wilcoxon Sign-test pvalue: {0}'.format(p_wilcox))
ax.axvline(rsc_df.groupby(by='site').mean()['diff'].mean(), color='grey', linestyle='--', lw=2, 
                            label='Grouped pop. mean, Wilcoxon Sign-test pvalue: {0}'.format(p_wilcox_g))

ax.legend(frameon=False, fontsize=12)
ax.set_xlabel(r"Mean $\Delta$noise correlation", fontsize=12)
ax.set_ylabel(r"$n$ bootstraps", fontsize=12)
f.tight_layout()

plt.show()

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
df['site'] = [d[:7] for d in df.index] 
wstat, pval = ss.wilcoxon(df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup0']], df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup']])
print("r vs. r0 for single cells: r: {0}, {1}\n r0: {2}, {3}\n pval: {4}, W: {5}".format(
                        df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup']].mean(),
                        df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup']].sem(),
                        df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup0']].mean(),
                        df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup0']].sem(),
                        pval,
                        wstat
))
print("\n")

# gain modulation
wstat, pval = ss.wilcoxon(df.groupby(by='site').mean().loc[:, pd.IndexSlice['gain_mod', 'st.pup']])
print("gain: mean: {0}, {1} \n pval: {2}, W: {3}".format(
                        df.groupby(by='site').mean().loc[:, pd.IndexSlice['gain_mod', 'st.pup']].mean(),
                        df.groupby(by='site').mean().loc[:, pd.IndexSlice['gain_mod', 'st.pup']].sem(),
                        pval,
                        wstat
))
print("\n")

# DC modulation
wstat, pval = ss.wilcoxon(df.groupby(by='site').mean().loc[:, pd.IndexSlice['dc_mod', 'st.pup']])
print("DC: mean: {0}, {1} \n pval: {2}, W: {3}".format(
                        df.groupby(by='site').mean().loc[:, pd.IndexSlice['dc_mod', 'st.pup']].mean(),
                        df.groupby(by='site').mean().loc[:, pd.IndexSlice['dc_mod', 'st.pup']].sem(),
                        pval,
                        wstat
))
print("\n")

# noise correlations
wstat, pval = ss.wilcoxon(rsc_df.groupby(by='site').mean()['bp'], rsc_df.groupby(by='site').mean()['sp'])
print("rsc: large: {0}, {1} \n small: {2}, {3}, \n pval: {4}, W: {5}".format(
                        rsc_df.groupby(by='site').mean()['bp'].mean(),
                        rsc_df.groupby(by='site').mean()['bp'].sem(),
                        rsc_df.groupby(by='site').mean()['sp'].mean(),
                        rsc_df.groupby(by='site').mean()['sp'].sem(),
                        pval,
                        wstat
))
print("\n")

plt.show()

