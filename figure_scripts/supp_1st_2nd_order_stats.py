"""
1) First order model schematic / example cell?. r_test for pup vs. pup0. Show gain results (and DC?)
    - Consider showing breakdown of FS vs. RS (if we see an effect for decoding weights)

2) Delta noise correlations.

3) (?) Can't predict delta noise correlations from first order effect magnitude.
    - Or show this as supplementary figure
"""
import charlieTools.statistics as stats
from single_cell_models.mod_per_state import get_model_results_per_state_model
import nems.db as nd
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as ss
import seaborn as sns
import load_results as ld
import matplotlib as mpl
from global_settings import ALL_SITES
from path_settings import PY_FIGURES_DIR
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True
fig_fn = PY_FIGURES_DIR + '/supp_1st_2nd_order_stats.svg'

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
# add SITE columns and exclude sites not used in decoding analysis (e.g. tungsten recording sites)
#df.loc[:, pd.IndexSlice['site', 'site']] = [idx[:7] for idx in df.index]
df['site'] = [idx[:7] for idx in df.index]
df = df[df.loc[:, 'site'].isin([s[:7] for s in ALL_SITES])]
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

# set up figures
f = plt.figure(figsize=(6, 2))
mp = plt.subplot2grid((1, 3), (0, 0))
gdc = plt.subplot2grid((1, 3), (0, 1))
nc = plt.subplot2grid((1, 3), (0, 2))

# plot model performance
mp.scatter(df.loc[:, pd.IndexSlice['r', 'st.pup0']],
            df.loc[:, pd.IndexSlice['r', 'st.pup']],
            color='grey', edgecolor='white', s=15)

mp.scatter(df.loc[sig, pd.IndexSlice['r', 'st.pup0']],
            df.loc[sig, pd.IndexSlice['r', 'st.pup']],
            color='k', edgecolor='white', s=15)
mp.legend([r"$n = %s $" % ntot, r"$n = %s $" % nsig], frameon=False, fontsize=8)

mp.axhline(0, linestyle='--', color='k')
mp.axvline(0, linestyle='--', color='k')
mp.plot([0, 1], [0, 1], 'k--')

mp.set_xlabel('Shuffled Pupil')
mp.set_ylabel('Full Model')
mp.set_title('Prediction Correlation')

# plot gain vs. dc
gdc.scatter(gain, dc, color='grey', edgecolor='white', s=15)
gdc.scatter(gain[sig], dc[sig], color='k', edgecolor='white', s=15)

gdc.axhline(0, linestyle='--', color='k')
gdc.axvline(0, linestyle='--', color='k')

gdc.set_xlabel('Gain')
gdc.set_ylabel('Baseline')
gdc.set_title('Pupil-dependent \n response modulation')

# plot change in noise correlations
#nc.bar([0, 1],
#       [rsc_df['bp'].mean(), rsc_df['sp'].mean()],
#       yerr=[rsc_df['bp'].sem(), rsc_df['sp'].sem()],
#       color=['firebrick', 'navy'],
#       edgecolor='k',
#       width=0.5)
nc.bar([0, 1],
       [rsc_df.groupby(by='site').mean()['bp'].mean(), rsc_df.groupby(by='site').mean()['sp'].mean()],
       yerr=[rsc_df.groupby(by='site').mean()['bp'].sem(), rsc_df.groupby(by='site').mean()['sp'].sem()],
       color=['firebrick', 'navy'],
       edgecolor='k',
       width=0.5)
nc.set_xticks([0, 1])
nc.set_xticklabels(['Large', 'Small'])
nc.set_ylabel("Noise Correlation")
nc.set_xlabel('Pupil State')

f.tight_layout()

if savefig:
    f.savefig(fig_fn)


# print out relevant stats


# ========================= Single cell / pairs of neurons stats (could be biased) ==================================
print("===================== SINGLE CELL STATS ====================== \n")
# full model vs. shuffled
wstat, pval = ss.wilcoxon(df.loc[:, pd.IndexSlice['r', 'st.pup0']], df.loc[:, pd.IndexSlice['r', 'st.pup']])
print("r vs. r0: r: {0}, {1}\n r0: {2}, {3}\n pval: {4}, W: {5}".format(
                        df.loc[:, pd.IndexSlice['r', 'st.pup']].mean(),
                        df.loc[:, pd.IndexSlice['r', 'st.pup']].sem(),
                        df.loc[:, pd.IndexSlice['r', 'st.pup0']].mean(),
                        df.loc[:, pd.IndexSlice['r', 'st.pup0']].sem(),
                        pval,
                        wstat
))
print("\n")

# gain modulation
wstat, pval = ss.wilcoxon(df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']])
print("gain: mean: {0}, {1} \n pval: {2}, W: {3}".format(
                        df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']].mean(),
                        df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']].sem(),
                        pval,
                        wstat
))
print("\n")

# DC modulation
wstat, pval = ss.wilcoxon(df.loc[:, pd.IndexSlice['dc_mod', 'st.pup']])
print("DC: mean: {0}, {1} \n pval: {2}, W: {3}".format(
                        df.loc[:, pd.IndexSlice['dc_mod', 'st.pup']].mean(),
                        df.loc[:, pd.IndexSlice['dc_mod', 'st.pup']].sem(),
                        pval,
                        wstat
))
print("\n")

# noise correlations
wstat, pval = ss.wilcoxon(rsc_df['bp'], rsc_df['sp'])
print("rsc: large: {0}, {1} \n small: {2}, {3}, \n pval: {4}, W: {5}".format(
                        rsc_df['bp'].mean(),
                        rsc_df['bp'].sem(),
                        rsc_df['sp'].mean(),
                        rsc_df['sp'].sem(),
                        pval,
                        wstat
))
print("\n")

# ==================================== hierarcichal statisistics ====================================================
np.random.seed(123)
print("============================= HEIRARCHICAL BOOTSTRAP STATS ========================= \n")
d = {s: df.loc[df.site==s, pd.IndexSlice['r', 'st.pup']].values - df.loc[df.site==s, pd.IndexSlice['r', 'st.pup0']].values 
                    for s in df.site.unique()}
mobs = np.mean([len(d[s]) for s in d.keys()])
sem = np.std([len(d[s]) for s in d.keys()]) / np.sqrt(len(d.keys()))
bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
print("r vs. r0 bootstrap probability of NULL hypothesis: {0},\n"
          "n recording sessions: {1},\n"
          "mean/sem observations per session: {2} +/- {3} \n".format(p, len(df.site.unique()), mobs, sem))
print("Full model mean: {0} +/- {1} after grouping within site".format(df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup']].mean(), 
                                                                       df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup']].sem()))
print("Shuffled model mean: {0} +/- {1} after grouping within site \n".format(df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup0']].mean(), 
                                                                           df.groupby(by='site').mean().loc[:, pd.IndexSlice['r', 'st.pup0']].sem()))

d = {s: df.loc[df.site==s, pd.IndexSlice['gain_mod', 'st.pup']].values
                    for s in df.site.unique()}
bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
print("Gain modulation probability of NULL hypothesis: {0} \n"
          "n recording sessions: {1},\n"
          "mean/sem observations per session: {2} +/- {3} \n".format(p, len(df.site.unique()), mobs, sem))
print("Gain modulation mean: {0} +/- {1} after grouping within site \n".format(df.groupby(by='site').mean().loc[:, pd.IndexSlice['gain_mod', 'st.pup']].mean(), 
                                                                            df.groupby(by='site').mean().loc[:, pd.IndexSlice['gain_mod', 'st.pup']].sem()))

d = {s: df.loc[df.site==s, pd.IndexSlice['dc_mod', 'st.pup']].values
                    for s in df.site.unique()}
bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
print("DC modulation bootstrap probability of NULL hypothesis: {0} \n"
          "n recording sessions: {1},\n"
          "mean/sem observations per session: {2} +/- {3} \n".format(p, len(df.site.unique()), mobs, sem))
print("DC modulation mean: {0} +/- {1} after grouping within site \n".format(df.groupby(by='site').mean().loc[:, pd.IndexSlice['dc_mod', 'st.pup']].mean(), 
                                                                          df.groupby(by='site').mean().loc[:, pd.IndexSlice['dc_mod', 'st.pup']].sem()))

d = {s: rsc_df.loc[rsc_df.site==s, 'sp'].values - rsc_df.loc[rsc_df.site==s, 'bp'].values
                    for s in df.site.unique()}
mobs = np.mean([len(d[s]) for s in d.keys()])
sem = np.std([len(d[s]) for s in d.keys()]) / np.sqrt(len(d.keys()))
bootstat = stats.get_bootstrapped_sample(d, nboot=5000)
p = 1 - stats.get_direct_prob(np.zeros(len(bootstat)), bootstat)[0]
print("Noise correlation probability of NULL hypothesis: {0} \n"
          "n recording sessions: {1},\n"
          "mean/sem observations per session: {2} +/- {3} \n".format(p, len(df.site.unique()), mobs, sem))
print("Noise corr. small mean: {0} +/- {1}, large mean: {2} +/- {3} \n after grouping within site \n".format(rsc_df.groupby(by='site')['sp'].mean().mean(), 
                                                                                                          rsc_df.groupby(by='site')['sp'].mean().sem(),
                                                                                                          rsc_df.groupby(by='site')['bp'].mean().mean(),
                                                                                                          rsc_df.groupby(by='site')['bp'].mean().sem()))

# after excluding the tungsten data, and sites not included in global_settings ALL_SITES
iso = [nd.get_isolation(cellid, batch=289).values[0][0] if cellid[:7] not in ['BOL005c', 'BOL006b'] else
                    nd.get_isolation(cellid, batch=294).values[0][0] for cellid in df.index]
SU, MU = sum(np.array(iso)>= 95), sum(np.array(iso)<95)
print("TOTAL n SU: {0}, and MU:{1}".format(SU, MU))

plt.show()

