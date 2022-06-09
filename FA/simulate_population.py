"""
Indep. noise FA models seem to capture decoding effects. We need correlated pop. activity for this to work, though.
If the pop. is uncorrelated, indep. noise changes can't explain decoding changes. In other words, if you leave out single 
neuron variance that can be explained by other neurons, we can't capture decoding effects.
Q: Is this because indep. noise changes in a corr. population cause changes in correlation?
    * Can changes in indep. noise cause a change in loading similarity?
Q: Is this because changes in low-D latent, shared process cause changes in FA estimated indep noise?
    * Most relevant question, in a simualted population with a change in loading sim. of a latent factor
     that depends on pupil, does the indep. model 

Simulate population activity with simple psth's, binary pupil and 10 "stimuli", to start:
    * only first order effects  
        * each neuron has independent noise (additive)
        * fixed correlations introduced with additive noise through single latent process 
    * pupil modulates indep. noise
        *** each neuron has independent noise (still additive), but depends on pupil based on set of weights
        *** fixed correlations from laten process, same as above
    * pupil modulates shared latent process, so correaltions depend on pupil
        *** option 1: do this with a rotation (change the weights of the modulator)
        * option 2: change variance of modulator (%shared variance change)
        * option 3: add more dimensions of latent modulation in one pupil state
"""
from operator import imod
import numpy as np
import matplotlib.pyplot as plt
from dDR.utils.surrogate_helpers import generate_lv_loading, generate_full_rank_cov
import sys
sys.path.append("/auto/users/hellerc/code/projects/")
import nat_pupil_ms.FA.pop_metric_helpers as pmh

from sklearn.decomposition import FactorAnalysis

np.random.normal(123)
# generate noise vectors, decide how to apply in next step
N = 50
S = 10
R = 2000
lv = np.random.normal(0, 1, (R, 1))
lvnoise = np.tile(lv, [1, N]).T
lvnoise = np.tile(lvnoise, [S, 1, 1]).transpose([1, 0, -1])
lv2 = np.random.normal(0, 1, (R, 1))
lvnoise2 = np.tile(lv2, [1, N]).T
lvnoise2 = np.tile(lvnoise2, [S, 1, 1]).transpose([1, 0, -1])
inoise = np.random.normal(0, 1, (N, S, R))

# build loading vector(s) for case where indep. noise fixed, but LV loading change
lv_s = generate_lv_loading(N, mean_loading=1, variance=0.5, mag=1)[:, :, np.newaxis] 
lv_l = generate_lv_loading(N, mean_loading=0, variance=0.5, mag=1)[:, :, np.newaxis] 

mod_ls = True
mod_indep = False

if mod_indep:
    # second simulation, we change indep variance only
    imodcoef_lg = np.random.normal(0.6, 0.4, (N, 1))
    imodcoef_lg[imodcoef_lg<=0.1] = 0.1
    imodcoef_lg = np.tile(imodcoef_lg, [1, S])[:, :, np.newaxis]
    imodcoef_sm = np.random.normal(0.3, 0.4, (N, 1))
    imodcoef_sm[imodcoef_sm<=0.1] = 0.1
    imodcoef_sm = np.tile(imodcoef_sm, [1, S])[:, :, np.newaxis]
    u_resp = np.random.poisson(2, (N, S, 1))
    Xbig = (u_resp * (lv_s * lvnoise[:, :, :1000])) * (imodcoef_lg * inoise[:, :, :1000])
    Xsmall = (u_resp * (lv_s * lvnoise[:, :, 1000:])) * (imodcoef_sm * inoise[:, :, 1000:])
    # get residual
    Xbig = Xbig - Xbig.mean(axis=-1, keepdims=True)
    Xsmall = Xsmall - Xsmall.mean(axis=-1, keepdims=True)

elif mod_ls:
    imodcoef = np.random.normal(0.7, 0.3, (N, 1))
    imodcoef[imodcoef<=0] = 0.1
    imodcoef = np.tile(imodcoef, [1, S])[:, :, np.newaxis]
    u_resp = np.random.poisson(2, (N, S, 1))
    Xbig = u_resp * (lv_l * lvnoise[:, :, :1000]) + (imodcoef * inoise[:, :, :1000])  
    Xsmall = u_resp * (lv_s * lvnoise[:, :, 1000:]) + (imodcoef * inoise[:, :, 1000:]) 
    # get residual
    Xbig = Xbig - Xbig.mean(axis=-1, keepdims=True)
    Xsmall = Xsmall - Xsmall.mean(axis=-1, keepdims=True)

# FA on the simulation
nfold = S
ncomp = 10
LLb = np.zeros((ncomp-1, nfold))
LLs = np.zeros((ncomp-1, nfold))
for ii in np.arange(1, ncomp):

    for nf in range(nfold):
        fit = [x for x in np.arange(0, S) if x != nf]    
        fa = FactorAnalysis(n_components=ii, random_state=0)
        # fit big data
        fa.fit(Xbig[:, fit, :].reshape(N, -1).T) # fit model
        # Get LL score
        LLb[ii-1, nf] = fa.score(Xbig[:, nf, :].reshape(N, -1).T)

        fa = FactorAnalysis(n_components=ii, random_state=0)
        # fit big data
        fa.fit(Xsmall[:, fit, :].reshape(N, -1).T) # fit model
        # Get LL score
        LLs[ii-1, nf] = fa.score(Xsmall[:, nf, :].reshape(N, -1).T)

LLs = np.mean(LLs, axis=-1)
LLb = np.mean(LLb, axis=-1)

# fit appropriate model
small_dims = pmh.get_dim(LLs)
small_fa = FactorAnalysis(n_components=small_dims, random_state=0)
small_fa.fit(Xsmall.reshape(N, -1).T)
# caculate metrics
small_sv = pmh.get_sv(small_fa)
small_ls = pmh.get_loading_similarity(small_fa)

large_dims = pmh.get_dim(LLs)
large_fa = FactorAnalysis(n_components=large_dims, random_state=0)
large_fa.fit(Xbig.reshape(N, -1).T)
# caculate metrics
large_sv = pmh.get_sv(large_fa)
large_ls = pmh.get_loading_similarity(large_fa)

f, ax = plt.subplots(1, 4, figsize=(10, 2.5))

ax[0].plot(LLs / -max((LLs)), "o-", label="LL small")
ax[0].plot(LLb / -max((LLb)), "o-", label="LL large")
ax[0].legend()

ax[1].scatter([0, 3], [small_sv, small_ls], s=75, color="k", label="small")
ax[1].scatter([1, 4], [large_sv, large_ls], s=75, color="r", label="large")
#ax[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
ax[1].set_xticks([0.5, 3.5])
ax[1].set_xticklabels(["%sv", "loading sim."])
ax[1].set_xlim([-1, 5])

# changes in independent cov matrix
ax[2].scatter(np.diag(pmh.sigma_ind(small_fa)), np.diag(pmh.sigma_ind(large_fa)), s=10, color="k")
mm = max(ax[2].get_xlim() + ax[2].get_ylim())
ax[2].plot([0, mm], [0, mm], "k--")
ax[2].set_xlabel("Small")
ax[2].set_ylabel("large")
# changes in shared cov matrix
ax[3].scatter(pmh.sigma_shared(small_fa).flatten(), pmh.sigma_shared(large_fa).flatten(), s=10, alpha=0.1, color="k")
mm = max(ax[3].get_xlim() + ax[3].get_ylim())
mi = min(ax[3].get_xlim() + ax[3].get_ylim())
ax[3].plot([mi, mm], [mi, mm], "k--")
ax[3].set_xlabel("Small")
ax[3].set_ylabel("large")

if mod_ls:
    f.suptitle("Explicitly change loading sim")
elif mod_indep:
    f.suptitle("Only changing indep. variance")

f.tight_layout()