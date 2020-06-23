"""
Illustrate efficacy of dimensionality reduction techinque in preserving information 
about small, differential correlations. Also illustrate ability of technique to be 
robust to overfitting, and work in low trial N regime.

Simulate a population with two noise PCs - one that projects along 
dU (differential) and another that is orthogonal (randomly).
Simulate for many (1000s) neurons, then run decoding for different subsets and 
show that information saturates as neurons are added.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.plotting as cplt

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = True
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/TDR_simulation.svg'

f = plt.figure(figsize=(9, 6))

dpax1 = plt.subplot2grid((2, 3), (0, 0))
dpax2 = plt.subplot2grid((2, 3), (1, 0))
scree1 = plt.subplot2grid((2, 3), (0, 1))
scree2 = plt.subplot2grid((2, 3), (1, 1))
cos1 = plt.subplot2grid((2, 3), (0, 2))
cos2 = plt.subplot2grid((2, 3), (1, 2))

np.random.seed(123)

def _do_jackknife_analysis(X, njacks=50):
    est, val = nat_preproc.get_est_val_sets(X, njacks=njacks)
    est, val = nat_preproc.scale_est_val(est, val)
    Ntrials = est[0].shape[1]

    test = np.zeros(njacks)
    train = np.zeros(njacks)
    test_tdr = np.zeros(njacks)
    train_tdr = np.zeros(njacks)
    for j in range(njacks):
        # run dprime on raw data
        try:
            dp_train, wopt_train, evals_train, evecs_train, evec_sim_train, dU_train = decoding.compute_dprime(est[j][:, :, 0], est[j][:, :, 1])
            dp_test, wopt_test, evals_test, evecs_test, evec_sim_train, dU_test = decoding.compute_dprime(val[j][:, :, 0], val[j][:, :, 1], wopt=wopt_train)

            test[j] = dp_test
            train[j] = dp_train
        except:
            test[j] = np.nan
            train[j] = np.nan

        y = dr.get_one_hot_matrix(2, Ntrials)
        xtrain = nat_preproc.flatten_X(est[j][:, :, :, np.newaxis])
        xtest = nat_preproc.flatten_X(val[j][:, :, :, np.newaxis])

        tdr = dr.TDR()
        tdr.fit(xtrain.T, y.T)
        Xtdr_train = xtrain.T.dot(tdr.weights.T).T
        Xtdr_train = nat_preproc.fold_X(Xtdr_train, nreps=Ntrials, nstim=2, nbins=1).squeeze()

        Xtdr_test = xtest.T.dot(tdr.weights.T).T
        Xtdr_test = nat_preproc.fold_X(Xtdr_test, nreps=Ntrials, nstim=2, nbins=1).squeeze()

        # run dprime analysis on the reduced data
        dp_tdr_train, wopt_tdr_train, evals_tdr_train, evecs_tdr_train, evec_sim_train, dU_tdr_train = \
                            decoding.compute_dprime(Xtdr_train[:, :, 0], Xtdr_train[:, :, 1])
        dp_tdr_test, wopt_tdr_test, evals_tdr_test, evecs_tdr_test, evec_sim_test, dU_tdr_test = \
                            decoding.compute_dprime(Xtdr_test[:, :, 0], Xtdr_test[:, :, 1], wopt=wopt_tdr_train)
        test_tdr[j] = dp_tdr_test
        train_tdr[j] = dp_tdr_train


    results = {}
    results['test'] = test.mean()
    results['test_sem'] = test.std() / np.sqrt(njacks)
    results['train'] = train.mean()
    results['train_sem'] = train.std() / np.sqrt(njacks)
    results['test_tdr'] = test_tdr.mean()
    results['test_tdr_sem'] = test_tdr.std() / np.sqrt(njacks)
    results['train_tdr'] = train_tdr.mean()
    results['train_tdr_sem'] = train_tdr.std() / np.sqrt(njacks)

    return results

# ========================================== low trial number example ============================================
Ndim = 500
maxDim = 500
Ntrials= 250 # 200 10000
step = 50
RandSubsets = 30
njacks = 10
var_ratio = 1.2 # pc1 has X times the variance as pc2

# simulated data
u1 = 4
u2 = 4
u = np.stack((np.random.poisson(u1, Ndim), np.random.poisson(u2, Ndim)))

# make two dimensional noise:
# one large dim ~orthogonal to dU and one smaller dim ~ parallel to dU
dU = u[[1], :] - u[[0], :]
dU = dU / np.linalg.norm(dU)

diff_cor = dU #+ np.random.normal(0, 0.0001, dU.shape)
diff_cor = diff_cor / np.linalg.norm(diff_cor) * 3
pc1 = np.random.normal(0, 1, dU.shape)
pc1 = (pc1 / np.linalg.norm(pc1)) * 3  * var_ratio

evecs = np.concatenate((diff_cor, pc1), axis=0)
cov = evecs.T.dot(evecs)

# simulate full data matrix
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
X1 = _X + u[0, :]
X2 = _X + u[1, :]
X_raw = np.stack((X1, X2)).transpose([-1, 1, 0])

# add random noise to data matrix
X_raw += np.random.normal(0, 0.5, X_raw.shape)

n_subsets = np.append([2], np.arange(step, maxDim, step))

results = {}
for nset in n_subsets:
    print('nset: {}'.format(nset))
    results[nset] = {}
    train = []
    test = []
    train_tdr = []
    test_tdr = []
    j = 0
    while j < RandSubsets:
        if (Ndim - nset) <= 0:
            j = RandSubsets
        else:
            j += 1
        # choose random subset of neurons
        neurons = np.random.choice(np.arange(0, Ndim), nset, replace=False)
        X = X_raw[neurons, :, :]
        r = _do_jackknife_analysis(X, njacks=njacks)
        train.append(r['train'])
        test.append(r['test'])
        train_tdr.append(r['train_tdr'])
        test_tdr.append(r['test_tdr'])

    results[nset]['train'] = np.nanmean(train)
    results[nset]['test'] = np.nanmean(test)
    results[nset]['train_sem'] = np.nanstd(train) / np.sqrt(RandSubsets)
    results[nset]['test_sem'] = np.nanstd(test) / np.sqrt(RandSubsets)

    results[nset]['train_tdr'] = np.nanmean(train_tdr)
    results[nset]['test_tdr'] = np.nanmean(test_tdr)
    results[nset]['train_tdr_sem'] = np.nanstd(train_tdr) / np.sqrt(RandSubsets)
    results[nset]['test_tdr_sem'] = np.nanstd(test_tdr) / np.sqrt(RandSubsets)

train = np.array([results[x]['train'] for x in results.keys()])
train_sem = np.array([results[x]['train_sem'] for x in results.keys()])
test = np.array([results[x]['test'] for x in results.keys()])
test_sem = np.array([results[x]['test_sem'] for x in results.keys()])

train_tdr = np.array([results[x]['train_tdr'] for x in results.keys()])
train_tdr_sem = np.array([results[x]['train_tdr_sem'] for x in results.keys()])
test_tdr = np.array([results[x]['test_tdr'] for x in results.keys()])
test_tdr_sem = np.array([results[x]['test_tdr_sem'] for x in results.keys()])

dpax1.plot(n_subsets, train, 'o-', label='train')
dpax1.fill_between(n_subsets, train-train_sem, train+train_sem, alpha=0.5, lw=0)
dpax1.plot(n_subsets, test, 'o-', label='test')
dpax1.fill_between(n_subsets, test-test_sem, test+test_sem, alpha=0.5, lw=0)

dpax1.plot(n_subsets, train_tdr, 'o-', label='train, TDR')
dpax1.fill_between(n_subsets, train_tdr-train_tdr_sem, train_tdr+train_tdr_sem, alpha=0.5, lw=0)
dpax1.plot(n_subsets, test_tdr, 'o-', label='test, TDR')
dpax1.fill_between(n_subsets, test_tdr-test_tdr_sem, test_tdr+test_tdr_sem, alpha=0.5, lw=0)

dpax1.set_ylabel(r"$d'^{2}$")
dpax1.set_xlabel(r"Neurons, $N$")
dpax1.set_title(r"$k = %s$" % (Ntrials))

dpax1.legend(frameon=False)

# scree plot
Xpca, _ = nat_preproc.scale_est_val([X_raw], [X_raw])
Xpca = Xpca[0]
pca = PCA()
pca.fit(Xpca[:, :, 0].T)

scree1.plot(pca.explained_variance_ratio_, color='lightgrey')
scree1.plot(1, pca.explained_variance_ratio_[1], 'o', color='k')
scree1.set_ylabel('Fraction Variance Explained')
scree1.set_xlabel(r"$\alpha$ (PC)")
scree1.set_title('Scree plot')

# noise vs. dU overlap
dU = Xpca[:, :, 0].mean(axis=1) - Xpca[:, :, 1].mean(axis=1)
dU /= np.linalg.norm(dU)
cos_dU_evecs = abs(dU.dot(pca.components_.T))

cos1.plot(cos_dU_evecs, color='lightgrey')
cos1.plot(1, cos_dU_evecs[1], 'o', color='k')
cos1.set_ylabel(r"$cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})$")
cos1.set_xlabel(r"$\alpha$ (PC)")
cos1.set_title("Noise vs. Signal Similarity")

# ========================================== high trial number example ============================================
np.random.seed(123)
Ndim = 500
maxDim = 500
Ntrials= 10000 # 200 10000
step = 50
RandSubsets = 30
njacks = 10
var_ratio = 1.2 # pc1 has X times the variance as pc2

# simulated data
u1 = 4
u2 = 4
u = np.stack((np.random.poisson(u1, Ndim), np.random.poisson(u2, Ndim)))

# make two dimensional noise:
# one large dim ~orthogonal to dU and one smaller dim ~ parallel to dU
dU = u[[1], :] - u[[0], :]
dU = dU / np.linalg.norm(dU)

diff_cor = dU #+ np.random.normal(0, 0.0001, dU.shape)
diff_cor = diff_cor / np.linalg.norm(diff_cor) * 3
pc1 = np.random.normal(0, 1, dU.shape)
pc1 = (pc1 / np.linalg.norm(pc1)) * 3  * var_ratio

evecs = np.concatenate((diff_cor, pc1), axis=0)
cov = evecs.T.dot(evecs)

# simulate full data matrix
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
X1 = _X + u[0, :]
X2 = _X + u[1, :]
X_raw = np.stack((X1, X2)).transpose([-1, 1, 0])

# add random noise to data matrix
X_raw += np.random.normal(0, 0.5, X_raw.shape)

n_subsets = np.append([2], np.arange(step, maxDim, step))

results = {}
for nset in n_subsets:
    print('nset: {}'.format(nset))
    results[nset] = {}
    train = []
    test = []
    train_tdr = []
    test_tdr = []
    j = 0
    while j < RandSubsets:
        if (Ndim - nset) <= 0:
            j = RandSubsets
        else:
            j += 1
        # choose random subset of neurons
        neurons = np.random.choice(np.arange(0, Ndim), nset, replace=False)
        X = X_raw[neurons, :, :]
        r = _do_jackknife_analysis(X, njacks=njacks)
        train.append(r['train'])
        test.append(r['test'])
        train_tdr.append(r['train_tdr'])
        test_tdr.append(r['test_tdr'])

    results[nset]['train'] = np.nanmean(train)
    results[nset]['test'] = np.nanmean(test)
    results[nset]['train_sem'] = np.nanstd(train) / np.sqrt(RandSubsets)
    results[nset]['test_sem'] = np.nanstd(test) / np.sqrt(RandSubsets)

    results[nset]['train_tdr'] = np.nanmean(train_tdr)
    results[nset]['test_tdr'] = np.nanmean(test_tdr)
    results[nset]['train_tdr_sem'] = np.nanstd(train_tdr) / np.sqrt(RandSubsets)
    results[nset]['test_tdr_sem'] = np.nanstd(test_tdr) / np.sqrt(RandSubsets)

train = np.array([results[x]['train'] for x in results.keys()])
train_sem = np.array([results[x]['train_sem'] for x in results.keys()])
test = np.array([results[x]['test'] for x in results.keys()])
test_sem = np.array([results[x]['test_sem'] for x in results.keys()])

train_tdr = np.array([results[x]['train_tdr'] for x in results.keys()])
train_tdr_sem = np.array([results[x]['train_tdr_sem'] for x in results.keys()])
test_tdr = np.array([results[x]['test_tdr'] for x in results.keys()])
test_tdr_sem = np.array([results[x]['test_tdr_sem'] for x in results.keys()])

dpax2.plot(n_subsets, train, 'o-', label='train')
dpax2.fill_between(n_subsets, train-train_sem, train+train_sem, alpha=0.5, lw=0)
dpax2.plot(n_subsets, test, 'o-', label='test')
dpax2.fill_between(n_subsets, test-test_sem, test+test_sem, alpha=0.5, lw=0)

dpax2.plot(n_subsets, train_tdr, 'o-', label='train, TDR')
dpax2.fill_between(n_subsets, train_tdr-train_tdr_sem, train_tdr+train_tdr_sem, alpha=0.5, lw=0)
dpax2.plot(n_subsets, test_tdr, 'o-', label='test, TDR')
dpax2.fill_between(n_subsets, test_tdr-test_tdr_sem, test_tdr+test_tdr_sem, alpha=0.5, lw=0)

dpax2.set_ylabel(r"$d'^{2}$")
dpax2.set_xlabel(r"Neurons, $N$")
dpax2.set_title(r"$k = %s$" % (Ntrials))

dpax2.legend(frameon=False)

# scree plot
Xpca, _ = nat_preproc.scale_est_val([X_raw], [X_raw])
Xpca = Xpca[0]
pca = PCA()
pca.fit(Xpca[:, :, 0].T)

scree2.plot(pca.explained_variance_ratio_, color='lightgrey')
scree2.plot(1, pca.explained_variance_ratio_[1], 'o', color='k')
scree2.set_ylabel('Fraction Variance Explained')
scree2.set_xlabel(r"$\alpha$ (PC)")
scree2.set_title('Scree plot')

# noise vs. dU overlap
dU = Xpca[:, :, 0].mean(axis=1) - Xpca[:, :, 1].mean(axis=1)
dU /= np.linalg.norm(dU)
cos_dU_evecs = abs(dU.dot(pca.components_.T))

cos2.plot(cos_dU_evecs, color='lightgrey')
cos2.plot(1, cos_dU_evecs[1], 'o', color='k')
cos2.set_ylabel(r"$cos(\Delta \mathbf{\mu}, \mathbf{e}_{\alpha})$")
cos2.set_xlabel(r"$\alpha$ (PC)")
cos2.set_title("Noise vs. Signal Similarity")

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()