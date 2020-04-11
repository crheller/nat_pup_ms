"""
Helper functions for dim reduction / decoding. These will get moved to charlieTools.
"""
import numpy as np
# ============================ data preprocessing fns ========================
def dict_to_X(d):
    """
    Transform dictionary of spike counts (returned by nems.recording.extract_epochs)
    into a matrix of shape: (Neuron X Repetition X Stimulus X Time)
    """
    
    epochs = d.keys()
    for i, epoch in enumerate(epochs):
        r_epoch = d[epoch].transpose(1, 0, -1)[:, :, np.newaxis, :]
        if i == 0:
            X = r_epoch
        else:
            # stack on stimuli (epochs)
            X = np.append(X, r_epoch, axis=2)
    return X

def flatten_X(X):
    """
    Transform X matrix (Neuron X Stimulus X Repetition X Time) into 
    matrix of Neurons X Repetition * Stimulus * Time
    """
    if len(X.shape) != 4: raise ValueError("Input has unexpected shape")
    return X.reshape(X.shape[0], -1)

def fold_X(X_flat, nstim, nreps, nbins):
    """
    Invert the transformation done by cat_stimuli. 
        X_flat is shape (Neurons X Stimulus * Repetition * Time)
        and this fn returns matrix of shape (Neurons X Repetion X Stimulus X Time)
    """
    if len(X_flat.shape) != 2: raise ValueError("Input has unexpected shape")
    return X_flat.reshape(X_flat.shape[0], nreps, nstim, nbins)


def get_one_hot_matrix(ncategories, nreps):
    # build Y matrix of one hot vectors
    Y = np.zeros((ncategories, ncategories * nreps))
    for stim in range(ncategories):
        yt = np.zeros((nreps, ncategories))
        yt[:, stim] = 1
        yt = yt.reshape(1, -1)
        Y[stim, :] = yt
    return Y


def compute_dprime(A, B, diag=False):
    """
    Compute discriminability between matrix A and matrix B
    where both are shape N neurons X N reps
    """
    usig = 0.5 * (np.cov((A.T - A.mean(axis=-1)).T) + np.cov((B.T - B.mean(axis=-1)).T))
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]

    # get decoding axis
    if diag is False:

        try:
            wopt = np.matmul(np.linalg.inv(usig), u_vec.T)
        except:
            print('WARNING, Singular Covariance, dprime infinite, set to np.nan')
            return np.nan, np.nan

        if np.linalg.det(usig)<0.001:
            import pdb; pdb.set_trace()

        dp2 = np.matmul(u_vec, wopt)[0][0]
        if dp2 < 0:
            dp2 = -dp2

        return wopt, np.sqrt(dp2)

    else:
        try:
            wopt = np.matmul(np.linalg.inv(usig), u_vec.T)
        except:
            print('WARNING, Singular Covariance, dprime infinite, set to np.nan')
            return np.nan, np.nan

        dp2 = np.matmul(u_vec, wopt)[0][0]

        if dp2 < 0:
            dp2 = -dp2

        num = dp2

        # get denominator
        usig_diag = np.zeros(usig.shape)
        np.fill_diagonal(usig_diag, np.diagonal(usig))

        den = u_vec @ np.linalg.inv(usig_diag) @ (usig @ np.linalg.inv(usig_diag)) @ u_vec.T
        if den < 0:
            den = -den

        dp2 = num / den

        wopt_diag = np.linalg.inv(usig_diag) @ u_vec.T

        return wopt_diag, np.sqrt(dp2)



        


