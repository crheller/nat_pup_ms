"""
What is the expected error in an estimate of a random vector in 2D space?
"""

import numpy as np
import matplotlib.pyplot as plt
from charlieTools.nat_sounds_ms.decoding import reflect_eigenvectors

niters = 1000
vec_err = []
for i in range(niters):
    nvecs = 10
    vecs = []
    for i in range(0, nvecs):
        v = np.random.normal(0, 1, size=(2,1))
        v /= np.linalg.norm(v)
        vecs.append(v)

    vecs = reflect_eigenvectors(np.stack(vecs)).squeeze()
    # get "standard error" of eigenvectors
    sem = np.std(vecs, axis=0) / np.sqrt(nvecs)
    # collapse into single number
    err = np.sqrt(np.sum(sem ** 2))
    vec_err.append(err)

# plot excmple random vectors and distribution of errors
f, ax = plt.subplots(1, 2, figsize=(8, 4))
for v in vecs:
    x = [0+v[0], 0-v[0]]
    y = [0+v[1], 0-v[1]]
    ax[0].plot(x, y, color='grey')
ax[0].set_title(f"Err: {round(vec_err[-1], 3)}")
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)

ax[1].set_title(f"Distribution of err over {niters} samples of {nvecs} vectors", fontsize=6)
ax[1].hist(vec_err, bins=10)

plt.show()