import numpy as np

def ld_numpy(G):

    means = G.mean(axis = 1)
    std = np.std(G, axis = 1, dtype=float, ddof = 1)
    G_std = (G - means[:, np.newaxis])/std[:, np.newaxis]

    ld_matrix = G_std @ G_std.T/ (G.shape[1]-1)

    return ld_matrix
