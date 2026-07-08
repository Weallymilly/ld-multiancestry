import numpy as np

G = np.array([[0, 1, 2],
              [0, 0, 1],
              [1, 1, 2],
              [2, 0, 0],
              [1, 2, 1]])

def ld_naive(G):
    n_variants, n_samples = G.shape
    means = np.array([sum(row)/len(G[0]) for row in G])
    G_fixed = G - means[:,np.newaxis]
    stds = (np.sum(G_fixed**2, axis = 1)/(len(G[0])-1))**0.5
    G_std = G_fixed/stds[:,np.newaxis]
    print(G_std)
    
    R = np.zeros((n_variants, n_variants))
    for i in range(n_variants):
        for j in range(n_variants):
            total = 0
            for k in range(n_samples):
                total += G_std[i,k] * G_std[j,k]
                print(G_std[i,k])
            R[i,j] = total/(n_samples - 1)
            R[j,i] = R[i,j]
    return R

print(ld_naive(G))

 

