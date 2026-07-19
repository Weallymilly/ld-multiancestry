import torch, numpy as np

def ld_torch_computation(G):

    with torch.no_grad():
        G = torch.tensor(G, dtype=float)

        means = torch.mean(G, dim = 1)
        std = torch.std(G, dim = 1, correction = 1)
        G_std = (G - means[:, None])/std[:, None]

        ld_matrix = G_std @ G_std.T/(G.shape[1]-1)

    return ld_matrix

def ld_torch_cpu(G, num_threads = 15):
    torch.set_num_threads(num_threads)
    return ld_torch_computation(G)

if __name__ == "__main__":
    G = np.random.randint(0,3,size=(100,200))
    ld_matrix = ld_torch_cpu(G)

    print(ld_matrix[0,:20])
