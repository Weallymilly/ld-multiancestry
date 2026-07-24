import torch, numpy as np

def ld_torch_gpu(G,device = None):

    with torch.inference_mode():
        G = torch.as_tensor(G, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            print("Using CUDA")

        g_GPU = G.to(device)

        means = torch.mean(g_GPU, dim = 1)
        std = torch.std(g_GPU, dim = 1, correction = 1)
        g_std = (g_GPU - means[:, None])/std[:, None]

        ld_matrix = g_std @ g_std.T/(g_GPU.shape[1]-1)

        if device.type == "cuda":
            torch.cuda.synchronize()

    return ld_matrix
