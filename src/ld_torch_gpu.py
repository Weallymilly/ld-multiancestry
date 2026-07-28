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

    return ld_matrix.cpu()

def time_decomposed(G):
    device = torch.device('cuda')

    with torch.inference_mode():
        G = torch.tensor(G, dtype=torch.float32)

        events = {
            "start_in": torch.cuda.Event(enable_timing=True),
            "end_in": torch.cuda.Event(enable_timing=True),
            "start_compute": torch.cuda.Event(enable_timing=True),
            "end_compute": torch.cuda.Event(enable_timing=True),
            "start_exit": torch.cuda.Event(enable_timing=True),
            "end_exit": torch.cuda.Event(enable_timing=True)
        }

        events["start_in"].record(stream=None)
        g_GPU = G.to(device)
        events["end_in"].record(stream=None)

        events["start_compute"].record(stream=None)
        means = torch.mean(g_GPU, dim = 1)
        std = torch.std(g_GPU, dim = 1, correction = 1)
        g_std = (g_GPU - means[:, None])/std[:, None]
        
        ld_matrix = g_std @ g_std.T/(g_GPU.shape[1]-1)

        events["end_compute"].record(stream=None)

        events["start_exit"].record(stream=None)
        ld_matrix = ld_matrix.cpu()
        events["end_exit"].record(stream=None)
        events["end_exit"].synchronize()

        total_elapsed_time = events["start_in"].elapsed_time(events["end_exit"])/1000

        separated_times = [events["start_in"].elapsed_time(events["end_in"]), events["start_compute"].elapsed_time(events["end_compute"]), events["start_exit"].elapsed_time(events["end_exit"])]

        separated_times = [x/1000 for x in separated_times]


    return total_elapsed_time, separated_times



