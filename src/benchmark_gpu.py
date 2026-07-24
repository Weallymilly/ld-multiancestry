import time, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import torch
from benchmark import time_one_run, generate_synthetic_G
from ld_torch_gpu import ld_torch_gpu

def benchmark_gpu(ld_func, window_sizes, n_samples, n_reps = 3):

    results = []

    for i in range(len(window_sizes)):
        G = generate_synthetic_G(window_sizes[i], n_samples)

        try:
            for j in range(n_reps):

                record = {
                    "implementation" : ld_func.__name__,
                    "n_variants" : window_sizes[i],
                    "elapsed_time" : time_one_run(ld_func, G)
                }

                results.append(record)

            print(f"Passed {ld_func.__name__} for window_size {window_sizes[i]}.")

        except torch.cuda.OutOfMemoryError:
            print(f"Caught CUDA OOM error at window size {window_sizes[i]}")
            torch.cuda.empty_cache()
            break

    df = pd.DataFrame(results)
    df = df.groupby(["implementation","n_variants"])["elapsed_time"].min()

    return df.reset_index()

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print("CUDA is available. Running GPU benchmark.")

    #Giving CUDA a warmp-up run before benchmarking
    small_G = np.random.randint(0,3,size = (50, 50))
    ld_torch_gpu(small_G)

    window_sizes = [5000, 8000, 10000, 15000, 20000, 30000, 40000]
    n_samples = 600

    df_gpu = benchmark_gpu(ld_torch_gpu, window_sizes, n_samples)

    sns.lineplot(data=df_gpu, x="n_variants", y="elapsed_time", hue="implementation", marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('runtime (seconds)')
    plt.title("LD Large-n Runtime GPU (log-log)", fontdict={'size': 8})

    plt.savefig("figs/fig2_gpu_large_n.png", dpi=300, bbox_inches='tight')

    df_gpu.to_csv("results/gpu_large_n.csv")