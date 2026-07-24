import time, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from ld_numpy import ld_numpy
from ld_naive import ld_naive
from ld_torch_cpu import ld_torch_cpu
from ld_torch_gpu import ld_torch_gpu

#Use sample size of ~633 matching 633 from EUR

def generate_synthetic_G(n_variants, n_samples):

    G = np.random.randint(0,3,size = (n_variants, n_samples))

    return G


def time_one_run(ld_func, G):
    start_time = time.perf_counter()
    ld_func(G)
    end_time = time.perf_counter()

    elapsed = end_time - start_time

    #print(f"The total elapsed time for function {ld_func.__name__} is {elapsed} nanoseconds.")

    return elapsed


def benchmark(ld_func, window_sizes, n_samples, n_reps = 3):

    results = []

    for i in range(len(window_sizes)):
        G = generate_synthetic_G(window_sizes[i], n_samples)
        
        for j in range(n_reps):

            record = {
                "implementation" : ld_func.__name__,
                "n_variants" : window_sizes[i],
                "elapsed_time" : time_one_run(ld_func, G)
            }

            results.append(record)

        print(f"Passed {ld_func.__name__} for window_size {window_sizes[i]}.")

    df = pd.DataFrame(results)
    df = df.groupby(["implementation","n_variants"])["elapsed_time"].min()

    return df.reset_index()

if __name__ == "__main__":

    #Giving CUDA a warmp-up run before benchmarking
    small_G = np.random.randint(0,3,size = (50, 50))
    ld_torch_gpu(small_G)

    #Starting the actual benchmarking
    window_sizes = [100, 200, 500, 1000, 2000, 5000]
    n_samples = 600

    df_gpu = benchmark(ld_torch_gpu, window_sizes, n_samples)
    df_torch_cpu = benchmark(ld_torch_cpu, window_sizes, n_samples)
    df_numpy = benchmark(ld_numpy, window_sizes, n_samples)
    df_naive = benchmark(ld_naive, window_sizes, n_samples, n_reps=1)


    result_table = pd.concat([df_naive, df_numpy, df_torch_cpu, df_gpu], ignore_index=True)

    sns.lineplot(data=result_table, x="n_variants", y="elapsed_time", hue="implementation", marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('runtime (seconds)')
    plt.title("LD Computation Runtime: Naive vs. NumPy vs. PyTorch_CPU vs. GPU (log-log)", fontdict={'size': 8})

    plt.savefig("figs/fig1_speedup_curves_v3_gpu.png", dpi=300, bbox_inches='tight')
    #The GPU run is done with ld_torch_cpu and ld_numpy running at 15 threads

    result_table.to_csv("results/all_vs_gpu.csv")
