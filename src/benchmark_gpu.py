import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import torch
from benchmark import generate_synthetic_G
from ld_torch_gpu import ld_torch_gpu, time_decomposed

def decomposed_benchmark(window_sizes, n_samples, n_reps = 3):

    results = []

    for i in range(len(window_sizes)):
        G = generate_synthetic_G(window_sizes[i], n_samples)

        try:
            for j in range(n_reps):

                total_t, sep = time_decomposed(G)

                record = {
                    "n_variants" : window_sizes[i],
                    "transfer_in_time" : sep[0],
                    "compute_time" : sep[1],
                    "transfer_out_time" : sep[2],
                    "total_time" : total_t
                }

                results.append(record)

            print(f"Passed for window_size {window_sizes[i]}.")

        except torch.cuda.OutOfMemoryError:
            print(f"Caught CUDA OOM error at window size {window_sizes[i]}")
            torch.cuda.empty_cache()
            break

    df = pd.DataFrame(results)

    #The following check is to ensure that the decomposed times sum to the total time with 1% relative tolerance. This is important because the decomposed times are measured separately and may have small discrepancies due to timing overheads or measurement inaccuracies. The check uses numpy's isclose function to compare the sum of the decomposed times with the total time for each record in the DataFrame. If all records pass this check, it confirms that the decomposition is consistent with the total time reported.
    stage_sum = df[['transfer_in_time', 'compute_time', 'transfer_out_time']].sum(axis=1)
    is_correct = np.isclose(stage_sum, df['total_time'], rtol=1e-2).all()

    print(f"Do the decomposed times sum to the total time? {is_correct}")


    df = df.groupby(["n_variants"])[["transfer_in_time", "compute_time", "transfer_out_time","total_time"]].min()

    return df.reset_index()

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print("CUDA is available. Running GPU benchmark.")

    #Giving CUDA a warmp-up run before benchmarking
    small_G = np.random.randint(0,3,size = (50, 50))
    time_decomposed(small_G)

    #Still using big-n to check for transfer times

    window_sizes = [5000, 8000, 10000, 15000, 20000, 30000, 40000]
    n_samples = 600

    df_gpu = decomposed_benchmark(window_sizes, n_samples)

    for col in ["transfer_in_time", "compute_time", "transfer_out_time"]:
        plt.plot(df_gpu['n_variants'], df_gpu[col], label = col)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('runtime (seconds)')
    plt.title("Decomposed GPU runtime (log-log)", fontdict={'size': 8})
    plt.legend()

    plt.savefig("figs/fig3_gpu_decomposed.png", dpi=300, bbox_inches='tight')

    y_cols = ['transfer_in_time', 'compute_time', 'transfer_out_time']
    df_percent = df_gpu[y_cols].div(df_gpu["total_time"], axis=0) * 100

    df_percent.index = df_gpu["n_variants"]

    ax = df_percent.plot(
        y= y_cols,
        kind='bar',
        stacked=True,
        ylabel="Percentage of total time",
        title="Stacked Contributions to Total"
    )

    #The figure produced, Fig4, may have bars not summing to 100% due to rounding errors in the percentage calculations. That is because of the grouby step taking the minimum of each time component, which can lead to slight discrepancies when calculating percentages.

    plt.savefig("figs/fig4_gpu_dec_stacked.png", dpi=300, bbox_inches='tight')

    df_gpu.to_csv("results/gpu_decomposed.csv")