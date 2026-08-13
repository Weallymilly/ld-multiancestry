import torch, pandas as pd, numpy as np
from src.ld_torch_gpu import transfer_chunked
from src.benchmark import generate_synthetic_G


def chunked_benchmark(window_sizes, n_samples, n_reps = 3):

    results = []

    rows_per_chunk = 1000
    staging = torch.empty((rows_per_chunk,max(window_sizes)), pin_memory=True)

    for i in range(len(window_sizes)):
        
        out_cpu = torch.empty((window_sizes[i],window_sizes[i]))
    
        G = generate_synthetic_G(window_sizes[i], n_samples)

        try:
            for j in range(n_reps):

                total_t, sep = transfer_chunked(G,
                                                out_cpu=out_cpu,
                                                staging=staging[:,:window_sizes[i]], 
                                                rows_per_chunk=rows_per_chunk)

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
    window_sizes = [30000, 40000]
    n_samples = 600
    df_chunked = chunked_benchmark(window_sizes, n_samples)
    print(df_chunked)
    df_chunked.to_csv('results/gpu_chunked.csv')