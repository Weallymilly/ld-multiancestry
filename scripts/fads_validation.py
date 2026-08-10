import numpy as np
import pandas as pd
from pathlib import Path

def load_aligned():
    archive = np.load('data/processed/fads_EAS_EUR_LD.npz')
    
    eas_ld = archive['EAS_LD']
    eur_ld = archive['EUR_LD']
    
    archive = np.load('data/processed/fads_shared_alignment.npz')

    #ids stores the ids of shared variants in eas and eur

    ids = archive['ids']

    #Gives pairwise kb positions of ids
    ids_dis = np.abs(ids[:, None] - ids[None, :])/1000

    return eas_ld, eur_ld, ids_dis

def flatten_to_pairs(eas_ld, eur_ld, ids_dis):
    iu = np.triu_indices(eas_ld.shape[0], k=1)

    eas_r2 = eas_ld[iu]**2
    eur_r2 = eur_ld[iu]**2

    if ids_dis is not None:
        dist_kb = ids_dis[iu]

        return eas_r2, eur_r2, dist_kb

    return eas_r2, eur_r2

def summary_stats(eas_r2, eur_r2):

    mean_eas_r2 = np.mean(eas_r2)
    mean_eur_r2 = np.mean(eur_r2)

    mask_eas = (eas_r2 >= 0.8)
    mask_eur = (eur_r2 >= 0.8)
    

    frac_eas = mask_eas.mean()
    frac_eur = mask_eur.mean()

    return mean_eas_r2, mean_eur_r2, frac_eas, frac_eur

def ld_decay(eas_r2, eur_r2, dist_kb, bins):
    ld_eas = []
    ld_eur = []
    n_pairs = []

    for lo, hi in bins:

        mask = (dist_kb >= lo) & (dist_kb < hi)

        ld_eas.append(np.mean(eas_r2[mask]))
        ld_eur.append(np.mean(eur_r2[mask]))
        n_pairs.append(mask.sum())

    return ld_eas, ld_eur, n_pairs

def discordance(eas_r2, eur_r2, strong=0.8, weak=0.2):
    strong_eas = (eas_r2 >= strong)
    strong_eur = (eur_r2 >= strong)

    weak_eas = (eas_r2 <= weak)
    weak_eur = (eur_r2 <= weak)

    shared_strong = strong_eas & strong_eur
    only_eas = strong_eas & weak_eur
    only_eur = strong_eur & weak_eas

    strong_count = shared_strong.sum()
    only_eas_count = only_eas.sum()
    only_eur_count = only_eur.sum()

    corr = np.corrcoef(eas_r2, eur_r2)[0,1]

    return strong_count, only_eas_count, only_eur_count, corr



if __name__ == "__main__":
    bins = [(0,5), (5,10), (10,25), (25,50), (50,100), (100,270)] #bin sizes in kb

    eas_ld, eur_ld, ids_dis = load_aligned()
    eas_r2, eur_r2, dist_kb = flatten_to_pairs(eas_ld, eur_ld, ids_dis)

    mean_eas_r2, mean_eur_r2, frac_eas, frac_eur = summary_stats(eas_r2, eur_r2)

    ld_eas, ld_eur, n_pairs = ld_decay(eas_r2, eur_r2, dist_kb, bins)

    strong_count, only_eas_count, only_eur_count, corr = discordance(eas_r2, eur_r2, strong=0.8, weak=0.2)

    assert sum(n_pairs) == eur_r2.shape[0]

    summary_df = pd.DataFrame([
        {
            "population": "EAS",
            "mean_r2": mean_eas_r2,
            "strong_pair_fraction": frac_eas,
        },
        {
            "population": "EUR",
            "mean_r2": mean_eur_r2,
            "strong_pair_fraction": frac_eur
        }

    ])
    discordance_df = pd.DataFrame([
        {
            "shared_strong_pair": strong_count,
            "eas_strong_eur_weak_pairs": only_eas_count,
            "eur_strong_eas_weak_pairs": only_eur_count,
            "corr_r2_between_pops": corr,
            "frac_r2_variance_shared": corr**2,
            "strong_threshold": 0.8,
            "weak_threshold": 0.2
        }
    ])

    decay_rows = []

    for (lo, hi), eas_mean, eur_mean, count in zip(
        bins, ld_eas, ld_eur, n_pairs
    ):
        decay_rows.append({
            "distance_start_kb": lo,
            "distance_end_kb": hi,
            "eas_mean_r2": eas_mean,
            "eur_mean_r2": eur_mean,
            "n_pairs": count
        })

    decay_df = pd.DataFrame(decay_rows)

    Path('results').mkdir(exist_ok=True)

    print(summary_df)
    print(discordance_df)
    print(decay_df)

    summary_df.to_csv('results/fads_summary.csv', index=False, float_format="%.6f")
    discordance_df.to_csv('results/fads_discordance.csv', index=False, float_format="%.6f")
    decay_df.to_csv('results/fads_ld_decay.csv', index=False, float_format="%.6f", na_rep="NA")
