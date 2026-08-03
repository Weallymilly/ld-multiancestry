import numpy as np

def find_shared_variant_ids():
    EAS = np.load('data/processed/fads_EAS_variant_ids.npy')
    EUR = np.load('data/processed/fads_EUR_variant_ids.npy')

    #EAS has 783 variants while EUR has 805 variants

    unique_eas, counts = np.unique(EAS, return_counts=True)
    single_eas = unique_eas[counts ==1]

    unique_eur, counts = np.unique(EUR, return_counts=True)
    single_eur = unique_eur[counts ==1]

    #drops multiallelic sites since input ID dataset doesn't preserve nature of alleles

    shared_ids, idx_eas, idx_eur = np.intersect1d(single_eas, single_eur, return_indices=True)
    idx_eas = np.nonzero(np.isin(EAS, shared_ids))[0]
    idx_eur = np.nonzero(np.isin(EUR, shared_ids))[0]

    assert np.array_equal(EUR[idx_eur], EAS[idx_eas])

    return shared_ids, idx_eas, idx_eur


if __name__ == "__main__":
    shared_ids, idx_eas, idx_eur = find_shared_variant_ids()
    