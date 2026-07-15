import numpy as np
from src.parse_vcf import compute_missingness, compute_maf, filter_variants, impute_mean
#AI_generated


def make_toy_G():
    # 3 variants x 5 samples
    return np.array([
        [0, 1, 2, np.nan, 1],
        [2, 2, np.nan, 0, 1],
        [1, np.nan, np.nan, 1, 0],
    ])

def test_compute_missingness():
    G = make_toy_G()
    result = compute_missingness(G)
    np.testing.assert_allclose(result, [0.2, 0.2, 0.4])

def test_compute_maf():
    G = make_toy_G()
    result = compute_maf(G)
    np.testing.assert_allclose(result, [0.5, 0.375, 1/3], atol=1e-6)

def test_filter_variants():
    G = make_toy_G()
    variant_ids = np.array([10, 20, 30])
    missingness = np.array([0.2, 0.2, 0.4])
    maf = np.array([0.5, 0.375, 1/3])
    G_f, ids_f = filter_variants(G, variant_ids, missingness, maf,
                                  maf_thres=0.01, missing_thres=0.3)
    assert G_f.shape[0] == 2
    np.testing.assert_array_equal(ids_f, [10, 20])

def test_impute_mean_no_nans_remain():
    G = make_toy_G()
    G_imputed = impute_mean(G)
    assert not np.isnan(G_imputed).any()

def test_impute_mean_preserves_row_mean():
    # Mean imputation shouldn't change a row's own average — that's the defining property of "fill with the mean."
    G = make_toy_G()
    original_means = np.nanmean(G, axis=1)
    G_imputed = impute_mean(G)
    np.testing.assert_allclose(G_imputed.mean(axis=1), original_means)

def test_impute_mean_does_not_mutate_input():
    G = make_toy_G()
    G_before = G.copy()
    impute_mean(G)
    np.testing.assert_array_equal(G, G_before)