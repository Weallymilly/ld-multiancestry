import numpy as np

def test_ld_naive_matches_corrcoef():
    G = np.array([[0,1,2],[0,0,1],[1,1,2],[2,0,0],[1,2,1]], dtype=float)
    R = ld_naive(G)
    expected = np.corrcoef(G)
    np.testing.assert_allclose(R, expected, atol=1e-6)
