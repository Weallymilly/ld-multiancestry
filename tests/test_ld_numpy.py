import numpy as np
from src.ld_numpy import ld_numpy

def test_ld_numpy_matches_corrcoef():
    G = np.array([[0,1,2],[0,0,1],[1,1,2],[2,0,0],[1,2,1]], dtype=float)
    R = ld_numpy(G)
    expected = np.corrcoef(G)
    np.testing.assert_allclose(R, expected, atol=1e-6)
