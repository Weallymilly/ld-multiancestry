import numpy as np, torch
from src.ld_torch_gpu import ld_torch_gpu

def test_ld_torch_gpu_matches_corrcoef():
    G = np.array([[0,1,2],[0,0,1],[1,1,2],[2,0,0],[1,2,1]], dtype=float)
    R = ld_torch_gpu(G)
    expected = np.corrcoef(G)
    np.testing.assert_allclose(R, expected, atol=1e-6)