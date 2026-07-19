import numpy as np
import cProfile
from src.ld_torch_cpu import ld_torch_cpu, ld_torch_computation

n_samples = 200
G_small = np.random.randint(0, 3, size=(200,n_samples))

cProfile.run("ld_torch_cpu(G_small)", sort='cumulative')