import numpy as np
import cProfile
from src.ld_naive import ld_naive

n_samples = 200
G_small = np.random.randint(0, 3, size=(200,n_samples))

cProfile.run("ld_naive(G_small)", sort='cumulative')