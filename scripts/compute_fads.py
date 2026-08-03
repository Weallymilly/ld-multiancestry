import numpy as np
from src.ld_torch_gpu import ld_torch_gpu

if __name__ == "__main__":
    archive = np.load('data/processed/fads_shared_alignment.npy.npz')

    shared_ids, idx_eas, idx_eur = archive['ids'], archive['idx_eas'], archive['idx_eur']

    EAS = np.load('data/processed/fads_EAS_G_imputed.npy')[idx_eas]
    EUR = np.load('data/processed/fads_EUR_G_imputed.npy')[idx_eur]

    EUR_LD = ld_torch_gpu(EUR)
    EAS_LD = ld_torch_gpu(EAS)

    np.savez('data/processed/fads_EAS_EUR_LD', EAS_LD = EAS_LD, EUR_LD = EUR_LD)