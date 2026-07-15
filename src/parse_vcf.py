import cyvcf2, numpy as np
from cyvcf2 import VCF


def parse_vcf_raw(vcf_path):
    vcf_path = VCF(vcf_path)

    sample_ids = vcf_path.samples
    variant_ids = []
    
    G = []
    
    for variant in vcf_path:
        variant_ids.append(variant.POS)
        row = []

        for gt in variant.genotypes:
            
            if gt[0] == -1 or gt[1] == -1: #Treating half-calls as completely missing
                dosage = np.nan
            else:
                dosage = gt[0] + gt[1]

            row.append(dosage)

        G.append(row)
    
    G = np.array(G, dtype=float)
    print(G.shape)

    return G, np.array(variant_ids), np.array(sample_ids)


def compute_missingness(G):
    missingness = np.zeros(G.shape[0])

    for i, row in enumerate(G):
        total_nans = np.sum(np.isnan(row))
        missingness[i] = total_nans/len(row)

    return missingness
        

def compute_maf(G):
    maf = []
    for i, row in enumerate(G):
        freq = np.nansum(row)/(np.count_nonzero(~np.isnan(row))*2)
        if freq > 0.5:
            freq = 1 - freq
        maf.append(freq)

    return np.array(maf)

def filter_variants(G, variant_ids, missingness, maf, maf_thres = 0.01, missing_thres = 0.05):

    #Discards variants with low MAF and high missingness

    mask = (maf > maf_thres) & (missingness < missing_thres)
    G_filtered = G[mask]
    variant_ids_filtered = variant_ids[mask]

    print(len(variant_ids_filtered), "variants passed the filter of MAF > 0.01 and missingness < 0.05")

    return  np.array(G_filtered), np.array(variant_ids_filtered)

def impute_mean(G):

    G_imputed = G.copy()

    row_means = np.nanmean(G_imputed, axis = 1, keepdims=True)

    G_imputed = np.where(np.isnan(G_imputed), row_means, G_imputed)

    return G_imputed


if __name__ == "__main__":
    G, variant_ids, sample_ids = parse_vcf_raw("data/1kg/chr11_phased.vcf.gz")

    print("Matrix shape:", G.shape)
    print("First variant position:", variant_ids[0])
    print("First variant genotypes:", G[0][:])