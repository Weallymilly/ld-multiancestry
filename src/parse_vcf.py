import numpy as np
from cyvcf2 import VCF
#Region chosen as : chr11 61,700,000-61,970,000 extracted with bcftools

def parse_vcf_raw(vcf_path):

    sample_ids = vcf_path.samples
    variant_ids = []
    variant_rsids = []
    ref = []
    alt = []
    
    G = []
    
    for variant in vcf_path:
        variant_ids.append(variant.POS)
        variant_rsids.append(variant.ID)

        assert len(variant.ALT) == 1

        ref.append(variant.REF)
        alt.append(variant.ALT[0])
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

    return G, np.array(variant_ids), np.array(variant_rsids), np.array(sample_ids), np.array(ref), np.array(alt)


def compute_missingness(G):
    missingness = np.zeros(G.shape[0])

    for i, row in enumerate(G):
        total_nans = np.sum(np.isnan(row))
        missingness[i] = total_nans/len(row)

    return missingness
        

def compute_maf(G):
    maf = []
    alt_freq = []

    #This calculates folded allele frequency in maf (Minor Allele Frequency)- it is ancestry blind, but it keeps alt_freq
    for i, row in enumerate(G):
        freq = np.nansum(row)/(np.count_nonzero(~np.isnan(row))*2)

        alt_freq.append(freq)
        if freq > 0.5:
            freq = 1 - freq
        maf.append(freq)

    return np.array(maf), np.array(alt_freq)

def filter_variants(G, variant_ids, variant_rsids, ref, alt, alt_freq, missingness, maf, maf_thres = 0.01, missing_thres = 0.05):

    #Discards variants with low MAF and high missingness

    mask = (maf > maf_thres) & (missingness < missing_thres)
    G_filtered = G[mask]
    variant_ids_filtered = variant_ids[mask]
    rsids_filtered = variant_rsids[mask]
    ref_filtered  = np.array(ref[mask])
    alt_filtered = np.array(alt[mask])
    alt_freq_filtered = np.array(alt_freq[mask])

    print(len(variant_ids_filtered), "variants passed the filter of MAF > 0.01 and missingness < 0.05")

    return  np.array(G_filtered), np.array(variant_ids_filtered), np.array(rsids_filtered), ref_filtered, alt_filtered, alt_freq_filtered

def impute_mean(G):

    G_imputed = G.copy()

    row_means = np.nanmean(G_imputed, axis = 1, keepdims=True)

    G_imputed = np.where(np.isnan(G_imputed), row_means, G_imputed)

    return G_imputed


if __name__ == "__main__":

    for name in ["EAS", "EUR"]:
        vcf_path = VCF(f"data/1kg/fads_{name}.vcf.gz")
        G, variant_ids, variant_rsids, sample_ids, ref, alt = parse_vcf_raw(vcf_path)
        assert variant_ids.shape == ref.shape == alt.shape
        print("Matrix shape:", G.shape)

        missingness = compute_missingness(G)
        maf, alt_freq = compute_maf(G)
        G_filtered, variant_ids_filtered, rsids_filtered, ref_filtered, alt_filtered, alt_freq_filtered = filter_variants(G, variant_ids, variant_rsids, ref, alt, alt_freq, missingness, maf)
        G_imputed = impute_mean(G_filtered)

        np.save(f"data/processed/fads_{name}_G_imputed.npy", G_imputed)
        np.save(f"data/processed/fads_{name}_variant_ids.npy", variant_ids_filtered)
        np.save(f"data/processed/fads_{name}_variant_rsids.npy", rsids_filtered)
        np.save(f"data/processed/fads_{name}_ref.npy", ref_filtered)
        np.save(f"data/processed/fads_{name}_alt.npy", alt_filtered)
        np.save(f"data/processed/fads_{name}_alt_freq.npy", alt_freq_filtered)

        print(f"Final imputed matrix shape for {name}:", G_imputed.shape)
        print(f"Any NaN remaining for {name}:", np.isnan(G_imputed).any())

