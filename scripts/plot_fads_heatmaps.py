import numpy as np, matplotlib.pyplot as plt
from fads_validation import discordance, summary_stats, flatten_to_pairs

#Position FADS1 on GRCh38: chr11. 61,799,627 - 61,817,003
#Position FADS2 on CRCh38: chr11. 61,828,300-61,867,354

def load_aligned_and_pos():
    archive = np.load('data/processed/fads_EAS_EUR_LD.npz')
    
    eas_ld = archive['EAS_LD']
    eur_ld = archive['EUR_LD']
    
    archive = np.load('data/processed/fads_shared_alignment.npz')

    #ids stores the ids of shared variants in eas and eur

    positions, idx_eas, idx_eur = archive['ids'], archive['idx_eas'], archive['idx_eur']

    assert np.all(positions[:-1] <= positions[1:])

    return eas_ld, eur_ld, positions, idx_eas, idx_eur

def bp_to_index(positions, bp):
    id = np.searchsorted(positions, bp, side = 'left', sorter=None)
    return id

def plot_ld_heatmaps(eas_ld, eur_ld, positions, gene_spans):

    fig, (ax1,ax2) = plt.subplots(1,2)

    genes_pos = []
    gene_centers = []
    gene_labels = []

    for i in range(len(gene_spans)):
        genes_pos.extend([gene_spans[i]["start_idx"], gene_spans[i]["end_idx"]])
        gene_centers.append(sum([gene_spans[i]["start_idx"], gene_spans[i]["end_idx"]])/2)
        gene_labels.append(gene_spans[i]["gene"])


    eas = ax1.imshow(eas_ld**2, interpolation='nearest', cmap='Reds', vmax=1, vmin=0)
    eur = ax2.imshow(eur_ld**2, interpolation='nearest', cmap='Reds', vmax=1, vmin=0)

    for ax in (ax1, ax2):
        #Set bottom midpoint labels
        ax.set_xticks(gene_centers, minor=True)
        ax.set_xticklabels(gene_labels, rotation=90, fontsize=5)

        #Set gene boundary labels
        ax.set_xticks(genes_pos)

        #Set top generic labels
        bp_pos = ax.secondary_xaxis("top")
        bp_pos.set_xticks(np.linspace(0,len(positions), 5, dtype=int))

        labels = ax.get_xticklabels()
        labels[0].set_y(-0.03)
        labels[1].set_y(-0.08)


    fig.colorbar(eur, ax=[ax1, ax2], label="r2")
    fig.supxlabel("Variant index (rank order, not bp)")
    fig.suptitle("EAS vs. EUR")

    return fig




if __name__ == "__main__":

    eas_ld, eur_ld, positions, idx_eas, idx_eur = load_aligned_and_pos()

    print(positions.shape)
    print(positions[:5])
    print(positions[-1])

    gene_spans = [{
            "gene": "FADS1",
            "start_idx": bp_to_index(positions, 61799627),
            "end_idx": bp_to_index(positions, 61817003)
        },{
            "gene": "FADS2",
            "start_idx": bp_to_index(positions, 61828300),
            "end_idx": bp_to_index(positions, 61867354)
        }]

    eas_r2, eur_r2 = flatten_to_pairs(eas_ld, eur_ld, ids_dis=None)

    strong_count, only_eas_count, only_eur_count, corr = discordance(eas_r2, eur_r2)

    print(strong_count, only_eas_count, only_eur_count, corr)

    mean_eas_r2, mean_eur_r2, frac_eas, frac_eur = summary_stats(eas_r2, eur_r2)

    print(mean_eas_r2, mean_eur_r2, frac_eas, frac_eur)

    fig = plot_ld_heatmaps(eas_ld, eur_ld, positions, gene_spans)

    plt.show()

    fig.savefig('figs/fig5_fads_ld_heatmaps.png', dpi=300, bbox_inches='tight')
