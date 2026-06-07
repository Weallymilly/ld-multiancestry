# Ultrafast Sample placement on Existing tRees (UShER) enables real-time phylogenetics for the SARS-CoV-2 pandemic
** https://doi.org/10.1038/s41588-021-00862-7 **
** Read:** June 5, 2026
** Repo:** github.com/yatisht/usher

## 1. Core Claim
Modelling the phylogenetic relationship in rapidly evolving specimen, such as viruses, is important for genomic contact tracing. Before UShER, real-time classification of viral-sequence samples was impossible because of algorithms/methods that were too time and memory inefficient to enable real-time classfication. UShER proved that by using MAT - a more practical data structure, bioinformatic analysis can be done in seconds rather than hours or days.

## 2. The Problem Before UShER
With the developing of high-throughput sequencing strategies, COVID-19 produced amounts of sequencing data existing processing methods cannot handle to provide real-time insight into the spread and evolution of the virus. Not only are existing tools computationally expensive, they are also memory-inefficient: the volume of sequencing data makes computing even on state-of-the-art hardware menial in contributing to real-time genomic contact tracing. 'Real-time' in UShER's context means seconds rather than the hours and days needed for prior tools.


## 3. The Mutation-Annotated Tree (MAT)
MAT stores the positions of the SNPs in a tree-based data object and omits storing the full sequence. This allows mutations to be represented sequentially, matching the shape of the phylogenetic tree. Attached to a tree is a table of mutations for each node. Mutations in SARS-CoV-2 are sparse relative to its size (roughly 30kb). This compressed representation saves compute and memory. This adopts a similar intuition to sparse matricies, where only the important non-zero values (i.e., the mutations) and their positions are stored, instead of the full matrix; in this analogy, the full matrix would be the MSA file type, while the sparse matrix is analogous to MAT.


## 4. The Placement Algorithm
UShER uses a maximum parsimony criterion when adding new samples, which adds the new sample to the tree with the most SNP efficient explanation of new traits. When placing in a new sample, UShER loads the MAT and computes an additional parsimony score. It uses mutation accumulation logic to prune the search space - the new sample can only be placed at a position with the valid accumulation of mutations. If there is a tie in the selection process, the place with the greatest number of descendent leaves is selected, or, if it is a parent vs. child case, the node with the most idiosyncratic descendent leaves is selected - the logic being a favoring of the more probabalistic/popular path of evolution.


## 5. Key Numbers
- 97.4% placement accuracy of placing SARS-CoV-2 samples with known true positions. Misplacement is always close to real position (on the adjacent node) with a median parsimony distance for misplacement of 0. 
- Roughly 400x faster than IQ-TREE. IQ-TREE operates on an hours time magnitude while UShER operates in seconds.
- 3.4MB (MAT) vs. 1.14 GB(full MSA) for the same dataset
- The benchmark dataset is the SARS-CoV-2 phylogeny from the UCSC Genome Browser.

## 6. UShER vs. pangoLEARN — The Algorithmic vs. ML Argument
pangoLEARN is a machine-learning approach that labels new samples by lineage based on sequence features. Unlike pangoLEARN, UShER does not need training data to function, rather it uses a fixed reference tree to which the new sample is added precise in relation to other samples. Although, in the SARS-CoV-2 application, data availability isn't a main bottleneck, the complexities that come with training ML models, such as feature section for a sparse mutation pattern, computational costs and used for training, and accounting for uncertainty in results, make UShER stand out. This classification problem is not a typical 'black box' problem that is best solved by ML methods but an iterative problem: instead, structured algorithms are more effective when dealing with this class of problems. This motivates a benchmark comparison of these two approaches when utilizing GPU-acceleration, answering the question 'when given the same hardware, which is more effective.'


## 7. The MAT Ecosystem (matOptimize, RIPPLES, matUtils)
- **matOptimize:** an algorithm that rearranges phylogenetic trees, optimizing for maximum parsimony.
- **RIPPLES:** detects recombination events/sequences in large MAT files
- **matUtils:** a toolset to operate on MAT files 


## 8. Connection to My Project
The engineering philosophy of this paper is to keep data lean and easy to work with: memory quickly becomes a bottleneck in bioinformatics problems. Thus, often, the core issue is knowing what information is relevant and pruning for it. MAT achieves this by omitting data that the placement algorithm will not need to read. The "hardware-software co-design" philosophy of the Turakhia lab is made evident to me through UShER: any real-world application of software must account for physical bottlenecks such as compute or memory as the purpose of developing these tools or infrastructure is to solve real problems. MAT is a data structure design that enables cache fit in CPUs - a realization of hardware situation that must be exploited for speed. This has inspired me to compare naive, NumPy, PyTorch CPU, and CUDA implementations. Reading UShER shaped how I designed my benchmark because it points out, despite the tech-space investing massively in GPUs and advanced hardware, clever utilization of existing hardware is what really needs to be studied and improved.
