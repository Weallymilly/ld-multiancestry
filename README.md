# Id-multiancestry

An exact-and-approximate LD pipeline that finds where GPU acceleration stops helping (memory bandwidth, not compute), then measures how far LD can be compressed before multi-ancestry fine-mapping at the FADS locus breaks.

## Motivation

Linkage disequilibrium (LD) is a calculation crucial to fine-mapping that becomes increasing large ($O(n^2)$) as the number of variants ($n$) grow, constituting an important bottleneck in fine-mapping. Multi-ancestry fine-mapping further requires separate LD matrices per ancestry panel. Given its the statistical importance and computational cost, we decided to investigate whether LD calculations can be optimized in similar bioinformatics pipelines. 

## Questions

In this repository, we are curious about:
1. Different implementations used in LD matrix creation affect the computational cost and speed.
2. 

## Data & Methods



## Repository Structure