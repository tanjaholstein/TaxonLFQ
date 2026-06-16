# TaxonLFQ

Benchmark comparing methods for allocating shared peptide evidence to taxa in metaproteomics. Four methods are evaluated on two published mock-community datasets (Kleiner 2017, Zhao 2023):

| Method | Shared peptide handling |
|---|---|
| **Uniques** | Only unique peptides used |
| **Uniform** | Shared intensity split equally |
| **Clades** | Distributed by phylogenetic clade |
| **StudentLFQ** | EM-based allocation with MaxLFQ-style taxon profiles |

## Setup

```bash
conda env create -f environment.yml
conda activate taxonlfq
```

## Run the analysis

Scripts must be run from the `analysis/` directory:

```bash
cd analysis

# Kleiner 2017 benchmark — composition plots + logFC error analysis
python kleiner_analysis.py

# Zhao 2023 benchmark — ΔlogFC comparison plots
python zhao_analysis.py
```

All output PDFs are saved to `analysis/results/`.

## Output files

**Kleiner 2017**
- `kleiner_composition_{method}_{condition}.pdf` — community composition bar charts (methods × conditions P/U/C)
- `kleiner_error_{method}_{contrast}.pdf` — logFC error vs. observed abundance scatter plots (methods × contrasts P_vs_U, P_vs_C, C_vs_U)

**Zhao 2023**
- `zhao_deltalogfc_{contrast}.pdf` — ΔlogFC comparison across all methods (contrasts S2_S1, S3_S1, S3_S2)

## Repository structure

```
data/
  recipes/          # Ground-truth community compositions (.tab)
  taxon_tables/     # Pre-computed organism × run matrices (.parquet)
analysis/
  config.py         # Paths and dataset parameters
  io_utils.py       # Data loading utilities
  metrics.py        # RMSE, logFC error computation
  plots.py          # All plotting functions
  kleiner_analysis.py
  zhao_analysis.py
src/                # Core StudentLFQ algorithm and shared peptide allocation
```

## Data

Pre-computed taxon tables for both datasets are included in `data/taxon_tables/`. These were generated from the original publications' peptide-level data using the methods listed above.

## References

- Kleiner M. et al. (2017) *Nature Communications* — mock community benchmark
- Zhao F. et al. (2023) — mock community benchmark
