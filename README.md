# TaxonLFQ

Benchmark comparing methods for allocating shared peptide evidence to taxa in metaproteomics. Four methods are evaluated on two published mock-community datasets (Kleiner 2017, Zhao 2023):

| Method | Shared peptide handling |
|---|---|
| **Uniques** | Only unique peptides used |
| **Uniform** | Shared intensity split equally |
| **Clades** | Distributed by phylogenetic clade |
| **TaxonLFQ** | EM-based allocation with MaxLFQ-style taxon profiles |

## Setup

```bash
git clone https://github.com/tanjaholstein/TaxonLFQ.git
cd TaxonLFQ
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
src/                # Core TaxonLFQ algorithm and shared peptide allocation
```

## Input data

All data files are included in the repository under `data/`.

### `data/taxon_tables/` — organism × run quantification matrices (`.parquet`)

One file per method × dataset × condition. Rows are organisms, columns are runs. Values are linear intensities.

| File pattern | Dataset | Conditions |
|---|---|---|
| `Kleiner_{method}_{condition}.parquet` | Kleiner 2017 | P (equal protein), U (uneven), C (equal cell number) |
| `Zhao_{method}_{sample}.parquet` | Zhao 2023 | S1, S2, S3 |

Methods in the filenames: `uniques`, `uniform`, `Clades`, `TaxonLFQ_direct` (Kleiner) / `LFQ_uniques`, `uniform`, `LFQ_clades`, `TaxonLFQ_noLFQ` (Zhao).

### `data/recipes/` — ground-truth community compositions (`.tab`)

Tab-separated files with the known protein and cell abundances for each organism in the Kleiner 2017 mock community. Used as reference for evaluating quantification accuracy.

| File | Community |
|---|---|
| `Composition_Of_EQUAL_PROTEIN_AMOUNT_Community.tab` | Condition P |
| `Composition_Of_UNEVEN_Community.tab` | Condition U |
| `Composition_Of_EQUAL_CELL_NUMBER_Community.tab` | Condition C |

### `data/peptide_tables/` — peptide-level XIC matrices (`.parquet`)

Per-peptide extracted ion chromatogram intensities for the Kleiner 2017 dataset (conditions P, U, C). These are the upstream input from which the taxon tables were derived.

## References

- Kleiner M. et al. (2017) *Nature Communications* — mock community benchmark
- Zhao F. et al. (2023) — mock community benchmark
