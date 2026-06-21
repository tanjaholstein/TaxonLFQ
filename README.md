# TaxonLFQ

EM-based taxon quantification for metaproteomics that handles shared peptide evidence. TaxonLFQ allocates XIC intensities across taxa using an expectation-maximisation algorithm followed by MaxLFQ-style profile fitting.

---

## Install

```bash
git clone https://github.com/tanjaholstein/TaxonLFQ.git
cd TaxonLFQ
conda env create -f environment.yml
conda activate taxonlfq
```

---

## Input

TaxonLFQ requires two files:

**1. Peptide intensities** — one of:

| Format | Description |
|---|---|
| Pre-pivoted TSV/CSV | First column header `peptide`, one column per sample, linear intensities |
| MaxQuant `evidence.txt` | Standard MaxQuant output; contaminants and reverse hits filtered automatically |
| AlphaPept `*_peptides.csv` | Expected columns: `sequence_naked`, `sample_group`, `ms1_int_sum_area` |

**2. Peptide-to-taxon mapping** — one of:

| Source | Description |
|---|---|
| Mapping TSV/CSV | Two columns: `peptide` and `taxon`. Shared peptides appear once per taxon. |
| Reference FASTA | Headers must follow `>TaxonPrefix_ProteinID` — the prefix before the first `_` becomes the taxon label. Mapping is constructed automatically. |

Example mapping TSV:

```
peptide        taxon
AELPDGTIVR     KF7
IGLAAQNTPVR    KF7
IGLAAQNTPVR    PD
```

---

## Output

Results are written to the output directory:

| File | Description |
|---|---|
| `taxonlfq_results.tsv` | Taxon × sample linear intensity matrix |
| `taxonlfq_diagnostics.tsv` | Per-taxon EM diagnostics (unique/shared peptide counts, convergence) |
| `taxonlfq_heatmap.pdf` | log₂-intensity heatmap |
| `taxonlfq_relative_abundance.pdf` | Stacked relative abundance bar chart |

---

## Usage

### Command-line interface

The CLI is the recommended entry point for scripting and workflow integration.
Progress messages go to **stderr**; the resolved output directory is printed to **stdout**.

**Quick start with the included example data:**

```bash
python cli.py \
  --intensities data/example/kleiner_example_intensities.tsv \
  --mapping     data/example/kleiner_example_taxa.tsv \
  --output-dir  results/
```

**MaxQuant `evidence.txt`:**

```bash
python cli.py \
  --intensities evidence.txt \
  --mapping     taxa.tsv \
  --max-pep     0.01
```

Format is auto-detected from the filename (`evidence.txt` → MaxQuant). Use `--no-filter-contaminants` / `--no-filter-reverse` to disable default filtering.

**AlphaPept `*_peptides.csv`:**

```bash
python cli.py \
  --format      alphapept \
  --intensities run_peptides.csv \
  --mapping     taxa.tsv
```

**Auto-construct mapping from a reference FASTA:**

```bash
python cli.py \
  --intensities evidence.txt \
  --fasta       reference.fasta \
  --output-dir  results/
```

**Capture the output path in a shell pipeline:**

```bash
OUT=$(python cli.py --intensities peptides.tsv --mapping taxa.tsv --quiet)
echo "Results in: $OUT"
```

**All options:**

```
input:
  --intensities PATH    Intensity file (evidence.txt, AlphaPept CSV, or pivoted TSV/CSV)
  --format              {evidence,alphapept,pivoted}  — auto-detected from filename when omitted
  --sep                 {tab,comma}  — separator for pre-pivoted files (default: tab)

mapping (supply one):
  --mapping PATH        Pre-built peptide→taxon TSV (columns: peptide, taxon)
  --fasta PATH          Reference FASTA for auto-mapping

MaxQuant evidence.txt:
  --no-filter-contaminants
  --no-filter-reverse
  --max-pep FLOAT       Drop PSMs above this PEP (e.g. 0.01)

AlphaPept:
  --no-filter-decoys
  --max-q-value FLOAT

algorithm:
  --max-iter INT        EM iterations (default: 3)
  --max-taxa INT        Max taxa per shared peptide (default: no limit)
  --min-ratios INT      Min peptide ratios for MaxLFQ fitting (default: 2)
  --tol FLOAT           EM convergence tolerance (default: 1e-4)

output:
  --output-dir PATH     Output folder (default: taxonlfq_results/)
  --no-plots            Skip PDF generation
  --quiet               Suppress progress output
```

---

### Streamlit app

An interactive UI for uploading files and exploring results visually.

```bash
conda activate taxonlfq
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The sidebar accepts the same input formats and mapping sources as the CLI. Results are shown as a heatmap, relative abundance chart, and interactive table, and can be downloaded or saved to a local folder.

---

## Example data

Small example files are included in `data/example/` (tracked in git):

| File | Description |
|---|---|
| `kleiner_example_intensities.tsv` | 14 peptides × 6 samples, pre-pivoted TSV |
| `kleiner_example_taxa.tsv` | Peptide→taxon mapping for 5 taxa |

Use these to verify the installation or explore the output format before running on your own data.

---
---

## Benchmark

The benchmark compares four shared-peptide allocation methods on two published mock-community datasets (Kleiner 2017, Zhao 2023):

| Method | Shared peptide handling |
|---|---|
| **Uniques** | Only unique peptides |
| **Uniform** | Equal split of shared intensity |
| **Clades** | Distributed by phylogenetic clade |
| **TaxonLFQ** | EM-based allocation with MaxLFQ-style taxon profiles |

### Reproduce the figures

Scripts must be run from the `analysis/` directory:

```bash
cd analysis

# Kleiner 2017 — composition plots + logFC error analysis
python kleiner_analysis.py

# Zhao 2023 — ΔlogFC comparison plots
python zhao_analysis.py
```

Output PDFs are saved to `analysis/results/`.

| Script | Output files |
|---|---|
| `kleiner_analysis.py` | `kleiner_composition_{method}_{condition}.pdf`, `kleiner_error_{method}_{contrast}.pdf` |
| `zhao_analysis.py` | `zhao_deltalogfc_{contrast}.pdf` |

### Benchmark data layout

```
data/
  taxon_tables/     # Pre-computed organism × run matrices (.parquet)
                    #   Kleiner_{method}_{condition}.parquet  (conditions: P, U, C)
                    #   Zhao_{method}_{sample}.parquet        (samples: S1, S2, S3)
  peptide_tables/   # Per-peptide XIC intensities for Kleiner 2017 (.parquet)
  recipes/          # Ground-truth community compositions (.tab)
```

### Repository structure

```
cli.py                    # Command-line interface
app.py                    # Streamlit app
src/
  TaxonLFQ.py             # Core EM algorithm: run_taxonlfq(), read_evidence_txt()
  maxlfq.py               # MaxLFQ solver
  Aho_Corasick.py         # FASTA → peptide-to-taxon mapping
  clades.py               # Phylogenetic clade-based allocation
data/
  example/                # Toy data for quick-start testing
  taxon_tables/           # Pre-computed benchmark matrices
  peptide_tables/         # Per-peptide XIC intensities
  recipes/                # Ground-truth compositions
analysis/
  config.py               # Paths and dataset parameters
  kleiner_analysis.py     # Kleiner 2017 benchmark driver
  zhao_analysis.py        # Zhao 2023 benchmark driver
  plots.py                # Plotting functions
```

### References

- Kleiner M. et al. (2017) *Nature Communications* — mock community benchmark (PRIDE PXD006118)
- Zhao F. et al. (2023) — mock community benchmark
