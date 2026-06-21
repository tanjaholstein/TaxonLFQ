# TaxonLFQ

EM-based taxon quantification for metaproteomics that handles shared peptide evidence. TaxonLFQ allocates XIC intensities across taxa using an expectation-maximisation algorithm followed by MaxLFQ-style profile fitting.

Four methods are evaluated on two published mock-community datasets (Kleiner 2017, Zhao 2023):

| Method | Shared peptide handling |
|---|---|
| **Uniques** | Only unique peptides |
| **Uniform** | Equal split of shared intensity |
| **Clades** | Distributed by phylogenetic clade |
| **TaxonLFQ** | EM-based allocation with MaxLFQ-style taxon profiles |

---

## Install

```bash
git clone https://github.com/tanjaholstein/TaxonLFQ.git
cd TaxonLFQ
conda env create -f environment.yml
conda activate taxonlfq
```

---

## Usage

### Command-line interface

The CLI is the recommended entry point for scripting and workflow integration.
Progress messages go to **stderr**; the resolved output directory is printed to **stdout**.

**Minimal usage — pre-pivoted TSV + mapping file:**

```bash
python cli.py \
  --intensities data/example/kleiner_example_intensities.tsv \
  --mapping     data/example/kleiner_example_taxa.tsv \
  --output-dir  results/
```

Outputs written to `results/`:
- `taxonlfq_results.tsv` — taxon × sample intensity matrix
- `taxonlfq_diagnostics.tsv` — per-taxon EM diagnostics
- `taxonlfq_heatmap.pdf` — log₂-intensity heatmap
- `taxonlfq_relative_abundance.pdf` — stacked bar chart

**MaxQuant `evidence.txt` input:**

```bash
python cli.py \
  --intensities evidence.txt \
  --mapping     taxa.tsv \
  --max-pep 0.01
```

Format is auto-detected from the filename (`evidence.txt` → MaxQuant). Contaminants and reverse hits are filtered by default; disable with `--no-filter-contaminants` / `--no-filter-reverse`.

**AlphaPept `*_peptides.csv` input:**

```bash
python cli.py \
  --format alphapept \
  --intensities run_peptides.csv \
  --mapping     taxa.tsv
```

**Auto-construct mapping from a reference FASTA:**

```bash
python cli.py \
  --intensities evidence.txt \
  --fasta       Mock_Comm_RefDB_V3.fasta \
  --output-dir  results/
```

FASTA headers must follow `>TaxonPrefix_ProteinID` — the prefix before the first `_` is used as the taxon label.

**Capture the output path in a pipeline:**

```bash
OUT=$(python cli.py --intensities peptides.tsv --mapping taxa.tsv --quiet)
echo "Results in: $OUT"
```

**All options:**

```
input:
  --intensities PATH    Intensity file (evidence.txt, AlphaPept CSV, or pivoted TSV/CSV)
  --format              {evidence,alphapept,pivoted}  — auto-detected when omitted
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

Open [http://localhost:8501](http://localhost:8501) in your browser.

The sidebar accepts the same three input formats (MaxQuant, AlphaPept, pre-pivoted TSV/CSV) and either a mapping TSV or a reference FASTA. Results are shown as a heatmap, relative abundance chart, and interactive table, and can be downloaded or saved to a folder.

---

## Example data

Small example files for testing are included under `data/example/`:

| File | Description |
|---|---|
| `kleiner_example_intensities.tsv` | 14 peptides × 6 samples (pre-pivoted TSV) |
| `kleiner_example_taxa.tsv` | Peptide→taxon mapping for 5 taxa |

Run with the example data:

```bash
python cli.py \
  --intensities data/example/kleiner_example_intensities.tsv \
  --mapping     data/example/kleiner_example_taxa.tsv
```

Or upload both files in the Streamlit app using the **Pre-pivoted TSV/CSV** format.

---

## Benchmark analysis

Scripts reproduce the figures from the paper and must be run from the `analysis/` directory:

```bash
cd analysis

# Kleiner 2017 — composition plots + logFC error analysis
python kleiner_analysis.py

# Zhao 2023 — ΔlogFC comparison plots
python zhao_analysis.py
```

Output PDFs are saved to `analysis/results/`.

**Output files:**

| Script | Output |
|---|---|
| `kleiner_analysis.py` | `kleiner_composition_{method}_{condition}.pdf`, `kleiner_error_{method}_{contrast}.pdf` |
| `zhao_analysis.py` | `zhao_deltalogfc_{contrast}.pdf` |

---

## Repository structure

```
cli.py                    # Command-line interface
app.py                    # Streamlit app
src/
  TaxonLFQ.py             # Core EM algorithm: run_taxonlfq(), read_evidence_txt()
  maxlfq.py               # MaxLFQ solver
  Aho_Corasick.py         # FASTA → peptide-to-taxon mapping
  clades.py               # Phylogenetic clade-based allocation
data/
  example/                # Toy data for testing (tracked in git)
    kleiner_example_intensities.tsv
    kleiner_example_taxa.tsv
  taxon_tables/           # Pre-computed organism × run matrices (.parquet)
  peptide_tables/         # Per-peptide XIC intensities (.parquet)
  recipes/                # Ground-truth community compositions (.tab)
analysis/
  config.py               # Paths and dataset parameters
  kleiner_analysis.py     # Kleiner 2017 benchmark driver
  zhao_analysis.py        # Zhao 2023 benchmark driver
  plots.py                # Plotting functions
```

---

## Input file formats

### Peptide intensities

| Format | Description |
|---|---|
| **Pre-pivoted TSV/CSV** | First column header `peptide`, one column per sample, linear intensities |
| **MaxQuant evidence.txt** | Standard MaxQuant output; contaminants/reverse filtered automatically |
| **AlphaPept `*_peptides.csv`** | Expected columns: `sequence_naked`, `sample_group`, `ms1_int_sum_area` |

### Peptide-to-taxon mapping

A two-column TSV/CSV with headers `peptide` and `taxon`. Shared peptides (assigned to multiple taxa) appear once per taxon:

```
peptide    taxon
AELPDGTIVR    KF7
IGLAAQNTPVR   KF7
IGLAAQNTPVR   PD
```

Alternatively, provide a reference FASTA with `--fasta` and the mapping is constructed automatically. Headers must follow `>TaxonPrefix_ProteinID`.

---

## References

- Kleiner M. et al. (2017) *Nature Communications* — mock community benchmark (PRIDE PXD006118)
- Zhao F. et al. (2023) — mock community benchmark
