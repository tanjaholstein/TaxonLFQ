#!/usr/bin/env python3
"""TaxonLFQ command-line interface.

Usage examples
--------------
# Pre-pivoted TSV + mapping file
python cli.py --intensities peptides.tsv --mapping taxa.tsv --output-dir results/

# MaxQuant evidence.txt + mapping file
python cli.py --format evidence --intensities evidence.txt --mapping taxa.tsv

# Pre-pivoted TSV + auto-construct mapping from FASTA
python cli.py --intensities peptides.tsv --fasta reference.fasta --output-dir results/

# AlphaPept peptides.csv + FASTA mapping
python cli.py --format alphapept --intensities peptides_peptides.csv --fasta ref.fasta
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _detect_format(path: Path) -> str:
    name = path.name.lower()
    if name == "evidence.txt" or name.endswith("evidence.txt"):
        return "evidence"
    if "peptide" in name and name.endswith(".csv"):
        return "alphapept"
    return "pivoted"


def _build_intensity_matrix(args, int_path: Path, tmp: Path):
    from TaxonLFQ import read_alphapept_peptides, read_evidence_txt

    fmt = args.format or _detect_format(int_path)

    if fmt == "evidence":
        _log(args, f"Reading MaxQuant evidence.txt: {int_path}")
        matrix = read_evidence_txt(
            int_path,
            filter_contaminants=args.filter_contaminants,
            filter_reverse=args.filter_reverse,
            min_pep=args.max_pep,
        )
        pivoted_path = tmp / "intensities.tsv"
        matrix.to_csv(pivoted_path, sep="\t")
        _log(args, f"  {len(matrix):,} peptides × {matrix.shape[1]} runs after reading")
        return pivoted_path, matrix, "\t"

    if fmt == "alphapept":
        _log(args, f"Reading AlphaPept peptides file: {int_path}")
        matrix = read_alphapept_peptides(
            int_path,
            filter_decoys=args.filter_decoys,
            max_q_value=args.max_q_value,
        )
        pivoted_path = tmp / "intensities.tsv"
        matrix.to_csv(pivoted_path, sep="\t")
        _log(args, f"  {len(matrix):,} peptides × {matrix.shape[1]} runs after aggregation")
        return pivoted_path, matrix, "\t"

    # pre-pivoted
    sep = "\t" if args.sep == "tab" else ","
    return int_path, None, sep


def _build_mapping(args, matrix, int_path: Path, sep: str, tmp: Path) -> Path:
    if args.fasta:
        from Aho_Corasick import fasta_by_prefix, map_peptides_ac

        fasta_path = Path(args.fasta)
        _log(args, f"Constructing peptide-to-taxon mapping from FASTA: {fasta_path}")

        if matrix is not None:
            observed = list(matrix.index)
        else:
            observed = list(pd.read_csv(int_path, sep=sep, index_col=0).index)

        org2seqs = fasta_by_prefix(str(fasta_path))
        raw_mapping = map_peptides_ac(observed, org2seqs)

        rows = [
            (pep, taxon)
            for pep, taxa in raw_mapping.items()
            for taxon in taxa
        ]
        mapping_df = pd.DataFrame(rows, columns=["peptide", "taxon"])
        map_path = tmp / "taxa.tsv"
        mapping_df.to_csv(map_path, sep="\t", index=False)
        _log(
            args,
            f"  Mapped {len(raw_mapping):,} peptides to "
            f"{mapping_df['taxon'].nunique():,} taxa",
        )
        return map_path

    return Path(args.mapping)


def _save_plots(T: pd.DataFrame, out_dir: Path, args):
    import matplotlib.pyplot as plt
    import numpy as np

    n_taxa, n_samples = T.shape

    # Heatmap
    T_float = T.astype(float).replace(0, float("nan"))
    log_T = np.log2(T_float + 1)

    fig, ax = plt.subplots(figsize=(max(5, n_samples * 0.9), max(4, n_taxa * 0.45)))
    im = ax.imshow(log_T.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(T.columns, rotation=45, ha="right")
    ax.set_yticks(range(n_taxa))
    ax.set_yticklabels(T.index)
    plt.colorbar(im, ax=ax, label="log₂(intensity + 1)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Taxon")
    ax.set_title("TaxonLFQ quantification")
    plt.tight_layout()
    hm_path = out_dir / "taxonlfq_heatmap.pdf"
    fig.savefig(hm_path, bbox_inches="tight")
    plt.close(fig)

    # Relative abundance bar chart
    totals = T.astype(float).sum(axis=0)
    rel = T.astype(float).div(totals, axis=1).fillna(0) * 100

    fig, ax = plt.subplots(figsize=(max(5, n_samples * 0.9), 5))
    bottom = np.zeros(n_samples)
    cmap = plt.cm.tab20
    for i, taxon in enumerate(T.index):
        vals = rel.loc[taxon].values
        ax.bar(T.columns, vals, bottom=bottom, label=taxon, color=cmap(i % 20))
        bottom += vals
    ax.set_ylabel("Relative abundance (%)")
    ax.set_xlabel("Sample")
    ax.set_title("Relative taxon abundance")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    bar_path = out_dir / "taxonlfq_relative_abundance.pdf"
    fig.savefig(bar_path, bbox_inches="tight")
    plt.close(fig)

    return hm_path, bar_path


def _log(args, msg: str):
    if not args.quiet:
        print(msg, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        prog="taxonlfq",
        description="EM-based taxon quantification with shared peptide evidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    inp = parser.add_argument_group("input")
    inp.add_argument(
        "--intensities", required=True, metavar="PATH",
        help=(
            "Peptide intensity file. Accepted formats: MaxQuant evidence.txt, "
            "AlphaPept *_peptides.csv, or a pre-pivoted TSV/CSV with a 'peptide' "
            "column and one column per sample."
        ),
    )
    inp.add_argument(
        "--format", choices=["evidence", "alphapept", "pivoted"], default=None,
        metavar="{evidence,alphapept,pivoted}",
        help=(
            "Input format. Auto-detected from filename when omitted: "
            "'evidence.txt' → evidence, '*peptide*.csv' → alphapept, else pivoted."
        ),
    )
    inp.add_argument(
        "--sep", choices=["tab", "comma"], default="tab",
        help="Column separator for pre-pivoted files (ignored for evidence/alphapept).",
    )

    # ── Mapping ────────────────────────────────────────────────────────────────
    mapp = parser.add_argument_group("peptide-to-taxon mapping (supply one)")
    mapp_ex = mapp.add_mutually_exclusive_group(required=True)
    mapp_ex.add_argument(
        "--mapping", metavar="PATH",
        help=(
            "Pre-built mapping TSV/CSV with columns 'peptide' and 'taxon'. "
            "Shared peptides appear once per taxon."
        ),
    )
    mapp_ex.add_argument(
        "--fasta", metavar="PATH",
        help=(
            "Reference FASTA to auto-construct the mapping. Headers must follow "
            "''>TaxonPrefix_ProteinID'' — the prefix before the first '_' becomes "
            "the taxon label."
        ),
    )

    # ── MaxQuant evidence options ──────────────────────────────────────────────
    ev = parser.add_argument_group("MaxQuant evidence.txt options")
    ev.add_argument(
        "--no-filter-contaminants", dest="filter_contaminants",
        action="store_false", default=True,
        help="Keep rows flagged as potential contaminants.",
    )
    ev.add_argument(
        "--no-filter-reverse", dest="filter_reverse",
        action="store_false", default=True,
        help="Keep rows flagged as reverse hits.",
    )
    ev.add_argument(
        "--max-pep", type=float, default=None, metavar="FLOAT",
        help="Drop PSMs with PEP above this threshold (e.g. 0.01).",
    )

    # ── AlphaPept options ──────────────────────────────────────────────────────
    ap = parser.add_argument_group("AlphaPept options")
    ap.add_argument(
        "--no-filter-decoys", dest="filter_decoys",
        action="store_false", default=True,
        help="Keep decoy hits.",
    )
    ap.add_argument(
        "--max-q-value", type=float, default=None, metavar="FLOAT",
        help="Drop peptides with q-value above this threshold.",
    )

    # ── Algorithm parameters ───────────────────────────────────────────────────
    alg = parser.add_argument_group("algorithm parameters")
    alg.add_argument(
        "--max-iter", type=int, default=3, metavar="INT",
        help="Maximum EM iterations.",
    )
    alg.add_argument(
        "--max-taxa", type=int, default=None, metavar="INT",
        help="Maximum taxa per shared peptide (no limit if omitted).",
    )
    alg.add_argument(
        "--min-ratios", type=int, default=2, metavar="INT",
        help="Minimum peptide ratios required for MaxLFQ profile fitting.",
    )
    alg.add_argument(
        "--tol", type=float, default=1e-4, metavar="FLOAT",
        help="EM convergence tolerance (max relative change in taxon profiles).",
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    out = parser.add_argument_group("output")
    out.add_argument(
        "--output-dir", default="taxonlfq_results", metavar="PATH",
        help="Directory to write results into (created if absent).",
    )
    out.add_argument(
        "--no-plots", action="store_true",
        help="Skip PDF plot generation (heatmap + relative abundance).",
    )
    out.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages (errors still go to stderr).",
    )

    args = parser.parse_args()

    int_path = Path(args.intensities)
    if not int_path.exists():
        print(f"error: intensities file not found: {int_path}", file=sys.stderr)
        sys.exit(1)

    if args.mapping and not Path(args.mapping).exists():
        print(f"error: mapping file not found: {args.mapping}", file=sys.stderr)
        sys.exit(1)

    if args.fasta and not Path(args.fasta).exists():
        print(f"error: FASTA file not found: {args.fasta}", file=sys.stderr)
        sys.exit(1)

    from TaxonLFQ import run_taxonlfq

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            pivoted_path, matrix, sep = _build_intensity_matrix(args, int_path, tmp)
            map_path = _build_mapping(args, matrix, pivoted_path, sep, tmp)

            _log(args, "Running TaxonLFQ…")
            T_final, diag = run_taxonlfq(
                str(pivoted_path),
                str(map_path),
                sep=sep,
                max_iter=args.max_iter,
                min_ratios=args.min_ratios,
                max_taxa_per_peptide=args.max_taxa,
                tol=args.tol,
            )

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    n_taxa, n_samples = T_final.shape
    _log(args, f"Done: {n_taxa} taxa × {n_samples} samples")

    results_path = out_dir / "taxonlfq_results.tsv"
    diag_path = out_dir / "taxonlfq_diagnostics.tsv"
    T_final.to_csv(results_path, sep="\t")
    diag.to_csv(diag_path, sep="\t")

    written = [results_path, diag_path]

    if not args.no_plots:
        try:
            hm_path, bar_path = _save_plots(T_final, out_dir, args)
            written += [hm_path, bar_path]
        except Exception as exc:
            print(f"warning: could not save plots: {exc}", file=sys.stderr)

    for p in written:
        _log(args, f"  wrote {p}")

    # Print output dir to stdout so callers can capture it
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
