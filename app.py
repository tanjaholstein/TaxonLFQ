import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from Aho_Corasick import fasta_by_prefix, map_peptides_ac
from TaxonLFQ import read_alphapept_peptides, read_evidence_txt, run_taxonlfq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="TaxonLFQ", layout="wide", page_icon="🔬")
st.title("TaxonLFQ")
st.caption("EM-based taxon quantification with shared peptide evidence")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Input files")

    input_format = st.radio(
        "Intensity input format",
        ["MaxQuant evidence.txt", "AlphaPept precursors.csv", "Pre-pivoted TSV/CSV"],
        horizontal=False,
    )
    use_evidence = input_format == "MaxQuant evidence.txt"
    use_alphapept = input_format == "AlphaPept precursors.csv"

    if use_evidence:
        int_file = st.file_uploader(
            "MaxQuant evidence.txt",
            type=["txt"],
            help=(
                "Standard MaxQuant output file. "
                "Uses the **Intensity** column (summed XIC per peptide per run). "
                "Contaminants and reverse hits are filtered automatically."
            ),
        )
        filter_contaminants = st.checkbox("Filter contaminants", value=True)
        filter_reverse = st.checkbox("Filter reverse hits", value=True)
        min_pep_raw = st.text_input("Max PEP (leave blank for no filter)", value="")
        min_pep = float(min_pep_raw) if min_pep_raw.strip() else None
        sep = "\t"
    elif use_alphapept:
        int_file = st.file_uploader(
            "AlphaPept *_peptides.csv",
            type=["csv", "tsv", "txt"],
            help=(
                "AlphaPept peptide-level export (``<results>_peptides.csv``). "
                "Expected columns: **`sequence_naked`**, **`sample_group`**, "
                "**`ms1_int_sum_area`**. Decoy hits are filtered automatically. "
                "Intensities are summed across charge states and modification forms "
                "to the unmodified stripped-sequence level."
            ),
        )
        ap_filter_decoys = st.checkbox("Filter decoy hits", value=True)
        ap_max_q_raw = st.text_input(
            "Max q-value (leave blank — FDR already applied in AlphaPept)", value=""
        )
        ap_max_q = float(ap_max_q_raw) if ap_max_q_raw.strip() else None
        sep = "\t"  # unused for AlphaPept path but keeps later code clean
    else:
        int_file = st.file_uploader(
            "Peptide intensities",
            type=["tsv", "csv", "txt"],
            help=(
                "Tab- or comma-separated. "
                "First column header must be **'peptide'**. "
                "Remaining columns are samples. "
                "Values must be **linear** intensities (not log-transformed). "
                "Missing values as empty or NaN."
            ),
        )
        sep_label = st.radio("Separator", ["Tab (TSV)", "Comma (CSV)"], horizontal=True)
        sep = "\t" if sep_label.startswith("Tab") else ","

    st.divider()
    st.header("1b · Peptide-to-taxon mapping")
    mapping_source = st.radio(
        "Mapping source",
        ["Upload TSV file", "Auto-construct from FASTA"],
        horizontal=False,
    )
    use_fasta_mapping = mapping_source == "Auto-construct from FASTA"

    if use_fasta_mapping:
        fasta_file = st.file_uploader(
            "Reference FASTA (protein database)",
            type=["fasta", "fa", "faa"],
            help=(
                "Protein database in FASTA format. "
                "Headers must follow **`>TaxonPrefix_ProteinID`** — "
                "everything before the first `_` is used as the taxon label."
            ),
        )
        map_file = None
    else:
        fasta_file = None
        map_file = st.file_uploader(
            "Peptide-to-taxon mapping",
            type=["tsv", "csv", "txt"],
            help=(
                "Two columns: **'peptide'** and **'taxon'**. "
                "Taxon values may be names or numeric IDs. "
                "Shared peptides appear on multiple rows (one per taxon)."
            ),
        )

    st.divider()
    st.header("2 · Parameters")
    max_iter = st.slider("EM iterations", min_value=1, max_value=10, value=3)
    max_taxa_raw = st.number_input(
        "Max taxa per shared peptide (0 = no limit)",
        min_value=0, value=0, step=1,
    )
    max_taxa = int(max_taxa_raw) if max_taxa_raw > 0 else None

    st.divider()
    st.header("3 · Run")
    mapping_ready = map_file if not use_fasta_mapping else fasta_file
    run_btn = st.button(
        "▶  Run TaxonLFQ",
        type="primary",
        use_container_width=True,
        disabled=not (int_file and mapping_ready),
    )

    if not int_file or not mapping_ready:
        st.caption("Upload both files above to enable the Run button.")


# ── Execute analysis ──────────────────────────────────────────────────────────
if run_btn and int_file and mapping_ready:
    int_file.seek(0)
    if map_file:
        map_file.seek(0)
    if fasta_file:
        fasta_file.seek(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Build intensity matrix ────────────────────────────────────────────
        if use_evidence:
            ev_bytes = int_file.read()
            peptide_matrix = read_evidence_txt(
                io.StringIO(ev_bytes.decode("utf-8")),
                filter_contaminants=filter_contaminants,
                filter_reverse=filter_reverse,
                min_pep=min_pep,
            )
            int_path = tmp / "intensities.tsv"
            peptide_matrix.to_csv(int_path, sep="\t")
        elif use_alphapept:
            ap_bytes = int_file.read()
            with st.spinner("Aggregating AlphaPept peptides to peptide × run matrix…"):
                peptide_matrix = read_alphapept_peptides(
                    io.StringIO(ap_bytes.decode("utf-8")),
                    filter_decoys=ap_filter_decoys,
                    max_q_value=ap_max_q,
                )
            st.info(
                f"AlphaPept: {len(peptide_matrix):,} peptides × "
                f"{peptide_matrix.shape[1]} runs after aggregation."
            )
            int_path = tmp / "intensities.tsv"
            peptide_matrix.to_csv(int_path, sep="\t")
        else:
            int_path = tmp / "intensities"
            int_path.write_bytes(int_file.read())
            peptide_matrix = None  # loaded lazily below if needed

        # ── Build mapping ─────────────────────────────────────────────────────
        if use_fasta_mapping:
            fasta_path = tmp / "reference.fasta"
            fasta_path.write_bytes(fasta_file.read())

            if peptide_matrix is not None:
                observed = list(peptide_matrix.index)
            else:
                observed = list(pd.read_csv(int_path, sep=sep, index_col=0).index)

            with st.spinner("Mapping peptides to taxa via FASTA…"):
                org2seqs = fasta_by_prefix(str(fasta_path))
                raw_mapping = map_peptides_ac(observed, org2seqs)

            rows = [
                (pep, taxon)
                for pep, taxa in raw_mapping.items()
                for taxon in taxa
            ]
            mapping_df = pd.DataFrame(rows, columns=["peptide", "taxon"])
            map_path = tmp / "taxa.tsv"
            map_path.write_text(mapping_df.to_csv(sep="\t", index=False))
            st.info(
                f"Mapped {len(raw_mapping):,} peptides to "
                f"{mapping_df['taxon'].nunique():,} taxa from FASTA."
            )
        else:
            map_path = tmp / "taxa"
            map_path.write_bytes(map_file.read())

        try:
            with st.spinner("Running TaxonLFQ…"):
                T_final, diag = run_taxonlfq(
                    str(int_path), str(map_path),
                    sep=sep,
                    max_iter=max_iter,
                    max_taxa_per_peptide=max_taxa,
                )
            st.session_state["T"] = T_final
            st.session_state["diag"] = diag
            st.session_state["ran"] = True
        except Exception as exc:
            st.error(f"**Error during analysis:** {exc}")
            st.session_state.pop("T", None)


# ── Waiting state ─────────────────────────────────────────────────────────────
if "T" not in st.session_state:
    if not (int_file and map_file):
        st.info("Upload both input files in the sidebar to get started.")
    else:
        st.info("Press **▶ Run TaxonLFQ** in the sidebar to run the analysis.")
    st.stop()


# ── Results ───────────────────────────────────────────────────────────────────
T: pd.DataFrame = st.session_state["T"]
diag: pd.DataFrame = st.session_state["diag"]

n_taxa, n_samples = T.shape
ok_count = int(diag["ok"].sum()) if "ok" in diag.columns else n_taxa

st.header("Results")
st.caption(
    f"{n_taxa} taxa · {n_samples} samples · "
    f"{ok_count}/{n_taxa} taxa with a successful LFQ profile"
)

tab_hm, tab_bar, tab_tbl, tab_diag = st.tabs(
    ["Heatmap", "Relative abundance", "Quantification table", "Diagnostics"]
)

# ── Tab 1: Heatmap ────────────────────────────────────────────────────────────
with tab_hm:
    T_float = T.astype(float).replace(0, np.nan)
    log_T = np.log2(T_float + 1)

    fig_hm, ax = plt.subplots(
        figsize=(max(5, n_samples * 0.9), max(4, n_taxa * 0.45))
    )
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
    st.pyplot(fig_hm)

    buf_hm = io.BytesIO()
    fig_hm.savefig(buf_hm, format="pdf", bbox_inches="tight")
    plt.close(fig_hm)
    st.download_button(
        "⬇ Download heatmap.pdf",
        buf_hm.getvalue(),
        "taxonlfq_heatmap.pdf",
        mime="application/pdf",
    )

# ── Tab 2: Relative abundance bar chart ──────────────────────────────────────
with tab_bar:
    totals = T.astype(float).sum(axis=0)
    rel = T.astype(float).div(totals, axis=1).fillna(0) * 100

    fig_bar, ax = plt.subplots(
        figsize=(max(5, n_samples * 0.9), 5)
    )
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
    st.pyplot(fig_bar)

    buf_bar = io.BytesIO()
    fig_bar.savefig(buf_bar, format="pdf", bbox_inches="tight")
    plt.close(fig_bar)
    st.download_button(
        "⬇ Download relative_abundance.pdf",
        buf_bar.getvalue(),
        "taxonlfq_relative_abundance.pdf",
        mime="application/pdf",
    )

# ── Tab 3: Quantification table ───────────────────────────────────────────────
with tab_tbl:
    st.dataframe(T.style.format("{:.3e}"), use_container_width=True)
    buf_tbl = io.StringIO()
    T.to_csv(buf_tbl, sep="\t")
    st.download_button(
        "⬇ Download taxonlfq_results.tsv",
        buf_tbl.getvalue(),
        "taxonlfq_results.tsv",
        mime="text/tab-separated-values",
        use_container_width=True,
    )

# ── Tab 4: Diagnostics ────────────────────────────────────────────────────────
with tab_diag:
    st.dataframe(diag, use_container_width=True)
    buf_diag = io.StringIO()
    diag.to_csv(buf_diag, sep="\t")
    st.download_button(
        "⬇ Download taxonlfq_diagnostics.tsv",
        buf_diag.getvalue(),
        "taxonlfq_diagnostics.tsv",
        mime="text/tab-separated-values",
        use_container_width=True,
    )


# ── Save to folder ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Save all results to folder")

col_path, col_btn = st.columns([4, 1])
with col_path:
    out_dir = st.text_input(
        "Output folder (will be created if it does not exist)",
        value=str(Path.home() / "taxonlfq_results"),
        label_visibility="collapsed",
        placeholder="Output folder path…",
    )
with col_btn:
    st.write("")
    save_btn = st.button("💾  Save", type="secondary", use_container_width=True)

if save_btn:
    try:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        T.to_csv(out_path / "taxonlfq_results.tsv", sep="\t")
        diag.to_csv(out_path / "taxonlfq_diagnostics.tsv", sep="\t")

        # Save plots as PDF
        T_float = T.astype(float).replace(0, np.nan)
        log_T = np.log2(T_float + 1)
        fig, ax = plt.subplots(figsize=(max(5, n_samples * 0.9), max(4, n_taxa * 0.45)))
        im = ax.imshow(log_T.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(n_samples)); ax.set_xticklabels(T.columns, rotation=45, ha="right")
        ax.set_yticks(range(n_taxa));   ax.set_yticklabels(T.index)
        plt.colorbar(im, ax=ax, label="log₂(intensity + 1)")
        ax.set_xlabel("Sample"); ax.set_ylabel("Taxon"); ax.set_title("TaxonLFQ quantification")
        plt.tight_layout()
        fig.savefig(out_path / "taxonlfq_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)

        rel = T.astype(float).div(T.astype(float).sum(axis=0), axis=1).fillna(0) * 100
        fig, ax = plt.subplots(figsize=(max(5, n_samples * 0.9), 5))
        bottom = np.zeros(n_samples)
        for i, taxon in enumerate(T.index):
            ax.bar(T.columns, rel.loc[taxon].values, bottom=bottom, label=taxon, color=plt.cm.tab20(i % 20))
            bottom += rel.loc[taxon].fillna(0).values
        ax.set_ylabel("Relative abundance (%)"); ax.set_xlabel("Sample")
        ax.set_title("Relative taxon abundance"); ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        fig.savefig(out_path / "taxonlfq_relative_abundance.pdf", bbox_inches="tight")
        plt.close(fig)

        saved_files = [
            "taxonlfq_results.tsv",
            "taxonlfq_diagnostics.tsv",
            "taxonlfq_heatmap.pdf",
            "taxonlfq_relative_abundance.pdf",
        ]
        st.success(
            f"Saved to `{out_path.resolve()}`\n\n"
            + "\n".join(f"- {f}" for f in saved_files)
        )
    except Exception as exc:
        st.error(f"Could not save: {exc}")
