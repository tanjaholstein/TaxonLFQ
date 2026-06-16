"""
Zhao 2023 benchmark analysis.

Loads method result matrices, computes all-pair fold changes, then produces
ΔlogFC comparison plots for all three contrasts (S2/S1, S3/S1, S3/S2).
"""

import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR,
    TAXON_TABLE_DIR,
    ZHAO_AUTHOR_OFFSETS,
    ZHAO_CONTRASTS,
    ZHAO_EXPECTED_FC,
    ZHAO_METHOD_FILES,
    ZHAO_METHODS,
    ZHAO_ORDER,
)
from io_utils import load_method_condition_tables
from metrics import compute_zhao_foldchanges, make_zhao_author_fc
from plots import plot_delta_logfc_sources_markers


def build_plot_datasets(zhao_results: dict) -> dict:
    fc_by_method = {
        method: compute_zhao_foldchanges(sample_tables, ZHAO_CONTRASTS)
        for method, sample_tables in zhao_results.items()
    }

    author_fc = {
        contrast: make_zhao_author_fc(
            ZHAO_AUTHOR_OFFSETS[contrast],
            ZHAO_EXPECTED_FC[contrast],
            ZHAO_ORDER,
        )
        for contrast in ZHAO_CONTRASTS
    }

    return {
        contrast: {
            "Proteins": author_fc[contrast],
            **{method: fc_by_method[method][contrast] for method in ZHAO_METHODS},
        }
        for contrast in ZHAO_CONTRASTS
    }


def run_delta_logfc_plots(plot_datasets: dict) -> None:
    for contrast, cfg in ZHAO_CONTRASTS.items():
        anchor = (0.001, 0.001) if contrast == "S3/S1" else (0.0, 0.0)
        fig, _, stats = plot_delta_logfc_sources_markers(
            datasets=plot_datasets[contrast],
            expected=ZHAO_EXPECTED_FC[contrast],
            title=cfg["title"],
            desired_order=ZHAO_ORDER,
            y_min=-3.0,
            y_max=3.0,
            log_base=2,
            figsize=(11, 4.5),
            title_fs=13,
            axis_label_fs=11,
            tick_fs=9,
            anno_fs=7.5,
            rmse_box_loc="lower left",
            rmse_box_decimals=3,
            rmse_box_anchor=anchor,
            rmse_box_fs=8,
            rmse_box_title=None,
            rmse_box_edgecolor="0.5",
            rmse_box_facecolor="white",
            rmse_box_alpha=1,
            legend_include_bias_sigma=True,
            legend_as_table=True,
            legend_hide_frame=False,
            show_n_in_legend=False,
            show=False,
        )
        safe_name = contrast.replace("/", "_")
        fig.savefig(RESULTS_DIR / f"zhao_deltalogfc_{safe_name}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    zhao_results = load_method_condition_tables(
        TAXON_TABLE_DIR,
        ZHAO_METHOD_FILES,
        normalize_columns=True,
    )

    plot_datasets = build_plot_datasets(zhao_results)
    run_delta_logfc_plots(plot_datasets)


if __name__ == "__main__":
    main()
