"""
Kleiner 2017 benchmark analysis.

Loads reference compositions and method result matrices, then produces:
  - Grid plot of community compositions across methods and conditions
  - Per-method logFC error summary table
  - Per-method / per-contrast error-vs-P_obs scatter plots
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    KLEINER_COMPOSITION_FILES,
    KLEINER_CONDITIONS,
    KLEINER_CONTRASTS,
    KLEINER_LABELS,
    KLEINER_METHOD_FILES,
    KLEINER_METHODS,
    KLEINER_ORDER,
    KLEINER_ORDER_COMPLETE,
    RECIPE_DIR,
    RESULTS_DIR,
    TAXON_TABLE_DIR,
)
from io_utils import (
    load_method_condition_tables,
    make_ref_from_df,
    normalize_reference_to_order,
    read_composition,
)
from metrics import compute_kleiner_logfc_result
from plots import plot_bar_chart, plot_err_vs_P, prep_med_sd


def load_reference_compositions() -> tuple[dict, dict]:
    """Return (ref_series, ref_props) dicts keyed by condition."""
    compositions = {
        cond: read_composition(RECIPE_DIR / filename)
        for cond, filename in KLEINER_COMPOSITION_FILES.items()
    }
    # The C file has trailing empty rows; keep only the 28 organisms
    compositions["C"] = compositions["C"].iloc[:28, :]
    for cond, labels in KLEINER_LABELS.items():
        compositions[cond].index = labels

    ref_series = {
        cond: make_ref_from_df(compositions[cond], KLEINER_ORDER_COMPLETE[cond], value_col="Protein_%")
        for cond in KLEINER_CONDITIONS
    }
    ref_props = {
        cond: normalize_reference_to_order(ref_series[cond], KLEINER_ORDER)
        for cond in KLEINER_CONDITIONS
    }
    return ref_series, ref_props


def run_composition_plot(kleiner_results: dict, ref_props: dict) -> None:
    ref_by_cond = {
        "P": 1 / len(KLEINER_ORDER),
        "U": ref_props["U"],
        "C": ref_props["C"],
    }
    cond_overrides = {"P": {"ytick_step": 0.05, "ymax_tick": 0.35}}
    base_kwargs = dict(
        fs_title=12,
        fs_metrics=8,
        fs_ticks=5.5,
        bar_color="#5799C7",
        metrics_loc="upper left",
        metrics_box_kw=dict(facecolor="white", edgecolor="none", alpha=0),
        ad_decimals=3,
        figsize=(3.0, 2.5),
        dpi=400,
        show=False,
    )

    for method in KLEINER_METHODS:
        for cond in KLEINER_CONDITIONS:
            df = kleiner_results[method].get(cond)
            if df is None or df.empty:
                continue
            med_dict, sd_dict = prep_med_sd(
                df, KLEINER_ORDER,
                lt2_div=3.0 if cond == "P" else None,
                normalize_cols=True,
            )
            panel_kwargs = {**base_kwargs, **cond_overrides.get(cond, {})}
            fig, _, _ = plot_bar_chart(
                med_dict,
                title=f"{method}: {cond}",
                ref=ref_by_cond.get(cond),
                sd_dict=sd_dict,
                **panel_kwargs,
            )
            fig.savefig(RESULTS_DIR / f"kleiner_composition_{method}_{cond}.pdf", bbox_inches="tight")
            plt.close(fig)


def run_logfc_analysis(kleiner_results: dict, ref_props: dict) -> pd.DataFrame:
    fc_results = {}
    summary_rows = []

    for method in KLEINER_METHODS:
        fc_results[method] = {}
        for contrast, cfg in KLEINER_CONTRASTS.items():
            result = compute_kleiner_logfc_result(
                kleiner_results[method],
                ref_props,
                numerator=cfg["numerator"],
                denominator=cfg["denominator"],
                order=KLEINER_ORDER,
            )
            fc_results[method][contrast] = result

            err = result["err"].to_numpy()
            summary_rows.append({
                "method": method,
                "contrast": contrast,
                "n": int(err.shape[0]),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.median(err)),
            })

    summary = pd.DataFrame(summary_rows).set_index(["method", "contrast"])
    print("\nKleiner 2017 logFC error summary:")
    print(summary.to_string())
    return fc_results, summary


def run_error_plots(fc_results: dict) -> None:
    for method in KLEINER_METHODS:
        for contrast, cfg in KLEINER_CONTRASTS.items():
            result = fc_results[method][contrast]
            fig, _ = plot_err_vs_P(
                result["numer_obs"],
                result["err"],
                title=f"{method}: {cfg['title']}",
                top_n_labels=0,
                xmin=-4.5,
                xmax=3.0,
                ymin=0,
                ymax=0.2,
                figsize=(2.5, 2.0),
                dpi=400,
                label_taxa=("BXL", "LT2"),
                top_k_by_err=20,
                y_label=cfg["obs_label"],
                fs_title=12,
                fs_axis_labels=9,
                fs_tick_labels=6,
                fs_metrics=8,
                fs_point_labels=6,
                x_label=None,
                show=False,
            )
            fig.savefig(RESULTS_DIR / f"kleiner_error_{method}_{contrast}.pdf", bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    _, ref_props = load_reference_compositions()

    kleiner_results = load_method_condition_tables(
        TAXON_TABLE_DIR,
        KLEINER_METHOD_FILES,
        normalize_columns=False,
    )

    run_composition_plot(kleiner_results, ref_props)
    fc_results, _ = run_logfc_analysis(kleiner_results, ref_props)
    run_error_plots(fc_results)


if __name__ == "__main__":
    main()
