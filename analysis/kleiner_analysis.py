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
    KLEINER_TAXON_NAMES,
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
from plots import (
    plot_bar_chart, plot_err_vs_P, prep_med_sd,
    plot_fc_obs_vs_true, plot_fc_bars_per_organism,
    plot_fc_error_grouped, plot_fc_heatmap,
)


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


def run_composition_plot(kleiner_results: dict, ref_props: dict, *, out_dir=None, show_metrics=True) -> None:
    if out_dir is None:
        out_dir = RESULTS_DIR
    out_dir.mkdir(exist_ok=True)

    ref_by_cond = {
        "P": 1 / len(KLEINER_ORDER),
        "U": ref_props["U"].rename(index=KLEINER_TAXON_NAMES),
        "C": ref_props["C"].rename(index=KLEINER_TAXON_NAMES),
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
        show_metrics=show_metrics,
    )

    for method in KLEINER_METHODS:
        for cond in KLEINER_CONDITIONS:
            df = kleiner_results[method].get(cond)
            if df is None or df.empty:
                continue
            med_dict, _ = prep_med_sd(
                df, KLEINER_ORDER,
                lt2_div=3.0 if cond == "P" else None,
                normalize_cols=True,
            )
            med_dict = {KLEINER_TAXON_NAMES.get(k, k): v for k, v in med_dict.items()}
            panel_kwargs = {**base_kwargs, **cond_overrides.get(cond, {})}
            fig, _, _ = plot_bar_chart(
                med_dict,
                title=f"{method}: {cond}",
                ref=ref_by_cond.get(cond),
                **panel_kwargs,
            )
            fig.savefig(out_dir / f"kleiner_composition_{method}_{cond}.svg", bbox_inches="tight")
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
            fig.savefig(RESULTS_DIR / f"kleiner_error_{method}_{contrast}.svg", bbox_inches="tight")
            plt.close(fig)


def run_fc_comparison_plots(fc_results: dict) -> None:
    fc_dir = RESULTS_DIR / "fold change per taxon comparison"
    fc_dir.mkdir(exist_ok=True)

    display_order = [KLEINER_TAXON_NAMES.get(t, t) for t in KLEINER_ORDER]

    for contrast, cfg in KLEINER_CONTRASTS.items():
        fc_by_method = {
            method: {
                k: (v.rename(index=KLEINER_TAXON_NAMES) if isinstance(v, pd.Series) else v)
                for k, v in fc_results[method][contrast].items()
            }
            for method in KLEINER_METHODS
        }
        title = cfg["title"]

        fig, _ = plot_fc_obs_vs_true(
            fc_by_method, f"Observed vs. true log2 FC: {title}",
            figsize=(13, 6), show=False,
        )
        fig.savefig(fc_dir / f"option1_obs_vs_true_{contrast}.svg", bbox_inches="tight")
        plt.close(fig)

        for method in KLEINER_METHODS:
            r = fc_by_method[method]
            fig, _ = plot_fc_bars_per_organism(
                r["logFC_hat"], r["logFC_true"],
                title=f"{method}: {title}",
                desired_order=display_order,
                figsize=(14, 4), show=False,
            )
            fig.savefig(fc_dir / f"option2_bars_{method}_{contrast}.svg", bbox_inches="tight")
            plt.close(fig)

        fig, _ = plot_fc_error_grouped(
            fc_by_method, f"log2 FC error: {title}",
            desired_order=display_order, figsize=(16, 4.5), show=False,
        )
        fig.savefig(fc_dir / f"option3_error_grouped_{contrast}.svg", bbox_inches="tight")
        plt.close(fig)

        fig, _ = plot_fc_heatmap(
            fc_by_method, f"log2 FC error: {title}",
            desired_order=display_order, figsize=(10, 7), show=False,
        )
        fig.savefig(fc_dir / f"option4_heatmap_{contrast}.svg", bbox_inches="tight")
        plt.close(fig)


def run_stacked_composition_plot(kleiner_results: dict, ref_props: dict, cond: str, ymax_tick: float = 0.15) -> None:
    ref = (
        1 / len(KLEINER_ORDER) if cond == "P"
        else ref_props[cond].rename(index=KLEINER_TAXON_NAMES)
    )
    lt2_div = 3.0 if cond == "P" else None

    fig, axes = plt.subplots(
        2, 1, sharex=True,
        figsize=(9, 5), dpi=400,
        gridspec_kw={"hspace": 0.06},
    )

    common_kw = dict(
        ref=ref,
        ytick_step=0.05, ymax_tick=ymax_tick,
        fs_title=12, fs_metrics=8, fs_ticks=9,
        bar_color="#5799C7",
        metrics_loc="upper left",
        metrics_box_kw=dict(facecolor="white", edgecolor="none", alpha=0),
        ad_decimals=3,
        ref_marker_kwargs=dict(s=40),
        show_metrics=False,
        show=False, tight=False,
    )

    for ax, method in zip(axes, ["Uniques", "TaxonLFQ"]):
        df = kleiner_results[method][cond]
        med_dict, _ = prep_med_sd(df, KLEINER_ORDER, lt2_div=lt2_div, normalize_cols=True)
        med_dict = {KLEINER_TAXON_NAMES.get(k, k): v for k, v in med_dict.items()}
        plot_bar_chart(med_dict, title="", ax=ax, **common_kw)
        ax.set_ylabel("")

    plt.setp(axes[0].get_xticklabels(), visible=False)
    fig.text(0.01, 0.5, "Relative abundance", va="center", rotation="vertical", fontsize=14)

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    fig.savefig(RESULTS_DIR / f"kleiner_stacked_Uniques_TaxonLFQ_{cond}.svg", bbox_inches="tight")
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
    run_stacked_composition_plot(kleiner_results, ref_props, cond="P")
    run_stacked_composition_plot(kleiner_results, ref_props, cond="U", ymax_tick=0.45)
    run_composition_plot(
        kleiner_results, ref_props,
        out_dir=RESULTS_DIR / "no_inplot_legend",
        show_metrics=False,
    )
    fc_results, _ = run_logfc_analysis(kleiner_results, ref_props)
    run_error_plots(fc_results)
    run_fc_comparison_plots(fc_results)


if __name__ == "__main__":
    main()
