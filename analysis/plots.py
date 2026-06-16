import math

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoLocator, FormatStrFormatter, FuncFormatter, MultipleLocator


def _strip_leading_zero(val, pos):
    s = f"{val:.2f}"
    if s in ("0.00", "-0.00"):
        return "0"
    if s.startswith("-0."):
        return "-" + s[2:]
    if s.startswith("0."):
        return s[1:]
    return s


def plot_bar_chart(
    data_dict,
    title,
    ref=None,
    dash_props=None,
    sd_dict=None,
    ref_marker_kwargs=None,
    ad_eps=1e-12,
    ad_normalize=True,
    ytick_step=0.05,
    ymax_tick=0.50,
    figsize=(5, 3),
    dpi=300,
    fs_title=None,
    fs_metrics=8,
    fs_ticks=None,
    metrics_loc="upper right",
    metrics_xy=None,
    metrics_coords="axes fraction",
    metrics_ha=None,
    metrics_va=None,
    metrics_box_kw=None,
    bar_color="#5799C7",
    ad_decimals=3,
    ax=None,
    show=True,
    tight=True,
):
    if dash_props is None:
        dash_props = dict(color="0.7", linewidth=1, linestyle="--")

    organisms = list(data_dict.keys())
    values = np.array([data_dict[o] for o in organisms], dtype=float)
    yerr = None
    if sd_dict is not None:
        yerr = np.array([sd_dict.get(org, 0.0) for org in organisms], dtype=float)

    def _align_ref_vec(ref_in):
        if ref_in is None:
            return None
        if np.isscalar(ref_in):
            return np.full(len(organisms), float(ref_in), dtype=float)
        if isinstance(ref_in, pd.Series):
            return ref_in.reindex(organisms).astype(float).to_numpy()
        if isinstance(ref_in, dict):
            return np.array([float(ref_in[o]) for o in organisms], dtype=float)
        arr = np.asarray(ref_in, dtype=float)
        if arr.shape[0] != len(organisms):
            raise ValueError("Vector 'ref' length must match number of organisms.")
        return arr

    ref_vec = _align_ref_vec(ref)

    def _aitchison_distance(p, q):
        p = np.clip(np.asarray(p, float), ad_eps, None)
        q = np.clip(np.asarray(q, float), ad_eps, None)
        p /= p.sum()
        q /= q.sum()
        clr = lambda x: np.log(x) - np.mean(np.log(x))
        diff = clr(p) - clr(q)
        if ad_normalize:
            diff = diff / np.sqrt(len(p))
        return np.linalg.norm(diff)

    if ref_vec is None:
        rmse = mae = ad = np.nan
    else:
        mask = np.isfinite(values) & np.isfinite(ref_vec)
        v = values[mask]
        r = ref_vec[mask]
        rmse = float(np.sqrt(np.mean((v - r) ** 2)))
        mae = float(np.mean(np.abs(v - r)))
        ad = float(_aitchison_distance(v, r))

    created_here = ax is None
    if created_here:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    x = np.arange(len(organisms))
    ax.bar(x, values, yerr=yerr, capsize=3, color=bar_color)
    try:
        ax.set_xticks(x, labels=organisms)
    except TypeError:
        ax.set_xticks(x)
        ax.set_xticklabels(organisms)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if fs_title is not None:
        ax.set_title(title, fontsize=fs_title)
    else:
        ax.set_title(title)

    if ref is None:
        pass
    elif np.isscalar(ref):
        ax.axhline(float(ref), **dash_props)
    else:
        mk = dict(marker="D", color="0.3", s=5, zorder=3, label="reference")
        if ref_marker_kwargs:
            mk.update(ref_marker_kwargs)
        ax.scatter(x, ref_vec, **mk)

    txt = f"RMSE = {rmse:.4f}"
    if np.isfinite(mae):
        txt += f"\nMAE = {mae:.4f}\nAitchison = {ad:.{ad_decimals}f}"

    LOCS = {
        "upper right":  ((0.98, 0.98), "right",  "top"),
        "upper left":   ((0.02, 0.98), "left",   "top"),
        "lower right":  ((0.98, 0.02), "right",  "bottom"),
        "lower left":   ((0.02, 0.02), "left",   "bottom"),
        "upper center": ((0.50, 0.98), "center", "top"),
        "lower center": ((0.50, 0.02), "center", "bottom"),
        "center right": ((0.98, 0.50), "right",  "center"),
        "center left":  ((0.02, 0.50), "left",   "center"),
        "center":       ((0.50, 0.50), "center", "center"),
    }

    if metrics_xy is not None:
        xy = metrics_xy
        transform = {"axes fraction": ax.transAxes, "data": ax.transData,
                     "figure fraction": fig.transFigure}[metrics_coords]
        ha = metrics_ha or "right"
        va = metrics_va or "top"
    elif isinstance(metrics_loc, str):
        if metrics_loc not in LOCS:
            raise ValueError(f"Unknown metrics_loc: {metrics_loc}")
        xy, def_ha, def_va = LOCS[metrics_loc]
        transform = ax.transAxes
        ha = metrics_ha or def_ha
        va = metrics_va or def_va
    else:
        xy = (0.98, 0.95)
        transform = ax.transAxes
        ha = metrics_ha or "right"
        va = metrics_va or "top"

    bbox_kw = dict(facecolor="white", edgecolor="0.8", alpha=0.85, boxstyle="round,pad=0.25")
    if metrics_box_kw:
        bbox_kw.update(metrics_box_kw)
    ax.text(xy[0], xy[1], txt, transform=transform, ha=ha, va=va,
            bbox=bbox_kw, fontsize=fs_metrics)

    ax.set_ylim(0.0, ymax_tick)
    ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
    ax.yaxis.set_major_formatter(FuncFormatter(_strip_leading_zero))

    if fs_ticks is not None:
        ax.tick_params(axis="both", which="both", labelsize=fs_ticks)

    if created_here:
        if tight:
            plt.tight_layout()
        if show:
            plt.show()

    return fig, ax, {"rmse": rmse, "mae": mae, "aitchison": ad}


def prep_med_sd(df, desired_order, *, lt2_div=None, normalize_cols=True):
    x = df.copy()
    if lt2_div is not None and "LT2" in x.index:
        x.loc["LT2"] = x.loc["LT2"] / float(lt2_div)
    if normalize_cols:
        x = x.div(x.sum(axis=0), axis=1)
    med = x.median(axis=1).reindex(desired_order).dropna()
    sd = x.std(axis=1, ddof=1).reindex(med.index).fillna(0.0)
    return med.to_dict(), sd.to_dict()


def plot_methods_grid(
    data_by_method: dict[str, dict[str, pd.DataFrame]],
    desired_order: list[str],
    *,
    conditions=("P", "U", "C"),
    ref_by_cond: dict | None = None,
    lt2_div_by_cond: dict | None = None,
    cond_overrides: dict | None = None,
    panel_size=(3.0, 2.5),
    dpi=400,
    row_gap=0.08,
    col_gap=0.18,
    fs_title=12,
    fs_metrics=8,
    fs_ticks=6,
    bar_color="#5799C7",
    metrics_loc="upper left",
    metrics_box_kw=None,
    ad_decimals=3,
):
    if metrics_box_kw is None:
        metrics_box_kw = dict(facecolor="white", edgecolor="none", alpha=0)
    ref_by_cond = {} if ref_by_cond is None else ref_by_cond
    lt2_div_by_cond = {} if lt2_div_by_cond is None else lt2_div_by_cond
    cond_overrides = {} if cond_overrides is None else cond_overrides

    methods = list(data_by_method.keys())
    n_rows, n_cols = len(methods), len(conditions)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        dpi=dpi, squeeze=False,
        sharex=False, sharey=False,
        gridspec_kw={"hspace": row_gap, "wspace": col_gap},
    )

    for i, method in enumerate(methods):
        for j, cond in enumerate(conditions):
            ax = axes[i, j]
            df = data_by_method[method].get(cond, None)
            if df is None or df.empty:
                ax.axis("off")
                continue

            med_dict, sd_dict = prep_med_sd(
                df, desired_order,
                lt2_div=lt2_div_by_cond.get(cond),
                normalize_cols=True,
            )

            panel_kwargs = dict(
                fs_title=fs_title, fs_metrics=fs_metrics, fs_ticks=fs_ticks,
                bar_color=bar_color, metrics_loc=metrics_loc,
                metrics_box_kw=metrics_box_kw, ad_decimals=ad_decimals,
            )
            panel_kwargs.update(cond_overrides.get(cond, {}))

            plot_bar_chart(
                med_dict, title=f"{method}: {cond}",
                ref=ref_by_cond.get(cond), sd_dict=sd_dict,
                ax=ax, show=False, tight=False,
                **panel_kwargs,
            )
            ax.tick_params(axis="y", labelleft=True)

    return fig, axes


def plot_err_vs_P(
    P_obs: pd.Series,
    err: pd.Series,
    *,
    title,
    top_n_labels: int = 25,
    label_taxa=(),
    annotate_kwargs: dict = None,
    figsize=(4, 4),
    dpi=300,
    xmin=-5.5,
    xmax=2.5,
    ymin=0,
    ymax=0.13,
    top_k_by_err: int = 20,
    y_violin: float = 0.1,
    violin_width=None,
    box_width=None,
    violin_frac: float = 0.18,
    box_frac: float = 0.12,
    violin_alpha: float = 0.40,
    box_alpha: float = 0.60,
    point_size: float = 18,
    point_alpha: float = 0.9,
    y_label: str = "P_obs",
    x_label=None,
    fs_title=None,
    fs_metrics=8,
    fs_point_labels=7,
    fs_tick_labels=None,
    fs_axis_labels=None,
    show: bool = True,
):
    df = pd.concat({"P_obs": P_obs, "err": err}, axis=1).dropna()
    x = df["err"].to_numpy()
    y = df["P_obs"].to_numpy()
    abs_err = df["err"].abs()

    rmse = float(np.sqrt(np.mean(x**2)))
    bias = float(np.median(x))
    mad = float(np.median(np.abs(x - bias)))
    sigma_mad = 1.4826 * mad
    q25, q75 = np.percentile(x, [25, 75])
    sigma_iqr = float((q75 - q25) / 1.349)

    k = min(top_k_by_err, len(df))
    if k > 0:
        x_k = df.loc[abs_err.nsmallest(k).index, "err"].to_numpy()
        rmse_topk = float(np.sqrt(np.mean(x_k**2))) if x_k.size else np.nan
    else:
        rmse_topk = np.nan

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, s=point_size, alpha=point_alpha, zorder=3)
    ax.axvline(0, color="0.8", ls="--", lw=1, zorder=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    yspan = ymax - ymin
    vw = violin_width if violin_width is not None else violin_frac * yspan
    bw = box_width if box_width is not None else box_frac * yspan

    parts = ax.violinplot([x], positions=[y_violin], vert=False, widths=vw,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(violin_alpha)
        body.set_zorder(1)
        body.set_linewidth(1.0)

    bp = ax.boxplot(x, vert=False, positions=[y_violin], widths=bw, patch_artist=True,
                    showfliers=False,
                    boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2), medianprops=dict(linewidth=1.6))
    for patch in bp["boxes"]:
        patch.set_alpha(box_alpha)
        patch.set_zorder(2)
    for key in ("medians", "whiskers", "caps"):
        for line in bp[key]:
            line.set_zorder(2)

    if fs_axis_labels is not None:
        if x_label is not None:
            ax.set_xlabel(x_label, fontsize=fs_axis_labels)
        ax.set_ylabel(y_label, fontsize=fs_axis_labels)
    else:
        if x_label is not None:
            ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    if fs_title is not None:
        ax.set_title(title, fontsize=fs_title)
    else:
        ax.set_title(title)

    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(_strip_leading_zero))
    ax.tick_params(axis="y", which="both", labelleft=True)

    if fs_tick_labels is not None:
        ax.tick_params(axis="both", which="both", labelsize=fs_tick_labels)

    txt = (
        rf"$\mathrm{{RMSE}}_{{\mathrm{{top}}20}}$" f" = {rmse_topk:.3f}\n"
        f"bias = {bias:.3f}\n"
        rf"$\hat\sigma_{{\rm IQR}}$" f" = {sigma_iqr:.3f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.0), fontsize=fs_metrics)

    ak = dict(xytext=(3, 3), textcoords="offset points", fontsize=fs_point_labels)
    if annotate_kwargs:
        ak.update(annotate_kwargs)

    if top_n_labels and top_n_labels > 0:
        for label, row in df.reindex(abs_err.sort_values(ascending=False).head(top_n_labels).index).iterrows():
            ax.annotate(str(label), (row["err"], row["P_obs"]), **ak)

    if label_taxa:
        label_taxa_set = set(map(str, label_taxa))
        for label in df.index[df.index.map(str).isin(label_taxa_set)]:
            row = df.loc[label]
            ax.annotate(str(label), (row["err"], row["P_obs"]), **ak)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, {
        "rmse": rmse,
        "rmse_topk": rmse_topk,
        "k_top": k,
        "bias": bias,
        "sigma_mad": sigma_mad,
        "sigma_iqr": sigma_iqr,
        "n": len(df),
    }


def plot_delta_logfc_sources_markers(
    datasets: dict[str, pd.DataFrame],
    expected: dict,
    title: str,
    *,
    desired_order: list,
    log_base: int = 2,
    y_min: float = -3.0,
    y_max: float = 3.0,
    annotate: bool = True,
    label_mode: str = "delta",
    top_label_y: float = 0.98,
    colors: dict | None = None,
    marker_size: float = 70.0,
    marker_edgecolor: str = "k",
    marker_edgewidth: float = 0.8,
    offset_span: float = 0.50,
    marker_zorder: int = 3,
    figsize=(10, 4),
    title_fs: int = 12,
    axis_label_fs: int = 10,
    tick_fs: int = 9,
    anno_fs: int = 7,
    xtick_rotation: float = 0.0,
    rmse_box_decimals: int = 2,
    rmse_box_title=None,
    rmse_box_fs: int = 9,
    rmse_box_title_fs=None,
    rmse_box_loc: str = "lower left",
    rmse_box_anchor=(0.01, 0.01),
    rmse_box_facecolor: str = "white",
    rmse_box_edgecolor: str = "0.6",
    rmse_box_alpha: float = 0.95,
    rmse_box_cols=None,
    legend_include_bias_sigma: bool = False,
    legend_as_table: bool = False,
    legend_hide_frame: bool = False,
    legend_sigma_math: bool = True,
    show_n_in_legend: bool = False,
    print_stats: bool = True,
    show: bool = True,
):
    palette = colors or {
        "Proteins": "#1f77b4",
        "Uniques": "#ff7f0e",
        "Uniform": "#2ca02c",
        "Clades": "#d62728",
        "NMF": "#9467bd",
        "St.LFQ": "#09D0EF",
        "StudentLFQ": "#09D0EF",
    }

    present = set().union(*(set(df["organism"].unique()) for df in datasets.values()))
    xorder = [t for t in desired_order if t in present]
    xs = np.arange(len(xorder), dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.6)
    ax.set_ylabel(f"Δlog{log_base} fold change", fontsize=axis_label_fs)
    ax.set_title(title, fontsize=title_fs)
    ax.set_xticks(xs)
    ax.set_xticklabels(xorder, fontsize=tick_fs, rotation=xtick_rotation)
    ax.tick_params(axis="y", labelsize=tick_fs)

    for i in range(len(xorder)):
        ax.hlines(0.0, i - 0.35, i + 0.35, linestyles="dashed", linewidth=1.2, color="0.25")

    keys = list(datasets.keys())
    shifts = np.linspace(-offset_span / 2.0, +offset_span / 2.0, len(keys)) if len(keys) > 1 else np.array([0.0])
    top_tf = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ln_base = np.log(log_base)
    def log_b(x):
        return np.log(x) / ln_base

    stats_rows = []
    legend_stats = {}

    for j, (key, df) in enumerate(datasets.items()):
        col = palette.get(key, f"C{j}")
        shift = shifts[j]

        med = (df.groupby("organism")["fold_change"]
               .median()
               .replace([np.inf, -np.inf], np.nan)
               .dropna())

        xs_draw, ys_delta, deltas = [], [], []

        for i, t in enumerate(xorder):
            if t not in med.index:
                continue
            exp = expected.get(t)
            if exp is None:
                continue
            obs_fc = float(med.loc[t])
            exp_fc = float(exp)
            if not (np.isfinite(obs_fc) and np.isfinite(exp_fc) and obs_fc > 0 and exp_fc > 0):
                continue

            dlt = log_b(obs_fc) - log_b(exp_fc)
            xs_draw.append(xs[i] + shift)
            ys_delta.append(dlt)
            deltas.append(dlt)

            if annotate:
                if label_mode == "delta":
                    text = f"{dlt:+.2f}"
                elif label_mode == "ratio":
                    text = f"×{(log_base ** dlt):.2f}"
                elif label_mode == "percent":
                    text = f"{((log_base ** dlt) - 1.0) * 100:+.0f}%"
                else:
                    text = f"{dlt:+.2f}"
                ax.text(xs[i] + shift, top_label_y, text,
                        transform=top_tf, rotation=90,
                        ha="center", va="top", fontsize=anno_fs, color=col)

        if xs_draw:
            ax.scatter(np.array(xs_draw), np.array(ys_delta),
                       c=col, edgecolors=marker_edgecolor,
                       s=marker_size, zorder=marker_zorder,
                       linewidths=marker_edgewidth, marker="D")

            d = np.array(deltas, dtype=float)
            rmse = float(np.sqrt(np.mean(d**2)))
            bias = float(np.median(d))
            if d.size >= 2:
                q1, q3 = np.percentile(d, [25, 75])
                sigma_iqr = float((q3 - q1) / 1.349)
            else:
                sigma_iqr = np.nan

            legend_stats[key] = (rmse, bias, sigma_iqr, len(d))
            stats_rows.append({"source": key, "n": len(d),
                                "rmse": rmse, "bias": bias, "sigma_IQR": sigma_iqr})

    stats_df = pd.DataFrame(stats_rows).set_index("source") if stats_rows else pd.DataFrame()

    if legend_stats:
        handles, labels = [], []
        name_w = max(len(k) for k in legend_stats.keys()) if legend_as_table else 0
        num_w = rmse_box_decimals + 4

        def fmt_num(x, signed=False):
            if x is None or (isinstance(x, float) and (math.isnan(x) or not np.isfinite(x))):
                return " " * num_w
            return (f"{x:+.{rmse_box_decimals}f}" if signed else f"{x:.{rmse_box_decimals}f}").rjust(num_w)

        sigma_label = r"$\sigma_{\mathrm{IQR}}$" if legend_sigma_math else "s_IQR"

        for j, key in enumerate(keys):
            if key not in legend_stats:
                continue
            col = palette.get(key, f"C{j}")
            handles.append(Line2D([], [], marker="D", linestyle="None",
                                  markersize=8, markeredgecolor=marker_edgecolor,
                                  markeredgewidth=marker_edgewidth, color=col))
            rmse, bias, sigma_iqr, n_val = legend_stats[key]
            left = f"{key:<{name_w}}  " if legend_as_table else f"{key}  "
            if legend_include_bias_sigma:
                row = (f"{left}RMSE {fmt_num(rmse)}  bias {fmt_num(bias, signed=True)}  "
                       f"{sigma_label} {fmt_num(sigma_iqr)}")
            else:
                row = f"{left}RMSE {fmt_num(rmse)}"
            if show_n_in_legend:
                row += f"  n={n_val:d}"
            labels.append(row)

        font_props = FontProperties(family="monospace" if legend_as_table else "sans-serif",
                                    size=rmse_box_fs)

        kwargs = dict(loc=rmse_box_loc, frameon=True, fancybox=True,
                      framealpha=rmse_box_alpha, title=rmse_box_title)
        if rmse_box_anchor is not None:
            kwargs["bbox_to_anchor"] = rmse_box_anchor
            kwargs["bbox_transform"] = ax.transAxes
        if rmse_box_cols is not None:
            kwargs["ncol"] = rmse_box_cols

        leg = ax.legend(handles=handles, labels=labels, prop=font_props, **kwargs)

        if rmse_box_title_fs is not None and rmse_box_title:
            try:
                leg.set_title(rmse_box_title, prop={"size": rmse_box_title_fs})
            except Exception:
                pass

        frame = leg.get_frame()
        if legend_hide_frame:
            frame.set_visible(False)
        else:
            frame.set_facecolor(rmse_box_facecolor)
            frame.set_edgecolor(rmse_box_edgecolor)

    if print_stats and not stats_df.empty:
        with pd.option_context("display.float_format", "{:.3f}".format):
            print("\nFold-change summary per source (Δlog units):")
            print(stats_df.loc[:, ["n", "rmse", "bias", "sigma_IQR"]])

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax, stats_df
