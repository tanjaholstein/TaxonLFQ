import numpy as np
import pandas as pd


def compositional_logFC(
    numer: pd.DataFrame,
    denom: pd.DataFrame,
    *,
    order: list | None = None,
    eps: float = 1e-12,
    return_parts: bool = False,
) -> pd.Series | dict:
    """
    Compute log2 fold-change between two conditions in composition space.

    Steps:
      1. Close each run to a composition (columns sum to 1).
      2. Replace zeros by eps and re-close.
      3. Center across runs with component-wise median.
      4. Re-close median vectors to valid compositions.
      5. Return log2(numer_obs) - log2(denom_obs).
    """
    def _intersect_in_order(idx_a, idx_b):
        bset = set(idx_b)
        return [i for i in idx_a if i in bset]

    if order is not None:
        common = [t for t in order if (t in numer.index) and (t in denom.index)]
    else:
        common = _intersect_in_order(numer.index, denom.index)
    if len(common) == 0:
        raise ValueError("No overlapping taxa between numer and denom.")

    N = numer.loc[common].astype(float)
    D = denom.loc[common].astype(float)

    def _close_cols(df):
        col_sums = df.sum(axis=0, skipna=True)
        return df.div(col_sums.where(col_sums != 0, np.nan), axis=1)

    N_comp = _close_cols(_close_cols(N).clip(lower=eps))
    D_comp = _close_cols(_close_cols(D).clip(lower=eps))

    N_med = N_comp.median(axis=1, skipna=True)
    D_med = D_comp.median(axis=1, skipna=True)

    numer_obs = N_med / N_med.sum(skipna=True)
    denom_obs = D_med / D_med.sum(skipna=True)

    logFC_hat = np.log2(numer_obs.clip(lower=eps)) - np.log2(denom_obs.clip(lower=eps))
    logFC_hat.name = "log2FC_numer_vs_denom"

    if not return_parts:
        return logFC_hat

    return {
        "logFC_hat": logFC_hat,
        "numer_obs": numer_obs,
        "denom_obs": denom_obs,
        "numer_comp": N_comp,
        "denom_comp": D_comp,
        "numer_med": N_med,
        "denom_med": D_med,
        "taxa_order": common,
        "eps": eps,
    }


def compute_kleiner_logfc_result(
    method_tables: dict[str, pd.DataFrame],
    ref_props: dict[str, pd.Series],
    *,
    numerator: str,
    denominator: str,
    order: list[str],
) -> dict:
    """Compute observed and expected compositional log2 fold-change for one Kleiner contrast."""
    parts = compositional_logFC(
        method_tables[numerator],
        method_tables[denominator],
        order=order,
        return_parts=True,
    )
    logFC_hat = parts["logFC_hat"]
    logFC_true = np.log2(ref_props[numerator]) - np.log2(ref_props[denominator])
    err = (logFC_hat - logFC_true).dropna()

    return {
        "parts": parts,
        "logFC_hat": logFC_hat,
        "logFC_true": logFC_true,
        "err": err,
        "numer_obs": parts["numer_obs"],
        "denom_obs": parts["denom_obs"],
    }


def foldchange_pairs(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """Return all run-pair fold changes B/A as a long table with columns [organism, pair, fold_change]."""
    taxa = A.index.intersection(B.index)
    A = A.loc[taxa].replace(0, np.nan)
    B = B.loc[taxa].replace(0, np.nan)

    fc = B.to_numpy()[:, :, None] / A.to_numpy()[:, None, :]
    pairs = [f"{b}/{a}" for b in B.columns for a in A.columns]

    long = (
        pd.DataFrame(fc.reshape(len(taxa), -1), index=taxa, columns=pairs)
        .stack()
        .rename("fold_change")
        .reset_index()
        .rename(columns={"level_0": "organism", "level_1": "pair"})
    )
    return long.replace([np.inf, -np.inf], np.nan).dropna(subset=["fold_change"])


def compute_zhao_foldchanges(sample_tables: dict[str, pd.DataFrame], contrasts: dict) -> dict[str, pd.DataFrame]:
    """Compute all configured Zhao fold-change contrasts for one method."""
    return {
        contrast: foldchange_pairs(
            sample_tables[cfg["denominator"]],
            sample_tables[cfg["numerator"]],
        )
        for contrast, cfg in contrasts.items()
    }


def make_zhao_author_fc(offsets: list[float], expected_fc: dict, order: list[str]) -> pd.DataFrame:
    """Back-calculate author median fold changes from stored offsets and expected fold changes."""
    dist = pd.Series(offsets, index=order, dtype=float)
    exp_series = pd.Series(expected_fc).reindex(order).astype(float)
    median_fc = exp_series + dist
    return pd.DataFrame({
        "organism": order,
        "pair": "authors_median",
        "fold_change": median_fc.values,
    })
