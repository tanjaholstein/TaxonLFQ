import datetime
import json
import re
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from maxlfq import get_protein_ratios, solve_profile


@dataclass
class BenchmarkResult:
    wall_time_s: float
    peak_memory_mb: float
    em_iterations: int
    T_final: pd.DataFrame
    diag: pd.DataFrame


def _save_run_stats(stats: dict, stats_dir: str | Path) -> Path:
    """Append a run-stats entry to run_stats/ and return the JSON path.

    Creates two files:
      <stats_dir>/run_log.tsv          — one TSV row per run (append mode)
      <stats_dir>/run_<timestamp>.json — full record for this run
    """
    out = Path(stats_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = stats["timestamp"]
    json_path = out / f"run_{ts.replace(':', '').replace('-', '').replace('T', '_')}.json"
    json_path.write_text(json.dumps(stats, indent=2))

    log_path = out / "run_log.tsv"
    tsv_cols = ["timestamp", "wall_time_s", "peak_memory_mb", "em_iterations",
                "n_taxa", "n_samples", "intensities_file", "mapping_file"]
    write_header = not log_path.exists()
    with log_path.open("a") as fh:
        if write_header:
            fh.write("\t".join(tsv_cols) + "\n")
        fh.write("\t".join(str(stats.get(c, "")) for c in tsv_cols) + "\n")

    return json_path


def lfq_taxon_unique(peptide_run_df, pep2org, exclude=None, min_ratios=2, solver='L-BFGS-B'):
    """
    Taxon-level MaxLFQ using UNIQUE peptides only.
    Inputs:
      - peptide_run_df: rows=peptides (sequence_naked), cols=runs (linear intensities, NaN for missing)
      - pep2org: dict {peptide -> organism} for unique peptides only
      - exclude: set of organism codes to drop (optional, already applied upstream)
      - min_ratios: minimum peptide overlap to compute a run-pair ratio (as in AlphaPept)
      - solver: optimizer for the profile solve (passed to solve_profile)
    Returns:
      - lfq_df: DataFrame [taxa x runs] with taxon-level LFQ intensities (linear units)
      - diag_df: diagnostics per taxon (n_peptides, n_pairs, ok flag)
    """
    if exclude is None:
        exclude = set()

    # Keep only unique peptides that map to a non-excluded taxon and exist in the matrix
    unique_items = [(p, t) for p, t in pep2org.items() if (t not in exclude) and (p in peptide_run_df.index)]
    if not unique_items:
        raise ValueError("No unique peptides found after filtering.")

    # Taxa present
    taxa = sorted(set(t for _, t in unique_items))
    runs = peptide_run_df.columns.tolist()

    # Pre-build all run pairs once
    run_idx = range(len(runs))
    run_pairs = list(combinations(run_idx, 2))

    # Build peptide lists per taxon
    taxon_to_peps = {t: [] for t in taxa}
    for p, t in unique_items:
        taxon_to_peps[t].append(p)

    # Prepare outputs
    lfq = pd.DataFrame(index=taxa, columns=runs, dtype=float)
    diag = pd.DataFrame(index=taxa, columns=['n_peptides', 'n_pairs', 'ok'], dtype=float)

    for tax in taxa:
        peps = taxon_to_peps[tax]
        if len(peps) == 0:
            lfq.loc[tax] = np.nan
            diag.loc[tax] = (0, 0, 0)
            continue

        sub = peptide_run_df.loc[peps].replace(0, np.nan)  # ensure no zeros
        # Compute run-pair ratios from this taxon's peptide set
        ratios = get_protein_ratios(sub.values, run_pairs, min_ratios)
        # Count how many ratios will be used (diagnostic)
        n_pairs = int(np.isfinite(ratios).sum())
        # Solve for the taxon profile
        I_hat, ok = solve_profile(ratios, solver)
        if not ok and np.isfinite(I_hat).any():
            warnings.warn(
                f"solve_profile did not converge for taxon {tax!r} (success=False); "
                "using best solution found.",
                stacklevel=2,
            )

        if np.isfinite(I_hat).any():
            # Optional rescaling to preserve the taxon’s total observed peptide signal
            scale_per_run = sub.sum(axis=0).values  # linear units
            total_scale = np.nansum(scale_per_run)
            if np.isfinite(I_hat).sum() > 0 and total_scale > 0:
                I_hat_rescaled = I_hat * (total_scale / np.nansum(I_hat))
            else:
                I_hat_rescaled = I_hat
            lfq.loc[tax] = I_hat_rescaled
            diag.loc[tax] = (len(peps), n_pairs, 1)
        else:
            lfq.loc[tax] = np.nan
            diag.loc[tax] = (len(peps), n_pairs, 0)

    lfq.index.name = 'organism'
    return lfq, diag

import numpy as np
import pandas as pd
from itertools import combinations

# ---- A. Split mapping into unique vs shared, apply "exclude", intersect with df_all rows
def prepare_mappings(df_all, pep2org, pep2orgs, exclude=None, max_taxa_per_peptide=None):
    """
    Returns:
      - pep2taxon_unique: dict {peptide -> taxon} (unique only, not excluded, in df_all)
      - pep2taxa_shared: dict {peptide -> tuple(taxa,...)} (shared only, not excluded, in df_all)
    """
    if exclude is None:
        exclude = set()
    peptides_in_matrix = set(df_all.index)

    # Unique peptides
    pep2taxon_unique = {
        p: t for p, t in pep2org.items()
        if (p in peptides_in_matrix) and (t not in exclude)
    }

    # Shared peptides (remove excluded taxa; drop if <=1 taxa remain)
    pep2taxa_shared = {}
    for p, taxa in pep2orgs.items():
        if p not in peptides_in_matrix:
            continue
        # Normalize container
        taxa_set = set(taxa)
        taxa_set = {t for t in taxa_set if t not in exclude}
        if len(taxa_set) >= 2:
            if max_taxa_per_peptide is None or len(taxa_set) <= max_taxa_per_peptide:
                pep2taxa_shared[p] = tuple(sorted(taxa_set))

    return pep2taxon_unique, pep2taxa_shared

# ---- B. Compute α_{p,t,s} given β and current taxon profiles T
def compute_alpha(df_all, T_taxon_run, pep2taxa_shared, beta):
    """
    Inputs:
      - df_all: peptide x run linear intensities (NaNs allowed)
      - T_taxon_run: taxon x run linear intensities (current profiles)
      - pep2taxa_shared: dict peptide -> tuple(taxa,...)
      - beta: dict peptide -> dict {taxon: beta_pt} (must cover each (p,t))
    Returns:
      - alpha: dict (p, t) -> np.ndarray of shape (n_runs,) with α_{p,t,s} per run
    """
    runs = T_taxon_run.columns.tolist()
    alpha = {}
    # Pre-extract T as array per taxon for speed
    T_by_tax = {t: T_taxon_run.loc[t].values.astype(float) for t in T_taxon_run.index if t in T_taxon_run.index}

    for p, taxa in pep2taxa_shared.items():
        # Numerators per taxon: β_{p,t} * T_{t,:}
        num = []
        taxa_list = list(taxa)
        for t in taxa_list:
            if t not in T_by_tax:
                num.append(np.full(len(runs), np.nan))
            else:
                num.append(beta[p][t] * T_by_tax[t])
        num = np.vstack(num)  # shape: (n_taxa_for_p, n_runs)
        denom = np.nansum(num, axis=0)  # shape: (n_runs,)
        # Avoid divide by zero: where denom==0 or NaN, set α to NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            for idx_t, t in enumerate(taxa_list):
                a = num[idx_t] / denom
                a[~np.isfinite(a)] = np.nan
                alpha[(p, t)] = a
    return alpha

# ---- C. Update β from virtual traces vs taxon profiles (robust intercepts)
def update_beta(df_all, T_taxon_run, pep2taxa_shared, alpha, prior_beta=None, shrink=0.0):
    """
    Returns updated beta dict with β normalized to sum to 1 per peptide.
    shrink in [0,1]: shrinkage toward prior_beta in log space (0=no shrink).
    prior_beta: dict peptide -> dict {taxon: prior_beta_pt} (optional).
    """
    runs = T_taxon_run.columns.tolist()
    logT = np.log(T_taxon_run.replace(0, np.nan))

    beta_new = {}
    for p, taxa in pep2taxa_shared.items():
        tilde = {}
        # Use per-tax robust intercept: median_s [ log(I) + log(α) - log(T) ]
        logI = np.log(df_all.loc[p].values.astype(float))
        for t in taxa:
            a = alpha.get((p, t))
            if a is None:
                tilde[t] = np.nan
                continue
            # per-run vector for this peptide/taxon
            vals = logI + np.log(a) - logT.loc[t].values
            # keep finite
            finite = np.isfinite(vals)
            if finite.sum() >= 2:
                c_hat = np.nanmedian(vals[finite])
            elif finite.sum() == 1:
                c_hat = vals[finite][0]
            else:
                c_hat = np.nan  # insufficient info
            tilde[t] = np.exp(c_hat) if np.isfinite(c_hat) else np.nan

        # If all NaN, fall back to prior or uniform
        # Combine with prior via shrinkage in log space
        if prior_beta and (p in prior_beta):
            for t in taxa:
                if not np.isfinite(tilde.get(t, np.nan)):
                    tilde[t] = prior_beta[p].get(t, 1.0)
                else:
                    if shrink > 0:
                        # shrink log tilde toward log prior
                        log_tilde = np.log(tilde[t])
                        log_prior = np.log(prior_beta[p].get(t, tilde[t]))
                        log_comb = (1 - shrink) * log_tilde + shrink * log_prior
                        tilde[t] = np.exp(log_comb)
        else:
            # No prior: replace NaNs by uniform
            n = len(taxa)
            for t in taxa:
                if not np.isfinite(tilde.get(t, np.nan)):
                    tilde[t] = 1.0

        # Normalize to sum 1
        s = sum(tilde.values())
        if s <= 0 or not np.isfinite(s):
            n = len(taxa)
            beta_new[p] = {t: 1.0 / n for t in taxa}
        else:
            beta_new[p] = {t: tilde[t] / s for t in taxa}
    return beta_new

def taxon_profiles_from_virtual(df_all, pep2taxon_unique, pep2taxa_shared, alpha,
                                min_ratios=2, solver='L-BFGS-B',
                                alpha_floor=1e-3, rescale_total=True):
    """
    Build per-taxon virtual peptide matrices and run MaxLFQ to get updated T.
    Returns:
      - T_new: DataFrame [taxa x runs] updated taxon profiles (linear units)
      - diag: diagnostics per taxon
    """
    taxa = sorted(set(pep2taxon_unique.values()) |
                  {t for _, taxa in pep2taxa_shared.items() for t in taxa})
    runs = df_all.columns.tolist()
    run_pairs = list(combinations(range(len(runs)), 2))

    T_new = pd.DataFrame(index=taxa, columns=runs, dtype=float)
    diag = pd.DataFrame(index=taxa, columns=['n_unique', 'n_shared', 'n_pairs', 'ok'], dtype=float)

    # Pre-index peptide lists
    taxon_to_unique = {t: [] for t in taxa}
    for p, t in pep2taxon_unique.items():
        taxon_to_unique[t].append(p)

    taxon_to_shared = {t: [] for t in taxa}
    for p, ts in pep2taxa_shared.items():
        for t in ts:
            taxon_to_shared[t].append(p)

    for t in taxa:
        unique_peps = taxon_to_unique.get(t, [])
        shared_peps = taxon_to_shared.get(t, [])
        # Build sub-matrix
        rows = []
        # Unique rows (raw)
        if unique_peps:
            rows.append(df_all.loc[unique_peps])
        # Shared rows (virtual)
        if shared_peps:
            # Construct per peptide virtual row: α_{p,t,:} * I_{p,:}, drop tiny α
            virt_rows = []
            for p in shared_peps:
                a = alpha.get((p, t))
                if a is None:
                    continue
                # Mask tiny α or missing I
                I = df_all.loc[p].values.astype(float)
                v = a * I
                v[(~np.isfinite(v)) | (a < alpha_floor)] = np.nan
                virt_rows.append(pd.Series(v, index=runs, name=p))
            if virt_rows:
                rows.append(pd.DataFrame(virt_rows))

        if not rows:
            T_new.loc[t] = np.nan
            diag.loc[t] = (0, 0, 0, 0)
            continue

        sub = pd.concat(rows, axis=0).replace(0, np.nan)
        # Compute MaxLFQ ratios and profile
        ratios = get_protein_ratios(sub.values, run_pairs, min_ratios)
        n_pairs = int(np.isfinite(ratios).sum())
        I_hat, ok = solve_profile(ratios, solver)
        if not ok and np.isfinite(I_hat).any():
            warnings.warn(
                f"solve_profile did not converge for taxon {t!r} (success=False); "
                "using best solution found.",
                stacklevel=2,
            )

        if np.isfinite(I_hat).any():
            if rescale_total:
                total_obs = np.nansum(sub.values)  # taxon’s observed peptide signal (virtual + unique)
                sum_hat = np.nansum(I_hat)
                if np.isfinite(total_obs) and (sum_hat > 0):
                    I_hat = I_hat * (total_obs / sum_hat)
            T_new.loc[t] = I_hat
            diag.loc[t] = (len(unique_peps), len(shared_peps), n_pairs, 1)
        else:
            T_new.loc[t] = np.nan
            diag.loc[t] = (len(unique_peps), len(shared_peps), n_pairs, 0)

    T_new.index.name = 'organism'
    return T_new, diag

def _log_max_change(T_new: pd.DataFrame, T_old: pd.DataFrame) -> float:
    """Max absolute log-ratio change between two taxon×run matrices (NaN-safe)."""
    a = np.log(T_new.values.astype(float))
    b = np.log(T_old.values.astype(float))
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return np.inf
    return float(np.max(np.abs(a[finite] - b[finite])))


def em_taxon_shared_lfq(df_all,
                        TaxonLFQ_Max_uniques,
                        pep2org, pep2orgs,
                        exclude=None,
                        max_iter=3,
                        min_ratios=2,
                        alpha_floor=1e-3,
                        beta_shrink=0.0,      # 0..0.5 typically; 0 means no shrink to priors
                        prior_beta=None,      # optional dict {peptide: {taxon: prior_beta}}
                        max_taxa_per_peptide=None,
                        solver='L-BFGS-B',
                        tol=1e-4):            # convergence: max log-change in T
    """
    EM-style refinement with shared peptides:
      - Initialize T from unique-only LFQ (TaxonLFQ_Max_uniques)
      - Initialize β (uniform or prior)
      - Iterate: α <- β,T ; T <- virtual LFQ ; β <- robust intercepts (with optional shrink)
    Returns:
      - T_final: DataFrame [taxa x runs]
      - beta: dict peptide -> dict {taxon: beta_pt}
      - alpha: dict (peptide,taxon) -> np.ndarray α over runs (last iteration)
      - diag: per-taxon diagnostics from last M-step(A)
      - n_iter: number of EM iterations actually run
    """
    if exclude is None:
        exclude = set()

    # 0) Prepare mappings
    pep2taxon_unique, pep2taxa_shared = prepare_mappings(
        df_all, pep2org, pep2orgs, exclude=exclude, max_taxa_per_peptide=max_taxa_per_peptide
    )
    if len(pep2taxa_shared) == 0:
        print("No shared peptides after filtering; nothing to EM-refine.")
        return TaxonLFQ_Max_uniques.copy(), {}, {}, pd.DataFrame(), 0

    # 1) Initialize T and β
    T = TaxonLFQ_Max_uniques.copy()
    # Keep only taxa present in T
    taxa_in_T = set(T.index)
    # Initialize β uniformly unless prior provided
    beta = {}
    for p, taxa in pep2taxa_shared.items():
        taxa = [t for t in taxa if t in taxa_in_T]
        if not taxa:
            continue
        if (prior_beta is not None) and (p in prior_beta):
            # Normalize prior to sum 1 over available taxa
            prior = {t: prior_beta[p].get(t, 0.0) for t in taxa}
            s = sum(prior.values())
            if s > 0:
                beta[p] = {t: prior[t] / s for t in taxa}
            else:
                beta[p] = {t: 1.0/len(taxa) for t in taxa}
        else:
            beta[p] = {t: 1.0/len(taxa) for t in taxa}

    # 2) Iterate
    n_iter = 0
    diag = pd.DataFrame()
    for it in range(1, max_iter + 1):
        T_prev = T.copy()

        # E-step: α from β and current T
        alpha = compute_alpha(df_all, T, pep2taxa_shared, beta)

        # M-step(A): update T via virtual intensities (unique + sharedα)
        T_new, diag = taxon_profiles_from_virtual(
            df_all, pep2taxon_unique, pep2taxa_shared, alpha,
            min_ratios=min_ratios, solver=solver,
            alpha_floor=alpha_floor, rescale_total=True
        )
        T = T_new
        n_iter = it

        # M-step(B): update β via robust intercepts with optional shrink to prior
        beta = update_beta(
            df_all, T, pep2taxa_shared, alpha,
            prior_beta=prior_beta, shrink=beta_shrink
        )

        delta = _log_max_change(T, T_prev)
        print(
            f"[EM] iteration {it} complete. "
            f"Taxa with OK profile: {int(diag['ok'].sum())}/{len(diag)}. "
            f"Max log-change: {delta:.4f}"
        )
        if delta < tol:
            print(f"[EM] Converged after {it} iteration(s).")
            break

    return T, beta, alpha, diag, n_iter

import numpy as np
import pandas as pd

def build_virtual_matrix(df_all, pep2org, pep2orgs, alpha_hat,
                         exclude=None, alpha_floor=1e-3, keep_shared_with_small_alpha=False):
    """
    Build a virtual, taxon-resolved peptide×run matrix using EM outputs.

    Parameters
    ----------
    df_all : DataFrame
        Peptide (index) × run (columns), linear intensities (NaN for missing).
    pep2org : dict
        Unique peptide -> single taxon.
    pep2orgs : dict
        Peptide -> iterable of taxa (for shared peptides).
    alpha_hat : dict
        (peptide, taxon) -> np.ndarray of alpha weights across runs (last EM iteration).
        Must align with df_all.columns order.
    exclude : set or None
        Taxa to exclude (already filtered upstream, but supported here).
    alpha_floor : float
        Contributions with alpha < alpha_floor are set to NaN (to avoid spraying tiny noise).
    keep_shared_with_small_alpha : bool
        If False (default), rows where all alphas < alpha_floor for a given (p,t) become all-NaN.

    Returns
    -------
    virt_df : DataFrame
        MultiIndex rows (peptide, taxon) × runs; virtual intensities (linear).
    virt_taxon_sums : DataFrame
        Taxon × runs; sums of virt_df per taxon (observed from apportioned peptides).
    """
    if exclude is None:
        exclude = set()

    runs = df_all.columns.tolist()
    rows = []

    # 1) Unique peptides: assign entirely to their taxon
    for p, t in pep2org.items():
        if t in exclude or p not in df_all.index:
            continue
        vals = df_all.loc[p].values.astype(float)
        # Keep as-is; zeros already NaN upstream
        rows.append(pd.Series(vals, index=runs, name=(p, t)))

    # 2) Shared peptides: split by alpha
    for p, taxa in pep2orgs.items():
        if p not in df_all.index:
            continue
        I = df_all.loc[p].values.astype(float)
        if not np.isfinite(I).any():
            continue
        for t in taxa:
            if t in exclude:
                continue
            a = alpha_hat.get((p, t))
            if a is None:
                continue
            v = a * I  # virtual intensities for this (p,t)
            # Drop tiny alpha contributions
            mask_small = (a < alpha_floor) | (~np.isfinite(a)) | (~np.isfinite(I))
            v = v.astype(float)
            v[mask_small] = np.nan
            if (not keep_shared_with_small_alpha) and np.all(~np.isfinite(v)):
                continue
            rows.append(pd.Series(v, index=runs, name=(p, t)))

    if not rows:
        virt_df = pd.DataFrame(columns=runs, dtype=float)
        virt_df.index = pd.MultiIndex.from_tuples([], names=['peptide', 'taxon'])
        virt_taxon_sums = pd.DataFrame(index=[], columns=runs, dtype=float)
        return virt_df, virt_taxon_sums

    virt_df = pd.DataFrame(rows)
    virt_df.index = pd.MultiIndex.from_tuples(virt_df.index, names=['peptide', 'taxon'])

    # Sums per taxon × run (what the apportioned evidence says)
    virt_taxon_sums = virt_df.groupby(level='taxon').sum(min_count=1)

    return virt_df, virt_taxon_sums


def reconcile_virtual_to_T(virt_df, T_final, min_total=0.0):
    """
    Scale virtual intensities within each (taxon, run) so that their sums equal T_final[t, s].

    Parameters
    ----------
    virt_df : DataFrame
        MultiIndex (peptide, taxon) × runs; virtual intensities.
    T_final : DataFrame
        Taxon × runs; final LFQ profiles from EM.
    min_total : float
        If the current virtual sum for (t, s) <= min_total, skip scaling (factor=1).

    Returns
    -------
    virt_df_scaled : DataFrame
        Same shape as virt_df, scaled per (taxon, run) to match T_final exactly.
    scale_factors : DataFrame
        Taxon × runs; the applied scaling factors gamma_{t,s} = T_final / sum(virt).
    """
    # Current sums per taxon-run from virt_df
    current = virt_df.groupby(level='taxon').sum(min_count=1).reindex(T_final.index).reindex(columns=T_final.columns)
    # Compute per (taxon, run) scaling factor
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = T_final / current
    gamma = gamma.where(np.isfinite(gamma) & (current > min_total), 1.0)

    # Apply scaling: for each taxon, multiply all its peptide rows by gamma[t, :]
    out = virt_df.copy()
    for t in T_final.index:
        if t in out.index.get_level_values('taxon'):
            out.loc[(slice(None), t), :] = out.loc[(slice(None), t), :].mul(gamma.loc[t], axis=1)

    return out, gamma


# ---------------------------------------------------------------------------
# Evidence.txt reader and top-level entry point
# ---------------------------------------------------------------------------

def read_evidence_txt(
    path,
    *,
    seq_col: str = "Sequence",
    run_col: str = "Raw file",
    intensity_col: str = "Intensity",
    filter_contaminants: bool = True,
    filter_reverse: bool = True,
    min_pep: float | None = None,
) -> pd.DataFrame:
    """Read a MaxQuant evidence.txt file and return a peptide × run intensity matrix.

    Rows are aggregated by (sequence, run) by summing intensities, so multiple
    charge states and PSMs for the same peptide in the same run are collapsed.

    Parameters
    ----------
    path:
        Path to evidence.txt (tab-separated MaxQuant output).
    seq_col:
        Column holding the unmodified peptide sequence (default ``"Sequence"``).
    run_col:
        Column identifying the raw file / run (default ``"Raw file"``).
    intensity_col:
        Intensity column to use (default ``"Intensity"``; do not use
        ``"LFQ intensity"`` — that only appears in proteinGroups.txt).
    filter_contaminants:
        Drop rows where ``Potential contaminant`` is ``"+"``.
    filter_reverse:
        Drop rows where ``Reverse`` is ``"+"``.
    min_pep:
        If set, drop rows where ``PEP`` exceeds this threshold.

    Returns
    -------
    DataFrame
        Index = peptide sequence (name ``"peptide"``), columns = run names,
        values = summed linear intensities (NaN for missing combinations).
    """
    ev = pd.read_csv(path, sep="\t", low_memory=False)

    if filter_contaminants and "Potential contaminant" in ev.columns:
        ev = ev[ev["Potential contaminant"].fillna("") != "+"]
    if filter_reverse and "Reverse" in ev.columns:
        ev = ev[ev["Reverse"].fillna("") != "+"]
    if min_pep is not None and "PEP" in ev.columns:
        ev = ev[pd.to_numeric(ev["PEP"], errors="coerce") <= min_pep]

    ev[intensity_col] = pd.to_numeric(ev[intensity_col], errors="coerce").replace(0, np.nan)

    agg = (
        ev.groupby([seq_col, run_col])[intensity_col]
        .sum(min_count=1)
        .reset_index()
    )

    matrix = agg.pivot(index=seq_col, columns=run_col, values=intensity_col)
    matrix.index.name = "peptide"
    matrix.columns.name = None
    return matrix


def read_alphapept_peptides(
    path,
    *,
    sequence_col: str = "sequence_naked",
    run_col: str = "sample_group",
    intensity_col: str = "ms1_int_sum_area",
    decoy_col: str = "decoy",
    filter_decoys: bool = True,
    max_q_value: float | None = None,
    q_value_col: str = "q_value",
    strip_run_extension: bool = True,
) -> pd.DataFrame:
    """Read an AlphaPept ``*_peptides.csv`` and return a peptide × run matrix.

    AlphaPept exports one row per scored precursor (charge state × modification
    form) in ``<results>_peptides.csv``.  This function aggregates to the
    unmodified stripped-sequence level by summing ``ms1_int_sum_area`` across
    all precursors sharing the same ``(sequence_naked, sample_group)`` pair,
    then pivoting to a peptide × run matrix.

    This matches the two-step aggregation described in the TaxonLFQ methods:
    precursor → modified peptide (collapse charge states), then modified
    peptide → unmodified sequence (collapse modification forms).  Because
    AlphaPept's ``sequence_naked`` is already fully stripped, a single
    ``groupby`` achieves both steps at once.

    Parameters
    ----------
    path:
        Path to AlphaPept ``*_peptides.csv`` (one row per scored precursor).
        Produced by the AlphaPept workflow with ``lfq_quantification: true``.
    sequence_col:
        Column holding the unmodified peptide sequence (default
        ``"sequence_naked"``).
    run_col:
        Column identifying the raw file / run (default ``"sample_group"``).
        AlphaPept sets this to the filename stem unless ``shortnames`` or
        ``sample_group`` were explicitly provided in the settings.
    intensity_col:
        Intensity column to aggregate (default ``"ms1_int_sum_area"``).
        Do **not** use any MaxLFQ-normalised column.
    decoy_col:
        Column flagging decoy hits (default ``"decoy"``).  Rows where this
        column equals ``1`` or ``True`` are dropped when *filter_decoys* is
        ``True``.
    filter_decoys:
        Drop decoy hits before aggregation (default ``True``).
    max_q_value:
        If set, drop rows where *q_value_col* exceeds this threshold.  Leave
        as ``None`` when FDR was already applied inside AlphaPept.
    q_value_col:
        Column holding per-PSM q-values (default ``"q_value"``).
    strip_run_extension:
        Strip file-system path and ``.raw`` / ``.mzML`` extension from run
        names so labels are clean (default ``True``).

    Returns
    -------
    DataFrame
        Index = peptide sequence (name ``"peptide"``), columns = run names,
        values = summed linear intensities (NaN for missing combinations).
    """
    prec = pd.read_csv(path, low_memory=False)

    if filter_decoys and decoy_col in prec.columns:
        flag = prec[decoy_col]
        prec = prec[~flag.isin([1, True, "1", "True", "true"])]

    if max_q_value is not None and q_value_col in prec.columns:
        prec = prec[pd.to_numeric(prec[q_value_col], errors="coerce") <= max_q_value]

    prec[intensity_col] = pd.to_numeric(prec[intensity_col], errors="coerce").replace(0, np.nan)

    if run_col in prec.columns:
        prec = prec.dropna(subset=[run_col])

    if strip_run_extension and run_col in prec.columns:
        prec[run_col] = (
            prec[run_col]
            .astype(str)
            .apply(lambda s: re.sub(r"\.(raw|mzML|mzml|RAW)$", "", Path(s).name))
        )

    agg = (
        prec.groupby([sequence_col, run_col])[intensity_col]
        .sum(min_count=1)
        .reset_index()
    )

    matrix = agg.pivot(index=sequence_col, columns=run_col, values=intensity_col)
    matrix.index.name = "peptide"
    matrix.columns.name = None
    return matrix


def _load_inputs(
    intensities_file, mapping_file, sep: str = "\t"
) -> tuple[pd.DataFrame, dict, dict]:
    """Load intensity matrix and peptide→taxon mappings from files."""
    df = pd.read_csv(intensities_file, sep=sep, index_col=0).apply(
        pd.to_numeric, errors="coerce"
    )
    df.index.name = "peptide"

    mapping = pd.read_csv(mapping_file, sep=sep)
    mapping.columns = [c.strip() for c in mapping.columns]

    counts = mapping.groupby("peptide")["taxon"].nunique()
    unique_peps = set(counts[counts == 1].index)
    pep2org = (
        mapping[mapping["peptide"].isin(unique_peps)]
        .set_index("peptide")["taxon"]
        .to_dict()
    )
    pep2orgs = mapping.groupby("peptide")["taxon"].apply(list).to_dict()
    return df, pep2org, pep2orgs


def _run_core(df, pep2org, pep2orgs, *, exclude, max_iter, min_ratios,
              alpha_floor, beta_shrink, max_taxa_per_peptide, solver, tol):
    """Shared compute kernel used by run_taxonlfq and benchmark_taxonlfq."""
    T_uniques, diag_uniques = lfq_taxon_unique(
        df, pep2org, exclude=exclude, min_ratios=min_ratios, solver=solver
    )
    T_em, _beta, _alpha, diag_em, n_iter = em_taxon_shared_lfq(
        df, T_uniques, pep2org, pep2orgs,
        exclude=exclude,
        max_iter=max_iter,
        min_ratios=min_ratios,
        alpha_floor=alpha_floor,
        beta_shrink=beta_shrink,
        max_taxa_per_peptide=max_taxa_per_peptide,
        solver=solver,
        tol=tol,
    )
    if T_em is None or T_em.empty:
        return T_uniques, diag_uniques, 0
    return T_em, diag_em, n_iter


def run_taxonlfq(
    intensities_file,
    mapping_file,
    *,
    sep: str = "\t",
    exclude: set | None = None,
    max_iter: int = 3,
    min_ratios: int = 2,
    alpha_floor: float = 1e-3,
    beta_shrink: float = 0.0,
    max_taxa_per_peptide: int | None = None,
    solver: str = "L-BFGS-B",
    tol: float = 1e-4,
    stats_dir: str | Path | None = "run_stats",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run TaxonLFQ end-to-end from pre-pivoted intensity and mapping files.

    Saves wall-time, EM iterations, and result shape to *stats_dir* after every
    run (pass ``stats_dir=None`` to disable).

    Parameters
    ----------
    intensities_file:
        Path to a tab- (or comma-) separated file where the first column is
        named ``"peptide"`` and the remaining columns are run/sample names
        containing linear intensities.  Produce this from a MaxQuant
        ``evidence.txt`` with :func:`read_evidence_txt`.
    mapping_file:
        Path to a tab- (or comma-) separated file with columns ``"peptide"``
        and ``"taxon"``.  Peptides mapping to a single taxon appear once;
        shared peptides appear once per taxon.
    sep:
        Column separator for both input files (default ``"\\t"``).

    Returns
    -------
    T_final : DataFrame
        Taxon × run linear LFQ intensity matrix.
    diag : DataFrame
        Per-taxon diagnostics (n_unique, n_shared, n_pairs, ok).
    """
    df, pep2org, pep2orgs = _load_inputs(intensities_file, mapping_file, sep)

    t0 = time.perf_counter()
    T_out, diag_out, n_iter = _run_core(
        df, pep2org, pep2orgs,
        exclude=exclude, max_iter=max_iter, min_ratios=min_ratios,
        alpha_floor=alpha_floor, beta_shrink=beta_shrink,
        max_taxa_per_peptide=max_taxa_per_peptide, solver=solver, tol=tol,
    )
    wall_time = time.perf_counter() - t0

    if stats_dir is not None:
        _save_run_stats({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_time_s": round(wall_time, 4),
            "peak_memory_mb": "",
            "em_iterations": n_iter,
            "n_taxa": int(T_out.shape[0]),
            "n_samples": int(T_out.shape[1]),
            "intensities_file": str(intensities_file),
            "mapping_file": str(mapping_file),
        }, stats_dir)

    return T_out, diag_out


def benchmark_taxonlfq(
    intensities_file,
    mapping_file,
    *,
    sep: str = "\t",
    exclude: set | None = None,
    max_iter: int = 3,
    min_ratios: int = 2,
    alpha_floor: float = 1e-3,
    beta_shrink: float = 0.0,
    max_taxa_per_peptide: int | None = None,
    solver: str = "L-BFGS-B",
    tol: float = 1e-4,
    stats_dir: str | Path | None = "run_stats",
) -> BenchmarkResult:
    """Run TaxonLFQ and return timing, peak memory, and iteration count.

    Parameters match :func:`run_taxonlfq`.  File I/O is excluded from the
    timing/memory window so that the benchmark reflects only compute cost.
    Results are saved to *stats_dir* (pass ``None`` to disable).

    Returns
    -------
    BenchmarkResult
        .wall_time_s    — elapsed wall-clock seconds (compute only, excl. I/O)
        .peak_memory_mb — peak Python heap allocation in MB (via tracemalloc)
        .em_iterations  — actual EM iterations run (≤ max_iter if converged)
        .T_final        — taxon × run LFQ matrix
        .diag           — per-taxon diagnostics DataFrame
    """
    df, pep2org, pep2orgs = _load_inputs(intensities_file, mapping_file, sep)

    tracemalloc.start()
    t0 = time.perf_counter()
    T_out, diag_out, n_iter = _run_core(
        df, pep2org, pep2orgs,
        exclude=exclude, max_iter=max_iter, min_ratios=min_ratios,
        alpha_floor=alpha_floor, beta_shrink=beta_shrink,
        max_taxa_per_peptide=max_taxa_per_peptide, solver=solver, tol=tol,
    )
    wall_time = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if stats_dir is not None:
        _save_run_stats({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_time_s": round(wall_time, 4),
            "peak_memory_mb": round(peak_bytes / 1024 ** 2, 3),
            "em_iterations": n_iter,
            "n_taxa": int(T_out.shape[0]),
            "n_samples": int(T_out.shape[1]),
            "intensities_file": str(intensities_file),
            "mapping_file": str(mapping_file),
        }, stats_dir)

    return BenchmarkResult(
        wall_time_s=wall_time,
        peak_memory_mb=peak_bytes / 1024 ** 2,
        em_iterations=n_iter,
        T_final=T_out,
        diag=diag_out,
    )