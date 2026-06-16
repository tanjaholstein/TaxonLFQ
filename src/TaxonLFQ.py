import numpy as np
import pandas as pd
from itertools import combinations

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

        if ok and np.isfinite(I_hat).any():
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

        if ok and np.isfinite(I_hat).any():
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
                        solver='L-BFGS-B'):
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
    """
    if exclude is None:
        exclude = set()

    # 0) Prepare mappings
    pep2taxon_unique, pep2taxa_shared = prepare_mappings(
        df_all, pep2org, pep2orgs, exclude=exclude, max_taxa_per_peptide=max_taxa_per_peptide
    )
    if len(pep2taxa_shared) == 0:
        print("No shared peptides after filtering; nothing to EM-refine.")
        return TaxonLFQ_Max_uniques.copy(), {}, {}, pd.DataFrame()

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
    for it in range(1, max_iter+1):
        # E-step: α from β and current T
        alpha = compute_alpha(df_all, T, pep2taxa_shared, beta)

        # M-step(A): update T via virtual intensities (unique + sharedα)
        T_new, diag = taxon_profiles_from_virtual(
            df_all, pep2taxon_unique, pep2taxa_shared, alpha,
            min_ratios=min_ratios, solver=solver,
            alpha_floor=alpha_floor, rescale_total=True
        )
        # Replace T (keep taxa that exist in T_new)
        T = T_new

        # M-step(B): update β via robust intercepts with optional shrink to prior
        beta = update_beta(
            df_all, T, pep2taxa_shared, alpha,
            prior_beta=prior_beta, shrink=beta_shrink
        )

        # (Optional) convergence check on T
        # We can stop early if max absolute log-change is tiny
        # Here we keep a fixed number of iterations for simplicity

        print(f"[EM] iteration {it} complete. Taxa with OK profile: {int(diag['ok'].sum())}/{len(diag)}")

    return T, beta, alpha, diag

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