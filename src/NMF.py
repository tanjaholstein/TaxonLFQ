import numpy as np

# ---- Euclidean projection onto the probability simplex (sum=1, >=0) ----
def _project_simplex(v, z=1.0):
    """Project 1D array v onto the simplex {x>=0, sum x = z} (Duchi et al., 2008)."""
    if z <= 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - z))[0][-1]
    theta = (cssv[rho] - z) / float(rho + 1)
    w = np.maximum(v - theta, 0)
    s = w.sum()
    return w if s > 0 else np.full_like(v, z / len(v))


def nmf_row_stochastic_totvar_pg(
    I_raw,
    taxon_idx_by_peptide,
    n_taxa,
    lambda_tot=5.0,      # strength of the totals-variance penalty
    step_tot=0.5,        # step size for the penalty gradient on P (0.1..1.0 works well)
    n_iter=800,
    tol=1e-5,
    init="uniform",      # "uniform" or "random" for shared rows
    init_jitter=1e-3,    # tiny noise added if init="uniform" to break symmetry
    n_starts=1,
    mask=None,           # 0/1 mask; default: ~np.isnan(I_raw)
    floor=1e-12,         # tiny floor to avoid zero-lock in multiplicative steps
    rho=0.0,             # shrink of shared rows toward a prior on allowed taxa
    prior_dict=None,     # {row_index -> prior vector over allowed taxa (sums to 1)}
    verbose=False,
    random_state=0,
    return_info=False,
):
    """
    NMF with row-stochastic P, unique rows one-hot, and a penalty on the
    run-to-run variance of taxon totals: (lambda_tot/R) * || P^T Zc ||_F^2,
    where Zc is the row-centered intensity matrix (under mask).

    Optimization:
      • G: multiplicative update for masked reconstruction term
      • P: (a) multiplicative reconstruction step on allowed taxa only
            then
            (b) projected-gradient step for the totals-variance penalty

    This avoids building S = Zc Zc^T (which is huge).
    """
    I_raw = np.asarray(I_raw, dtype=float)
    n_pep, n_run = I_raw.shape

    # observation mask
    if mask is None:
        M = (~np.isnan(I_raw)).astype(float)
    else:
        M = np.asarray(mask, dtype=float)
        if M.shape != I_raw.shape:
            raise ValueError("mask must have same shape as I_raw")

    # replace NaNs with 0 for arithmetic; they are ignored by M
    I = np.nan_to_num(I_raw, nan=0.0)

    # row-centering under mask: Zc = (I - row_mean) * M
    m_counts = M.sum(axis=1, keepdims=True)
    row_mean = np.divide((I * M).sum(axis=1, keepdims=True),
                         np.maximum(m_counts, 1.0))
    Zc = (I - row_mean) * M   # (n_pep × n_run)

    # unique rows bookkeeping
    unique_mask = np.array([len(t) == 1 for t in taxon_idx_by_peptide], dtype=bool)

    # default uniform prior for shared rows if requested
    if rho > 0.0 and prior_dict is None:
        prior_dict = {
            i: np.full(len(taxa), 1.0 / len(taxa), dtype=float)
            for i, taxa in enumerate(taxon_idx_by_peptide) if len(taxa) > 1
        }

    def _init_P_G(rng):
        P0 = np.zeros((n_pep, n_taxa), dtype=float)
        for i, taxa in enumerate(taxon_idx_by_peptide):
            if len(taxa) == 1:
                P0[i, taxa[0]] = 1.0
            else:
                if init == "uniform":
                    p = np.full(len(taxa), 1.0 / len(taxa))
                    if init_jitter > 0:
                        noise = rng.random(len(taxa))
                        noise /= noise.sum()
                        p = (1.0 - init_jitter) * p + init_jitter * noise
                    P0[i, taxa] = p
                else:  # random Dirichlet-like
                    v = rng.random(len(taxa)); v /= v.sum()
                    P0[i, taxa] = v
        G0 = np.maximum(np.random.default_rng(rng.integers(1<<31)).random((n_taxa, n_run)), floor)
        return P0, G0

    def _objective(P, G):
        PG   = P @ G
        resid = M * (I - PG)
        loss_rec = (resid**2).sum() / max(M.sum(), 1.0)
        T = P.T @ Zc
        loss_tot = (T**2).sum() / max(n_run, 1)
        return loss_rec + lambda_tot * loss_tot, loss_rec, loss_tot

    def _run_once(seed):
        rng = np.random.default_rng(seed)
        P, G = _init_P_G(rng)
        prev_obj = np.inf

        for it in range(n_iter):
            # ---------- G update: masked multiplicative reconstruction ----------
            PG   = P @ G
            numG = P.T @ (M * I)
            denG = P.T @ (M * PG) + floor
            G *= numG / denG
            G  = np.maximum(G, floor)

            # ---------- P update (a): multiplicative reconstruction ----------
            PG      = P @ G
            numPrec = (M * I) @ G.T
            denPrec = (M * PG) @ G.T + floor
            # per-row, on allowed taxa only
            for i, taxa in enumerate(taxon_idx_by_peptide):
                if len(taxa) == 1:
                    # keep one-hot
                    j = taxa[0]; P[i, :] = 0.0; P[i, j] = 1.0
                    continue
                upd = numPrec[i, taxa] / denPrec[i, taxa]
                p_i = P[i, taxa] * upd
                p_i = np.maximum(p_i, floor)
                p_i = p_i / p_i.sum()
                P[i, :] = 0.0; P[i, taxa] = p_i

            # ---------- P update (b): gradient step for totals-variance ----------
            # grad_tot = 2*lambda_tot/R * Zc @ (Zc^T @ P)
            ZtP      = Zc.T @ P             # (n_run × n_taxa)
            grad_tot = (2.0 * lambda_tot / max(n_run, 1.0)) * (Zc @ ZtP)  # (n_pep × n_taxa)

            # global step using infinity-norm for safety
            gscale = np.max(np.abs(grad_tot))
            eta    = step_tot / (gscale + 1e-12)

            for i, taxa in enumerate(taxon_idx_by_peptide):
                if len(taxa) == 1:
                    continue
                p_i = P[i, taxa] - eta * grad_tot[i, taxa]
                # project onto simplex over allowed taxa
                p_i = _project_simplex(p_i, z=1.0)
                # optional shrink toward prior
                if rho > 0.0:
                    prior_i = prior_dict.get(i, np.full(len(taxa), 1.0/len(taxa)))
                    p_i = (1.0 - rho) * p_i + rho * prior_i
                    p_i = _project_simplex(p_i, z=1.0)
                P[i, :] = 0.0; P[i, taxa] = p_i

            # ---------- objective & stopping ----------
            obj, loss_rec, loss_tot = _objective(P, G)
            if verbose and (it % 50 == 0 or it == n_iter - 1):
                print(f"{it:4d}  rec={loss_rec:.4e}  totVar={loss_tot:.4e}  obj={obj:.4e}")
            rel = abs(prev_obj - obj) / max(prev_obj, 1e-12)
            if rel < tol:
                if verbose:
                    print(f"Converged at iter {it}, obj={obj:.4e}")
                break
            prev_obj = obj

        return P, G, obj

    # -------------- multi-start --------------
    best = {"loss": np.inf, "P": None, "G": None, "seed": None}
    for s in range(n_starts):
        P_s, G_s, obj_s = _run_once(random_state + s)
        if obj_s < best["loss"]:
            best.update({"loss": obj_s, "P": P_s, "G": G_s, "seed": random_state + s})

    if return_info:
        return best["P"], best["G"], {"best_loss": best["loss"], "seed": best["seed"], "n_starts": n_starts}
    return best["P"], best["G"]


import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1.  Helper: build taxon-index mapping in peptide-row order
# ----------------------------------------------------------------------
def build_taxon_idx_by_peptide(pep2orgs_clean, peptide_index):
    """
    Returns
      peptides_used          – list of peptides that *did* get a mapping
      all_taxa               – sorted list of taxon names
      taxon_idx_by_peptide   – list-of-lists of integer taxa indices
                               (same order as peptides_used)
    """
    all_taxa = sorted({t for v in pep2orgs_clean.values() for t in v})
    tax2idx = {t: j for j, t in enumerate(all_taxa)}

    peptides_used, taxon_idx_by_peptide = [], []
    for pep in peptide_index:
        taxa = pep2orgs_clean.get(pep)
        if taxa is None:                     # peptide had no mapping → skip
            continue
        peptides_used.append(pep)
        taxon_idx_by_peptide.append([tax2idx[t] for t in taxa])

    return peptides_used, all_taxa, taxon_idx_by_peptide


# ----------------------------------------------------------------------
# 2.  Row-stochastic NMF with variance penalty
# ----------------------------------------------------------------------
def nmf_row_stochastic(
    I,
    taxon_idx_by_peptide,
    n_taxa,
    lambda_var=1e-2,
    n_iter=500,
    tol=1e-4,
    verbose=False,
    random_state=0,
):
    """
    Factorises I (n_peptides × n_runs) as I ≈ P @ G with:

        • P ≥ 0,   rows sum to 1
        • G ≥ 0
        • variance penalty on rows of G
        • peptides mapping to a single taxon are held fixed  (one-hot rows)

    Parameters
    ----------
    I : (n_peptides, n_runs) ndarray
        Intensity matrix (use zeros or NaN-masked values for missing entries).
    taxon_idx_by_peptide : list[list[int]]
        For each peptide row, the list of taxa (column indices) it can map to.
    n_taxa : int
        Total number of taxa (columns in P, rows in G).
    lambda_var : float, default 1e-2
        Strength of the taxon-variance penalty.
    n_iter : int, default 500
    tol : float, default 1e-4
        Relative tolerance for early stopping.
    verbose : bool, default False
        If True, prints loss components every 10 iterations.
    random_state : int, default 0
        Seed for reproducible initialisation.

    Returns
    -------
    P : (n_peptides, n_taxa) ndarray
    G : (n_taxa, n_runs) ndarray
    """
    rng = np.random.default_rng(random_state)
    n_pep, n_run = I.shape

    # ---- initial P (row-stochastic) and G ----------------------------------
    P = np.zeros((n_pep, n_taxa), dtype=float)
    for i, taxa in enumerate(taxon_idx_by_peptide):
        if len(taxa) == 1:  # unique peptide
            P[i, taxa[0]] = 1.0
        else:  # shared peptide
            rand = rng.random(len(taxa))
            rand /= rand.sum()
            P[i, taxa] = rand
    G = rng.random((n_taxa, n_run))

    # mask for fixed (unique) rows
    unique_mask = P.sum(1) == 1.0

    prev_loss = np.inf
    for it in range(n_iter):
        # ---- update G (multiplicative) -----------------------------------
        num = P.T @ I
        den = P.T @ P @ G + 1e-12
        G *= num / den

        # variance-shrink step
        if lambda_var > 0.0:
            alpha = lambda_var / (1.0 + lambda_var)
            row_mean = G.mean(axis=1, keepdims=True)
            G = (1.0 - alpha) * G + alpha * row_mean
        G = np.maximum(G, 1e-12)

        # ---- update P (skip fixed rows) -----------------------------------
        num = I @ G.T
        den = P @ G @ G.T + 1e-12
        P[~unique_mask] *= num[~unique_mask] / den[~unique_mask]
        P[~unique_mask] /= P[~unique_mask].sum(axis=1, keepdims=True)

        # ---- loss & convergence -------------------------------------------
        recon = I - P @ G
        loss_rec = np.nanmean(recon**2)  # use nanmean if I has NaNs
        loss_var = np.mean((G - G.mean(1, keepdims=True)) ** 2)
        loss = loss_rec + lambda_var * loss_var

        if verbose and it % 10 == 0:
            print(f"{it:4d}  rec={loss_rec:9.3g}  var={loss_var:9.3g}")

        if abs(prev_loss - loss) < tol * prev_loss:
            if verbose:
                print(f"Converged at iter {it}, loss={loss:.4g}")
            break
        prev_loss = loss

    return P, G


def nmf_taxon_totals(P_hat: np.ndarray,
                     pt: pd.DataFrame,
                     peptides_used: list[str],
                     all_taxa: list[str],
                     use_scaled: bool):
    """
    Return (T_raw, T_prop):
      T_raw  : taxa × runs absolute sums
      T_prop : taxa × runs per-run proportions (columns sum to 1, unless all-zero)
    """
    runs = pt.columns

    if use_scaled:
        # NMF ran on per-run proportions → reconstruct proportions first
        col_sums = pt.sum(axis=0).astype(float)
        col_sums_safe = col_sums.replace(0.0, np.nan)

        pt_scaled = pt.div(col_sums_safe, axis=1)
        I = pt_scaled.loc[peptides_used].fillna(0.0).to_numpy(dtype=float)

        T_prop = pd.DataFrame(P_hat.T @ I, index=all_taxa, columns=runs)
        # back to absolute totals
        T_raw  = T_prop.mul(col_sums, axis=1)

    else:
        # NMF ran on raw intensities → totals first, proportions second
        I = pt.loc[peptides_used].fillna(0.0).to_numpy(dtype=float)

        T_raw = pd.DataFrame(P_hat.T @ I, index=all_taxa, columns=runs)

        denom = T_raw.sum(axis=0).replace(0.0, np.nan)
        T_prop = T_raw.div(denom, axis=1)

    return T_raw, T_prop


def nmf_weighting_pipeline(
    pt: pd.DataFrame,
    pep2orgs_clean: dict[str, list[str] | set[str]],
    *,
    method: str = "simple",          # "simple" -> nmf_row_stochastic, "totvar" -> nmf_row_stochastic_totvar_pg
    use_scaled: bool = False,        # if True, NMF runs on per-run proportions
    nmf_kwargs: dict | None = None,  # forwarded to the chosen NMF function
):
    """
    Convenience pipeline for NMF-based peptide→taxon weighting.

    Parameters
    ----------
    pt : DataFrame
        Peptide × run intensity table. Index = peptide IDs, columns = runs.
    pep2orgs_clean : dict
        Mapping peptide ID -> iterable of taxon names (organisms / clades).
        Only peptides present in `pt.index` and in this dict are used.
    method : {"simple", "totvar"}, default "simple"
        Which NMF variant to use:
        - "simple":  nmf_row_stochastic          (variance penalty on G rows, so no coupling between taxa in the penalty; only within each taxon across runs)
        - "totvar":  nmf_row_stochastic_totvar_pg (totals-variance penalty via PG)
    use_scaled : bool, default False
        If True, columns of `pt` are scaled to sum to 1 before NMF, so NMF
        sees per-run proportions. Taxon totals are then rescaled back to
        absolute intensities inside `nmf_taxon_totals`.
    nmf_kwargs : dict or None
        Extra keyword arguments forwarded to the chosen NMF function
        (e.g. lambda_var, n_iter, tol, lambda_tot, step_tot, etc.).

    Returns
    -------
    result : dict
        {
          "P":            peptide×taxon weight matrix (np.ndarray),
          "G":            taxon×run factor matrix (np.ndarray),
          "T_raw":        taxa×runs totals (DataFrame),
          "T_prop":       taxa×runs proportions (DataFrame),
          "peptides_used": list of peptides in P (row order),
          "taxa":         list of taxa in P (column order),
        }
    """
    if nmf_kwargs is None:
        nmf_kwargs = {}

    # 1) Build taxon index mapping in peptide-row order
    peptide_order = list(pt.index)
    peptides_used, all_taxa, taxon_idx_by_peptide = build_taxon_idx_by_peptide(
        pep2orgs_clean,
        peptide_order,
    )
    if not peptides_used:
        raise ValueError("No peptides found that have a taxon mapping and are present in `pt`.")

    n_taxa = len(all_taxa)

    # 2) Build the matrix fed into NMF: either raw intensities or per-run proportions
    if use_scaled:
        col_sums = pt.sum(axis=0).astype(float)
        col_sums_safe = col_sums.replace(0.0, np.nan)
        I_mat = (
            pt.div(col_sums_safe, axis=1)
              .loc[peptides_used]
              .fillna(0.0)
              .to_numpy(dtype=float)
        )
    else:
        I_mat = (
            pt.loc[peptides_used]
              .fillna(0.0)
              .to_numpy(dtype=float)
        )

    # 3) Run the chosen NMF variant
    method_norm = method.lower()
    if method_norm == "simple":
        P_hat, G_hat = nmf_row_stochastic(
            I_mat,
            taxon_idx_by_peptide=taxon_idx_by_peptide,
            n_taxa=n_taxa,
            **nmf_kwargs,
        )
    elif method_norm in ("totvar", "totvar_pg", "pg"):
        P_hat, G_hat = nmf_row_stochastic_totvar_pg(
            I_mat,
            taxon_idx_by_peptide=taxon_idx_by_peptide,
            n_taxa=n_taxa,
            **nmf_kwargs,
        )
    else:
        raise ValueError(f"Unknown NMF method '{method}'; use 'simple' or 'totvar'.")

    # 4) Aggregate to taxon×run totals and proportions
    T_raw, T_prop = nmf_taxon_totals(
        P_hat=P_hat,
        pt=pt,
        peptides_used=peptides_used,
        all_taxa=all_taxa,
        use_scaled=use_scaled,
    )

    return {
        "P": P_hat,
        "G": G_hat,
        "T_raw": T_raw,
        "T_prop": T_prop,
        "peptides_used": peptides_used,
        "taxa": all_taxa,
    }

# Usage:
# res = nmf_weighting_pipeline(
#     pt=peptide_run_df,               # peptides × runs
#     pep2orgs_clean=pep2orgs_clean,   # peptide -> {taxa}
#     method="totvar",                 # "simple" | "totvar"
#     use_scaled=False,
#     nmf_kwargs=dict(lambda_tot=5.0, n_iter=800, tol=1e-5, verbose=False),
# )
#
# P      = res["P"]        # peptide→taxon weights
# T_raw  = res["T_raw"]    # taxa×runs intensity
# T_prop = res["T_prop"]   # taxa×runs proportions