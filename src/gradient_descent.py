from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

__all__ = [
    "proj_simplex",
    "prepare_gd_design",
    "run_gd_optimization",
    "weights_from_solution",
    "compute_taxon_totals",
    "gradient_descent_weighting_pipeline",
]


# ----------------------------------------------------------------------
# 0. Simplex projection
# ----------------------------------------------------------------------
def proj_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a 1D array v onto the probability simplex:

        { w >= 0,  sum(w) = 1 }

    using the standard Euclidean projection (Duchi et al., 2008).

    Parameters
    ----------
    v : 1D ndarray

    Returns
    -------
    w : 1D ndarray
        Projected vector.
    """
    v = np.asarray(v, float)
    if v.ndim != 1:
        raise ValueError("proj_simplex expects a 1D array.")

    s = v.sum()
    if s == 1.0 and np.all(v >= 0):
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w


# ----------------------------------------------------------------------
# 1. Build design matrices for GD optimisation
# ----------------------------------------------------------------------
def prepare_gd_design(
    pt: pd.DataFrame,
    pep2orgs: dict[str, set[str] | list[str]],
    *,
    pre_scale: bool = False,
    verbose: bool = False,
):
    """
    Prepare all structures needed for the GD-based weighting.

    Parameters
    ----------
    pt : DataFrame
        Peptide × run intensity table. Index = peptide IDs, columns = runs.
    pep2orgs : dict
        Mapping peptide -> iterable of taxa (e.g. organisms).
    pre_scale : bool, default False
        If True, each run (column) is scaled to sum to 1 before optimisation.
        If False, raw intensities are used.
    verbose : bool, default False
        If True, prints some basic size information.

    Returns
    -------
    I_tab : DataFrame
        Peptide × run intensities actually used for optimisation
        (optionally column-scaled).
    runs : Index
        Run labels (columns of pt).
    taxa : list[str]
        Sorted list of taxa.
    pep2taxa_used : dict[str, set[str]]
        Cleaned mapping peptide -> set of taxa (only peptides present in pt
        with any positive intensity are retained).
    shared_peps : list[str]
        Peptides mapping to ≥2 taxa.
    unique_peps : list[str]
        Peptides mapping to exactly 1 taxon.
    col_idx : dict[(str, str), int]
        Mapping (peptide, taxon) -> column index in the weight vector w
        (only for shared peptide–taxon pairs).
    B_run : dict[str, csr_matrix]
        Per-run sparse design matrices (shape = n_taxa × n_vars) for shared
        peptides only.
    U_run : dict[str, np.ndarray]
        Per-run vectors of unique-peptide totals, shape = (n_taxa,).
    peptides_used : list[str]
        Peptides actually used by the optimisation (shared + unique),
        in the same order as they appear in pt.index.
    """
    # Runs (columns) and taxa (union over mapping)
    runs = pt.columns
    taxa = sorted({t for lst in pep2orgs.values() for t in lst})
    taxon_index = {t: i for i, t in enumerate(taxa)}
    n_taxa = len(taxa)

    # Intensities used internally
    if pre_scale:
        # per-run scaling to total = 1
        I_tab = pt.div(pt.sum(axis=0), axis=1)
    else:
        I_tab = pt.copy()

    # Keep only peptides present in pt AND with at least some positive signal
    pep2taxa_used: dict[str, set[str]] = {}
    shared_peps: list[str] = []
    unique_peps: list[str] = []

    for pep in pt.index:
        if pep not in pep2orgs:
            continue
        intens = I_tab.loc[pep, runs]
        if not np.isfinite(intens).any() or not (intens > 0).any():
            continue

        taxa_set = set(pep2orgs[pep])
        if len(taxa_set) == 0:
            continue
        pep2taxa_used[pep] = taxa_set
        if len(taxa_set) == 1:
            unique_peps.append(pep)
        elif len(taxa_set) >= 2:
            shared_peps.append(pep)

    # Deterministic column indexing for shared peptide–taxon pairs
    col_idx: dict[tuple[str, str], int] = {}
    for pep in shared_peps:
        for t in sorted(pep2taxa_used[pep]):
            col_idx[(pep, t)] = len(col_idx)
    n_vars = len(col_idx)

    # Build per-run B and U
    B_run: dict[str, csr_matrix] = {}
    U_run: dict[str, np.ndarray] = {}

    for run in runs:
        data: list[float] = []
        rows: list[int] = []
        cols: list[int] = []
        u_vec = np.zeros(n_taxa, dtype=float)

        # shared peptides → go into B_run
        for pep in shared_peps:
            I_val = I_tab.at[pep, run]
            if not np.isfinite(I_val) or I_val <= 0:
                continue
            for t in sorted(pep2taxa_used[pep]):
                rows.append(taxon_index[t])
                cols.append(col_idx[(pep, t)])
                data.append(float(I_val))

        # unique peptides → go into U_run
        for pep in unique_peps:
            I_val = I_tab.at[pep, run]
            if not np.isfinite(I_val) or I_val <= 0:
                continue
            t = next(iter(pep2taxa_used[pep]))
            u_vec[taxon_index[t]] += float(I_val)

        B_run[run] = csr_matrix((data, (rows, cols)), shape=(n_taxa, n_vars))
        U_run[run] = u_vec

    if verbose:
        nnz_total = sum(mat.nnz for mat in B_run.values())
        print(f"[GD] n_taxa = {n_taxa}, n_vars (shared weights) = {n_vars}")
        print(f"[GD] total non-zeros in B blocks = {nnz_total:,}")

    # Peptides actually used, in pt-index order
    peptides_used = [p for p in pt.index if p in pep2taxa_used]

    return (
        I_tab,
        runs,
        taxa,
        pep2taxa_used,
        shared_peps,
        unique_peps,
        col_idx,
        B_run,
        U_run,
        peptides_used,
    )


# ----------------------------------------------------------------------
# 2. Gradient-descent optimisation of shared-peptide weights
# ----------------------------------------------------------------------
def run_gd_optimization(
    B_run: dict[str, csr_matrix],
    U_run: dict[str, np.ndarray],
    shared_peps: list[str],
    pep2taxa_used: dict[str, set[str]],
    col_idx: dict[tuple[str, str], int],
    runs,
    taxa: list[str],
    *,
    ref_value: float | None = None,
    max_iters: int = 4000,
    lr0: float = 3e-2,
    check_every: int = 200,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    seed: int = 13,
    verbose: bool = True,
):
    """
    Run Adam-style gradient descent on the shared-peptide weights.

    Objective:
        minimize run-to-run variance of *shared* taxon totals S (taxa × runs):

            Φ = (1/R) ∑_r ∑_t (S[t,r] - mean_t)^2

        where S[t,r] = (B_run[r] @ w)[t].

    A diagnostic RMSE against a scalar reference composition can be recorded:

        - Compute total taxon intensities T[t,r] = S[t,r] + unique[t,r]
        - Convert each column to proportions
        - Take median proportion per taxon across runs
        - RMSE vs `ref_value` (if provided)

    Parameters
    ----------
    B_run, U_run : dict
        From `prepare_gd_design`.
    shared_peps : list[str]
        List of shared peptides.
    pep2taxa_used : dict[str, set[str]]
        Cleaned peptide → taxa mapping.
    col_idx : dict[(peptide, taxon), int]
        Mapping into the weight vector.
    runs : sequence
        Run labels (same keys as B_run/U_run).
    taxa : list[str]
        Taxon labels (for diagnostics).
    ref_value : float or None, default None
        Scalar reference composition (e.g. 1/T). If None, RMSE is recorded
        as NaN and not used for stopping.
    max_iters : int, default 4000
    lr0 : float, default 3e-2
        Base learning rate; decays as lr0 / sqrt(iter).
    check_every : int, default 200
        Iteration interval for verbose logging and one of the convergence
        checks (relative Φ change).
    beta1, beta2, eps : Adam hyperparameters.
    seed : int, default 13
    verbose : bool, default True

    Returns
    -------
    w : 1D ndarray
        Optimised shared-peptide weight vector (length = n_vars).
    history : dict
        {
          "phi":  list of Φ per iteration,
          "rmse": list of RMSE per iteration (NaN if ref_value is None),
        }
    """
    rng = np.random.default_rng(seed)
    n_taxa = len(taxa)

    # number of weight variables = number of columns in any B_run
    any_run = next(iter(B_run.keys()))
    n_vars = B_run[any_run].shape[1]

    # Precompute column indices per shared peptide
    pep_cols: dict[str, np.ndarray] = {
        pep: np.array(
            [col_idx[(pep, t)] for t in sorted(pep2taxa_used[pep])],
            dtype=int,
        )
        for pep in shared_peps
    }

    # --- initial weights: random simplex per peptide -----------------------
    w = np.zeros(n_vars, dtype=float)
    for pep in shared_peps:
        cols = pep_cols[pep]
        random_vec = rng.random(len(cols))
        w[cols] = proj_simplex(random_vec)

    # Adam buffers
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    history_phi: list[float] = []
    history_rmse: list[float] = []
    grad_inf_0 = None

    R = len(runs)

    for it in range(1, max_iters + 1):
        # ---------- shared-only totals ----------
        Smat = np.column_stack([B_run[run] @ w for run in runs])  # n_taxa × R

        # ---------- variance objective Φ ----------
        row_mean = Smat.mean(axis=1, keepdims=True)
        phi = ((Smat - row_mean) ** 2).mean() * n_taxa  # same as your script

        # ---------- gradient wrt w ----------
        grad_w = np.zeros_like(w)
        for r_idx, run in enumerate(runs):
            mat = B_run[run]
            grad_T = 2.0 * (Smat[:, r_idx] - row_mean[:, 0]) / R
            grad_w += mat.T @ grad_T

        grad_inf = float(np.max(np.abs(grad_w)))
        if grad_inf_0 is None:
            grad_inf_0 = grad_inf

        # ---------- Adam update ----------
        lr = lr0 / np.sqrt(it)
        m = beta1 * m + (1.0 - beta1) * grad_w
        v = beta2 * v + (1.0 - beta2) * (grad_w ** 2)
        m_hat = m / (1.0 - beta1 ** it)
        v_hat = v / (1.0 - beta2 ** it)
        w_new = w - lr * m_hat / (np.sqrt(v_hat) + eps)

        # ---------- simplex projection per peptide ----------
        for pep in shared_peps:
            cols = pep_cols[pep]
            w_new[cols] = proj_simplex(w_new[cols])

        # ---------- diagnostics: add uniques, compute RMSE ----------
        Tmat = Smat + np.column_stack([U_run[run] for run in runs])  # n_taxa × R

        if ref_value is not None:
            col_sums = Tmat.sum(axis=0, keepdims=True)
            # avoid division by zero
            col_sums[col_sums == 0] = np.nan
            props = Tmat / col_sums
            median_per_taxon = np.nanmedian(props, axis=1)
            rmse = float(np.sqrt(((median_per_taxon - ref_value) ** 2).mean()))
        else:
            rmse = np.nan

        history_phi.append(float(phi))
        history_rmse.append(rmse)

        # ---------- convergence tests ----------
        dw = float(np.max(np.abs(w_new - w)))
        if it > check_every:
            dphi = abs(history_phi[-1] - history_phi[-check_every]) / max(
                abs(history_phi[-1]), 1e-12
            )
        else:
            dphi = np.inf

        if ref_value is not None and it > 1 and np.all(
            np.isfinite(history_rmse[-2:])
        ):
            drmse = abs(history_rmse[-1] - history_rmse[-2])
        else:
            drmse = np.inf

        if verbose and (it % check_every == 0 or it == 1):
            print(
                f"iter {it:4d}  Φ = {phi:.4e}  RMSE = {rmse:.5f}  "
                f"∥∇Φ∥∞ = {grad_inf:.2e}  Δw∞ = {dw:.2e}"
            )

        w[:] = w_new  # commit

        stop = (
            grad_inf < 1e-4 * grad_inf_0
            and dphi < 1e-5
            and drmse < 1e-4
            and dw < 1e-4
        )

        if stop:
            if verbose:
                print(f"\n[GD] converged at iter {it}")
            break
    else:
        if verbose:
            print("\n[GD] maximum iterations reached without convergence.")

    if verbose and np.isfinite(history_rmse[-1]):
        print(f"\n[GD] Final RMSE: {history_rmse[-1]:.5f}")

    history = {"phi": history_phi, "rmse": history_rmse}
    return w, history


# ----------------------------------------------------------------------
# 3. Convert solution to peptide→taxon weights & taxon totals
# ----------------------------------------------------------------------
def weights_from_solution(
    w: np.ndarray,
    shared_peps: list[str],
    unique_peps: list[str],
    pep2taxa_used: dict[str, set[str]],
    col_idx: dict[tuple[str, str], int],
) -> dict[str, dict[str, float]]:
    """
    Build a peptide→taxon weight dictionary from the optimised weight vector.

    Parameters
    ----------
    w : 1D ndarray
        Optimised weight vector (length = number of shared peptide–taxon pairs).
    shared_peps, unique_peps : list[str]
        Peptides partitioned by multiplicity.
    pep2taxa_used : dict[str, set[str]]
        Cleaned mapping peptide -> set of taxa.
    col_idx : dict[(peptide, taxon), int]
        Same mapping that was used to build B_run.

    Returns
    -------
    weights_dict : dict
        { peptide: { taxon: weight, ... }, ... }.
        Shared-peptide weights are normalised to sum to 1 across owning taxa.
        Unique peptides get weight 1.0 on their single taxon.
    """
    w = np.asarray(w, float)
    weights_dict: dict[str, dict[str, float]] = {}

    # Shared peptides
    for pep in shared_peps:
        taxa_sorted = sorted(pep2taxa_used[pep])
        cols = [col_idx[(pep, t)] for t in taxa_sorted]
        w_vec = np.asarray([w[c] for c in cols], dtype=float)

        s = w_vec.sum()
        if not np.isfinite(s) or s <= 0:
            w_vec = np.ones_like(w_vec) / len(w_vec)
        else:
            w_vec = w_vec / s

        weights_dict[pep] = {t: float(v) for t, v in zip(taxa_sorted, w_vec)}

    # Unique peptides
    for pep in unique_peps:
        t = next(iter(pep2taxa_used[pep]))
        weights_dict[pep] = {t: 1.0}

    return weights_dict


def compute_taxon_totals(
    w: np.ndarray,
    B_run: dict[str, csr_matrix],
    U_run: dict[str, np.ndarray],
    runs,
    taxa: list[str],
) -> pd.DataFrame:
    """
    Compute final taxon × run totals from shared weights + unique intensities.

    Parameters
    ----------
    w : 1D ndarray
        Optimised shared-peptide weights.
    B_run, U_run : dict
        From `prepare_gd_design`.
    runs : sequence
        Run labels.
    taxa : list[str]
        Taxon labels.

    Returns
    -------
    taxon_totals : DataFrame
        Index = taxa, columns = runs.
    """
    Smat = np.column_stack([B_run[run] @ w for run in runs])
    Tmat = Smat + np.column_stack([U_run[run] for run in runs])
    taxon_totals = pd.DataFrame(Tmat, index=taxa, columns=runs)
    return taxon_totals


# ----------------------------------------------------------------------
# 4. High-level pipeline
# ----------------------------------------------------------------------
def gradient_descent_weighting_pipeline(
    pt: pd.DataFrame,
    pep2orgs_clean: dict[str, set[str] | list[str]],
    *,
    pre_scale: bool = False,
    ref_value: float | None = None,
    max_iters: int = 4000,
    lr0: float = 3e-2,
    check_every: int = 200,
    seed: int = 13,
    verbose: bool = True,
):
    """
    High-level convenience function to run the GD-based weighting.

    Parameters
    ----------
    pt : DataFrame
        Peptide × run intensity table (raw or pre-normalised).
    pep2orgs_clean : dict
        Mapping peptide -> iterable of taxa.
    pre_scale : bool, default False
        If True, each run is scaled to sum to 1 before optimisation.
    ref_value : float or None, default None
        Scalar reference composition for the diagnostic RMSE
        (e.g. 1 / number_of_taxa). If None, RMSE is recorded but not used
        for stopping.
    max_iters, lr0, check_every, seed, verbose :
        Passed to `run_gd_optimization`.

    Returns
    -------
    result : dict
        {
          "pt_used":         peptide×run DataFrame used internally
                             (possibly column-scaled and filtered to peptides
                             with mappings and signal),
          "weights":         { peptide: { taxon: weight } } dict,
          "taxon_totals":    taxa×runs DataFrame,
          "runs":            list of runs,
          "taxa":            list of taxa,
          "shared_peptides": list of shared peptides,
          "unique_peptides": list of unique peptides,
          "history":         {"phi": [...], "rmse": [...]},
        }
    """
    (
        I_tab,
        runs,
        taxa,
        pep2taxa_used,
        shared_peps,
        unique_peps,
        col_idx,
        B_run,
        U_run,
        peptides_used,
    ) = prepare_gd_design(pt, pep2orgs_clean, pre_scale=pre_scale, verbose=verbose)

    w, history = run_gd_optimization(
        B_run=B_run,
        U_run=U_run,
        shared_peps=shared_peps,
        pep2taxa_used=pep2taxa_used,
        col_idx=col_idx,
        runs=runs,
        taxa=taxa,
        ref_value=ref_value,
        max_iters=max_iters,
        lr0=lr0,
        check_every=check_every,
        seed=seed,
        verbose=verbose,
    )

    weights_dict = weights_from_solution(
        w=w,
        shared_peps=shared_peps,
        unique_peps=unique_peps,
        pep2taxa_used=pep2taxa_used,
        col_idx=col_idx,
    )

    taxon_totals = compute_taxon_totals(
        w=w,
        B_run=B_run,
        U_run=U_run,
        runs=runs,
        taxa=taxa,
    )

    pt_used = I_tab.loc[peptides_used].copy()

    return {
        "pt_used": pt_used,                # peptide × run intensities used
        "weights": weights_dict,          # peptide → taxon weights
        "taxon_totals": taxon_totals,     # taxa × runs
        "runs": list(runs),
        "taxa": taxa,
        "shared_peptides": shared_peps,
        "unique_peptides": unique_peps,
        "history": history,
    }

# Usage:

# from gd_weighting import gradient_descent_weighting_pipeline

# res = gradient_descent_weighting_pipeline(
#     pt=pt_normed_linear,
#     pep2orgs_clean=pep2orgs_clean,
#     pre_scale=False,               # use True if you want per-run normalization
#     ref_value=None,                # or e.g. 1.0 / len(set.union(*pep2orgs_clean.values()))
#     max_iters=4000,
#     lr0=3e-2,
#     check_every=200,
#     seed=13,
#     verbose=True,
# )

# # Peptide × run table actually used in optimisation
# pt_gd = res["pt_used"]

# # Peptide → taxon weights
# weights_GD = res["weights"]

# # Taxon × run totals implied by these weights (shared + unique)
# taxon_totals_GD = res["taxon_totals"]

# # If you want to reuse your existing totals_from_weights_loop:
# # totals_again = totals_from_weights_loop(pt_gd, weights_GD)
