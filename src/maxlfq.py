"""
Core MaxLFQ math extracted from normalization_in_ratio_space.py.

Only the two functions needed by TaxonLFQ are included here so that
the rest of normalization_in_ratio_space.py (which depends on alphapept)
does not need to be imported.
"""

import warnings

import numpy as np
from numba import njit
from scipy.optimize import least_squares, minimize


@njit
def get_protein_ratios(signal: np.ndarray, column_combinations: list, minimum_ratios: int = 1) -> np.ndarray:
    """Calculates the protein ratios between samples for one protein.

    Args:
        signal (np.ndarray): np.array[:,:] containing peptide intensities for each sample.
        column_combinations (list): list of all index combinations to compare (usually all sample combinations).
        minimum_ratios (int, optional): minimum number of peptide ratios necessary to calculate a protein ratio. Defaults to 1.

    Returns:
        np.ndarray: np.array[:,:] matrix comparing the ratios for all column combinations.
    """
    n_samples = signal.shape[1]
    ratios = np.empty((n_samples, n_samples))
    ratios[:] = np.nan

    for element in column_combinations:
        i = element[0]
        j = element[1]

        ratio = signal[:, j] / signal[:, i]

        non_nan = np.sum(~np.isnan(ratio))

        if non_nan >= minimum_ratios:
            ratio_median = np.nanmedian(ratio)
        else:
            ratio_median = np.nan

        ratios[j, i] = ratio_median

    return ratios


@njit
def triangle_error(normalization: np.ndarray, ratios: np.ndarray) -> float:
    """Calculates the difference between calculated ratios and expected ratios.

    Args:
        normalization (np.ndarray): Used normalization.
        ratios (np.ndarray): Peptide ratios.

    Returns:
        float: summed quadratic difference.
    """
    int_matrix = np.repeat(normalization, len(normalization)).reshape(
        (len(normalization), len(normalization))
    ).transpose()
    x = (np.log(ratios) - np.log(int_matrix.T) + np.log(int_matrix)) ** 2
    return np.nansum(x)


def solve_profile(ratios: np.ndarray, method: str) -> tuple[np.ndarray, bool]:
    """Calculates protein pseudointensities with a specified solver.

    Args:
        ratios (np.ndarray): np.array[:,:] matrix containing all estimated protein ratios between samples.
        method (str): string specifying which solver to use.

    Raises:
        NotImplementedError: if the solver is not implemented.

    Returns:
        tuple[np.ndarray, bool]: the protein pseudointensities and a success flag.
    """
    if method not in ["L-BFGS-B", "SLSQP", "Powell", "trust-constr", "trf"]:
        raise NotImplementedError(method)

    x0 = np.ones(ratios.shape[1])
    bounds = [(min(np.nanmin(ratios), 1 / np.nanmax(ratios)), 1) for _ in x0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if method == "trf":
            bounds = (x0 * 0 + 0.01, x0)
            res_wrapped = least_squares(
                triangle_error, args=[ratios], x0=x0, bounds=bounds, verbose=0, method="trf"
            )
        else:
            ncor = max(20, int(2 * np.ceil(np.sqrt(ratios.shape[0]))))
            if method == "L-BFGS-B":
                options = {
                    "maxiter": int(1e6),
                    "maxfun": int(ratios.shape[0] * 2e4),
                    "eps": 1e-06,
                    "maxcor": ncor,
                }
            elif method == "SLSQP":
                options = {"maxiter": int(1e6)}
            else:
                options = {}

            res_wrapped = minimize(
                triangle_error, args=ratios, x0=x0, bounds=bounds, method=method, options=options
            )

    solution = res_wrapped.x / np.max(res_wrapped.x)
    return solution, res_wrapped.success
