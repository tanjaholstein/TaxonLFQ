"""
Clade-based allocation of peptide signal to taxa.

Main entry points
-----------------
bifurcate(tree, seed=0)
    Randomly resolve multifurcations so the tree becomes strictly binary.

allocate_by_clade(tree, org_to_peps, pep2orgs, xic, intensity_col="median_area")
    Allocate peptide intensities directly to taxa (one total per taxon)
    by recursively splitting each internal node according to the unique
    signal in each child clade.

clade_weights(tree, pep2orgs, xic_df, intensity_col="median_area")
    Compute per-peptide weights across taxa using the same clade-split
    logic; weights for each peptide sum to 1 over its owning taxa.

totals_from_weights_loop(pt, weights_dict, taxa=None)
    Given a peptide×runs intensity table and clade weights, compute the
    taxa×runs intensity table (unique + appropriately weighted shared
    peptides).
"""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Dict, Iterable, Mapping, Set

import numpy as np
import pandas as pd
from Bio import Phylo


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def bifurcate(tree: Phylo.BaseTree.Tree, seed: int = 0) -> None:
    """
    In-place: randomly resolve all multifurcations (nodes with >2 children)
    into a binary tree.

    This doesn't change leaf labels, only how internal clades are grouped.
    Run this once on a tree to simplify clade-based allocation.
    """
    random.seed(seed)
    for clade in tree.find_clades(order="preorder"):
        # While there are more than two children, merge two of them under
        # a new internal node.
        while len(clade.clades) > 2:
            a, b = random.sample(clade.clades, 2)
            clade.clades.remove(a)
            clade.clades.remove(b)
            new = Phylo.BaseTree.Clade()
            new.clades.extend([a, b])
            clade.clades.append(new)


def _taxa_below(
    node: Phylo.BaseTree.Clade,
    cache: Dict[Phylo.BaseTree.Clade, Set[str]] | None = None
) -> Set[str]:
    """
    Return the set of leaf names (taxa) under a given node.

    Results are cached in `cache` (if provided), so you can reuse the
    same cache across multiple calls for efficiency.
    """
    if cache is None:
        cache = {}
    if node in cache:
        return cache[node]
    if node.is_terminal():
        s = {node.name}
    else:
        s = set().union(*(_taxa_below(ch, cache) for ch in node.clades))
    cache[node] = s
    return s


# ---------------------------------------------------------------------------
# 1. Direct intensity allocation per taxon
# ---------------------------------------------------------------------------

def allocate_by_clade(
    tree:          Phylo.BaseTree.Tree,
    org_to_peps:   Dict[str, Set[str]],
    pep2orgs:      Dict[str, Set[str]],
    xic:           pd.DataFrame,
    intensity_col: str = "median_area",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Allocate peptide intensities to taxa by recursively splitting each
    internal node according to the unique signal in each child clade.

    Parameters
    ----------
    tree
        Phylogenetic tree with leaf names matching taxon IDs in `pep2orgs`.
        Should be strictly binary; call `bifurcate(tree)` first if needed.
    org_to_peps
        Mapping taxon -> set of peptides observed in that taxon. Used only
        to restrict the peptide universe to those we care about.
    pep2orgs
        Mapping peptide -> set of owning taxa.
    xic
        DataFrame with at least columns ['sequence_naked', intensity_col].
        Intensities are taken from the specified column.
    intensity_col
        Column name in `xic` holding peptide intensities.
    verbose
        If True, prints a summary for each internal split.

    Returns
    -------
    allocation : dict
        {taxon: total_intensity} summed over all peptides.
    """
    # peptide → intensity
    I = dict(zip(xic["sequence_naked"], xic[intensity_col]))
    taxa_cache: Dict[Phylo.BaseTree.Clade, Set[str]] = {}

    def taxa_below(node: Phylo.BaseTree.Clade) -> Set[str]:
        return _taxa_below(node, taxa_cache)

    def recurse(node: Phylo.BaseTree.Clade,
                peptide_pool: Set[str]) -> Dict[str, float]:
        """
        Return a dict of intensity allocations for the sub-tree rooted at `node`.

        peptide_pool
            Peptides that still carry residual intensity and have at least one
            owner below this node.
        """
        if node.is_terminal():
            taxon = node.name
            gain  = sum(I[p] for p in peptide_pool if taxon in pep2orgs[p])
            return {taxon: gain}

        L, R = node.clades
        taxa_L = taxa_below(L)
        taxa_R = taxa_below(R)

        # Intensities of peptides whose owners are entirely in L or R
        U_L = sum(I[p] for p in peptide_pool if pep2orgs[p] <= taxa_L)
        U_R = sum(I[p] for p in peptide_pool if pep2orgs[p] <= taxa_R)

        # Shared peptides whose owners span both sides
        shared = {
            p for p in peptide_pool
            if pep2orgs[p] & taxa_L and pep2orgs[p] & taxa_R
        }

        w_L = U_L / (U_L + U_R) if (U_L + U_R) else 0.5
        w_R = 1.0 - w_L

        if verbose:
            print(
                f"Node split  |  U_L={U_L:.1f}  U_R={U_R:.1f}  "
                f"shared={sum(I[p] for p in shared):.1f}  w_L={w_L:.3f}"
            )

        shared_total = sum(I[p] for p in shared)
        total_L = U_L + w_L * shared_total
        total_R = U_R + w_R * shared_total

        # Recurse on peptides that touch each side
        pool_L = {p for p in peptide_pool if pep2orgs[p] & taxa_L}
        pool_R = {p for p in peptide_pool if pep2orgs[p] & taxa_R}

        alloc_L = recurse(L, pool_L)
        alloc_R = recurse(R, pool_R)

        # Rescale each side so its sum matches total_L / total_R
        def rescale(alloc: Dict[str, float], target: float) -> Dict[str, float]:
            s = sum(alloc.values())
            if s == 0.0:
                return alloc
            f = target / s
            return {k: v * f for k, v in alloc.items()}

        alloc_L = rescale(alloc_L, total_L)
        alloc_R = rescale(alloc_R, total_R)

        out: Dict[str, float] = defaultdict(float)
        for d in (alloc_L, alloc_R):
            for k, v in d.items():
                out[k] += v
        return dict(out)

    # Peptides that have at least one owner we care about and appear in xic
    observed_taxa = set(org_to_peps)
    all_peptides = {
        p for p, owners in pep2orgs.items()
        if (owners & observed_taxa) and p in I
    }

    allocation = recurse(tree.root, all_peptides)
    return dict(allocation)


# ---------------------------------------------------------------------------
# 2. Per-peptide clade weights (weights sum to 1 per peptide)
# ---------------------------------------------------------------------------

def clade_weights(
    tree: Phylo.BaseTree.Tree,
    pep2orgs: Dict[str, Set[str]],         # peptide → set of owning taxa
    xic_df: pd.DataFrame,                  # must have columns: 'sequence_naked', intensity_col
    intensity_col: str = "median_area",
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-peptide clade weights using the same clade-split logic.

    For each internal node:
      1. Compute U_child = sum of intensities of peptides unique to that
         child (owners ⊆ child taxa).
      2. Convert to child priors w_child = U_child / Σ U_child
         (fallback: equal weights if ΣU == 0).
      3. For peptides whose owners span several children, split their mass
         across children proportionally to w_child at that node.
      4. Recurse until leaves. Peptides unique to a leaf naturally end up
         with weight 1 on that leaf.

    Returns
    -------
    weights : dict
        { peptide: { taxon: weight, ... }, ... }
        For each peptide, weights over owning taxa sum (approximately) to 1.
    """
    # peptide → intensity (used only for U_child)
    I = dict(zip(xic_df["sequence_naked"], xic_df[intensity_col]))

    taxa_cache: Dict[Phylo.BaseTree.Clade, Set[str]] = {}

    def taxa_below(node: Phylo.BaseTree.Clade) -> Set[str]:
        return _taxa_below(node, taxa_cache)

    all_taxa = taxa_below(tree.root)

    # Peptides in play: mapped, owners intersect the tree, and present in xic_df
    peptides: list[str] = [
        p for p, owners in pep2orgs.items()
        if owners and (owners & all_taxa) and p in I
    ]

    # ---------- Precompute split weights for each internal node ----------
    node_info: Dict[Phylo.BaseTree.Clade, dict] = {}

    def prepare(node: Phylo.BaseTree.Clade, peptide_pool: Iterable[str]) -> None:
        """Fill node_info[node] with (children, taxa_below_each, w_child)."""
        if node.is_terminal():
            return

        children = list(node.clades)
        child_taxa = [taxa_below(ch) for ch in children]

        # U_child = sum intensities of peptides unique to that child
        U: list[float] = []
        for taxa in child_taxa:
            u = 0.0
            for p in peptide_pool:
                owners = pep2orgs[p]
                if owners <= taxa:        # owners ⊆ child taxa
                    u += float(I.get(p, 0.0))
            U.append(u)

        sumU = sum(U)
        if sumU > 0.0:
            w = [u / sumU for u in U]
        else:
            # No unique signal on this split → uninformative prior
            m = len(children)
            w = [1.0 / m] * m

        node_info[node] = dict(children=children, taxa=child_taxa, w=w)

        # Recurse into children with only the peptides that touch each child
        for ch, taxa in zip(children, child_taxa):
            child_pool = [p for p in peptide_pool if pep2orgs[p] & taxa]
            prepare(ch, child_pool)

    prepare(tree.root, peptides)

    # ---------- Push unit mass for each peptide down the tree ----------
    weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def distribute(node: Phylo.BaseTree.Clade,
                   mass: float,
                   owners: Set[str],
                   pep_id: str) -> None:
        """
        Push `mass` of peptide `pep_id` down the tree from `node`.

        At each internal node, split `mass` over the children that actually
        contain at least one owner, renormalising the precomputed child
        weights to that subset.
        """
        if node.is_terminal():
            t = node.name
            if t in owners:
                weights[pep_id][t] += mass
            return

        info      = node_info[node]
        children  = info["children"]
        child_tax = info["taxa"]
        child_w   = info["w"]   # prior weights for this split

        # Children that contain at least one owner
        idxs = [i for i, taxa in enumerate(child_tax) if owners & taxa]
        k = len(idxs)

        if k == 0:
            # No owners below this node (should not happen if inputs are consistent)
            return
        elif k == 1:
            # All owners live under a single child → no split at this node.
            i = idxs[0]
            distribute(children[i], mass, owners, pep_id)
        else:
            # Split mass over relevant children, renormalising w to this subset.
            w_sub = [child_w[i] for i in idxs]
            s = sum(w_sub)
            if s == 0.0:
                w_norm = [1.0 / k] * k
            else:
                w_norm = [x / s for x in w_sub]
            for i, frac in zip(idxs, w_norm):
                distribute(children[i], mass * frac, owners, pep_id)

    for p in peptides:
        distribute(tree.root, 1.0, pep2orgs[p], p)

    # Cast nested defaultdicts to plain dicts
    return {p: dict(tdict) for p, tdict in weights.items()}


# ---------------------------------------------------------------------------
# 3. Rebuild a taxa×runs table from clade weights
# ---------------------------------------------------------------------------

def totals_from_weights_loop(
    pt: pd.DataFrame,
    weights_dict: Dict[str, Dict[str, float]],
    taxa: list[str] | None = None,
) -> pd.DataFrame:
    """
    Sum unique + weighted shared peptide intensities into a taxa×runs table.

    Parameters
    ----------
    pt
        DataFrame of shape (peptides × runs). Index = peptide IDs, columns = runs,
        values = intensities (e.g. XIC areas).
    weights_dict
        { peptide: { taxon: weight, ... }, ... }  (rows sum to ~1),
        e.g. returned by `clade_weights`.
    taxa
        Optional list of taxa to include and define the output order.
        If None, taxa are inferred from `weights_dict`.

    Returns
    -------
    totals : DataFrame
        Index = taxa, columns = runs. Each cell is the sum over peptides of
        (weight[taxon] × peptide_intensity_in_run).
    """
    runs = list(pt.columns)
    if taxa is None:
        taxa = sorted({t for w in weights_dict.values() for t in w})

    totals = pd.DataFrame(0.0, index=taxa, columns=runs)

    # Only peptides that exist in pt and have weights
    peps = pt.index.intersection(weights_dict.keys())
    skipped = len(weights_dict) - len(peps)

    for pep in peps:
        intens = pt.loc[pep].astype(float).fillna(0.0)  # Series (runs)
        wd = weights_dict[pep]
        for taxon, w in wd.items():
            if taxon not in totals.index:
                # Optionally add unseen taxa on the fly
                totals.loc[taxon] = 0.0
            if w and np.isfinite(w):
                totals.loc[taxon] = totals.loc[taxon].add(w * intens, fill_value=0.0)

    if skipped:
        print(f"{skipped} peptides in weights_dict not found in pt (skipped).")

    return totals

__all__ = [
    "bifurcate",
    "allocate_by_clade",
    "clade_weights",
    "totals_from_weights_loop",
]