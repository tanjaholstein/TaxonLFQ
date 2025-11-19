# Okay, apparently I modified these because I had issues with some modules. Some of these were called differently, like:
# tracefilter.exclude_unconnected_samples

import numpy as np
from numba import njit

def exclude_unconnected_samples(distance_matrix): #ensures that every sample in the matrix is connected to every other sample at arbitrary degree
    if distance_matrix.shape[0] < 2:
        return
    unconnected_sample_idxs = get_unconnected_sample_idxs(distance_matrix)
    distance_matrix[unconnected_sample_idxs, :] = np.inf
    distance_matrix[:, unconnected_sample_idxs] = np.inf

def get_unconnected_sample_idxs(lower_matrix):
    full_matrix = convert_lower_to_full_matrix(lower_matrix)
    sums = np.sum(np.isfinite(full_matrix), axis=1)
    starting_sample = np.argmax(sums) #one of the traces with the most neighbors
    num_samples = full_matrix.shape[0]

    connected_samples = np.zeros(num_samples, dtype=np.bool_)
    check_connected_traces(full_matrix, starting_sample, connected_samples) #fills the set
    unconnected_samples = np.where(~connected_samples)[0]
    return unconnected_samples


def convert_lower_to_full_matrix(lower_matrix):
    full_matrix = np.copy(lower_matrix)
    rows, cols = np.where((lower_matrix != np.inf))
    full_matrix[cols, rows] = lower_matrix[rows, cols]
    return full_matrix

@njit
def check_connected_traces(matrix, trace_idx, visited):
    neighbors = np.where(matrix[trace_idx] != np.inf)[0]
    for neighbor in neighbors:
        if not visited[neighbor]:
            visited[neighbor] = True
            check_connected_traces(matrix, neighbor, visited)

# %% ../nbdev_nbs/02_normalization.ipynb 16

import numpy as np

def merge_distribs(anchor_distrib, shifted_distrib, count_anchor, count_shifted):
    """
    Merge two peptide-intensity distributions (e.g., log2 vectors) by position.

    Rules:
      - both NaN           → NaN
      - only anchor valid  → anchor value
      - only shifted valid → shifted value
      - both valid         → weighted average by (count_anchor, count_shifted)

    Parameters
    ----------
    anchor_distrib : array-like
    shifted_distrib : array-like
    count_anchor : float or int
    count_shifted : float or int

    Returns
    -------
    np.ndarray
        Merged distribution (float), same shape as inputs.
    """
    a = np.asarray(anchor_distrib, dtype=float)
    b = np.asarray(shifted_distrib, dtype=float)
    res = np.empty_like(a)

    na = np.isnan(a)
    nb = np.isnan(b)

    both_nan  = na & nb
    only_a    = (~na) & nb
    only_b    = na & (~nb)
    both_good = (~na) & (~nb)

    res[both_nan] = np.nan
    res[only_a]   = a[only_a]
    res[only_b]   = b[only_b]

    denom = (count_anchor + count_shifted)
    if denom == 0:
        # fallback: simple mean if no weights provided
        res[both_good] = 0.5 * (a[both_good] + b[both_good])
    else:
        res[both_good] = (a[both_good] * count_anchor + b[both_good] * count_shifted) / denom

    return res

def get_normfacts(samples):##row is the sample column is the features

    "finds optimal scaling factors for samples measured in the same condition and corrects the samples by these scaling factors. Takes a 2d numpy array as input  "
    set_samples_with_only_single_intensity_to_nan(samples)
    num_samples = samples.shape[0]
    mergedsamples = np.copy(samples) #the virtual "merged" samples will be stored in this array
    sampleidx2shift = dict(zip(range(num_samples), np.zeros(num_samples))) #the scaling factors applied to the samples are stored here
    sampleidx2counts = dict(zip(range(num_samples), np.ones(num_samples)))#keeps track of how many distributions are merged
    sampleidx2anchoridx = {} #keeps track of the shifted samples
    exclusion_set = set() #already clustered samples are stored here
    distance_matrix = create_distance_matrix(samples)
    variance_matrix = create_distance_matrix(samples, metric = 'variance')
    exclude_unconnected_samples(distance_matrix)
    exclude_unconnected_samples(variance_matrix)
    #print(f"distance matrix start\n{distance_matrix}")

    for rep in range(num_samples-1):
        #anchor_idx, shift_idx, min_distance = get_bestmatch_pair(mergedsamples, exclusion_set, sampleidx2counts)
        anchor_idx, shift_idx, min_distance = get_bestmatch_pair(distance_matrix,variance_matrix, sampleidx2counts)
        
        # #determine the closest pair of samples (one "shift" sample to be shifted and one "anchor sample which stays the same") and the distance between this pair
        #update the sets

        if(anchor_idx == None):
            break
        sampleidx2anchoridx.update({shift_idx : anchor_idx})
        sampleidx2shift.update({shift_idx : min_distance })
        exclusion_set.add(shift_idx)

        anchor_sample = mergedsamples[anchor_idx]
        shift_sample = samples[shift_idx]
        shifted_sample = shift_sample + min_distance
        
        merged_sample = merge_distribs(anchor_sample, shifted_sample, sampleidx2counts[anchor_idx], sampleidx2counts[shift_idx])
        mergedsamples[anchor_idx] = merged_sample


        update_distance_matrix(variance_matrix, mergedsamples, anchor_idx, shift_idx, metric='variance')
        update_distance_matrix(distance_matrix, mergedsamples, anchor_idx, shift_idx)

        #print(f"distance matrix after\n{distance_matrix}")
        sampleidx2counts[anchor_idx]+=1

    sampleidx2totalshift = {}
    for i in exclusion_set:
        shift = get_total_shift(sampleidx2anchoridx, sampleidx2shift, i)
        sampleidx2totalshift[i] = shift
        #samples[i] = samples[i]+shift
    return sampleidx2totalshift
    #return samples

def set_samples_with_only_single_intensity_to_nan(samples):
    for idx in range(len(samples)):
        sample = samples[idx]
        if sum(~np.isnan(sample)) <2:
            sample[:] = np.nan
            

def apply_sampleshifts(samples, sampleidx2shift):
    for idx in sampleidx2shift.keys():
        samples[idx] = samples[idx] + sampleidx2shift.get(idx)
    return samples

# %% ../nbdev_nbs/02_normalization.ipynb 6
def get_bestmatch_pair(distance_matrix, variance_matrix, sample2counts):
    
    i,j = np.unravel_index(np.argmin(variance_matrix, axis=None), variance_matrix.shape)
    min_distance = distance_matrix[i,j]
    #print(f"idxs are {i}, {j} median is {distance_matrix[i][j]} variance is {variance_matrix[i][j]}")
    if(min_distance == np.inf):
        return None, None, None
    anchor_idx, shift_idx, min_distance = determine_anchor_and_shift_sample(sample2counts,i, j, min_distance) #direction flip of distance if necessary
    return anchor_idx, shift_idx, min_distance

# %% ../nbdev_nbs/02_normalization.ipynb 7
def create_distance_matrix(samples, metric = 'median'):
    num_samples = samples.shape[0]
    distance_matrix = np.full((num_samples, num_samples), np.inf)
    for i in range(num_samples):
        for j in range(i+1, num_samples):#do every comparison once
            distance_matrix[i,j] = calc_distance(metric, samples[i], samples[j]) #the median of the shifted distribution is taken as the distance measure
            
    return distance_matrix

# %% ../nbdev_nbs/02_normalization.ipynb 8
from numba import njit

def calc_distance(metric, samples_1, samples_2):
    res = None

    if metric == 'median':
        res = calc_nanmedian(get_fcdistrib(samples_1, samples_2))#the median of the shifted distribution is taken as the distance measure
    if(metric == 'variance'):
        fcdist = get_fcdistrib(samples_1, samples_2)
        #if sum(~np.isnan(fcdist))<3:
         #   return 1000.0
        res = calc_nanvar(fcdist)
    if metric == 'overlap':
        fcdist = get_fcdistrib(samples_1, samples_2)
        res = sum(~np.isnan(fcdist))
    if res == None:
        raise Exception(f"distance metric {metric} not implemented")
    if(np.isnan(res)):
        return np.inf
    else:
        return res

@njit
def calc_nanvar(fcdist):
    return np.nanvar(fcdist)

@njit
def calc_nanmedian(fcdist):
    return np.nanmedian(fcdist)

# %% ../nbdev_nbs/02_normalization.ipynb 9
def update_distance_matrix(distance_matrix, merged_samples, merged_sample_idx, shift_idx,metric ='median'):
    "determine the distances to the newly merged sample"
    for i in range(0, merged_sample_idx):#update rows of distance matrix
        if distance_matrix[i, merged_sample_idx]==np.inf:#do not compare already merged samples
            continue
        distance = calc_distance(metric,merged_samples[i], merged_samples[merged_sample_idx])
        distance_matrix[i, merged_sample_idx] = distance
    
    for j in range(merged_sample_idx+1, merged_samples.shape[0]):#update columns of distance matrix
        if distance_matrix[merged_sample_idx, j] == np.inf:
            continue
        distance = calc_distance(metric,merged_samples[merged_sample_idx], merged_samples[j])
        distance_matrix[merged_sample_idx, j] = distance
    
    distance_matrix[shift_idx] = np.inf #shifted samples are excluded by setting distance to infinity
    distance_matrix[:, shift_idx] = np.inf
        

# %% ../nbdev_nbs/02_normalization.ipynb 10
def get_fcdistrib(logvals_rep1, logvals_rep2):
    "generates difference distribution between two samples"
    dist = np.subtract(logvals_rep1, logvals_rep2)
    return dist

# %% ../nbdev_nbs/02_normalization.ipynb 11
def determine_anchor_and_shift_sample(sample2counts, i_min, j_min, min_distance):
    "given two samples, declare the sample with fewer merges as the shift"
    counts_i = sample2counts[i_min]
    counts_j = sample2counts[j_min]
    anchor_idx = i_min if counts_i>=counts_j else j_min
    shift_idx = j_min if anchor_idx == i_min else i_min
    flip = 1 if anchor_idx == i_min else -1
    return anchor_idx, shift_idx, flip*min_distance

# %% ../nbdev_nbs/02_normalization.ipynb 13
def shift_samples(samples, sampleidx2anchoridx, sample2shift):
    for sample_idx in range(samples.shape[0]):
        samples[sample_idx] = samples[sample_idx]+get_total_shift(sampleidx2anchoridx, sample2shift, sample_idx)

# %% ../nbdev_nbs/02_normalization.ipynb 14
def get_total_shift(sampleidx2anchoridx, sample2shift,sample_idx):

    total_shift = 0.0

    while(True):
        total_shift +=sample2shift[sample_idx]
        if sample_idx not in sampleidx2anchoridx: #every shifted sample has an anchor
            break
        sample_idx = sampleidx2anchoridx[sample_idx]

    return total_shift

import numpy as np
import pandas as pd

def normalize_pt_log2_linear(
    pt: pd.DataFrame,
    *,
    replace_zeros: bool = True,
    return_shifts: bool = False,
):
    """
    Run-level normalization wrapper (no timing).

    Parameters
    ----------
    pt : DataFrame
        Peptide × run intensity table (linear scale). Rows = peptides, columns = runs.
    replace_zeros : bool, default True
        Replace 0 with NaN before log2 to avoid -inf.
    return_shifts : bool, default False
        Also return the per-run additive shifts (log2) produced by get_normfacts().

    Returns
    -------
    pt_normed_log2 : DataFrame
        Normalized log2 intensities (same shape as `pt`).
    pt_normed_linear : DataFrame
        Normalized linear intensities (same shape as `pt`).
    shifts (optional) : array-like or dict
        Per-run additive log2 shifts as returned by `get_normfacts`, if return_shifts=True.
    """
    # 0) prep: cast to float, handle zeros, log2, orient as samples × features
    if replace_zeros:
        pt_log2 = np.log2(pt.astype(float).replace(0, np.nan))
    else:
        pt_log2 = np.log2(pt.astype(float))

    samples_x_features = pt_log2.T.to_numpy()  # rows = runs, cols = peptides

    # 1) run the normalizer (expects samples × features in log space)
    sample_idx2shift = get_normfacts(samples_x_features)          # our variance+median loop
    samples_norm     = apply_sampleshifts(samples_x_features.copy(), sample_idx2shift)

    # 2) reconstruct DataFrame in original orientation (peptides × runs)
    pt_normed_log2 = (
        pd.DataFrame(samples_norm, index=pt_log2.columns, columns=pt_log2.index)
        .T
    )

    # 3) back to linear (keep NaNs where peptide/run was missing)
    pt_normed_linear = np.exp2(pt_normed_log2)

    if return_shifts:
        return pt_normed_log2, pt_normed_linear, sample_idx2shift
    else:
        return pt_normed_log2, pt_normed_linear
    
# ------------------------------------------------------------
# 4 · TAXON-LEVEL INTENSITIES (uniques only, directLFQ-style)
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from collections import defaultdict

# Inputs assumed present from the script:
# - pt_normed_log2 : DataFrame (rows=peptides, cols=runs) after step 1–3
# - unique_map     : dict {peptide_seq: [taxon]} — each list has length 1 (unique peptides)

# ---------- helpers ----------

def build_taxon_to_unique_peptides(unique_map, peptides_index):
    """Return {taxon: [unique_peptides_present_in_df]}"""
    taxon_to_peps = defaultdict(list)
    peps_in_df = set(peptides_index)
    for pep, taxa in unique_map.items():
        if not taxa or len(taxa) != 1:
            continue
        if pep in peps_in_df:
            taxon_to_peps[taxa[0]].append(pep)
    return taxon_to_peps

def align_peptide_traces_log2(peptide_df_log2, min_nonan=1):
    """
    Align ion traces (rows=peptides, cols=runs) with get_normfacts/apply_sampleshifts,
    then compute per-run median and rescale so total intensity is preserved.
    Returns (shifted_peptides_df_log2, taxon_profile_log2_vector).
    """
    if peptide_df_log2.shape[0] == 0:
        return peptide_df_log2, np.full(peptide_df_log2.shape[1], np.nan)

    # Keep a copy; work on numpy for speed
    arr = peptide_df_log2.to_numpy(copy=True)  # shape: (num_peptides, num_runs)

    # directLFQ requires at least two non-NaN points to contribute; get_normfacts enforces that per row
    shifts = get_normfacts(arr.copy()) or {}   # dict: row_idx -> additive log2 shift
    arr_shifted = apply_sampleshifts(arr.copy(), shifts)

    shifted_df = pd.DataFrame(arr_shifted, index=peptide_df_log2.index, columns=peptide_df_log2.columns)

    # per-run medians with a minimum number of non-NaN peptides
    med = []
    for run in shifted_df.columns:
        col = shifted_df[run].to_numpy()
        if np.sum(~np.isnan(col)) >= min_nonan:
            med.append(np.nanmedian(col))
        else:
            med.append(np.nan)
    taxon_profile_log2 = np.array(med, dtype=float)

    # intensity-preserving rescale (match sum over runs of peptides vs. profile)
    summed_pepints = np.nansum(2.0 ** peptide_df_log2.to_numpy())
    summed_profile = np.nansum(2.0 ** taxon_profile_log2)
    if summed_profile > 0 and np.isfinite(summed_pepints):
        taxon_profile_log2 = taxon_profile_log2 + np.log2(summed_pepints / summed_profile)

    return shifted_df, taxon_profile_log2

def estimate_taxon_intensities_unique(pt_normed_log2, unique_map, min_nonan=1, return_shifted_ions=False):
    """
    Build taxon LFQ intensities from unique peptides only.
    Returns (taxon_lfq_log2: DataFrame, taxon_lfq_linear: DataFrame[, ion_shifted_df: DataFrame]).
    """
    # 1) map taxa -> peptides present
    taxon_to_peps = build_taxon_to_unique_peptides(unique_map, pt_normed_log2.index)

    taxon_profiles = {}
    shifted_ion_blocks = []

    for taxon, peps in taxon_to_peps.items():
        sub = pt_normed_log2.loc[peps]  # rows = peptides, cols = runs
        # optional: cap number of peptides for speed (like ProtvalCutter); skip unless needed
        shifted_df, prof_log2 = align_peptide_traces_log2(sub, min_nonan=min_nonan)

        # store taxon profile (keep even if all-NaN; we can drop later)
        taxon_profiles[taxon] = prof_log2

        if return_shifted_ions:
            tmp = shifted_df.copy()
            tmp["taxon"] = taxon
            tmp["ion"] = tmp.index
            tmp = tmp.set_index(["taxon", "ion"])
            shifted_ion_blocks.append(tmp)

    # 2) assemble outputs
    runs = list(pt_normed_log2.columns)
    taxon_lfq_log2 = pd.DataFrame.from_dict(taxon_profiles, orient="index", columns=runs)
    # (optional) drop taxa with all-NaN
    # taxon_lfq_log2 = taxon_lfq_log2.dropna(how="all")

    taxon_lfq_linear = (2.0 ** taxon_lfq_log2).replace(np.nan, 0.0)

    if return_shifted_ions:
        ion_shifted_df = pd.concat(shifted_ion_blocks, axis=0) if shifted_ion_blocks else pd.DataFrame(columns=runs)
        return taxon_lfq_log2, taxon_lfq_linear, ion_shifted_df
    else:
        return taxon_lfq_log2, taxon_lfq_linear