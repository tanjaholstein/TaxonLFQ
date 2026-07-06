from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RECIPE_DIR = DATA_DIR / "recipes"
PEPTIDE_TABLE_DIR = DATA_DIR / "peptide_tables"
TAXON_TABLE_DIR = DATA_DIR / "taxon_tables"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Kleiner 2017
# ---------------------------------------------------------------------------

KLEINER_CONDITIONS = ["P", "U", "C"]
KLEINER_METHODS = ["Uniques", "Uniform", "Clades", "TaxonLFQ"]

KLEINER_COMPOSITION_FILES = {
    "P": "Composition_Of_EQUAL_PROTEIN_AMOUNT_Community.tab",
    "U": "Composition_Of_UNEVEN_Community.tab",
    "C": "Composition_Of_EQUAL_CELL_NUMBER_Community.tab",
}

KLEINER_METHOD_FILES = {
    "Uniques": {
        "P": "Kleiner_uniques_P.parquet",
        "U": "Kleiner_uniques_U.parquet",
        "C": "Kleiner_uniques_C.parquet",
    },
    "Uniform": {
        "P": "Kleiner_uniform_P.parquet",
        "U": "Kleiner_uniform_U.parquet",
        "C": "Kleiner_uniform_C.parquet",
    },
    "Clades": {
        "P": "Kleiner_Clades_P.parquet",
        "U": "Kleiner_Clades_U.parquet",
        "C": "Kleiner_Clades_C.parquet",
    },
    "TaxonLFQ": {
        "P": "Kleiner_TaxonLFQ_direct_P.parquet",
        "U": "Kleiner_TaxonLFQ_direct_U.parquet",
        "C": "Kleiner_TaxonLFQ_direct_C.parquet",
    },
}

KLEINER_LABELS = {
    "P": [
        "PD", "BS", "PaD", "AK199", "KF7", "CV", "ATN", "SMS",
        "Cup", "Pfl", "BXL", "137", "259", "Am2", "LT2a",
        "K12", "LT2b", "LT2c", "HB2", "CRH", "M13", "F2", "P22",
        "F0", "ES18", "841", "VF", "NV",
    ],
    "U": [
        "PD", "BS", "PaD", "AK199", "KF7", "CV", "ATN", "SMS",
        "Cup", "Pfl", "BXL", "DVH", "137", "259", "Am2", "LT2a",
        "K12", "LT2b", "LT2c", "HB2", "CRH", "M13", "F2", "P22",
        "F0", "ES18", "841", "VF", "Ne1", "Nu1", "Nm1", "NV",
    ],
    "C": [
        "PD", "BS", "PaD", "AK199", "KF7", "CV", "ATN", "SMS",
        "Cup", "Pfl", "BXL", "137", "259", "Am2", "LT2a",
        "K12", "LT2b", "LT2c", "HB2", "CRH", "M13", "F2", "P22",
        "F0", "ES18", "841", "VF", "NV",
    ],
}

KLEINER_ORDER_COMPLETE = {
    "P": [
        "NV", "841", "VF", "ATN", "AK199", "PaD", "Am2", "K12", "LT2",
        "KF7", "PD", "Pfl", "SMS", "BXL", "Cup", "CV", "CRH",
        "259", "137", "BS", "HB2", "ES18", "cRAP", "F0", "P22", "F2", "M13",
    ],
    "U": [
        "NV", "841", "VF", "ATN", "AK199", "PaD", "Am2", "K12", "LT2",
        "KF7", "PD", "Pfl", "SMS", "BXL", "Cup", "Ne1", "Nm1", "Nu1", "DVH", "CV", "CRH",
        "259", "137", "BS", "HB2", "ES18", "cRAP", "F0", "F2", "P22", "M13",
    ],
    "C": [
        "NV", "841", "VF", "ATN", "AK199", "PaD", "Am2", "K12", "LT2",
        "KF7", "PD", "Pfl", "SMS", "BXL", "Cup", "CV", "CRH",
        "259", "137", "BS", "HB2", "ES18", "cRAP", "F0", "P22", "F2", "M13",
    ],
}

KLEINER_ORDER = [
    "NV", "841", "VF", "ATN", "AK199", "PaD", "Am2", "K12", "LT2",
    "KF7", "PD", "Pfl", "SMS", "BXL", "Cup", "CV", "CRH",
    "259", "137", "BS", "HB2",
]

KLEINER_TAXON_NAMES = {
    "NV":    "N. viennensis",
    "841":   "R. leguminosarum 3841",
    "VF":    "R. leguminosarum VF39",
    "ATN":   "A. tumefaciens NTL4",
    "AK199": "Roseobacter AK199",
    "PaD":   "Pa. denitrificans",
    "Am2":   "A. macleodii",
    "K12":   "E. coli K12",
    "LT2":   "S. typhimurium LT2",
    "KF7":   "P. pseudoalcaligenes",
    "PD":    "Ps. denitrificans",
    "Pfl":   "P. fluorescens",
    "SMS":   "S. maltophilia",
    "BXL":   "B. xenovorans",
    "Cup":   "C. metallidurans",
    "CV":    "C. violaceum",
    "CRH":   "C. reinhardtii",
    "259":   "S. aureus 25923",
    "137":   "S. aureus 13709",
    "BS":    "B. subtilis 168",
    "HB2":   "T. thermophilus HB27",
}

KLEINER_CONTRASTS = {
    "P_vs_U": {"numerator": "P", "denominator": "U", "title": "P vs. U", "obs_label": r"P$_{\mathrm{obs}}$"},
    "P_vs_C": {"numerator": "P", "denominator": "C", "title": "P vs. C", "obs_label": r"P$_{\mathrm{obs}}$"},
    "C_vs_U": {"numerator": "C", "denominator": "U", "title": "C vs. U", "obs_label": r"C$_{\mathrm{obs}}$"},
}

# ---------------------------------------------------------------------------
# Zhao 2023
# ---------------------------------------------------------------------------

ZHAO_SAMPLES = ["S1", "S2", "S3"]
ZHAO_METHODS = ["Uniques", "Uniform", "Clades", "TaxonLFQ"]
ZHAO_ORDER = ["Bfr", "Cbu", "Cfr", "Eas", "Eca", "Eco", "Efa", "Kae", "Kpn", "Lac", "Mmo", "Pae"]

ZHAO_METHOD_FILES = {
    "TaxonLFQ": {
        "S1": "Zhao_TaxonLFQ_noLFQ_S1.parquet",
        "S2": "Zhao_TaxonLFQ_noLFQ_S2.parquet",
        "S3": "Zhao_TaxonLFQ_noLFQ_S3.parquet",
    },
    "Uniques": {
        "S1": "Zhao_LFQ_uniques_S1.parquet",
        "S2": "Zhao_LFQ_uniques_S2.parquet",
        "S3": "Zhao_LFQ_uniques_S3.parquet",
    },
    "Uniform": {
        "S1": "Zhao_uniform_S1.parquet",
        "S2": "Zhao_uniform_S2.parquet",
        "S3": "Zhao_uniform_S3.parquet",
    },
    "Clades": {
        "S1": "Zhao_LFQ_clades_S1.parquet",
        "S2": "Zhao_LFQ_clades_S2.parquet",
        "S3": "Zhao_LFQ_clades_S3.parquet",
    },
}

ZHAO_CONTRASTS = {
    "S3/S1": {"numerator": "S3", "denominator": "S1", "title": "Sample 3 vs. Sample 1"},
    "S3/S2": {"numerator": "S3", "denominator": "S2", "title": "Sample 3 vs. Sample 2"},
    "S2/S1": {"numerator": "S2", "denominator": "S1", "title": "Sample 2 vs. Sample 1"},
}

ZHAO_EXPECTED_FC = {
    "S2/S1": {
        "Kpn": 1/5, "Bfr": 5/1, "Pae": 2/1, "Mmo": 1/5,
        "Eca": 5/2, "Cbu": 2/1, "Efa": 2/1, "Eas": 1/5,
        "Cfr": 5/2, "Lac": 2/5, "Eco": 3/2, "Kae": 2/1,
    },
    "S3/S1": {
        "Kpn": 2/5, "Bfr": 2/1, "Pae": 5/1, "Mmo": 2/5,
        "Eca": 1/2, "Cbu": 5/1, "Efa": 5/1, "Eas": 2/5,
        "Cfr": 1/2, "Lac": 1/5, "Eco": 1/2, "Kae": 5/1,
    },
}
ZHAO_EXPECTED_FC["S3/S2"] = {
    k: ZHAO_EXPECTED_FC["S3/S1"][k] / ZHAO_EXPECTED_FC["S2/S1"][k]
    for k in set(ZHAO_EXPECTED_FC["S2/S1"]) & set(ZHAO_EXPECTED_FC["S3/S1"])
}

ZHAO_AUTHOR_OFFSETS = {
    "S3/S1": [0.62, 0.81, 0.04, 0.03, 0.08, 0.06, 1.61, -1.62, 0.07, 0.01, 0.03, 1.98],
    "S3/S2": [0.01, 0.79, -0.02, 0.15, -0.02, 0.08, 0.25, -0.14, -1.31, 0.05, 0.39, 0.56],
    "S2/S1": [1.37, -0.23, 0.54, 0.01, 0.75, -0.17, 0.41, 0.0, 0.49, -0.04, -0.02, 0.31],
}
