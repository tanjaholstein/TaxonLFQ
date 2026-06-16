from pathlib import Path

import numpy as np
import pandas as pd


def read_composition(path: str | Path) -> pd.DataFrame:
    """Load and tidy a composition *.tab file."""
    df = pd.read_csv(path, sep="\t", quotechar='"', encoding="latin1", engine="python")
    cols = (
        pd.Series(df.columns)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df.columns = cols

    rename_map = {
        "Protein input per biological replicate (mug)": "Protein_µg",
        "Protein input per biological replicate (µg)": "Protein_µg",
        "Protein abundance %": "Protein_%",
        "Cell number input per biological replicate": "Cells",
        "Cell abundance %": "Cell_%",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    num_cols = [c for c in ["Protein_µg", "Protein_%", "Cells", "Cell_%"] if c in df.columns]
    if num_cols:
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    if "Label" in df.columns:
        df = df.set_index("Label")
    return df


def make_ref_from_df(df: pd.DataFrame, desired_order_complete: list[str], value_col: str = "Protein_%") -> pd.Series:
    """Return reference proportions aligned to desired_order_complete, merging LT2a/LT2b/LT2c into LT2."""
    s = df[value_col].copy()
    lt2_parts = [x for x in ["LT2a", "LT2b", "LT2c"] if x in s.index]
    if lt2_parts:
        s["LT2"] = s.get("LT2", 0.0) + s.loc[lt2_parts].sum()
        s = s.drop(lt2_parts)
    return (s / 100.0).reindex(desired_order_complete).dropna()


def normalize_reference_to_order(ref: pd.Series, order: list[str]) -> pd.Series:
    """Subset a reference Series to order and renormalize to sum to one."""
    out = ref.loc[order].astype(float)
    return out / out.sum()


def read_parquet_matrix(path: str | Path, *, normalize_columns: bool = False) -> pd.DataFrame:
    """Read an organism × run matrix, optionally closing each run to proportions."""
    df = pd.read_parquet(path).apply(pd.to_numeric, errors="coerce")
    if normalize_columns:
        df = df.div(df.sum(axis=0), axis=1)
    return df


def load_method_condition_tables(
    input_dir: Path,
    file_map: dict,
    *,
    normalize_columns: bool = False,
) -> dict:
    """Load nested {method: {condition: DataFrame}} parquet files from one directory."""
    return {
        method: {
            condition: read_parquet_matrix(input_dir / filename, normalize_columns=normalize_columns)
            for condition, filename in condition_files.items()
        }
        for method, condition_files in file_map.items()
    }
