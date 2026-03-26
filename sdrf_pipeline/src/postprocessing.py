import os
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Set, Optional
import json
from .eval import load_sdrf, build_clusters, _string_similarity


def accumulate_unique_per_column_via_load_sdrf(
    folder_path: str,
    file_pattern: str = ".csv",
    include_cols = None,
    exclude_cols: Optional[List[str]] = None
) -> Dict[str, Set[str]]:
    """
    Accumulate unique values per column across all CSV files in a folder,
    using `load_sdrf` for normalization, stripping, and NT= handling.
    Ignores PXD grouping.

    Args:
        folder_path: folder containing CSV files
        file_pattern: suffix filter for files
        exclude_cols: optional list of columns to skip; if None, keep all

    Returns:
        Dict[column_name, Set[unique_values]]
    """
    accumulated: Dict[str, Set[str]] = defaultdict(set)

    for fname in os.listdir(folder_path):
        if not fname.endswith(file_pattern):
            continue
        fpath = os.path.join(folder_path, fname)
        df = pd.read_csv(fpath)
        try:
            sdrf_dict = load_sdrf(df)

        except Exception:
            continue  # skip files without PXD column
        if include_cols is not None:
            include_cols = list(include_cols)
        if exclude_cols is not None:
            exclude_cols = list(exclude_cols)
        # flatten across PXDs
        for cols in sdrf_dict.values():
            for col, values in cols.items():
                if include_cols and col not in include_cols:
                    continue
                if exclude_cols and col in exclude_cols:
                    continue
                accumulated[col].update(values)

    return accumulated
    

def dump_dictionary_to_json(accum: Dict[str, Dict[str, List[str]]], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(accum, f, indent=2, ensure_ascii=False)


def load_dictionary_from_json(filepath: str) -> Dict[str, Dict[str, List[str]]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Build expansion dictionaries per column ---
def build_column_expansion_dict(accumulated: Dict[str, set],
                                threshold: float = 0.8) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns a dict mapping column_name -> expansion_dict (key -> list of variants).
    """

    # Step 2: cluster values per column and build expansion dictionaries
    column_expansions: Dict[str, Dict[str, List[str]]] = {}
    for col, vals in accumulated.items():
        clusters = build_clusters(list(vals), threshold=threshold)
        expansion_dict: Dict[str, List[str]] = defaultdict(list)
        for cluster in clusters:
            for v in cluster:
                key = v.lower()
                for other in cluster:
                    if other != v and other not in expansion_dict[key]:
                        expansion_dict[key].append(other)
        # keep only top 2 variants per key
        for k in expansion_dict:
            expansion_dict[k] = expansion_dict[k][:2]
        column_expansions[col] = expansion_dict

    return column_expansions


def find_expansions(val: str, expansion_dict: Dict[str, List[str]], threshold: float = 0.8) -> List[str]:
    out = []
    val_lower = val.lower()
    for k, variants in expansion_dict.items():
        if _string_similarity(val_lower, k) >= threshold:
            out.extend(variants)
    return out


def expand_value_learned(val: str, expansion_dict: Dict[str, List[str]], threshold: float = 0.8, size=3) -> str:
    if not val:
        return val
    variants = find_expansions(val, expansion_dict, threshold)
    all_vals = [val] + variants
    seen = []
    for x in all_vals:
        if x not in seen:
            seen.append(x)
    return "; ".join(seen[:size])


def expand_dataframe_values(
    df: pd.DataFrame,
    expansion_dict: Dict[str, Dict[str, List[str]]],
    threshold: float = 0.8,
    size: int = 3
) -> pd.DataFrame:
    """
    Expand values in a DataFrame using an expansion dictionary.
    
    Args:
        df: input DataFrame
        expansion_dict: Dict[column_name, Dict[val, List[expansions]]]
        threshold: similarity threshold for finding expansions (passed to expand_value_learned)
        size: maximum number of expansions to keep (passed to expand_value_learned)
    
    Returns:
        DataFrame with expanded values (joined by "; ")
    """
    df_expanded = df.copy()

    for col in df.columns:
        if col not in expansion_dict:
            continue

        col_dict = expansion_dict[col]

        def _expand_cell(val):
            val_str = str(val) if pd.notna(val) else ""
            return expand_value_learned(val_str, col_dict, threshold=threshold, size=size)

        df_expanded[col] = df_expanded[col].apply(_expand_cell)

    return df_expanded


def filter_by_set_size(accum: Dict[str, Set[str]], min_size=1, max_size=2) -> Dict[str, Set[str]]:
    """
    Keep only entries whose set size is within [min_size, max_size].
    """
    return {k: v for k, v in accum.items() if min_size <= len(v) <= max_size}
