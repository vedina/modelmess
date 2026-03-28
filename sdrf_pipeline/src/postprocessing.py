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


from difflib import SequenceMatcher

def merge_with_similarity(dfs, threshold=0.8):
    """
    Merges multiple DataFrames. If values are > threshold similar, 
    only the most descriptive (longest) one is kept.
    """
    def is_similar(a, b):
        # Calculate ratio of similarity between two strings
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

    def combine_and_dedup(values):
        valid_parts = []
        for val in values:
            s_val = str(val).strip()
            if s_val.lower() not in ['not applicable', 'nan', 'none', '']:
                # Split current cell by semicolon
                parts = [p.strip() for p in s_val.split(';') if p.strip()]
                valid_parts.extend(parts)
        
        if not valid_parts:
            return "Not Applicable"

        # Deduplicate based on similarity threshold
        unique_parts = []
        for new_p in valid_parts:
            is_duplicate = False
            for i, existing_p in enumerate(unique_parts):
                if is_similar(new_p, existing_p):
                    is_duplicate = True
                    # If the new string is more descriptive/longer, replace the old one
                    if len(new_p) > len(existing_p):
                        unique_parts[i] = new_p
                    break
            if not is_duplicate:
                unique_parts.append(new_p)
                
        return "; ".join(unique_parts)

    # 1. Start with structure of first DF
    merged_df = dfs[0].copy()
    key_cols = ['ID', 'PXD', 'Raw Data File']
    meta_cols = [c for c in merged_df.columns if c not in key_cols]

    # 2. Merge columns row-by-row
    for col in meta_cols:
        col_data = []
        for i in range(len(merged_df)):
            # Gather values from all DFs for this specific row/column
            row_values = [df.iloc[i][col] for df in dfs]
            col_data.append(combine_and_dedup(row_values))
        merged_df[col] = col_data
        
    return merged_df

# --- Usage ---
# df_list = [df_peak, df_current]
# final_df = merge_with_similarity(df_list, threshold=0.85)


def competition_aware_merge(df_peak, df_new):
    """
    Optimized for the Kaggle SDRF-F1 Cluster Evaluator.
    Priority: Peak Run > New Run. 
    Only fills 'Not Applicable' holes. Never appends.
    """
    result = df_peak.copy()
    meta_cols = [c for c in result.columns if c not in ['ID', 'PXD', 'Raw Data File']]
    
    for col in meta_cols:
        # Define what counts as 'Missing'
        is_missing = result[col].astype(str).str.lower().isin(['not applicable', 'nan', 'none', ''])
        
        # Define what the new run has to offer
        has_info = ~df_new[col].astype(str).str.lower().isin(['not applicable', 'nan', 'none', ''])
        
        # Only update where Peak is missing but New has data
        mask = is_missing & has_info
        result.loc[mask, col] = df_new.loc[mask, col]
        
    return result