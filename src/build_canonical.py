import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from sklearn.cluster import AgglomerativeClustering

# Columns the metric skips entirely — mirror load_sdrf behaviour
SKIP_COLUMNS = {"ID", "PXD", "Raw Data File", "Usage"}

# Columns that are free-text / high-cardinality and won't cluster meaningfully
FREETEXT_COLUMNS = {
    "Comment[data file]",
    "Comment[file uri]",
    "Source Name",
    "Assay Name",
    "comment[data file]",
}

MAX_UNIQUE_FOR_CLUSTERING = 300  # skip clustering above this cardinality


def normalize(x: str) -> str:
    return str(x).lower().strip().replace("  ", " ")


def build_canonical_from_sdrf_dict(
    sdrf_dict: Dict[str, Dict[str, List[str]]],
    threshold: float = 0.80,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]], Dict[str, Counter]]:
    """
    Build a canonical map from a pre-loaded SDRF dict (output of load_sdrf).

    Uses the identical clustering algorithm as the competition metric:
      - pairwise distance matrix via rapidfuzz.process.cdist (falls back to difflib)
      - AgglomerativeClustering, metric='precomputed', linkage='average'
      - distance_threshold = 1.0 - threshold

    For each cluster the canonical representative is the most frequent member.

    Returns
    -------
    canonical_map    : col -> {raw_normalised_value: canonical_value}
    canonical_values : col -> [canonical_value, ...]  (frequency-ordered)
    freq_per_column  : col -> Counter of normalised values
    """

    # ------------------------------------------------------------------ #
    # 1. Collect normalised values across all PXDs                        #
    # ------------------------------------------------------------------ #
    values_per_column: Dict[str, List[str]] = defaultdict(list)

    for pxd, col_dict in sdrf_dict.items():
        for col, vals in col_dict.items():
            if col in SKIP_COLUMNS or col in FREETEXT_COLUMNS:
                continue
            for v in vals:
                values_per_column[col].append(normalize(v))

    freq_per_column = {
        col: Counter(vals) for col, vals in values_per_column.items()
    }

    # ------------------------------------------------------------------ #
    # 2. Try to import rapidfuzz; fall back to difflib                    #
    # ------------------------------------------------------------------ #
    try:
        from rapidfuzz import process as rf_process
        from rapidfuzz.distance import Levenshtein

        def build_distance_matrix(vals: List[str]) -> np.ndarray:
            # cdist returns similarity scores in [0, 100]; convert to distance
            sim_matrix = rf_process.cdist(
                vals, vals,
                scorer=Levenshtein.normalized_similarity,
                score_cutoff=0.0,
            )
            return (1.0 - sim_matrix).astype(float)

        print("Using rapidfuzz for distance computation.")

    except ImportError:
        import difflib

        def build_distance_matrix(vals: List[str]) -> np.ndarray:  # type: ignore[misc]
            n = len(vals)
            dist = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    d = 1.0 - difflib.SequenceMatcher(None, vals[i], vals[j]).ratio()
                    dist[i, j] = d
                    dist[j, i] = d
            return dist

        print("rapidfuzz not available — falling back to difflib (will be slower).")

    # ------------------------------------------------------------------ #
    # 3. Cluster per column                                               #
    # ------------------------------------------------------------------ #
    canonical_map: Dict[str, Dict[str, str]] = {}
    canonical_values: Dict[str, List[str]] = {}

    for col, counter in freq_per_column.items():
        # Most-frequent first so ties resolve consistently
        unique_vals: List[str] = [v for v, _ in counter.most_common()]

        # ---- trivial cases ------------------------------------------- #
        if len(unique_vals) == 0:
            canonical_map[col] = {}
            canonical_values[col] = []
            continue

        if len(unique_vals) == 1:
            v = unique_vals[0]
            canonical_map[col] = {v: v}
            canonical_values[col] = [v]
            continue

        # ---- high-cardinality guard ----------------------------------- #
        if len(unique_vals) > MAX_UNIQUE_FOR_CLUSTERING:
            # Identity map — no clustering attempted
            canonical_map[col] = {v: v for v in unique_vals}
            canonical_values[col] = unique_vals
            print(
                f"  Skipping clustering for '{col}' "
                f"({len(unique_vals)} unique values > {MAX_UNIQUE_FOR_CLUSTERING})"
            )
            continue

        # ---- distance matrix + agglomerative clustering --------------- #
        dist = build_distance_matrix(unique_vals)

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
        labels = clusterer.fit_predict(dist)

        # Group by cluster label
        cluster_members: Dict[int, List[str]] = defaultdict(list)
        for val, label in zip(unique_vals, labels):
            cluster_members[label].append(val)

        col_map: Dict[str, str] = {}
        col_canonicals: List[str] = []

        for members in cluster_members.values():
            # Most frequent member is the canonical representative
            representative = max(members, key=lambda x: counter[x])
            col_canonicals.append(representative)
            for v in members:
                col_map[v] = representative

        # Order canonical list by descending frequency of representative
        col_canonicals.sort(key=lambda x: counter[x], reverse=True)

        canonical_map[col] = col_map
        canonical_values[col] = col_canonicals

    return canonical_map, canonical_values, freq_per_column


# ---------------------------------------------------------------------- #
# Postprocessing                                                          #
# ---------------------------------------------------------------------- #

def postprocess_sdrf(
    df: pd.DataFrame,
    canonical_map: Dict[str, Dict[str, str]],
    similarity_threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Apply the canonical map to a submission DataFrame.

    For each cell value:
      1. Exact match in canonical_map  →  use canonical directly
      2. Fuzzy match above threshold   →  use canonical of best match
      3. No match                      →  leave original value unchanged
         (never fabricate a value via fallback)

    The metric scores on the *set* of unique values per (PXD, column),
    so row order doesn't matter — we normalise in place.
    """
    import difflib

    try:
        from rapidfuzz import process as rf_process, fuzz
        def best_fuzzy_match(v: str, candidates: List[str]):
            result = rf_process.extractOne(
                v, candidates,
                scorer=fuzz.ratio,
                score_cutoff=similarity_threshold * 100,
            )
            return result  # (match, score, index) or None
    except ImportError:
        def best_fuzzy_match(v: str, candidates: List[str]):  # type: ignore[misc]
            best = max(
                candidates,
                key=lambda c: difflib.SequenceMatcher(None, v, c).ratio(),
            )
            score = difflib.SequenceMatcher(None, v, best).ratio()
            return (best, score * 100, 0) if score >= similarity_threshold else None

    df = df.copy()

    for col in df.columns:
        if col not in canonical_map or col in SKIP_COLUMNS or col in FREETEXT_COLUMNS:
            continue

        col_map = canonical_map[col]
        candidates = list(col_map.keys())  # all known raw values

        def map_cell(cell):
            if pd.isna(cell):
                return cell
            v_norm = normalize(str(cell))
            # 1. exact
            if v_norm in col_map:
                return col_map[v_norm]
            # 2. fuzzy
            result = best_fuzzy_match(v_norm, candidates)
            if result is not None:
                return col_map[result[0]]
            # 3. unchanged
            return cell

        df[col] = df[col].map(map_cell)

    return df
