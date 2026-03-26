import numpy as np
import pandas as pd
import difflib
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering


class ParticipantVisibleError(Exception):
    pass


def load_sdrf(sdrf_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """Load SDRF-like dataframe into nested dict keyed by PXD then column."""
    # print(f'\n{"#"*50} Loading SDRF data {"#"*50}')
    if "PXD" not in sdrf_df.columns:
        raise ParticipantVisibleError("Both solution and submission must include a 'PXD' column.")

    sdrf_dict: Dict[str, Dict[str, List[str]]] = {}
    for pxd, pxd_df in sdrf_df.groupby('PXD'):
        sdrf_dict[pxd] = {}
        for col in pxd_df.columns:

            if col in ['Raw Data File', 'Usage', 'PXD']:
                continue  # skip these columns entirely

            # Collect unique stringified values; drop NaNs safely
            uniq = pd.Series(pxd_df[col]).dropna().astype(str).unique().tolist()

            # Skip harmonization for 'Not Applicable' if it's the only unique value
            if uniq == ['Not Applicable']:
                continue
            
            values: List[str] = []
            for v in uniq:
                if 'NT=' in v:
                    parts = [r for r in v.split(';') if 'NT=' in r]
                    values.append(parts[0].replace('NT=', '').strip() if parts else v.strip())
                else:
                    values.append(v.strip())
                    
            # strip .1, .2, etc. suffixes from column names
            if '.' in col:
                col = col.split('.')[0].strip()

            if col in sdrf_dict[pxd]:
                sdrf_dict[pxd][col] += values  # append to existing list if column already exists
            else:
                sdrf_dict[pxd][col] = values   # ← initialize (DO NOT use += before init)
            # print(f"Processing PXD={pxd}, column={col}, unique values (pre-harmonization): {values}")
   
    return sdrf_dict


def _string_similarity(a: str, b: str) -> float:
    """Return similarity in [0,1] using difflib.SequenceMatcher (pure stdlib)."""
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()

def Harmonize_and_Evaluate_datasets(
    A: Dict[str, Dict[str, List[str]]],
    B: Dict[str, Dict[str, List[str]]],
    threshold: float = 0.80,
    method: str = 'RapidFuzz',
    CompleteAbsence: float = float('nan'),
) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]], pd.DataFrame]:

    # if method != 'RapidFuzz':
    #    raise ParticipantVisibleError("This metric only supports method='RapidFuzz' in the Kaggle sandbox.")

    from sklearn.metrics import precision_score, recall_score, f1_score
    eval_metrics = {'pxd': [], 'AnnotationType': [], 'precision': [], 'recall': [], 'f1': [], 'jacc': []}
    harmonized_A: Dict[str, Dict[str, List[int]]] = {}
    harmonized_B: Dict[str, Dict[str, List[int]]] = {}

    common_pubs = set(A) & set(B)
    for pub in common_pubs:
        harmonized_A[pub], harmonized_B[pub] = {}, {}

        ## A is the ground truth data and B is the submission
        for category in set(A[pub]):  
            vals_A = A[pub][category]
            
            ## check if sol column is in submission and if not return an empty list.  
            vals_B = B[pub].get(category, [])

            ## merge values into a single set of values
            all_vals = vals_A + [v for v in vals_B if v not in vals_A]

            if len(vals_A) == 0 and len(vals_B) == 0:
                harmA: List[int] = []
                harmB: List[int] = []
            elif len(all_vals) == 1:
                labels = np.array([0])
                str2cid = {all_vals[0]: 0}
                harmA = [str2cid[s] for s in vals_A]
                harmB = [str2cid[s] for s in vals_B]
            else:
                N = len(all_vals)
                dist = np.zeros((N, N), dtype=float)
                for i in range(N):
                    for j in range(i + 1, N):
                        # sim = fuzz.ratio(all_vals[i], all_vals[j]) / 100.0
                        sim = _string_similarity(all_vals[i], all_vals[j])
                        d = 1.0 - sim
                        dist[i, j] = d
                        dist[j, i] = d
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    metric='precomputed',
                    linkage='average',
                    distance_threshold=1.0 - threshold
                )
                labels = clusterer.fit_predict(dist)
                str2cid = {s: int(labels[i]) for i, s in enumerate(all_vals)}
                harmA = [str2cid[s] for s in vals_A]
                harmB = [str2cid[s] for s in vals_B]

            harmonized_A[pub][category] = harmA
            harmonized_B[pub][category] = harmB

            uniq = sorted(set(harmA) | set(harmB))
            if not uniq:
                p = r = f = CompleteAbsence
                j = 1.0
            else:
                y_true = [1 if u in harmA else 0 for u in uniq]
                y_pred = [1 if u in harmB else 0 for u in uniq]
                p = precision_score(y_true, y_pred, average='macro', zero_division=0)
                r = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f = f1_score(y_true, y_pred, average='macro', zero_division=0)
                setA, setB = set(harmA), set(harmB)
                j = 1.0 if (not setA and not setB) else len(setA & setB) / len(setA | setB)

            eval_metrics['pxd'].append(pub)
            eval_metrics['AnnotationType'].append(category)
            eval_metrics['precision'].append(p)
            eval_metrics['recall'].append(r)
            eval_metrics['f1'].append(f)
            eval_metrics['jacc'].append(j)

    return harmonized_A, harmonized_B, pd.DataFrame(eval_metrics)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute the competition score for a submission vs. solution.

    This metric clusters text values per (PXD, column) using a string-similarity
    distance (difflib) and agglomerative clustering, then compares the presence
    of cluster IDs between solution and submission to produce macro-averaged F1
    per (PXD, column). The final score is the mean F1 across all evaluated pairs.

    Requirements:
      - Both dataframes must include a 'PXD' column.
      - Kaggle passes an alignment key name in `row_id_column_name`; if present,
        it will be dropped from both frames before scoring.

    Returns:
      float: A single finite score in [0, 1].

    Example
    -------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> sol = pd.DataFrame({"id":[0,1], "PXD":["P1","P1"], "Characteristics[Organism]":["human","mouse"]})
    >>> sub = sol.copy()  # identical -> perfect score
    >>> score(sol.copy(), sub.copy(), row_id_column_name)
    1.0
    """
    # Safely drop the alignment key if present
    if row_id_column_name and row_id_column_name in solution.columns:
        solution = solution.drop(columns=[row_id_column_name])
    if row_id_column_name and row_id_column_name in submission.columns:
        submission = submission.drop(columns=[row_id_column_name])

    if "PXD" not in solution.columns or "PXD" not in submission.columns:
        raise ParticipantVisibleError("Both solution and submission must include a 'PXD' column.")

    sol = load_sdrf(solution)
    sub = load_sdrf(submission)
    harmonized_A, harmonized_B, eval_df = Harmonize_and_Evaluate_datasets(sol, sub, threshold=0.80)

    vals = eval_df["f1"].dropna()
    return float(vals.mean()) if not vals.empty else 0.0, harmonized_A, harmonized_B, eval_df


# expansion dict
def build_clusters(values: List[str], threshold: float = 0.8) -> List[List[str]]:
    if not values:
        return []

    values = [v.strip() for v in values if v and v.strip()]
    N = len(values)
    if N == 1:
        return [values]

    dist = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            sim = _string_similarity(values[i], values[j])
            d = 1.0 - sim
            dist[i, j] = d
            dist[j, i] = d

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=1.0 - threshold
    )
    labels = clusterer.fit_predict(dist)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for val, lbl in zip(values, labels):
        clusters[lbl].append(val)

    return list(clusters.values())