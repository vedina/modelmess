"""
Helpers to clean LLM output → produce plain dicts for SDRF rows.
No Pydantic dependency — works standalone.
"""
import re
import json
import logging

log = logging.getLogger('sdrf')


def _fix_json_string(text: str) -> str:
    """
    Best-effort repair of common LLM JSON formatting problems:
      1. Invalid backslash escapes  (\a etc.)
      2. Python None/True/False     -> null/true/false
      3. Single-quoted strings      -> double-quoted
      4. Trailing commas            -> removed
      5. Truncated JSON             -> close open braces/brackets
    """
    # 0. Fix invalid backslash escapes (char-by-char — preserves \n, \t, \uXXXX, \\)
    _fixed = []
    _i = 0
    while _i < len(text):
        if text[_i] == '\\':
            _nxt = text[_i+1] if _i+1 < len(text) else ''
            if _nxt in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                _fixed.append(text[_i]); _fixed.append(_nxt); _i += 2
            else:
                _fixed.append('\\\\'); _i += 1
        else:
            _fixed.append(text[_i]); _i += 1
    text = ''.join(_fixed)

    # 1. Python literals
    for py, js in (
        (': None',  ': null'),  (': True',  ': true'),  (': False',  ': false'),
        (':None',   ':null'),   (':True',   ':true'),   (':False',   ':false'),
        ('[None',   '[null'),   (', None',  ', null'),  ('None]',    'null]'),
        ('None,',   'null,'),   ('[True',   '[true'),   (', True',   ', true'),
        ('True]',   'true]'),   ('True,',   'true,'),   ('[False',   '[false'),
        (', False', ', false'), ('False]',  'false]'),  ('False,',   'false,'),
    ):
        text = text.replace(py, js)

    # 2. Single-quote → double-quote (char-by-char to preserve apostrophes)
    if "'" in text:
        out = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'":
                out.append('"')
                i += 1
                while i < len(text) and text[i] != "'":
                    if text[i] == '\\' and i + 1 < len(text):
                        out.append(text[i]); i += 1
                    elif text[i] == '"':
                        out.append('\\"'  )
                    else:
                        out.append(text[i])
                    i += 1
                out.append('"')
                i += 1
            else:
                out.append(ch); i += 1
        text = ''.join(out)

    # 3. Trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # 4. Close truncated JSON
    opens_b = text.count('{') - text.count('}')
    opens_k = text.count('[') - text.count(']')
    if opens_b > 0 or opens_k > 0:
        candidates = [
            text.rfind('",'), text.rfind('"}'), text.rfind('"\n'),
            text.rfind('},'), text.rfind('],'), text.rfind(','),
            text.rfind('{'), text.rfind('['),
        ]
        boundary = max((c for c in candidates if c >= 0), default=-1)
        if boundary > 0:
            text = text[:boundary]
        text = text.rstrip(' ,\n\r\t')
        opens_b = text.count('{') - text.count('}')
        opens_k = text.count('[') - text.count(']')
        text += '}' * max(opens_b, 0) + ']' * max(opens_k, 0)

    return text


def _safe_parse_json(text: str):
    """
    Robustly parse JSON from LLM output.
    Tries json.loads first, then applies _fix_json_string repair, then
    a targeted None/True/False substitution before a final json.loads.
    Never uses ast.literal_eval (can't handle bare None/True/False identifiers).
    Returns parsed value or None on failure.
    """
    if not text or not text.strip():
        return None

    # Strip markdown fences
    text = re.sub(r'```(?:json)?', '', text)
    text = re.sub(r'```', '', text).strip()

    # Try as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Apply full repair
    repaired = _fix_json_string(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        log.debug(f"JSON still broken after repair: {e}. Snippet: {repaired[:200]}")
        return None


def clean_json(text: str) -> str:
    """Strip markdown, repair, extract first JSON object (prefers object over array)."""
    text = re.sub(r'```(?:json)?', '', text)
    text = re.sub(r'```', '', text).strip()
    for pattern in (r'\{.*\}', r'\[.*\]'):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return _fix_json_string(m.group(0))
    return _fix_json_string(text)


def clean_json_array(text: str) -> str:
    """Strip markdown, repair, extract first JSON array (prefers array over object)."""
    text = re.sub(r'```(?:json)?', '', text)
    text = re.sub(r'```', '', text).strip()
    for pattern in (r'\[.*\]', r'\{.*\}'):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return _fix_json_string(m.group(0))
    return _fix_json_string(text)


def _coerce_to_dict(data) -> dict:
    """Turn whatever the LLM returned into a plain dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        if all(isinstance(x, dict) for x in data):
            merged = {}
            for item in data: merged.update(item)
            return merged
        if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in data):
            return dict(data)
    log.warning(f"Cannot coerce to dict: {type(data)} — {str(data)[:200]}")
    return {}


def _serialise_globals(globals_dict: dict) -> str:
    """Serialise globals dict to JSON string (handles nested dicts/lists)."""
    return json.dumps(globals_dict, default=str)


def _parse_kv_string(s: str) -> dict:
    """Parse NT=...;AC=... string into a dict. Plain name → {'NT': name}."""
    if not isinstance(s, str) or '=' not in s:
        return {'NT': s.strip()} if s.strip() else {}
    result = {}
    for part in s.split(';'):
        part = part.strip()
        if '=' in part:
            k, _, v = part.partition('=')
            result[k.strip()] = v.strip()
        elif part:
            result['NT'] = part
    return result


def parse_globals(raw_json: str) -> dict:
    """
    Parse globals JSON from LLM output.
    Returns a plain dict — no Pydantic models.
    Instrument and CleavageAgent stay as dicts or strings.
    """
    data = _safe_parse_json(clean_json(raw_json))
    if data is None:
        log.warning(f"globals parse failed. Raw snippet: {raw_json[:200]}")
        return {}
    return _coerce_to_dict(data)


def parse_samples(raw_json: str, pxd: str = '') -> list:
    """
    Parse samples JSON array from LLM output.
    Returns a list of plain dicts — no Pydantic models.
    """
    data = _safe_parse_json(clean_json_array(raw_json))
    if data is None:
        log.warning(f"[{pxd}] samples parse failed. Raw snippet: {raw_json[:200]}")
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    return []


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
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


import difflib

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
    _, _, eval_df = Harmonize_and_Evaluate_datasets(sol, sub, threshold=0.80)

    vals = eval_df["f1"].dropna()
    return float(vals.mean()) if not vals.empty else 0.0