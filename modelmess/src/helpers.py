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


def parse_globals(raw_json: str, resolve: bool = True) -> dict:
    """
    Parse globals JSON from LLM output.
    Returns a plain dict - no Pydantic models.

    If resolve=True (default), passes the parsed dict through
    resolve_globals_structure() in resolve.py to deterministically
    canonicalise instrument, cleavage_agent, modification, and
    plain-string CV fields.  Set resolve=False to get the raw LLM
    dict for debugging or gold-label construction.
    """
    data = _safe_parse_json(clean_json(raw_json))
    if data is None:
        log.warning(f"globals parse failed. Raw snippet: {raw_json[:200]}")
        return {}
    d = _coerce_to_dict(data)
    if resolve:
        try:
            from resolve import resolve_globals_structure
            d = resolve_globals_structure(d)
        except ImportError:
            log.debug("resolve.py not on path -- skipping structure resolution")
        except Exception as e:
            log.warning(f"resolve_globals_structure failed: {e} -- returning raw dict")
    return d


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
    harmonized_A, harmonized_B, eval_df = Harmonize_and_Evaluate_datasets(sol, sub, threshold=0.80)

    vals = eval_df["f1"].dropna()
    return float(vals.mean()) if not vals.empty else 0.0, harmonized_A, harmonized_B, eval_df

"""
compare_submissions

Columnwise comparison of two (or more) submission CSV files.
Usage:
    python compare_submissions.py sub1.csv sub2.csv [sub3.csv ...]

Output:
    - Per-column table: NA%, agreement%, category
    - Summary by category
"""


NA = 'Not Applicable'
ID_COLS = {'ID', 'PXD', 'Raw Data File', 'Usage'}


def load(path):
    return pd.read_csv(path)


def compare_two(a: pd.DataFrame, b: pd.DataFrame,
                name_a='A', name_b='B') -> pd.DataFrame:
    """
    Compare two submissions column by column, aligned on ID.
    Returns a DataFrame with one row per data column.
    """
    data_cols = [c for c in a.columns if c not in ID_COLS]

    # Align on ID
    merged = a[['ID'] + data_cols].merge(
        b[['ID'] + data_cols].rename(columns={c: c + '_b' for c in data_cols}),
        on='ID'
    )

    rows = []
    for col in data_cols:
        bc = col + '_b'
        na_a = (merged[col] == NA).mean()
        na_b = (merged[bc]  == NA).mean()
        agree = (merged[col] == merged[bc]).mean()

        # Categorise
        if na_a > 0.95 and na_b > 0.95:
            cat = 'both_NA'
        elif na_a > 0.5 and na_b < 0.5:
            cat = 'A_missing'    # A=NA, B has values
        elif na_a < 0.5 and na_b > 0.5:
            cat = 'B_missing'    # A has values, B=NA
        elif agree > 0.90:
            cat = 'agree'
        elif agree > 0.50:
            cat = 'partial'
        else:
            cat = 'differ'

        rows.append({
            'column'        : col,
            f'NA%_{name_a}' : f'{na_a:.0%}',
            f'NA%_{name_b}' : f'{na_b:.0%}',
            'agree%'        : f'{agree:.0%}',
            'category'      : cat,
        })

    return pd.DataFrame(rows)


# compare_submissions(
#    ("baseline", df1),
#    ("model_v2", df2),
#    ("ensemble", df3),
#)
def compare_submissions(*named_subs):
    if len(named_subs) < 2:
        raise ValueError("Provide at least two (name, DataFrame) pairs")

    names = [name for name, _ in named_subs]
    subs  = [df for _, df in named_subs]

    print(f"\n{'='*70}")
    for name, sub in zip(names, subs):
        print(f"  {name}: {sub.shape[0]} rows, {sub.shape[1]} cols")
    print(f"{'='*70}\n")

    df = compare_two(subs[0], subs[1], names[0], names[1])

    print(df.to_string(index=False))
    print()

    print("=== Summary by category ===")
    for cat, grp in df.groupby('category'):
        cols = grp['column'].tolist()
        print(f"\n  {cat} ({len(cols)}):")
        for c in cols:
            print(f"    {c}")

    if len(subs) >= 3:
        print(f"\n{'='*70}")
        print("Multi-way agreement analysis (columns with <80% all-agree):")

        data_cols = [c for c in subs[0].columns if c not in ID_COLS]
        m = subs[0][['ID'] + data_cols].copy()

        for i, sub in enumerate(subs[1:], 1):
            m = m.merge(
                sub[['ID'] + data_cols].rename(
                    columns={c: f'{c}_{i}' for c in data_cols}
                ),
                on='ID'
            )

        for col in data_cols:
            cols_i = [col] + [f'{col}_{i}' for i in range(1, len(subs))]
            all_agree = (m[cols_i].nunique(axis=1) == 1).mean()
            if all_agree < 0.8:
                print(f"  {col:50} all_agree={all_agree:.0%}")


def assign_char2factor(df: pd.DataFrame):
    pattern = re.compile(r"\[([^\]]+)\]")
    # Build mapping: key -> column name
    factor_cols = {
        pattern.search(col).group(1): col
        for col in df.columns
        if col.startswith("FactorValue[")
    }
    char_cols = {
        pattern.search(col).group(1): col
        for col in df.columns
        if col.startswith("Characteristics[")
    }
    common_keys = factor_cols.keys() & char_cols.keys()
    for key in common_keys:
        df[factor_cols[key]] = df[char_cols[key]]    
    return df


# Define NA-like values
NA_VALUES = {'not available', 'Not Applicable', 'not applicable', '', 'nan', 'NaN', 'NA', 'N/A', 'none', 'None'}


def load_clean_rulesdf(file, columns=None):
    df = pd.read_csv(file)
    if columns is not None:
        df = df.reindex(columns=columns)
        df = df[columns]
    df.replace("Not Applicable", np.nan, inplace=True)
    return assign_char2factor(df)                


def _is_na(v):
    if pd.isna(v):
        return True
    return str(v).strip() in NA_VALUES


# Function to get columns with at least one NA-like value
def cols_with_na_like(df):
    return [col for col in df.columns if df[col].apply(_is_na).any()]


def constant_columns(df: pd.DataFrame, cardinality=1) -> list[str]:
    """Return a list of columns that have the same value in all rows."""
    return [col for col in df.columns if df[col].nunique(dropna=False) <=cardinality]


def get_cols_by_type(cols : []):
    pattern = re.compile(r"\[([^\]]+)\]")
    result = {"FactorValue": [], "Characteristics" : [], "Comment": []}
    # Build mapping: key -> column name
    result["FactorValue"] = {
        pattern.search(col).group(1): col
        for col in cols
        if col.startswith("FactorValue[")
    }
    result["Characteristics"] = {
        pattern.search(col).group(1): col
        for col in cols
        if col.startswith("Characteristics[")
    }
    result["Comment"] = {
        pattern.search(col).group(1): col
        for col in cols
        if col.startswith("Comment[")
    }    
    result["Other"] = [col for col in cols if not pattern.search(col)]
    return result


def compare_unique_values(df1: pd.DataFrame,
                          df2: pd.DataFrame,
                          cols,
                          name1="df1",
                          name2="df2",
                          max_display: int = 10):
    
    for col in cols:
        print(f"\n=== {col} ===")

        if col not in df1.columns or col not in df2.columns:
            print(f"Missing in one of the dataframes")
            continue

        s1 = pd.Series(df1[col].unique())
        s2 = pd.Series(df2[col].unique())

        set1 = set(s1)
        set2 = set(s2)

        only_1 = list(set1 - set2)
        only_2 = list(set2 - set1)
        common = list(set1 & set2)

        def show(vals):
            if len(vals) <= max_display:
                return vals
            return vals[:max_display] + [f"... (+{len(vals)-max_display} more)"]

        print(f"{name1} unique ({len(set1)}): {show(list(set1))}")
        print(f"{name2} unique ({len(set2)}): {show(list(set2))}")
        print(f"common ({len(common)}): {show(common)}")
        if len(only_1)>0:
            print(f"only in {name1} ({len(only_1)}): {show(only_1)}")
        if len(only_2)>0:
            print(f"only in {name2} ({len(only_2)}): {show(only_2)}")


def to_snake_case(col_name: str) -> str:
    """
    Convert a submission column name to snake_case similar to _COL_TO_SNAKE.
    Examples:
        'Characteristics[NumberOfSamples]' -> 'number_of_samples'
        'Characteristics[Modification].2' -> 'modification_2'
        'Comment[MS2MassAnalyzer]'         -> 'ms2_mass_analyzer'
    """
    # Step 1: Remove prefix before [ and keep what's inside
    m = re.match(r'(?:\w+)\[(.+?)\](?:\.(\d+))?', col_name)
    if m:
        base, number = m.groups()
        s = base
        if number:
            s += f"_{number}"
    else:
        s = col_name

    # Step 2: Convert CamelCase or PascalCase to snake_case
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    s = s.replace('-', '_')
    
    # Step 3: lowercase everything
    return s.lower()
                

SCHEMA_DICT: dict[str, str] = {
    # CHARACTERISTICS
    "age": "Age of the donor or developmental stage of the organism (e.g. '45 years', 'E14.5 embryo')",
    "alkylation_reagent": "A chemical (like Iodoacetamide (IAA) or N-ethylmaleimide (NEM)) that irreversibly adds an alkyl group to the free sulfhydryl (-SH) of cysteine residues, blocking disulfide bonds and preventing protein re-folding",
    "anatomic_site_tumor": "Anatomical location from which a tumor sample was taken (e.g. 'left lung lobe')",
    "ancestry_category": "Donor ancestry or ethnicity category (e.g. 'European', 'East Asian')",
    "bait": "The protein or molecule used as bait in an affinity-purification experiment",
    "bmi": "Body-Mass Index of the donor (kg/m²)",
    "biological_replicate": "Identifier for biological replicates (e.g. 'bioRep1', 'bioRep2')",
    "cell_line": "Name of the immortalized cell line (e.g. 'HEK293T', 'U2OS')",
    "cell_part": "Subcellular compartment or fraction (e.g. 'nucleus', 'mitochondria')",
    "cell_type": "Primary cell type or lineage (e.g. 'neurons', 'fibroblasts')",
    "cleavage_agent": "Protease or chemical used to digest proteins (e.g. 'trypsin', 'chymotrypsin')",
    "compound": "Chemical or small molecule added to the sample (e.g. drug, inhibitor) as a perturbation agent",
    "concentration_of_compound": "Concentration of the Compound used (e.g. '10 µM')",
    "concentration_of_compound_1": "Additional Concentration of the Compound used (e.g. '10 µM')",
    "depletion": "Method used to remove high-abundance proteins (e.g. 'albumin depletion kit')",
    "developmental_stage": "Stage of development for the sample source (e.g. 'adult', 'P7 pup')",
    "disease": "Disease state or diagnosis (e.g. 'breast cancer', 'Type 2 diabetes')",
    "disease_treatment": "Pre-treatment applied to diseased samples (e.g. 'chemotherapy', 'radiation')",
    "genetic_modification": "Any genetic alteration in the source organism/cells (e.g. 'GFP-tagged', 'knockout of gene X')",
    "genotype": "Genotypic background (e.g. 'C57BL/6J', 'BRCA1-mutant')",
    "growth_rate": "Doubling time or growth rate of cell cultures (e.g. '24 h doubling')",
    "label": "Isobaric or metabolic label applied (e.g. 'TMT-126', 'SILAC heavy')",
    "material_type": "Broad class of material (e.g. 'tissue', 'cell line', 'biofluid')",
    "modification": "Post-translational modification enrichment or tagging (e.g. 'phosphorylation', 'ubiquitination')",
    "modification_3": "Post-translational modification enrichment or tagging (e.g. 'phosphorylation', 'ubiquitination')",
    "modification_4": "Post-translational modification enrichment or tagging (e.g. 'phosphorylation', 'ubiquitination')",
    "modification_5": "Post-translational modification enrichment or tagging (e.g. 'phosphorylation', 'ubiquitination')",
    "modification_6": "Post-translational modification enrichment or tagging (e.g. 'phosphorylation', 'ubiquitination')",
    "number_of_biological_replicates": "Total number of biological replicates in the study for samples of the same context (not per-sample)",
    "number_of_samples": "Total number of samples processed in total in the study (not per-sample)",
    "number_of_technical_replicates": "Total number of technical replicates for the sample",
    "organism": "Source species (NCBI Taxonomy ID and name, e.g. '9606 (Homo sapiens)')",
    "organism_part": "Tissue or organ of origin (Uberon term, e.g. 'UBERON:0002107 (liver)')",
    "origin_site_disease": "Anatomical site of disease origin (e.g. 'colon', 'prostate')",
    "pooled_sample": "Indicates if multiple samples were pooled (e.g. 'pool1 of reps1–3')",
    "reduction_reagent": "Chemical used to reduce disulfide bonds (e.g. 'DTT', 'TCEP')",
    "sampling_time": "Time point of sample collection (e.g. 'T0', '24 h post-treatment')",
    "sex": "Donor sex (e.g. 'male', 'female')",
    "specimen": "Description of biological specimen (e.g. 'biopsy', 'plasma')",
    "spiked_compound": "Exogenous standard or spike-in added (e.g. 'iRT peptides')",
    "staining": "Any staining applied to the sample prior to mass spec that may still be present in the sample",
    "strain": "Animal strain (e.g. 'BALB/c', 'FVB/N')",
    "synthetic_peptide": "Indicates a synthetic peptide sample (e.g. 'synthetic phosphopeptide')",
    "technical_replicate": "Identifier for technical replicates (e.g. 'techRep1', 'techRep2')",
    "temperature": "Growth temperature of the samples or perturbation temperature if a differential study",
    "time": "Broad time parameter (e.g. 'day 5', 'week 2')",
    "treatment": "Experimental treatment (e.g. 'drug X 5 µM 24 h')",
    "tumor_cellularity": "Percentage of tumor cells in the sample (e.g. '80%')",
    "tumor_grade": "Histological grade (e.g. 'Grade II')",
    "tumor_size": "Physical size of the tumor (e.g. '3 cm diameter')",
    "tumor_site": "Anatomical site of tumor (e.g. 'breast', 'pancreas')",
    "tumor_stage": "Clinical staging (e.g. 'Stage III')",

    # COMMENT
    "acquisition_method": "MS acquisition scheme (e.g. 'DDA', 'DIA', 'PRM')",
    "collision_energy": "Collision energy applied in MS/MS (e.g. '27 eV')",
    "enrichment_method": "Peptide/enrichment protocol used (e.g. 'TiO₂ phosphopeptide enrichment')",
    "flow_rate_chromatogram": "LC flow rate (e.g. '300 nL/min')",
    "fractionation_method": "Any off-line method used to fraction bulk sample into the primary samples used in the MS or LC/MS",
    "fraction_identifier": "Numeric or text ID of each fraction (e.g. 'F1', 'F2')",
    "fragmentation_method": "Ion-fragmentation technique (e.g. 'HCD', 'CID', 'ETD')",
    "fragment_mass_tolerance": "Mass tolerance for fragment matching (e.g. '0.02 Da')",
    "gradient_time": "Total LC gradient length (e.g. '120 min')",
    "instrument": "Mass spec make/model (e.g. 'Thermo Q-Exactive Plus')",
    "ionization_type": "Ionization source (e.g. 'nanoESI', 'MALDI')",
    "ms2_mass_analyzer": "Analyzer used for MS2 (e.g. 'orbitrap', 'ion trap')",
    "number_of_missed_cleavages": "Max missed cleavages allowed in database search (e.g. '2')",
    "number_of_fractions": "Total number of fractions generated from each sample",
    "precursor_mass_tolerance": "Mass tolerance for precursor matching (e.g. '10 ppm')",
    "separation": "Any on-line method used to separate the samples into fractions right before MS",
    
   "factor_bait"                   :  "Experimental factor: protein or molecule used as bait in affinity-purification",
    "factor_cell_part"               :  "Experimental factor: subcellular compartment or fraction (e.g. 'nucleus', 'mitochondria')",
    "factor_compound"                :  "Experimental factor:chemical or small molecule used as perturbation",
    "factor_concentration_of_compound_1":  "Experimental factor:concentration of the compound in the sample (e.g. 10 µM)",
    "factor_fraction_identifier"     :  "Experimental factor: numeric/text ID of each fraction",
    "factor_genetic_modification"    :  "Experimental factor:genetic alteration (e.g. 'GFP-tagged', 'knockout of gene X')",
    "factor_temperature"             :  "Experimental factor: experimental or perturbation temperature",
    "factor_treatment"               :  "Experimental factor: experimental treatment applied",
    "factor_chemical_entity"         :  "Experimental factor: chemical entity applied to sample",
    "factor_induced_by"              :  "Experimental factor: stimulus/agent inducing the condition",
    "factor_isolation_width"         :  "Experimental factor: MS isolation width parameter",
    "factor_multiplicities_of_infection" :  "Experimental factor: infection MOI (e.g. 1, 5)",
    "factor_overproduction"          :  "Experimental factor: protein overexpression condition",
    "factor_overproduction_1"        :  "Experimental factor: secondary overproduction (if multiple)",
    "factor_protocol"                :  "Experimental factor: experimental protocol or procedure" 
}


# Modules - will be optimised separately
# Sample-level field → module mapping
SAMPLE_FIELD_MODULES = {
    # Module 1 — Basic Demographics / Organism Info
    "age": "demographics",
    "sex": "demographics",
    "bmi": "demographics",
    "developmental_stage": "demographics",
    "strain": "demographics",
    "genotype": "demographics",
    "ancestry_category": "demographics",
    "organism": "demographics",
    "organism_part": "demographics",

    # Module 2 — Sample Source / Tissue / Tumor
    "anatomic_site_tumor": "tissue_tumor",
    "tumor_site": "tissue_tumor",
    "tumor_stage": "tissue_tumor",
    "tumor_grade": "tissue_tumor",
    "tumor_size": "tissue_tumor",
    "tumor_cellularity": "tissue_tumor",
    "origin_site_disease": "tissue_tumor",
    "specimen": "tissue_tumor",
    "pooled_sample": "tissue_tumor",

    # Module 3 — Cell / Material Type
    "cell_line": "cell_material",
    "cell_type": "cell_material",
    "cell_part": "cell_material",
    "material_type": "cell_material",

    # Module 4 — Treatment / Experiment Conditions
    "treatment": "treatment",
    "compound": "treatment",
    "concentration_of_compound": "treatment",
    "temperature": "treatment",
    "time": "treatment",
    "disease_treatment": "treatment",
    "sampling_time": "treatment",
    "spiked_compound": "treatment",
    "depletion": "treatment",
    "staining": "treatment",
    "label": "treatment",
    "synthetic_peptide": "treatment",

    # Module 5 — Sample Preparation / Chemistry
    "cleavage_agent": "prep_chemistry",
    "reduction_reagent": "prep_chemistry",
    "alkylation_reagent": "prep_chemistry",
    "modification_6": "prep_chemistry",

    # Module 6 — Replicates / Numbers
    "number_of_samples": "replicates",
    "number_of_biological_replicates": "replicates",
    "number_of_technical_replicates": "replicates"
}

GLOBAL_FIELD_MODULES = {
    # Experiment-level — MS / instrument
    "instrument": "ms_instrument",
    "acquisition_method": "ms_instrument",
    "fragmentation_method": "ms_instrument",
    "ms2_mass_analyzer": "ms_instrument",
    "ionization_type": "ms_instrument",
    "collision_energy": "ms_instrument",

    # Experiment-level — LC / separation
    "flow_rate_chromatogram": "chromatography",
    "gradient_time": "chromatography",
    "separation": "chromatography",
    "fraction_identifier": "chromatography",
    "fractionation_method": "chromatography",
    "number_of_fractions": "chromatography",

    # Experiment-level — Search / digestion
    "precursor_mass_tolerance": "search_params",
    "fragment_mass_tolerance": "search_params",
    "number_of_missed_cleavages": "search_params",
    "enrichment_method": "search_params",
}