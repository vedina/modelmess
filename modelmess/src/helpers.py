"""
Helpers to clean output → validate with Pydantic → produce SDRFRow list.
"""
import re
import json
import logging
from datamodel import (
    InstrumentRef, CleavageAgent, ProteinModification, SDRFRow
)


log = logging.getLogger('sdrf')


def _fix_json_string(text: str) -> str:
    """
    Best-effort repair of common LLM JSON formatting problems:
      1. Python None/True/False  -> null/true/false
      2. Single-quoted strings   -> double-quoted
      3. Trailing commas         -> removed
      4. Truncated JSON          -> close open braces/brackets
    """
    # 0. Fix invalid backslash escapes (e.g. \a from unicode \u00a0 mangling)
    #    Walk char-by-char: only double-escape \ not followed by a valid JSON
    #    escape char (" \ / b f n r t u). Preserves \n, \t, \uXXXX, \\.
    _fixed = []
    _i = 0
    while _i < len(text):
        if text[_i] == '\\':
            _nxt = text[_i+1] if _i+1 < len(text) else ''
            if _nxt in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                _fixed.append(text[_i]); _fixed.append(_nxt); _i += 2
            else:
                _fixed.append('\\\\'); _i += 1   # invalid → escaped
        else:
            _fixed.append(text[_i]); _i += 1
    text = ''.join(_fixed)

    # 1. Python literals — simple string replacement covers the common patterns
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
                        out.append(text[i]); i += 1   # keep escape sequence
                    elif text[i] == '"':
                        out.append('\\"')              # escape inner double-quote
                    else:
                        out.append(text[i])
                    i += 1
                out.append('"')
                i += 1   # skip closing single-quote
            else:
                out.append(ch)
                i += 1
        text = ''.join(out)

    # 3. Trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # 4. Close truncated JSON
    opens_b = text.count('{') - text.count('}')
    opens_k = text.count('[') - text.count(']')
    if opens_b > 0 or opens_k > 0:
        # Find last COMPLETE value: closing quote+comma is safer than bare comma
        candidates = [
            text.rfind('",'),   # end of string value, more to follow
            text.rfind('"}'),   # end of string value, closing object
            text.rfind('",\n'), # same with newline
            text.rfind('},'),   # end of sub-object
            text.rfind('],'),   # end of array
            text.rfind(','),    # plain comma fallback
            text.rfind('{'),
            text.rfind('['),
        ]
        boundary = max(c for c in candidates if c >= 0) if any(c >= 0 for c in candidates) else -1
        if boundary > 0:
            text = text[:boundary]
        text = text.rstrip(' ,\n\r\t')
        opens_b = text.count('{') - text.count('}')
        opens_k = text.count('[') - text.count(']')
        text += '}' * max(opens_b, 0) + ']' * max(opens_k, 0)

    return text


def clean_json(text: str) -> str:
    """Strip markdown, repair, extract first JSON object or array (prefers object)."""
    text = re.sub(r'```(?:json)?', '', text)
    text = re.sub(r'```', '', text).strip()
    for pattern in (r'\{.*\}', r'\[.*\]'):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return _fix_json_string(m.group(0))
    return _fix_json_string(text)


def clean_json_array(text: str) -> str:
    """Strip markdown, repair, extract first JSON array (prefers array)."""
    text = re.sub(r'```(?:json)?', '', text)
    text = re.sub(r'```', '', text).strip()
    for pattern in (r'\[.*\]', r'\{.*\}'):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return _fix_json_string(m.group(0))
    return _fix_json_string(text)


def _coerce_to_dict(data) -> dict:
    """
    Turn whatever the LLM returned into a plain dict.
    Handles: dict, single-element list, list of {k:v} dicts, list of [k,v] pairs.
    """
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        if all(isinstance(x, dict) for x in data):
            merged = {}
            for item in data:
                merged.update(item)
            return merged
        if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in data):
            return dict(data)
    log.warning(f"Cannot coerce to dict: {type(data)} — {str(data)[:200]}")
    return {}


def _serialise_globals(globals_dict: dict) -> str:
    """Serialise globals dict (may contain Pydantic sub-models) to JSON string."""
    out = {}
    for k, v in globals_dict.items():
        if hasattr(v, 'model_dump'):
            out[k] = v.model_dump()
        elif isinstance(v, list):
            out[k] = [x.model_dump() if hasattr(x, 'model_dump') else x for x in v]
        else:
            out[k] = v
    return json.dumps(out, default=str)


def _parse_kv_string(s: str) -> dict:
    """
    Parse NT=...;AC=...;TA=... string into a dict.
    Also handles plain name strings like 'Trypsin' -> {'NT': 'Trypsin'}.
    """
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


def _coerce_instrument(v):
    """Accept dict, kv-string, or plain name → InstrumentRef (or None)."""
    if v is None:
        return None
    if isinstance(v, InstrumentRef):
        return v
    if isinstance(v, str):
        v = _parse_kv_string(v)
    if isinstance(v, dict) and v:
        try:
            return InstrumentRef(**v)
        except Exception as e:
            log.warning(f"InstrumentRef coerce failed: {e} — {v}")
    return None


def _coerce_cleavage(v):
    """Accept dict, kv-string, or plain name → CleavageAgent (or None)."""
    if v is None:
        return None
    if isinstance(v, CleavageAgent):
        return v
    if isinstance(v, str):
        v = _parse_kv_string(v)
    if isinstance(v, dict) and v:
        try:
            return CleavageAgent(**v)
        except Exception as e:
            log.warning(f"CleavageAgent coerce failed: {e} — {v}")
    return None


def _coerce_modification(m):
    """Accept dict, kv-string, or plain name → ProteinModification (or None)."""
    if m is None:
        return None
    if isinstance(m, ProteinModification):
        return m
    if isinstance(m, str):
        m = _parse_kv_string(m)
    if isinstance(m, dict) and m:
        if 'NT' not in m and 'name' in m:
            m['NT'] = m.pop('name')
        if 'NT' not in m:
            log.warning(f"ProteinModification missing NT: {m}")
            return None
        try:
            return ProteinModification(**m)
        except Exception as e:
            log.warning(f"ProteinModification coerce failed: {e} — {m}")
    return None


def parse_globals(raw_json: str) -> dict:
    """Parse and validate globals JSON. Returns a plain dict with Pydantic sub-models."""
    cleaned = clean_json(raw_json)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"globals JSON parse error: {e}. Raw: {cleaned[:300]}")
        try:
            import ast as _ast
            data = _ast.literal_eval(cleaned)
            log.info("globals recovered via ast.literal_eval")
        except Exception:
            return {}

    data = _coerce_to_dict(data)

    if 'instrument' in data:
        data['instrument'] = _coerce_instrument(data['instrument'])
    if 'cleavage_agent' in data:
        data['cleavage_agent'] = _coerce_cleavage(data['cleavage_agent'])
    if 'modifications' in data and isinstance(data['modifications'], list):
        data['modifications'] = [
            m2 for m in data['modifications']
            if (m2 := _coerce_modification(m)) is not None
        ]
    return data


def parse_samples(raw_json: str, globals_dict: dict,
                  file_list: list, pxd: str) -> list:
    """
    Parse samples JSON array → list of validated SDRFRow objects.
    Merges experiment globals into each row.
    Falls back to minimal rows per file if extraction fails.
    """
    cleaned = clean_json_array(raw_json)
    try:
        samples = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"[{pxd}] samples JSON parse error: {e}. Raw: {cleaned[:300]}")
        try:
            import ast as _ast
            samples = _ast.literal_eval(cleaned)
            log.info(f"[{pxd}] samples recovered via ast.literal_eval")
        except Exception:
            samples = []

    if isinstance(samples, dict):
        samples = [samples]
    elif not isinstance(samples, list):
        samples = []

    rows = []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            continue
        merged = {**globals_dict, **s}

        if not merged.get('raw_data_file') and file_list:
            merged['raw_data_file'] = file_list[i % len(file_list)]
        if not merged.get('source_name'):
            merged['source_name'] = f"Sample {i+1}"
        if not merged.get('assay_name'):
            merged['assay_name'] = f"run {i+1}"
        if not merged.get('organism'):
            merged['organism'] = 'not available'

        # Coerce sub-model fields that arrived as strings
        for key, coerce_fn in (('instrument',    _coerce_instrument),
                                ('cleavage_agent', _coerce_cleavage)):
            if key in merged and isinstance(merged[key], str):
                merged[key] = coerce_fn(merged[key])
        if 'modifications' in merged and isinstance(merged['modifications'], list):
            merged['modifications'] = [
                m2 for m in merged['modifications']
                if (m2 := _coerce_modification(m)) is not None
            ]

        # Strip None values so Pydantic field defaults kick in
        # (model returning null for 'label' must not override default=LABEL_FREE)
        merged = {k: v for k, v in merged.items() if v is not None}

        # Coerce NT=... kv-strings in plain scalar string fields
        for _pf in ('fragmentation_method', 'acquisition_method',
                    'ionization_type', 'ms2_mass_analyzer', 'enrichment_method'):
            _v = merged.get(_pf)
            if isinstance(_v, str) and '=' in _v:
                merged[_pf] = _parse_kv_string(_v).get('NT', _v)

        try:
            rows.append(SDRFRow(**merged))
        except Exception as e:
            log.warning(f"[{pxd}] Row {i} validation failed: {e}. merged={str(merged)[:200]}")

    # Fallback: produce minimal rows for every file
    if not rows and file_list:
        log.warning(f"[{pxd}] Falling back to minimal rows for {len(file_list)} files.")
        for i, fname in enumerate(file_list):
            minimal = {
                k: v for k, v in globals_dict.items()
                if k in ('organism', 'label', 'instrument', 'cleavage_agent',
                         'modifications', 'fragmentation_method',
                         'precursor_mass_tolerance', 'fragment_mass_tolerance')
            }
            minimal.update({
                'source_name'  : f"Sample {i+1}",
                'assay_name'   : f"run {i+1}",
                'raw_data_file': fname,
                'organism'     : globals_dict.get('organism', 'not available')
                                 if isinstance(globals_dict.get('organism'), str)
                                 else 'not available',
                'usage'        : 'Raw Data File',
            })
            try:
                rows.append(SDRFRow(**minimal))
            except Exception as e:
                log.warning(f"[{pxd}] minimal row {i} failed: {e}")
    return rows

