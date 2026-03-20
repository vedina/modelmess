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