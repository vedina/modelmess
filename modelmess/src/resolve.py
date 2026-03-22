"""
resolve.py — Deterministic structure resolver for LLM globals output.

Bridges the gap between what the LLM produces (semantic names, plain strings,
partial or malformed NT/AC dicts) and what the SDRF format requires (canonical
NT=name;AC=accession kv-strings, correct UNIMOD accessions, Fixed/Variable
capitalisation, etc.).

The LLM's job is reduced to recognising concepts correctly.
This module handles all formatting deterministically.

Call `resolve_globals_structure(raw_dict)` immediately after `parse_globals()`
in helpers.py — before the dict is serialised or passed to the TSV writer.

Design principles
-----------------
- Static tables first (fast, offline, deterministic, testable)
- OLS UNIMOD cache as fallback for unknown modifications
- Never silently discard an LLM value — if lookup fails, keep it as-is
- All functions are pure (no side effects, no global state)

Public API
----------
    resolve_globals_structure(d: dict, ols_client=None) -> dict
        Main entry point. Returns a new dict with all structured fields resolved.

    resolve_modification(entry, ols_client=None) -> dict
        Resolve a single modification entry (dict or plain string).

    resolve_instrument(value: str, ols_client=None) -> dict
        Resolve an instrument name to {NT, AC} dict.

    resolve_cleavage_agent(value: str) -> str
        Resolve enzyme name(s) to canonical NT=;AC= kv-string.
"""

from __future__ import annotations

import re
import logging
import difflib
from typing import Optional, Union

log = logging.getLogger('sdrf.resolve')

# ---------------------------------------------------------------------------
# Modification table
# ---------------------------------------------------------------------------
# Keys are normalised (lowercase, no spaces/punctuation) for fuzzy matching.
# MT_default is used only when the LLM did not supply MT at all.
# TA may be a comma-separated list of one-letter amino acid codes.
# PP (position) is Anywhere unless the modification is specifically N/C-terminal.

_MOD_TABLE: dict[str, dict] = {
    # ── Alkylation ────────────────────────────────────────────────────────
    'carbamidomethyl':          {'NT': 'Carbamidomethyl',   'AC': 'UNIMOD:4',   'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'carbamidomethylation':     {'NT': 'Carbamidomethyl',   'AC': 'UNIMOD:4',   'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'iodoacetamide':            {'NT': 'Carbamidomethyl',   'AC': 'UNIMOD:4',   'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'iaa':                      {'NT': 'Carbamidomethyl',   'AC': 'UNIMOD:4',   'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'propionamide':             {'NT': 'Propionamide',      'AC': 'UNIMOD:24',  'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'nethylmaleimide':          {'NT': 'Nethylmaleimide',   'AC': 'UNIMOD:108', 'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'nem':                      {'NT': 'Nethylmaleimide',   'AC': 'UNIMOD:108', 'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'chloroacetamide':          {'NT': 'Carbamidomethyl',   'AC': 'UNIMOD:4',   'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'caa':                      {'NT': 'Carbamidomethyl',   'AC': 'UNIMOD:4',   'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Fixed'},

    # ── Oxidation / reduction ─────────────────────────────────────────────
    'oxidation':                {'NT': 'Oxidation',         'AC': 'UNIMOD:35',  'TA': 'M',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'oxidizedmethionine':       {'NT': 'Oxidation',         'AC': 'UNIMOD:35',  'TA': 'M',       'PP': 'Anywhere',       'MT_default': 'Variable'},

    # ── Phosphorylation ───────────────────────────────────────────────────
    'phospho':                  {'NT': 'Phospho',           'AC': 'UNIMOD:21',  'TA': 'S,T,Y',   'PP': 'Anywhere',       'MT_default': 'Variable'},
    'phosphorylation':          {'NT': 'Phospho',           'AC': 'UNIMOD:21',  'TA': 'S,T,Y',   'PP': 'Anywhere',       'MT_default': 'Variable'},
    'phosphoserine':            {'NT': 'Phospho',           'AC': 'UNIMOD:21',  'TA': 'S',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'phosphothreonine':         {'NT': 'Phospho',           'AC': 'UNIMOD:21',  'TA': 'T',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'phosphotyrosine':          {'NT': 'Phospho',           'AC': 'UNIMOD:21',  'TA': 'Y',       'PP': 'Anywhere',       'MT_default': 'Variable'},

    # ── Acetylation ───────────────────────────────────────────────────────
    'acetyl':                   {'NT': 'Acetyl',            'AC': 'UNIMOD:1',   'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'acetylation':              {'NT': 'Acetyl',            'AC': 'UNIMOD:1',   'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'acetylproteinnterminus':   {'NT': 'Acetyl',            'AC': 'UNIMOD:1',   'TA': '',        'PP': 'Protein N-term',  'MT_default': 'Variable'},
    'acetylnterm':              {'NT': 'Acetyl',            'AC': 'UNIMOD:1',   'TA': '',        'PP': 'Protein N-term',  'MT_default': 'Variable'},

    # ── Methylation ───────────────────────────────────────────────────────
    'methylation':              {'NT': 'Methylation',       'AC': 'UNIMOD:34',  'TA': 'K,R',     'PP': 'Anywhere',       'MT_default': 'Variable'},
    'dimethyl':                 {'NT': 'Dimethyl',          'AC': 'UNIMOD:36',  'TA': 'K,R',     'PP': 'Anywhere',       'MT_default': 'Variable'},
    'dimethylation':            {'NT': 'Dimethyl',          'AC': 'UNIMOD:36',  'TA': 'K,R',     'PP': 'Anywhere',       'MT_default': 'Variable'},
    'trimethyl':                {'NT': 'Trimethyl',         'AC': 'UNIMOD:37',  'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'trimethylation':           {'NT': 'Trimethyl',         'AC': 'UNIMOD:37',  'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'monomethyl':               {'NT': 'Methylation',       'AC': 'UNIMOD:34',  'TA': 'K,R',     'PP': 'Anywhere',       'MT_default': 'Variable'},

    # ── Deamidation ───────────────────────────────────────────────────────
    'deamidated':               {'NT': 'Deamidated',        'AC': 'UNIMOD:7',   'TA': 'N,Q',     'PP': 'Anywhere',       'MT_default': 'Variable'},
    'deamidation':              {'NT': 'Deamidated',        'AC': 'UNIMOD:7',   'TA': 'N,Q',     'PP': 'Anywhere',       'MT_default': 'Variable'},

    # ── Ubiquitination ────────────────────────────────────────────────────
    'glygly':                   {'NT': 'GlyGly',            'AC': 'UNIMOD:121', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'ubiquitination':           {'NT': 'GlyGly',            'AC': 'UNIMOD:121', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'diglycine':                {'NT': 'GlyGly',            'AC': 'UNIMOD:121', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},

    # ── TMT labels ────────────────────────────────────────────────────────
    'tmt6plex':                 {'NT': 'TMT6plex',          'AC': 'UNIMOD:737', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'tmt':                      {'NT': 'TMT6plex',          'AC': 'UNIMOD:737', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'tmt10plex':                {'NT': 'TMT6plex',          'AC': 'UNIMOD:737', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'tmt11plex':                {'NT': 'TMT6plex',          'AC': 'UNIMOD:737', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'tmt16plex':                {'NT': 'TMT6plex',          'AC': 'UNIMOD:737', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'tmt6plexonk':              {'NT': 'TMT6plex',          'AC': 'UNIMOD:737', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'tmt6plexonnterm':          {'NT': 'TMT6plex',          'AC': 'UNIMOD:737', 'TA': '',        'PP': 'Any N-term',      'MT_default': 'Fixed'},

    # ── iTRAQ labels ──────────────────────────────────────────────────────
    'itraq4plex':               {'NT': 'iTRAQ4plex',        'AC': 'UNIMOD:214', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'itraq8plex':               {'NT': 'iTRAQ8plex',        'AC': 'UNIMOD:730', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'itraq':                    {'NT': 'iTRAQ4plex',        'AC': 'UNIMOD:214', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},

    # ── SILAC labels ──────────────────────────────────────────────────────
    'silacheavylysine':         {'NT': 'Label:13C(6)15N(2)', 'AC': 'UNIMOD:259','TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'silacheavyarginine':       {'NT': 'Label:13C(6)15N(4)', 'AC': 'UNIMOD:267','TA': 'R',       'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'silacmediumlysine':        {'NT': 'Label:2H(4)',        'AC': 'UNIMOD:481', 'TA': 'K',      'PP': 'Anywhere',       'MT_default': 'Fixed'},
    'silacmediumarginine':      {'NT': 'Label:13C(6)',       'AC': 'UNIMOD:188', 'TA': 'R',      'PP': 'Anywhere',       'MT_default': 'Fixed'},

    # ── Pyro modifications ────────────────────────────────────────────────
    'pyroglu':                  {'NT': 'Glu->pyro-Glu',     'AC': 'UNIMOD:27',  'TA': 'E',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'pyroglufromq':             {'NT': 'Gln->pyro-Glu',     'AC': 'UNIMOD:28',  'TA': 'Q',       'PP': 'Any N-term',     'MT_default': 'Variable'},
    'pyrocarbamidomethyl':      {'NT': 'Pyro-carbamidomethyl','AC':'UNIMOD:26',  'TA': 'C',       'PP': 'Any N-term',     'MT_default': 'Variable'},

    # ── Misc common ───────────────────────────────────────────────────────
    'ammonia-loss':             {'NT': 'Ammonia-loss',      'AC': 'UNIMOD:385', 'TA': 'Q,C,N',   'PP': 'Anywhere',       'MT_default': 'Variable'},
    'ammonialoss':              {'NT': 'Ammonia-loss',      'AC': 'UNIMOD:385', 'TA': 'Q,C,N',   'PP': 'Anywhere',       'MT_default': 'Variable'},
    'sulfo':                    {'NT': 'Sulfo',             'AC': 'UNIMOD:40',  'TA': 'S,T,Y',   'PP': 'Anywhere',       'MT_default': 'Variable'},
    'sulfation':                {'NT': 'Sulfo',             'AC': 'UNIMOD:40',  'TA': 'S,T,Y',   'PP': 'Anywhere',       'MT_default': 'Variable'},
    'formylation':              {'NT': 'Formyl',            'AC': 'UNIMOD:122', 'TA': 'K,S,T',   'PP': 'Anywhere',       'MT_default': 'Variable'},
    'succinylation':            {'NT': 'Succinyl',          'AC': 'UNIMOD:64',  'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'propionylation':           {'NT': 'Propionyl',         'AC': 'UNIMOD:58',  'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'crotonylation':            {'NT': 'Crotonyl',          'AC': 'UNIMOD:1363','TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'hydroxyproline':           {'NT': 'Hydroxy',           'AC': 'UNIMOD:35',  'TA': 'P',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'glutathione':              {'NT': 'Glutathione',       'AC': 'UNIMOD:55',  'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'nitrosylation':            {'NT': 'Nitrosyl',          'AC': 'UNIMOD:275', 'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'citrullination':           {'NT': 'Citrullination',    'AC': 'UNIMOD:7',   'TA': 'R',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'farnesylation':            {'NT': 'Farnesyl',          'AC': 'UNIMOD:44',  'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'myristoylation':           {'NT': 'Myristoyl',         'AC': 'UNIMOD:45',  'TA': 'K,G',     'PP': 'Anywhere',       'MT_default': 'Variable'},
    'palmitoylation':           {'NT': 'Palmitoyl',         'AC': 'UNIMOD:47',  'TA': 'C,K,S,T', 'PP': 'Anywhere',       'MT_default': 'Variable'},
    'geranylgeranylation':      {'NT': 'GeranylGeranyl',    'AC': 'UNIMOD:48',  'TA': 'C',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'biotinylation':            {'NT': 'Biotin',            'AC': 'UNIMOD:3',   'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'sumoylation':              {'NT': 'SUMO',              'AC': 'UNIMOD:1301','TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'neddylation':              {'NT': 'NEDD8',             'AC': 'UNIMOD:1303','TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'dihydroalanine':           {'NT': 'Didehydro',         'AC': 'UNIMOD:401', 'TA': 'C,S',     'PP': 'Anywhere',       'MT_default': 'Variable'},
    'crosslink':                {'NT': 'Xlink:SSD',         'AC': 'UNIMOD:209', 'TA': 'K',       'PP': 'Anywhere',       'MT_default': 'Variable'},
    'heavyisotope':             {'NT': 'Label:18O(1)',      'AC': 'UNIMOD:258', 'TA': '',        'PP': 'Anywhere',       'MT_default': 'Variable'},
    'o18':                      {'NT': 'Label:18O(1)',      'AC': 'UNIMOD:258', 'TA': '',        'PP': 'Anywhere',       'MT_default': 'Variable'},
    'o18x2':                    {'NT': 'Label:18O(2)',      'AC': 'UNIMOD:193', 'TA': '',        'PP': 'Anywhere',       'MT_default': 'Variable'},
}


def _normalise_key(s: str) -> str:
    """Lowercase, strip whitespace and punctuation for table lookup."""
    return re.sub(r'[\s\-_/,\.\(\)]+', '', s.lower())


def _extract_name(entry) -> str:
    """Pull the human-readable name out of whatever the LLM produced."""
    if isinstance(entry, str):
        # Could be 'Carbamidomethyl', 'NT=Carbamidomethyl;AC=UNIMOD:4', plain text
        if 'NT=' in entry:
            parts = [p for p in entry.split(';') if p.strip().startswith('NT=')]
            if parts:
                return parts[0].replace('NT=', '').strip()
        return entry.strip()
    if isinstance(entry, dict):
        # Prefer NT, then name, then first string value found
        for k in ('NT', 'name', 'Name', 'modification', 'mod'):
            if k in entry and isinstance(entry[k], str):
                return entry[k].strip()
        for v in entry.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ''


def _extract_mt(entry) -> Optional[str]:
    """Pull Fixed/Variable out of LLM entry. Returns None if not found."""
    if isinstance(entry, str):
        m = re.search(r'\b(fixed|variable)\b', entry, re.IGNORECASE)
        return m.group(1).capitalize() if m else None
    if isinstance(entry, dict):
        for k in ('MT', 'mt', 'type', 'Type', 'mod_type', 'modification_type'):
            v = entry.get(k, '')
            if isinstance(v, str) and re.search(r'\b(fixed|variable)\b', v, re.IGNORECASE):
                return 'Fixed' if v.lower().startswith('f') else 'Variable'
    return None


def _extract_ta(entry) -> Optional[str]:
    """Pull target amino acid(s) out of LLM entry. Returns None if not found."""
    if isinstance(entry, dict):
        for k in ('TA', 'ta', 'target', 'Target', 'residue', 'Residue',
                  'amino_acid', 'site', 'position'):
            v = entry.get(k, '')
            if isinstance(v, str) and re.match(r'^[A-Z,]+$', v.strip()):
                return v.strip()
    if isinstance(entry, str):
        # e.g. "Carbamidomethyl on C" or "Phospho (S/T/Y)"
        m = re.search(r'\bon\s+([A-Z](?:,\s*[A-Z])*)', entry)
        if m:
            return m.group(1).replace(' ', '')
        m = re.search(r'\(([ACDEFGHIKLMNPQRSTVWY](?:[,/][ACDEFGHIKLMNPQRSTVWY])*)\)', entry)
        if m:
            return m.group(1).replace('/', ',')
    return None


def _fuzzy_lookup(name: str) -> Optional[dict]:
    """Lookup modification by normalised name with fuzzy fallback."""
    key = _normalise_key(name)

    # Exact match first
    if key in _MOD_TABLE:
        return _MOD_TABLE[key]

    # Fuzzy match (difflib, threshold 0.82)
    best_score = 0.0
    best_entry = None
    for table_key, entry in _MOD_TABLE.items():
        score = difflib.SequenceMatcher(None, key, table_key).ratio()
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= 0.82:
        log.debug(f'Fuzzy mod match: {name!r} → {best_entry["NT"]} (score={best_score:.2f})')
        return best_entry

    return None


def resolve_modification(
    entry: Union[str, dict],
    ols_client=None,
) -> dict:
    """
    Resolve a single modification entry (string or dict from LLM) to a
    fully-structured SDRF modification dict with NT, AC, TA, MT, PP.

    The LLM's Fixed/Variable and target residue override table defaults
    when present, since those are context-specific.

    Falls back to OLS UNIMOD search if not found in static table.
    Returns the raw entry (wrapped in a dict) if all lookups fail.
    """
    name = _extract_name(entry)
    if not name:
        log.warning(f'resolve_modification: cannot extract name from {entry!r}')
        return entry if isinstance(entry, dict) else {'NT': str(entry)}

    # --- Static table lookup ---
    template = _fuzzy_lookup(name)

    # --- OLS fallback ---
    if template is None and ols_client is not None:
        try:
            hits = ols_client.cache_search(name, 'unimod', full_search=False)
            if not hits:
                hits = ols_client.cache_search(name, 'unimod', full_search=True)
            if hits:
                h   = hits[0]
                acc = h.get('obo_id', h.get('accession', ''))
                lbl = h.get('label', name)
                template = {'NT': lbl, 'AC': acc, 'TA': '', 'PP': 'Anywhere', 'MT_default': 'Variable'}
        except Exception as e:
            log.debug(f'OLS mod lookup failed for {name!r}: {e}')

    if template is None:
        # Can't resolve — keep what the LLM gave us, just ensure dict form
        log.debug(f'resolve_modification: no match for {name!r}, keeping as-is')
        if isinstance(entry, dict):
            return {k: v for k, v in entry.items() if k != 'MT_default'}
        return {'NT': name}

    # --- Build resolved dict ---
    resolved = {
        'NT': template['NT'],
        'AC': template['AC'],
    }

    # PP: table value (N-terminal mods must keep it)
    pp = template.get('PP', 'Anywhere')
    if pp and pp != 'Anywhere':
        resolved['PP'] = pp

    # TA: LLM override > table value (LLM may know exact residue for multi-site mods)
    llm_ta = _extract_ta(entry)
    ta = llm_ta if llm_ta else template.get('TA', '')
    if ta:
        resolved['TA'] = ta

    # MT: LLM value > table default
    llm_mt = _extract_mt(entry)
    mt = llm_mt if llm_mt else template.get('MT_default', 'Variable')
    resolved['MT'] = mt

    return resolved


# ---------------------------------------------------------------------------
# Instrument resolver
# ---------------------------------------------------------------------------

# Ordered from most to least specific — first match wins.
_INSTRUMENT_TABLE = [
    (r'orbitrap\s*astral',                   'Orbitrap Astral',          'MS:1003378'),
    (r'q\s*exactive\s*hf[\s-]*x',           'Q Exactive HF-X',          'MS:1002877'),
    (r'q\s*exactive\s*hf\b',                'Q Exactive HF',            'MS:1002523'),
    (r'q\s*exactive\s*plus',                'Q Exactive Plus',           'MS:1002634'),
    (r'q\s*exactive',                        'Q Exactive',               'MS:1001911'),
    (r'orbitrap\s*fusion\s*lumos',          'Orbitrap Fusion Lumos',     'MS:1002732'),
    (r'orbitrap\s*fusion\s*eclipse',        'Orbitrap Eclipse',          'MS:1003029'),
    (r'orbitrap\s*fusion',                  'Orbitrap Fusion',           'MS:1002416'),
    (r'orbitrap\s*exploris\s*480',          'Orbitrap Exploris 480',     'MS:1003028'),
    (r'orbitrap\s*exploris\s*240',          'Orbitrap Exploris 240',     'MS:1003360'),
    (r'orbitrap\s*elite',                   'LTQ Orbitrap Elite',        'MS:1001910'),
    (r'orbitrap\s*velos',                   'LTQ Orbitrap Velos',        'MS:1001742'),
    (r'ltq\s*orbitrap\b',                   'LTQ Orbitrap',              'MS:1000449'),
    (r'ltq\s*xl',                           'LTQ XL',                   'MS:1000854'),
    (r'\bltq\b',                            'LTQ',                      'MS:1000447'),
    (r'timstof\s*pro\s*2',                  'timsTOF Pro 2',             'MS:1003230'),
    (r'timstof\s*pro',                      'timsTOF Pro',               'MS:1003005'),
    (r'timstof\s*scp',                      'timsTOF SCP',               'MS:1003231'),
    (r'\btimstof\b',                        'timsTOF',                   'MS:1002817'),
    (r'mabaldia\s*ims',                     'MALDI-2 timsTOF',           'MS:1003380'),
    (r'triple\s*tof\s*6600',               'TripleTOF 6600',            'MS:1002533'),
    (r'triple\s*tof\s*5600',               'TripleTOF 5600',            'MS:1002532'),
    (r'sciex\s*7600',                       'ZenoTOF 7600',              'MS:1003294'),
    (r'synapt\s*g2[-\s]*s',               'Synapt G2-S MS',             'MS:1002726'),
    (r'synapt\s*g2',                       'Synapt G2 MS',              'MS:1002280'),
    (r'\bsynapt\b',                        'Synapt MS',                 'MS:1001490'),
    (r'xevo\s*g2',                         'Xevo G2 QTOF',              'MS:1001535'),
    (r'maxi\s*ms',                         'maXis',                     'MS:1001534'),
    (r'impact\s*ii',                       'impact II',                 'MS:1002818'),
    (r'\bimpact\b',                        'impact',                    'MS:1002055'),
    (r'velos\s*pro',                       'Velos Pro',                 'MS:1001820'),
    (r'eclipse',                           'Orbitrap Eclipse',          'MS:1003029'),
    (r'exploris',                          'Orbitrap Exploris 480',     'MS:1003028'),
]


def resolve_instrument(value: str, ols_client=None) -> dict:
    """
    Resolve an instrument name (plain string or partial NT=... string) to
    {'NT': canonical_name, 'AC': 'MS:xxxxxxx'}.

    Falls back to OLS psi-ms search if not found in static table.
    Returns {'NT': value} (unresolved) if all lookups fail.
    """
    if not value:
        return {}

    # Extract plain name if already in kv-string format
    name = value
    if 'NT=' in value:
        parts = [p for p in value.split(';') if p.strip().startswith('NT=')]
        name  = parts[0].replace('NT=', '').strip() if parts else value

    check = name.lower().strip()

    # If it's already an AC= string with accession, trust it
    if re.search(r'AC=MS:\d+', value):
        ac_m = re.search(r'AC=(MS:\d+)', value)
        nt_m = re.search(r'NT=([^;]+)', value)
        if ac_m and nt_m:
            return {'NT': nt_m.group(1).strip(), 'AC': ac_m.group(1).strip()}

    # Static table
    for pattern, canonical_name, accession in _INSTRUMENT_TABLE:
        if re.search(pattern, check, re.IGNORECASE):
            return {'NT': canonical_name, 'AC': accession}

    # OLS fallback
    if ols_client is not None:
        try:
            hits = ols_client.cache_search(name, 'ms', full_search=False)
            if not hits:
                hits = ols_client.cache_search(name, 'ms', full_search=True)
            if hits:
                h   = hits[0]
                acc = h.get('obo_id', h.get('accession', ''))
                lbl = h.get('label', name)
                if acc:
                    return {'NT': lbl, 'AC': acc}
        except Exception as e:
            log.debug(f'OLS instrument lookup failed for {name!r}: {e}')

    log.debug(f'resolve_instrument: no match for {name!r}')
    return {'NT': name}


# ---------------------------------------------------------------------------
# Cleavage agent resolver
# ---------------------------------------------------------------------------

_CLEAVAGE_TABLE = [
    # Combined enzymes — check before single
    (r'lys.?c.*trypsin|trypsin.*lys.?c',    'Trypsin/Lys-C',    'MS:1001251|MS:1001309'),
    (r'lys.?c.*lys.?n|lys.?n.*lys.?c',     'Lys-C/Lys-N',      'MS:1001309|MS:1001309'),
    # Single enzymes
    (r'\btrypsin\b',                         'Trypsin',          'MS:1001251'),
    (r'\blys.?c\b',                          'Lys-C',            'MS:1001309'),
    (r'\blys.?n\b',                          'Lys-N',            'MS:1001309'),
    (r'\basp.?n\b',                          'Asp-N',            'MS:1001305'),
    (r'\barg.?c\b',                          'Arg-C',            'MS:1001303'),
    (r'\bglu.?c\b',                          'Glu-C',            'MS:1001917'),
    (r'\bchymotrypsin\b',                    'Chymotrypsin',     'MS:1001306'),
    (r'\bcnbr\b|cyanogen\s+bromide',         'CNBr',             'MS:1001308'),
    (r'\bpepsin\b',                          'Pepsin',           'MS:1001375'),
    (r'\bthermolysin\b',                     'Thermolysin',      'MS:1001376'),
    (r'no\s+enzyme|unspecific|non.specific|none', 'No cleavage', 'MS:1001955'),
]


def resolve_cleavage_agent(value) -> str:
    """
    Resolve enzyme name to canonical NT=name;AC=accession kv-string.
    Handles plain strings, dicts, and partial kv-strings from the LLM.
    For combined enzymes (Trypsin/Lys-C), returns a pipe-joined pair.
    Returns the original value as NT=value if no match found.
    """
    if not value:
        return ''

    # Extract name from dict or kv-string
    if isinstance(value, dict):
        name = value.get('NT') or value.get('name') or next(
            (v for v in value.values() if isinstance(v, str)), '')
    elif isinstance(value, str):
        name = value
        if 'NT=' in value:
            parts = [p for p in value.split(';') if p.strip().startswith('NT=')]
            name  = parts[0].replace('NT=', '').strip() if parts else value
    else:
        name = str(value)

    check = name.lower().strip()
    if not check:
        return ''

    for pattern, canonical_name, accession in _CLEAVAGE_TABLE:
        if re.search(pattern, check, re.IGNORECASE):
            # Handle combined enzymes (pipe-separated accessions)
            if '|' in accession:
                enzymes = canonical_name.split('/')
                accs    = accession.split('|')
                parts   = [f'NT={e};AC={a}' for e, a in zip(enzymes, accs)]
                return '|'.join(parts)
            return f'NT={canonical_name};AC={accession}'

    # Already canonical?
    if re.search(r'NT=.+;AC=MS:\d+', str(value)):
        return str(value)

    log.debug(f'resolve_cleavage_agent: no match for {name!r}')
    return f'NT={name}'


# ---------------------------------------------------------------------------
# Plain-string CV resolvers (thin wrappers reusing cv_map tables)
# ---------------------------------------------------------------------------

_FRAGMENTATION_TABLE = [
    (r'\bethcd\b|eth?hcd\b',                         'EThcD',   'MS:1002631'),
    (r'\bhcd\b|higher.energy\s+collision',           'HCD',     'MS:1000422'),
    (r'\bcid\b|collision.induced',                   'CID',     'MS:1000133'),
    (r'\betd\b|electron\s+transfer\s+dissoc',        'ETD',     'MS:1001848'),
    (r'\becd\b|electron\s+capture\s+dissoc',         'ECD',     'MS:1000250'),
    (r'\buvpd\b',                                    'UVPD',    'MS:1003246'),
    (r'\bpd\b|photodissociation',                    'PD',      'MS:1000435'),
]

_ACQUISITION_TABLE = [
    (r'\bdda\b|data.dependent',                       'DDA'),
    (r'\bdia\b|data.independent',                     'DIA'),
    (r'\bprm\b|parallel\s+reaction\s+monitor',        'PRM'),
    (r'\bsrm\b|\bmrm\b|multiple\s+reaction\s+monitor','SRM'),
]

_MS2_ANALYZER_TABLE = [
    (r'orbitrap',                'orbitrap',         'MS:1000484'),
    (r'ion\s*trap|iontrap',      'ion trap',         'MS:1000264'),
    (r'\btof\b|time.of.flight',  'time-of-flight',   'MS:1000084'),
    (r'\bquadrupole\b',          'quadrupole',       'MS:1000081'),
    (r'\bfticr\b|ft.?icr',       'FT-ICR',           'MS:1000079'),
]

_LABEL_TABLE = [
    (r'label.free|lfq\b',         'label free sample', 'MS:1002038'),
    (r'\btmt\s*16|tmt16plex',     'TMT16plex',         'PRIDE:0000543'),
    (r'\btmt\s*11|tmt11plex',     'TMT11plex',         'MS:1002229'),
    (r'\btmt\s*10|tmt10plex',     'TMT10plex',         'MS:1002228'),
    (r'\btmt\s*6|tmt6plex',       'TMT6plex',          'MS:1001736'),
    (r'\btmt\s*2|tmt2plex',       'TMT2plex',          'MS:1002723'),
    (r'\btmt\b',                  'TMT6plex',          'MS:1001736'),
    (r'\bitraq\s*8|itraq8plex',   'iTRAQ8plex',        'MS:1001985'),
    (r'\bitraq\s*4|itraq4plex',   'iTRAQ4plex',        'MS:1001522'),
    (r'\bitraq\b',                'iTRAQ4plex',        'MS:1001522'),
    (r'silac.heavy',              'SILAC heavy',       'MS:1002038'),
    (r'silac.medium',             'SILAC medium',      'MS:1002038'),
    (r'silac.light',              'SILAC light',       'MS:1002038'),
    (r'\bsilac\b',                'SILAC light',       'MS:1002038'),
    (r'\bdimethyl\b',             'Dimethyl',          'PRIDE:0000534'),
]

_IONIZATION_TABLE = [
    (r'\bnano.?esi\b|nanoesi|nanospray',  'nanoelectrospray',                                     'MS:1000398'),
    (r'\besi\b|electrospray',             'electrospray ionization',                              'MS:1000073'),
    (r'\bmaldi\b',                        'matrix-assisted laser desorption ionization',          'MS:1000075'),
    (r'\bapci\b',                         'atmospheric pressure chemical ionization',             'MS:1000070'),
]

_FRACTIONATION_TABLE = [
    (r'no\s+fraction|without\s+fraction|single.shot|unfractionated', 'No fractionation',                       'PRIDE:0000552'),
    (r'high.?ph\s+r[pl]|basic\s+r[pl]',                              'high pH RPLC',                           'PRIDE:0000564'),
    (r'\bscx\b',                                                      'SCX',                                    'PRIDE:0000558'),
    (r'\bsax\b',                                                      'SAX',                                    'PRIDE:0000557'),
    (r'sds.?page|gel.*frac|in.gel',                                  'SDS-PAGE',                               'PRIDE:0000568'),
    (r'size.exclusion|sec\b',                                         'Size-exclusion chromatography',          'PRIDE:0000560'),
    (r'off.?line\s+r[pl]',                                            'Off-line RP',                            'PRIDE:0000563'),
    (r'isoelectric|ief',                                              'IEF',                                    'PRIDE:0000562'),
]

_SEPARATION_TABLE = [
    (r'reversed?.phase|rplc|rp.?lc|nano\s*lc|nanoflow|c18|c8\b',  'Reversed-phase chromatography', 'PRIDE:0000563'),
    (r'\bscx\b|strong\s+cation',                                    'SCX',                          'PRIDE:0000558'),
    (r'\bsax\b|strong\s+anion',                                     'SAX',                          'PRIDE:0000557'),
    (r'\bhilic\b',                                                   'HILIC',                        'PRIDE:0000551'),
    (r'size.exclusion|sec\b',                                        'Size exclusion chromatography','PRIDE:0000560'),
    (r'ion\s+exchange',                                              'Ion exchange chromatography',  'PRIDE:0000554'),
    (r'hplc|high.performance\s+liquid',                             'HPLC',                         'PRIDE:0000565'),
    (r'\bcze\b|capillary\s+zone',                                    'CZE',                          'PRIDE:0000566'),
]


def _resolve_plain(value: str, table: list) -> Optional[str]:
    """Match against a (pattern, canonical) or (pattern, canonical, AC) table."""
    if not value:
        return None
    check = value.lower().strip()
    # Strip existing NT= wrapper for matching
    if 'NT=' in check:
        m = re.search(r'nt=([^;]+)', check)
        check = m.group(1).strip() if m else check

    for row in table:
        pattern, canonical = row[0], row[1]
        if re.search(pattern, check, re.IGNORECASE):
            return canonical
    return None


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve_globals_structure(d: dict, ols_client=None) -> dict:
    """
    Resolve all structured fields in a globals dict returned by parse_globals().

    Operates on a copy — does not mutate the input.

    Handles:
      instrument      → {'NT': ..., 'AC': ...}
      cleavage_agent  → 'NT=...;AC=...' string
      modification    → list of {'NT','AC','TA','MT',['PP']} dicts
      fragmentation_method, acquisition_method, ms2_mass_analyzer,
      label, ionization_type, fractionation_method, separation
                      → canonical plain strings

    Leaves all other fields (tolerances, organism, numbers, etc.) untouched.
    """
    out = dict(d)  # shallow copy

    # ── instrument ──────────────────────────────────────────────────────────
    if 'instrument' in out and out['instrument']:
        raw = out['instrument']
        # LLM may produce a plain string, a dict, or an NT/AC dict
        if isinstance(raw, str):
            out['instrument'] = resolve_instrument(raw, ols_client)
        elif isinstance(raw, dict):
            # Already structured — try to canonicalise the NT name
            name = raw.get('NT') or raw.get('name', '')
            resolved = resolve_instrument(name, ols_client)
            if resolved.get('AC'):
                out['instrument'] = resolved
            # else keep original dict

    # ── cleavage_agent ───────────────────────────────────────────────────────
    if 'cleavage_agent' in out and out['cleavage_agent']:
        out['cleavage_agent'] = resolve_cleavage_agent(out['cleavage_agent'])

    # ── modification list ────────────────────────────────────────────────────
    if 'modification' in out:
        raw_mods = out['modification']
        if isinstance(raw_mods, list):
            out['modification'] = [
                resolve_modification(m, ols_client)
                for m in raw_mods
                if m is not None
            ]
        elif raw_mods is not None:
            # LLM returned a single dict or string instead of a list
            out['modification'] = [resolve_modification(raw_mods, ols_client)]

    # ── plain-string CV fields ───────────────────────────────────────────────
    for field, table in [
        ('fragmentation_method', _FRAGMENTATION_TABLE),
        ('acquisition_method',   _ACQUISITION_TABLE),
        ('ms2_mass_analyzer',    _MS2_ANALYZER_TABLE),
        ('label',                _LABEL_TABLE),
        ('ionization_type',      _IONIZATION_TABLE),
        ('fractionation_method', _FRACTIONATION_TABLE),
        ('separation',           _SEPARATION_TABLE),
    ]:
        if field in out and out[field]:
            raw = str(out[field]).strip()
            resolved = _resolve_plain(raw, table)
            if resolved:
                out[field] = resolved

    return out


# ---------------------------------------------------------------------------
# Convenience: simplified schema hint for LLM (used in optimize_dspy.py)
# ---------------------------------------------------------------------------

GLOBALS_SCHEMA_SIMPLE = """
Extract experiment-level metadata shared across ALL samples.
Return a JSON object with EXACTLY these snake_case keys:

  organism               : Latin binomial, e.g. "Homo sapiens"
  label                  : e.g. "label free sample", "TMT10plex", "SILAC heavy"
  instrument             : instrument model name only, e.g. "Q Exactive HF"
  fragmentation_method   : "HCD", "CID", "ETD", "EThcD", or null
  acquisition_method     : "DDA", "DIA", "PRM", "SRM", or null
  ms2_mass_analyzer      : "Orbitrap", "ion trap", "TOF", or null
  ionization_type        : "nanoESI", "ESI", "MALDI", or null
  cleavage_agent         : enzyme name only, e.g. "Trypsin", "Lys-C"
  modification           : list of objects, each with:
                             name   : modification name, e.g. "Carbamidomethyl"
                             residue: target amino acid(s), e.g. "C" or "S,T,Y"
                             type   : "Fixed" or "Variable"
  precursor_mass_tolerance: e.g. "10 ppm" or "0.05 Da"
  fragment_mass_tolerance : e.g. "0.02 Da" or "20 ppm"
  missed_cleavages        : integer or null
  enrichment_method       : plain text or null
  fractionation_method    : e.g. "high pH RPLC", "SCX", or null
  alkylation_reagent      : e.g. "iodoacetamide", "IAA", or null
  reduction_reagent       : e.g. "DTT", "TCEP", or null
  separation              : e.g. "nano LC", "RPLC", or null
  number_of_biological_replicates: integer or null
  number_of_samples       : integer or null

Return ONLY valid JSON. Use null for unknown fields.
Extract ONLY from Title, Abstract, and Methods sections.
DO NOT include NT=, AC=, UNIMOD:, or MS: identifiers — use plain names only.
""".strip()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')

    test_globals = {
        'organism'           : 'human',
        'label'              : 'label-free',
        'instrument'         : 'Q Exactive HF',
        'fragmentation_method': 'higher-energy collisional dissociation',
        'acquisition_method' : 'data-dependent acquisition',
        'ms2_mass_analyzer'  : 'Orbitrap',
        'ionization_type'    : 'nanoESI',
        'cleavage_agent'     : 'Trypsin',
        'modification'       : [
            {'name': 'Carbamidomethyl', 'residue': 'C', 'type': 'Fixed'},
            {'name': 'Oxidation',       'residue': 'M', 'type': 'Variable'},
            {'name': 'Phospho',         'residue': 'S,T,Y', 'type': 'Variable'},
            'Acetyl (Protein N-term)',   # plain string form
            'TMT6plex on K fixed',
        ],
        'precursor_mass_tolerance': '10 ppm',
        'fragment_mass_tolerance' : '0.02 Da',
        'missed_cleavages'        : 2,
    }

    print('Input:')
    print(json.dumps(test_globals, indent=2))
    print()

    resolved = resolve_globals_structure(test_globals)

    print('Resolved:')
    print(json.dumps(resolved, indent=2))
    print()

    # Check cleavage agent
    print('Cleavage agent tests:')
    for ca in ['Trypsin', 'trypsin/Lys-C', {'NT': 'LysC', 'AC': 'MS:1001309'},
               'NT=Glu-C;AC=MS:1001917', 'no enzyme']:
        print(f'  {str(ca):40} → {resolve_cleavage_agent(ca)}')

    print()
    print('Instrument tests:')
    for inst in ['Q Exactive HF', 'Orbitrap Fusion Lumos', 'timsTOF Pro',
                 'AC=MS:1002523;NT=Q Exactive HF', 'Bruker Impact II']:
        r = resolve_instrument(inst)
        print(f'  {inst:35} → {r}')
