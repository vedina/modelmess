"""
cv_map.py  —  Controlled vocabulary normaliser for SDRF submission fields.

Uses sdrf-pipelines' OlsClient (which caches psi-ms.parquet, pride.parquet,
unimod.parquet locally) to look up canonical CV terms, then applies
keyword-based mappers for fields where exact OLS lookup is too slow or
the LLM output needs structural transformation.

Usage in notebook:

    from cv_map import build_cv_normaliser, normalise_submission

    # Build once per session (downloads & caches ontology parquets on first run)
    cv = build_cv_normaliser()

    # Post-process a submission DataFrame in-place
    submission_df = normalise_submission(submission_df, cv)

Where to insert:
    Same place as vocab.snap_submission — at the end of the submission
    builder cell, before writing submission.csv.  Run AFTER snap_submission
    (cv_map handles format normalisation, snap handles value selection).
"""

from __future__ import annotations

import re
import logging
import difflib
from typing import Optional

import pandas as pd

log = logging.getLogger('sdrf.cv')

NA = 'Not Applicable'


# ── Helper ────────────────────────────────────────────────────────────────────

def _nt_ac(name: str, accession: str) -> str:
    """Format a CV term as NT=name;AC=accession."""
    return f'NT={name};AC={accession}'


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ── OLS lookup helpers ────────────────────────────────────────────────────────

def _ols_lookup(client, term: str, ontology: str) -> Optional[dict]:
    """
    Search OLS cache for a term in a given ontology.
    Returns the best hit dict or None.
    """
    try:
        hits = client.cache_search(term, ontology, full_search=False)
        if hits:
            return hits[0]
        # Try broader search
        hits = client.cache_search(term, ontology, full_search=True)
        return hits[0] if hits else None
    except Exception as e:
        log.debug(f'OLS lookup failed for {term!r} in {ontology}: {e}')
        return None


def _extract_nt(value: str) -> str:
    """Extract the NT= component from a kv-string."""
    if 'NT=' in value:
        parts = [p for p in value.split(';') if 'NT=' in p]
        if parts:
            return parts[0].replace('NT=', '').strip()
    return value.strip()


# ── Keyword lookup tables (fast, offline, no OLS needed) ─────────────────────
# These cover the high-frequency controlled fields where the LLM reliably
# produces the right concept but wrong format.
# Patterns are matched case-insensitively against the LLM output.

_FRAGMENTATION_MAP = [
    # (regex pattern,  canonical NT=...;AC=... string)
    (r'\bhcd\b|higher.energy\s+collision',          'NT=HCD;AC=MS:1000422'),
    (r'\bcid\b|collision.induced',                   'NT=CID;AC=MS:1000133'),
    (r'\betd\b|electron\s+transfer\s+dissoc',        'NT=ETD;AC=MS:1001848'),
    (r'\becd\b|electron\s+capture\s+dissoc',         'NT=ECD;AC=MS:1000250'),
    (r'\bethcd\b|eth?hcd\b',                         'NT=EThcD;AC=MS:1002631'),
    (r'\buvpd\b',                                    'NT=UVPD;AC=MS:1003246'),
]

_ACQUISITION_MAP = [
    (r'\bdda\b|data.dependent',     'DDA'),
    (r'\bdia\b|data.independent',   'DIA'),
    (r'\bprm\b|parallel\s+reaction\s+monitor', 'PRM'),
    (r'\bsrm\b|\bmrm\b|multiple\s+reaction\s+monitor', 'SRM'),
]

_SEPARATION_MAP = [
    (r'reversed?.phase|rplc|rp.?lc|nano\s*lc|nanoflow|c18|c8\b',
     'NT=Reversed-phase chromatography;AC=PRIDE:0000563'),
    (r'\bscx\b|strong\s+cation',
     'NT=SCX;AC=PRIDE:0000558'),
    (r'\bsax\b|strong\s+anion',
     'NT=SAX;AC=PRIDE:0000557'),
    (r'\bhilic\b',
     'NT=HILIC;AC=PRIDE:0000551'),
    (r'size.exclusion|sec\b',
     'NT=Size exclusion chromatography;AC=PRIDE:0000560'),
    (r'ion\s+exchange',
     'NT=Ion exchange chromatography;AC=PRIDE:0000554'),
    (r'hplc|high.performance\s+liquid',
     'NT=High-performance liquid chromatography;AC=PRIDE:0000565'),
]

_FRACTIONATION_MAP = [
    (r'no\s+fraction|without\s+fraction|single.shot|not\s+fraction',
     'NT=No fractionation;AC=PRIDE:0000552'),
    (r'high.?ph\s+rp|basic\s+rp|high\s+ph\s+revers',
     'NT=high pH RPLC;AC=PRIDE:0000564'),
    (r'\bscx\b.*(frac|chrom)|strong\s+cation.*frac',
     'NT=SCX;AC=PRIDE:0000558'),
    (r'sds.?page|gel.*frac|in.gel',
     'NT=SDS-PAGE;AC=PRIDE:0000568'),
    (r'off.?line\s+rplc|off.?line.*reverse',
     'NT=Off-line RP;AC=PRIDE:0000563'),
]

_IONIZATION_MAP = [
    (r'\bnano.?esi\b|nanoesi|nanospray',  'NT=nanoelectrospray;AC=MS:1000398'),
    (r'\besi\b|electrospray',             'NT=electrospray ionization;AC=MS:1000073'),
    (r'\bmaldi\b',                        'NT=matrix-assisted laser desorption ionization;AC=MS:1000075'),
    (r'\bapci\b',                         'NT=atmospheric pressure chemical ionization;AC=MS:1000070'),
]

_CLEAVAGE_MAP = [
    (r'lys.?c.*trypsin|trypsin.*lys.?c',
     'NT=Trypsin;AC=MS:1001251|NT=Lys-C;AC=MS:1001309'),
    (r'\blys.?c\b',        'NT=Lys-C;AC=MS:1001309'),
    (r'\btrypsin\b',       'NT=Trypsin;AC=MS:1001251'),
    (r'\bchymotrypsin\b',  'NT=Chymotrypsin;AC=MS:1001306'),
    (r'\bglu.?c\b',        'NT=Glu-C;AC=MS:1001917'),
    (r'\basp.?n\b',        'NT=Asp-N;AC=MS:1001305'),
    (r'\barg.?c\b',        'NT=Arg-C;AC=MS:1001303'),
    (r'\bbnps.skatole\b',  'NT=CNBr;AC=MS:1001308'),
    (r'\bcnbr\b',          'NT=CNBr;AC=MS:1001308'),
]

_LABEL_MAP = [
    (r'label.free|lfq\b',                   'NT=label free sample;AC=MS:1002038'),
    (r'\btmt\s*16|tmt16',                   'NT=TMT16plex;AC=PRIDE:0000543'),
    (r'\btmt\s*11|tmt11',                   'NT=TMT11plex;AC=MS:1002229'),
    (r'\btmt\s*10|tmt10',                   'NT=TMT10plex;AC=MS:1002228'),
    (r'\btmt\s*6|tmt6',                     'NT=TMT6plex;AC=MS:1001736'),
    (r'\btmt\s*2|tmt2',                     'NT=TMT2plex;AC=MS:1002723'),
    (r'\btmt\b',                            'NT=TMT;AC=MS:1002038'),
    (r'\bitraq\s*8|itraq8',                 'NT=iTRAQ8plex;AC=MS:1001985'),
    (r'\bitraq\s*4|itraq4',                 'NT=iTRAQ4plex;AC=MS:1001522'),
    (r'\bitraq\b',                          'NT=iTRAQ;AC=MS:1001522'),
    (r'\bsilac\b',                          'NT=SILAC;AC=MS:1002038'),
    (r'\bdimethyl\b',                       'NT=Dimethyl;AC=PRIDE:0000534'),
    (r'\breductive\s+dimethylation',        'NT=Dimethyl;AC=PRIDE:0000534'),
]

_MS2_ANALYZER_MAP = [
    (r'orbitrap',          'NT=orbitrap;AC=MS:1000484'),
    (r'ion\s*trap|iontrap', 'NT=ion trap;AC=MS:1000264'),
    (r'\btof\b|time.of.flight', 'NT=time-of-flight;AC=MS:1000084'),
    (r'\bquadrupole\b|\bq\b', 'NT=quadrupole;AC=MS:1000081'),
    (r'\bfticr\b|ft.?icr|fourier', 'NT=fourier transform ion cyclotron resonance mass spectrometer;AC=MS:1000079'),
]

# Instrument: matched against LLM output, returns NT=...;AC=... string
_INSTRUMENT_MAP = [
    (r'orbitrap\s*astral',                  'NT=Orbitrap Astral;AC=MS:1003378'),
    (r'q\s*exactive\s*hf.?x',              'NT=Q Exactive HF-X;AC=MS:1002877'),
    (r'q\s*exactive\s*hf',                 'NT=Q Exactive HF;AC=MS:1002523'),
    (r'q\s*exactive\s*plus',               'NT=Q Exactive Plus;AC=MS:1002634'),
    (r'q\s*exactive',                      'NT=Q Exactive;AC=MS:1001911'),
    (r'orbitrap\s*fusion\s*lumos',         'NT=Orbitrap Fusion Lumos;AC=MS:1002732'),
    (r'orbitrap\s*fusion',                 'NT=Orbitrap Fusion;AC=MS:1002416'),
    (r'orbitrap\s*exploris\s*480',         'NT=Orbitrap Exploris 480;AC=MS:1003028'),
    (r'orbitrap\s*eclipse',                'NT=Orbitrap Eclipse;AC=MS:1003029'),
    (r'orbitrap\s*elite',                  'NT=LTQ Orbitrap Elite;AC=MS:1001910'),
    (r'orbitrap\s*velos',                  'NT=LTQ Orbitrap Velos;AC=MS:1001742'),
    (r'ltq\s*orbitrap\b',                  'NT=LTQ Orbitrap;AC=MS:1000449'),
    (r'ltq\s*xl',                          'NT=LTQ XL;AC=MS:1000854'),
    (r'\bltq\b',                           'NT=LTQ;AC=MS:1000447'),
    (r'timstof\s*pro\s*2',                 'NT=timsTOF Pro 2;AC=MS:1003230'),
    (r'timstof\s*pro',                     'NT=timsTOF Pro;AC=MS:1003005'),
    (r'\btimstof\b',                       'NT=timsTOF;AC=MS:1002817'),
    (r'triple\s*tof\s*6600|sciex\s*6600',  'NT=TripleTOF 6600;AC=MS:1002533'),
    (r'triple\s*tof\s*5600|sciex\s*5600',  'NT=TripleTOF 5600;AC=MS:1002532'),
    (r'synapt\s*g2',                       'NT=Synapt G2 MS;AC=MS:1002280'),
    (r'\bsynapt\b',                        'NT=Synapt MS;AC=MS:1001490'),
    (r'xevo\s*g2',                         'NT=Xevo G2 QTOF;AC=MS:1001535'),
    (r'maxi\s*ms',                         'NT=maXis;AC=MS:1001534'),
]

# Map LLM alkylation reagent names to canonical short forms
_ALKYLATION_MAP = [
    (r'\biaa\b|iodoacetamide',             'IAA'),
    (r'\bcaa\b|chloroacetamide',           'CAA'),
    (r'\bnem\b|n.ethylmaleimide',          'NEM'),
    (r'\bmpaq\b',                          'MPAQ'),
]

_REDUCTION_MAP = [
    (r'\bdtt\b|dithiothreitol',            'DTT'),
    (r'\btcep\b',                          'TCEP'),
    (r'\bbme\b|beta.mercaptoethanol|2.mercaptoethanol', 'beta-mercaptoethanol'),
    (r'\bdtnb\b',                          'DTNB'),
]


def _apply_map(value: str, mapping: list) -> Optional[str]:
    """
    Apply a list of (regex, canonical) pairs to a value.
    Returns the first matching canonical string, or None.
    Case-insensitive.
    """
    if not value or value == NA:
        return None
    # Check if already in NT=...;AC=... format — extract NT part for matching
    check = _extract_nt(value).lower()
    for pattern, canonical in mapping:
        if re.search(pattern, check, re.IGNORECASE):
            return canonical
    return None


# ── OLS-based instrument/modification lookup ─────────────────────────────────

def _ols_instrument(client, value: str) -> Optional[str]:
    """Look up an instrument name in psi-ms ontology via OLS cache."""
    if client is None:
        return None
    name = _extract_nt(value)
    hit  = _ols_lookup(client, name, 'ms')
    if hit:
        acc = hit.get('obo_id', hit.get('accession', ''))
        lbl = hit.get('label', name)
        if acc:
            return f'NT={lbl};AC={acc}'
    return None


def _ols_modification(client, value: str) -> Optional[str]:
    """
    Look up a modification in UNIMOD via OLS cache.
    Preserves TA=, MT=, PP= fields from original value if present.
    """
    if client is None:
        return None
    name = _extract_nt(value)
    hit  = _ols_lookup(client, name, 'unimod')
    if hit:
        acc = hit.get('obo_id', hit.get('accession', ''))
        lbl = hit.get('label', name)
        if acc:
            # Rebuild with original TA/MT/PP fields preserved
            extra = {}
            for part in value.split(';'):
                part = part.strip()
                for k in ('TA', 'MT', 'PP', 'CF', 'MM'):
                    if part.startswith(f'{k}='):
                        extra[k] = part[len(k)+1:]
            parts = [f'NT={lbl}', f'AC={acc}']
            for k in ('PP', 'TA', 'MT', 'CF', 'MM'):
                if k in extra:
                    parts.append(f'{k}={extra[k]}')
            return ';'.join(parts)
    return None


# ── Main normaliser ───────────────────────────────────────────────────────────

class CvNormaliser:
    """
    Normalises SDRF submission values to canonical CV terms.

    Uses keyword maps for fast offline lookup, with OLS cache as fallback
    for instruments and modifications.
    """

    def __init__(self, ols_client=None):
        self.ols = ols_client

    def normalise(self, col: str, value: str) -> str:
        """
        Normalise a single cell value for a given submission column.
        Returns the canonical string, or the original value if no mapping found.
        """
        if not value or value == NA:
            return value
        v = str(value).strip()
        if v in (NA, '', 'nan'):
            return v

        result = None

        if col == 'Comment[FragmentationMethod]':
            result = _apply_map(v, _FRAGMENTATION_MAP)

        elif col == 'Comment[AcquisitionMethod]':
            result = _apply_map(v, _ACQUISITION_MAP)

        elif col == 'Comment[Separation]':
            result = _apply_map(v, _SEPARATION_MAP)

        elif col == 'Comment[FractionationMethod]':
            result = _apply_map(v, _FRACTIONATION_MAP)

        elif col == 'Comment[IonizationType]':
            result = _apply_map(v, _IONIZATION_MAP)

        elif col == 'Characteristics[CleavageAgent]':
            result = _apply_map(v, _CLEAVAGE_MAP)

        elif col == 'Characteristics[Label]':
            result = _apply_map(v, _LABEL_MAP)

        elif col == 'Comment[MS2MassAnalyzer]':
            result = _apply_map(v, _MS2_ANALYZER_MAP)

        elif col == 'Comment[Instrument]':
            # Try keyword map first (fast), then OLS
            result = _apply_map(v, _INSTRUMENT_MAP)
            if result is None:
                result = _ols_instrument(self.ols, v)

        elif col == 'Characteristics[AlkylationReagent]':
            result = _apply_map(v, _ALKYLATION_MAP)

        elif col == 'Characteristics[ReductionReagent]':
            result = _apply_map(v, _REDUCTION_MAP)

        elif col.startswith('Characteristics[Modification'):
            # Try OLS for modification normalisation
            if self.ols is not None:
                result = _ols_modification(self.ols, v)

        if result is not None and result != v:
            log.debug(f'{col}: {v!r} → {result!r}')
            return result
        return v


def build_cv_normaliser(use_ols: bool = True) -> CvNormaliser:
    """
    Build a CvNormaliser, optionally initialising the OLS client.

    Args:
        use_ols : if True, attempt to load sdrf-pipelines OLS cache
                  (downloads parquet files on first run, ~50 MB).
                  Set False for fully offline / no-network environments.

    Returns:
        CvNormaliser instance ready to use.
    """
    ols_client = None
    if use_ols:
        try:
            from sdrf_pipelines.ols.ols import OlsClient, OLS_AVAILABLE
            if OLS_AVAILABLE:
                ols_client = OlsClient()
                log.info('OLS cache loaded')
            else:
                log.warning('OLS dependencies not available — using keyword maps only')
        except Exception as e:
            log.warning(f'OLS init failed ({e}) — using keyword maps only')
    return CvNormaliser(ols_client)


def normalise_submission(submission: pd.DataFrame,
                          cv: CvNormaliser) -> pd.DataFrame:
    """
    Apply CV normalisation to all relevant columns of a submission DataFrame.

    Args:
        submission : submission CSV as DataFrame
        cv         : CvNormaliser from build_cv_normaliser()

    Returns:
        Modified copy (original not mutated).
    """
    out  = submission.copy()
    cols = [c for c in out.columns if c not in ('ID', 'PXD', 'Raw Data File', 'Usage')]

    for col in cols:
        if col not in (
            'Comment[FragmentationMethod]',
            'Comment[AcquisitionMethod]',
            'Comment[Separation]',
            'Comment[FractionationMethod]',
            'Comment[IonizationType]',
            'Comment[MS2MassAnalyzer]',
            'Comment[Instrument]',
            'Characteristics[CleavageAgent]',
            'Characteristics[Label]',
            'Characteristics[AlkylationReagent]',
            'Characteristics[ReductionReagent]',
        ) and not col.startswith('Characteristics[Modification'):
            continue

        out[col] = out[col].apply(lambda v: cv.normalise(col, str(v).strip()))

    return out


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')

    cv = build_cv_normaliser(use_ols=False)   # offline test

    tests = [
        ('Comment[FragmentationMethod]',      'HCD'),
        ('Comment[FragmentationMethod]',      'higher-energy collisional dissociation'),
        ('Comment[FragmentationMethod]',      'CID'),
        ('Comment[FragmentationMethod]',      'NT=ETD;AC=MS:1001848'),   # already canonical
        ('Comment[AcquisitionMethod]',        'data-dependent acquisition'),
        ('Comment[AcquisitionMethod]',        'DIA'),
        ('Comment[Separation]',               'nano LC'),
        ('Comment[Separation]',               'RPLC'),
        ('Comment[Separation]',               'reversed-phase chromatography'),
        ('Comment[FractionationMethod]',      'no fractionation'),
        ('Comment[FractionationMethod]',      'high pH RP fractionation'),
        ('Comment[IonizationType]',           'ESI'),
        ('Comment[IonizationType]',           'nanoESI'),
        ('Comment[Instrument]',               'Q Exactive HF'),
        ('Comment[Instrument]',               'Orbitrap Fusion Lumos'),
        ('Comment[Instrument]',               'timsTOF Pro'),
        ('Comment[Instrument]',               'LTQ Orbitrap Velos'),
        ('Characteristics[CleavageAgent]',    'Trypsin'),
        ('Characteristics[CleavageAgent]',    'trypsin/Lys-C'),
        ('Characteristics[Label]',            'label free sample'),
        ('Characteristics[Label]',            'label-free'),
        ('Characteristics[Label]',            'TMT16'),
        ('Characteristics[AlkylationReagent]','iodoacetamide'),
        ('Characteristics[ReductionReagent]', 'dithiothreitol'),
        ('Comment[MS2MassAnalyzer]',          'Orbitrap'),
        ('Comment[MS2MassAnalyzer]',          'ion trap'),
    ]

    print(f"{'Column':45} {'Input':35} → Output")
    print('-' * 110)
    for col, val in tests:
        result = cv.normalise(col, val)
        marker = '✓' if result != val else '·'
        print(f"  {marker} {col:43} {val!r:35} → {result!r}")