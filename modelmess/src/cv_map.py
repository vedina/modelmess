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
    return f'AC={accession};NT={name}'


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
    (r'\bhcd\b|higher.energy\s+collision',          'AC=MS:1000422;NT=HCD'),
    (r'\bcid\b|collision.induced',                   'AC=MS:1000133;NT=CID'),
    (r'\betd\b|electron\s+transfer\s+dissoc',        'AC=MS:1001848;NT=ETD'),
    (r'\becd\b|electron\s+capture\s+dissoc',         'AC=MS:1000250;NT=ECD'),
    (r'\bethcd\b|eth?hcd\b',                         'AC=MS:1002631;NT=EThcD'),
    (r'\buvpd\b',                                    'AC=MS:1003246;NT=UVPD'),
]

_ACQUISITION_MAP = [
    (r'\bdda\b|data.dependent',     'DDA'),
    (r'\bdia\b|data.independent',   'DIA'),
    (r'\bprm\b|parallel\s+reaction\s+monitor', 'PRM'),
    (r'\bsrm\b|\bmrm\b|multiple\s+reaction\s+monitor', 'SRM'),
]

_SEPARATION_MAP = [
    (r'reversed?.phase|rplc|rp.?lc|nano\s*lc|nanoflow|c18|c8\b',
     'AC=PRIDE:0000563;NT=Reversed-phase chromatography'),
    (r'\bscx\b|strong\s+cation',
     'AC=PRIDE:0000558;NT=SCX'),
    (r'\bsax\b|strong\s+anion',
     'AC=PRIDE:0000557;NT=SAX'),
    (r'\bhilic\b',
     'AC=PRIDE:0000551;NT=HILIC'),
    (r'size.exclusion|sec\b',
     'AC=PRIDE:0000560;NT=Size exclusion chromatography'),
    (r'ion\s+exchange',
     'AC=PRIDE:0000554;NT=Ion exchange chromatography'),
    (r'hplc|high.performance\s+liquid',
     'AC=PRIDE:0000565;NT=High-performance liquid chromatography'),
]

_FRACTIONATION_MAP = [
    (r'no\s+fraction|without\s+fraction|single.shot|not\s+fraction',
     'AC=PRIDE:0000552;NT=No fractionation'),
    (r'high.?ph\s+rp|basic\s+rp|high\s+ph\s+revers',
     'AC=PRIDE:0000564;NT=high pH RPLC'),
    (r'\bscx\b.*(frac|chrom)|strong\s+cation.*frac',
     'AC=PRIDE:0000558;NT=SCX'),
    (r'sds.?page|gel.*frac|in.gel',
     'AC=PRIDE:0000568;NT=SDS-PAGE'),
    (r'off.?line\s+rplc|off.?line.*reverse',
     'AC=PRIDE:0000563;NT=Off-line RP'),
]

_IONIZATION_MAP = [
    (r'\bnano.?esi\b|nanoesi|nanospray',  'AC=MS:1000398;NT=nanoelectrospray'),
    (r'\besi\b|electrospray',             'AC=MS:1000073;NT=electrospray ionization'),
    (r'\bmaldi\b',                        'AC=MS:1000075;NT=matrix-assisted laser desorption ionization'),
    (r'\bapci\b',                         'AC=MS:1000070;NT=atmospheric pressure chemical ionization'),
]

_CLEAVAGE_MAP = [
    (r'lys.?c.*trypsin|trypsin.*lys.?c',
     'AC=MS:1001251;NT=Trypsin|AC=MS:1001309;NT=Lys-C'),
    (r'\blys.?c\b',        'AC=MS:1001309;NT=Lys-C'),
    (r'\btrypsin\b',       'AC=MS:1001251;NT=Trypsin'),
    (r'\bchymotrypsin\b',  'AC=MS:1001306;NT=Chymotrypsin'),
    (r'\bglu.?c\b',        'AC=MS:1001917;NT=Glu-C'),
    (r'\basp.?n\b',        'AC=MS:1001305;NT=Asp-N'),
    (r'\barg.?c\b',        'AC=MS:1001303;NT=Arg-C'),
    (r'\bbnps.skatole\b',  'AC=MS:1001308;NT=CNBr'),
    (r'\bcnbr\b',          'AC=MS:1001308;NT=CNBr'),
]

_LABEL_MAP = [
    (r'label.free|lfq\b',                   'AC=MS:1002038;NT=label free sample'),
    (r'\btmt\s*16|tmt16',                   'AC=PRIDE:0000543;NT=TMT16plex'),
    (r'\btmt\s*11|tmt11',                   'AC=MS:1002229;NT=TMT11plex'),
    (r'\btmt\s*10|tmt10',                   'AC=MS:1002228;NT=TMT10plex'),
    (r'\btmt\s*6|tmt6',                     'AC=MS:1001736;NT=TMT6plex'),
    (r'\btmt\s*2|tmt2',                     'AC=MS:1002723;NT=TMT2plex'),
    (r'\btmt\b',                            'AC=MS:1002038;NT=TMT'),
    (r'\bitraq\s*8|itraq8',                 'AC=MS:1001985;NT=iTRAQ8plex'),
    (r'\bitraq\s*4|itraq4',                 'AC=MS:1001522;NT=iTRAQ4plex'),
    (r'\bitraq\b',                          'AC=MS:1001522;NT=iTRAQ'),
    (r'\bsilac\b',                          'AC=MS:1002038;NT=SILAC'),
    (r'\bdimethyl\b',                       'AC=PRIDE:0000534;NT=Dimethyl'),
    (r'\breductive\s+dimethylation',        'AC=PRIDE:0000534;NT=Dimethyl'),
]

_MS2_ANALYZER_MAP = [
    (r'orbitrap',          'AC=MS:1000484;NT=orbitrap'),
    (r'ion\s*trap|iontrap', 'AC=MS:1000264;NT=ion trap'),
    (r'\btof\b|time.of.flight', 'AC=MS:1000084;NT=time-of-flight'),
    (r'\bquadrupole\b|\bq\b', 'AC=MS:1000081;NT=quadrupole'),
    (r'\bfticr\b|ft.?icr|fourier', 'AC=MS:1000079;NT=fourier transform ion cyclotron resonance mass spectrometer'),
]

# Instrument: matched against LLM output, returns NT=...;AC=... string
_INSTRUMENT_MAP = [
    (r'orbitrap\s*astral',                  'AC=MS:1003378;NT=Orbitrap Astral'),
    (r'q\s*exactive\s*hf.?x',              'AC=MS:1002877;NT=Q Exactive HF-X'),
    (r'q\s*exactive\s*hf',                 'AC=MS:1002523;NT=Q Exactive HF'),
    (r'q\s*exactive\s*plus',               'AC=MS:1002634;NT=Q Exactive Plus'),
    (r'q\s*exactive',                      'AC=MS:1001911;NT=Q Exactive'),
    (r'orbitrap\s*fusion\s*lumos',         'AC=MS:1002732;NT=Orbitrap Fusion Lumos'),
    (r'orbitrap\s*fusion',                 'AC=MS:1002416;NT=Orbitrap Fusion'),
    (r'orbitrap\s*exploris\s*480',         'AC=MS:1003028;NT=Orbitrap Exploris 480'),
    (r'orbitrap\s*eclipse',                'AC=MS:1003029;NT=Orbitrap Eclipse'),
    (r'orbitrap\s*elite',                  'AC=MS:1001910;NT=LTQ Orbitrap Elite'),
    (r'orbitrap\s*velos',                  'AC=MS:1001742;NT=LTQ Orbitrap Velos'),
    (r'ltq\s*orbitrap\b',                  'AC=MS:1000449;NT=LTQ Orbitrap'),
    (r'ltq\s*xl',                          'AC=MS:1000854;NT=LTQ XL'),
    (r'\bltq\b',                           'AC=MS:1000447;NT=LTQ'),
    (r'timstof\s*pro\s*2',                 'AC=MS:1003230;NT=timsTOF Pro 2'),
    (r'timstof\s*pro',                     'AC=MS:1003005;NT=timsTOF Pro'),
    (r'\btimstof\b',                       'AC=MS:1002817;NT=timsTOF'),
    (r'triple\s*tof\s*6600|sciex\s*6600',  'AC=MS:1002533;NT=TripleTOF 6600'),
    (r'triple\s*tof\s*5600|sciex\s*5600',  'AC=MS:1002532;NT=TripleTOF 5600'),
    (r'synapt\s*g2',                       'AC=MS:1002280;NT=Synapt G2 MS'),
    (r'\bsynapt\b',                        'AC=MS:1001490;NT=Synapt MS'),
    (r'xevo\s*g2',                         'AC=MS:1001535;NT=Xevo G2 QTOF'),
    (r'maxi\s*ms',                         'AC=MS:1001534;NT=maXis'),
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
            return f'AC={acc};NT={lbl}'
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
        ('Comment[FragmentationMethod]',      'AC=MS:1001848;NT=ETD'),   # already canonical
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