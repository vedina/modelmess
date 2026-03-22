"""
sdrf_refine.py — DSPy refinement layer for a rules-produced submission.csv.

Architecture
------------
The rules pipeline already does a good job.  The submission.csv contains two
types of columns:

  Globals cols  — same value on every row of a PXD (instrument, cleavage agent,
                  modifications, tolerances, label, ...).  Rules often miss a few
                  of these (CleavageAgent, NumberOfMissedCleavages, EnrichmentMethod).

  Sample cols   — can vary across rows of a PXD (Disease, Treatment, Genotype, ...).
                  For PXDs where these vary, the rules pipeline already fills them
                  correctly from filename patterns.  For PXDs where they are uniform
                  (e.g. one disease across all samples), the rules often fill them too.
                  The LLM adds value only when they are uniformly empty.

In both cases the fix is identical: one LLM call per PXD that returns a flat
corrections dict, then broadcast every correction to all rows of that PXD.

Specifically:
  • Never send per-row arrays to the LLM — too fragile and row count can be 1376.
  • Never overwrite a non-empty cell (unless allow_overwrite=True).
  • For sample cols that VARY across rows: skip them entirely (rules already filled
    them; LLM cannot assign row→condition without filename reasoning).
  • For sample cols that are UNIFORMLY empty across all rows: treat like globals —
    ask LLM once, broadcast result.

This collapses globals and samples into a single LLM call per PXD.

Usage in notebook
-----------------
    from sdrf_refine import SDRFRefiner, RefineConfig
    import pandas as pd

    # Configure once, after dspy.configure(lm=...)
    refiner = SDRFRefiner(pub_text_dir=TEST_TEXT_DIR)

    # Optionally load BFRS-optimised demos
    refiner.load_optimised('optim_output/refine_optimized.json')

    submission = pd.read_csv('submission_rules.csv')
    refined    = refiner.refine(submission)
    refined.to_csv('submission.csv', index=False)

Optimisation
------------
    from sdrf_refine import SDRFRefineOptimiser

    opt = SDRFRefineOptimiser(
        pub_text_dir   = TRAIN_TEXT_DIR,
        train_sdrf_dir = TRAIN_SDRF_DIR,
        rules_sub_path = Path('submission_rules.csv'),
        output_dir     = Path('optim_output'),
    )
    opt.run(n_train=30, n_val=10, num_candidates=8)
    # saves: optim_output/refine_optimized.json
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger('sdrf.refine')

# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

# Globals: same value on every row of a PXD.
# Rules often miss several of these.
GLOBALS_COLS = [
    'Characteristics[CleavageAgent]',
    'Characteristics[Label]',
    'Characteristics[Modification]',
    'Characteristics[Modification].1',
    'Characteristics[Modification].2',
    'Characteristics[Modification].3',
    'Characteristics[Modification].4',
    'Characteristics[Modification].5',
    'Characteristics[Modification].6',
    'Characteristics[AlkylationReagent]',
    'Characteristics[ReductionReagent]',
    'Characteristics[NumberOfBiologicalReplicates]',
    'Characteristics[NumberOfSamples]',
    'Characteristics[NumberOfTechnicalReplicates]',
    'Characteristics[Organism]',
    'Comment[AcquisitionMethod]',
    'Comment[EnrichmentMethod]',
    'Comment[FractionationMethod]',
    'Comment[FragmentMassTolerance]',
    'Comment[FragmentationMethod]',
    'Comment[Instrument]',
    'Comment[IonizationType]',
    'Comment[MS2MassAnalyzer]',
    'Comment[NumberOfMissedCleavages]',
    'Comment[PrecursorMassTolerance]',
    'Comment[Separation]',
]

# Sample cols: can vary per row, but are ONLY sent to the LLM when they are
# uniformly empty across ALL rows of the PXD.  If any row has a non-empty
# value, the column is skipped (rules already handled it).
SAMPLE_UNIFORM_COLS = [
    'Characteristics[BiologicalReplicate]',   # sometimes uniformly 1
    'Characteristics[CellLine]',
    'Characteristics[CellType]',
    'Characteristics[Disease]',
    'Characteristics[DevelopmentalStage]',
    'Characteristics[Genotype]',
    'Characteristics[GeneticModification]',
    'Characteristics[MaterialType]',
    'Characteristics[OrganismPart]',
    'Characteristics[Sex]',
    'Characteristics[Specimen]',
    'Characteristics[Strain]',
    'Characteristics[Treatment]',
    'Characteristics[Compound]',
    'Characteristics[ConcentrationOfCompound]',
    'FactorValue[Disease]',
    'FactorValue[Treatment]',
    'FactorValue[Compound]',
    'FactorValue[GeneticModification]',
    'FactorValue[Temperature]',
]

SKIP_COLS = {'ID', 'PXD', 'Raw Data File', 'Usage'}

NA_VALUES = frozenset({
    'not available', 'Not Applicable', 'not applicable',
    'Text Span', 'TextSpan', '', 'nan', 'NaN', 'NA', 'N/A', 'none', 'None',
})


def _is_empty(v) -> bool:
    if v is None:
        return True
    try:
        import math
        if math.isnan(float(str(v))):
            return True
    except (ValueError, TypeError):
        pass
    return str(v).strip() in NA_VALUES


def _all_empty(series: pd.Series) -> bool:
    """True if every non-null value in the series is an NA sentinel."""
    return series.apply(_is_empty).all()


def _any_filled(series: pd.Series) -> bool:
    return not _all_empty(series)


# ---------------------------------------------------------------------------
# Column ↔ snake_case mapping
# ---------------------------------------------------------------------------

_COL_TO_SNAKE: dict[str, str] = {
    'Characteristics[CleavageAgent]'              : 'cleavage_agent',
    'Characteristics[Label]'                      : 'label',
    'Characteristics[Modification]'               : 'modification',
    'Characteristics[Modification].1'             : 'modification_1',
    'Characteristics[Modification].2'             : 'modification_2',
    'Characteristics[Modification].3'             : 'modification_3',
    'Characteristics[Modification].4'             : 'modification_4',
    'Characteristics[Modification].5'             : 'modification_5',
    'Characteristics[Modification].6'             : 'modification_6',
    'Characteristics[AlkylationReagent]'          : 'alkylation_reagent',
    'Characteristics[ReductionReagent]'           : 'reduction_reagent',
    'Characteristics[NumberOfBiologicalReplicates]': 'number_of_biological_replicates',
    'Characteristics[NumberOfSamples]'            : 'number_of_samples',
    'Characteristics[NumberOfTechnicalReplicates]': 'number_of_technical_replicates',
    'Characteristics[Organism]'                   : 'organism',
    'Comment[AcquisitionMethod]'                  : 'acquisition_method',
    'Comment[EnrichmentMethod]'                   : 'enrichment_method',
    'Comment[FractionationMethod]'                : 'fractionation_method',
    'Comment[FragmentMassTolerance]'              : 'fragment_mass_tolerance',
    'Comment[FragmentationMethod]'                : 'fragmentation_method',
    'Comment[Instrument]'                         : 'instrument',
    'Comment[IonizationType]'                     : 'ionization_type',
    'Comment[MS2MassAnalyzer]'                    : 'ms2_mass_analyzer',
    'Comment[NumberOfMissedCleavages]'            : 'number_of_missed_cleavages',
    'Comment[PrecursorMassTolerance]'             : 'precursor_mass_tolerance',
    'Comment[Separation]'                         : 'separation',
    'Characteristics[BiologicalReplicate]'        : 'biological_replicate',
    'Characteristics[CellLine]'                   : 'cell_line',
    'Characteristics[CellType]'                   : 'cell_type',
    'Characteristics[Disease]'                    : 'disease',
    'Characteristics[DevelopmentalStage]'         : 'developmental_stage',
    'Characteristics[Genotype]'                   : 'genotype',
    'Characteristics[GeneticModification]'        : 'genetic_modification',
    'Characteristics[MaterialType]'               : 'material_type',
    'Characteristics[OrganismPart]'               : 'organism_part',
    'Characteristics[Sex]'                        : 'sex',
    'Characteristics[Specimen]'                   : 'specimen',
    'Characteristics[Strain]'                     : 'strain',
    'Characteristics[Treatment]'                  : 'treatment',
    'Characteristics[Compound]'                   : 'compound',
    'Characteristics[ConcentrationOfCompound]'    : 'concentration_of_compound',
    'FactorValue[Disease]'                        : 'factor_disease',
    'FactorValue[Treatment]'                      : 'factor_treatment',
    'FactorValue[Compound]'                       : 'factor_compound',
    'FactorValue[GeneticModification]'            : 'factor_genetic_modification',
    'FactorValue[Temperature]'                    : 'factor_temperature',
}
_SNAKE_TO_COL = {v: k for k, v in _COL_TO_SNAKE.items()}


# ---------------------------------------------------------------------------
# Schema hints (token-efficient)
# ---------------------------------------------------------------------------

SCHEMA_HINT = """
cleavage_agent      : enzyme name, e.g. "Trypsin", "Lys-C", "Trypsin/Lys-C"
label               : "label free sample", "TMT10plex", "SILAC heavy", etc.
modification        : {"name":"Carbamidomethyl","residue":"C","type":"Fixed"}
modification_1      : second modification, e.g. {"name":"Oxidation","residue":"M","type":"Variable"}
alkylation_reagent  : "iodoacetamide", "IAA", "CAA", "NEM", or null
reduction_reagent   : "DTT", "TCEP", or null
number_of_biological_replicates: integer
number_of_samples   : integer  
number_of_technical_replicates : integer
acquisition_method  : "DDA", "DIA", "PRM", or "SRM"
enrichment_method   : "TiO2", "IMAC", "immunoprecipitation", or null
fractionation_method: "high pH RPLC", "SCX", "SDS-PAGE", or null
fragment_mass_tolerance: e.g. "0.02 Da" or "20 ppm"
fragmentation_method: "HCD", "CID", "ETD", "EThcD", or null
instrument          : model name only, e.g. "Q Exactive HF", "Orbitrap Fusion Lumos"
ionization_type     : "nanoESI", "ESI", or "MALDI"
ms2_mass_analyzer   : "Orbitrap", "ion trap", or "TOF"
number_of_missed_cleavages: integer
precursor_mass_tolerance: e.g. "10 ppm" or "0.05 Da"
separation          : "nano LC", "RPLC", or null
biological_replicate: integer (1, 2, 3, ...) — only if uniform across all samples
cell_line           : cell line name, e.g. "HeLa" — null for primary cells/tissues
cell_type           : primary cell type, e.g. "neurons", "hepatocytes", or null
disease             : disease name or "normal" — only if uniform across all samples
developmental_stage : "adult", "embryonic", or null
genotype            : e.g. "wild-type" or null — only if uniform across all samples
genetic_modification: e.g. "GFP-tagged" or null — only if uniform across all samples
material_type       : "tissue", "cell line", "biofluid", or "primary cells"
organism_part       : tissue/organ, e.g. "brain", "liver", "plasma"
sex                 : "male", "female", or null — only if uniform across all samples
specimen            : "biopsy", "plasma", "urine", or null
strain              : animal strain, e.g. "C57BL/6" — only if uniform
treatment           : experimental treatment — only if uniform across all samples
factor_disease      : same as disease if it is the experimental variable, else null
factor_treatment    : same as treatment if it is the experimental variable, else null
""".strip()


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

def _build_module():
    """Build the single RefineExtraction DSPy module. Called lazily."""
    import dspy

    class RefineExtraction(dspy.Signature):
        """
        You are given a proteomics publication and an existing metadata
        extraction produced by pattern-matching rules.  Some fields are
        missing ("not available").

        Fill in ONLY the fields listed in fields_to_fix.
        Return a JSON object with ONLY those fields.
        Use plain names — no NT=, AC=, UNIMOD: identifiers.
        Use null for fields you cannot determine from the text.

        Important: fields marked as "uniform only" should only be filled
        if the value is the same for ALL samples in the experiment.
        """
        pub_text      : str = dspy.InputField(
            desc='Publication title, abstract and methods section')
        current_values: str = dspy.InputField(
            desc='JSON: current field values (empty = "not available")')
        fields_to_fix : str = dspy.InputField(
            desc='Comma-separated snake_case field names to fill or correct')
        schema_hint   : str = dspy.InputField(
            desc='Field descriptions and expected value formats')
        corrections   : str = dspy.OutputField(
            desc='JSON object with filled/corrected fields only. '
                 'Omit fields you cannot determine. No NT=/AC= format.')

    class ExtractionRefiner(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(RefineExtraction)

        def forward(self, pub_text, current_values, fields_to_fix, schema_hint):
            return self.predict(
                pub_text       = pub_text,
                current_values = current_values,
                fields_to_fix  = fields_to_fix,
                schema_hint    = schema_hint,
            )

    return ExtractionRefiner


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RefineConfig:
    # If True, overwrite cells that already have a value (use for fixing known-bad rules)
    allow_overwrite: bool = False
    # Retry count for LLM failures
    max_retries: int = 2
    # Text section limits (chars) for the prompt
    title_limit   : int = 300
    abstract_limit: int = 800
    methods_limit : int = 1000
    # Restrict which columns to attempt (None = all GLOBALS_COLS + uniform SAMPLE_UNIFORM_COLS)
    cols_filter: Optional[list] = None
    # Paths to optimised program JSON files
    program_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Publication text loader
# ---------------------------------------------------------------------------

def _load_pub_text(pxd: str, pub_text_dir: Path, cfg: RefineConfig) -> str:
    for suffix in ('_PubText.json', '_PubText.txt'):
        path = pub_text_dir / f'{pxd}{suffix}'
        if not path.exists():
            continue
        if suffix.endswith('.json'):
            data = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
            parts = []
            t = (data.get('TITLE')    or '')[:cfg.title_limit]
            a = (data.get('ABSTRACT') or '')[:cfg.abstract_limit]
            m = (data.get('METHODS')  or '')[:cfg.methods_limit]
            if t: parts.append(f'TITLE:\n{t}')
            if a: parts.append(f'ABSTRACT:\n{a}')
            if m: parts.append(f'METHODS:\n{m}')
            return '\n\n'.join(parts)
        else:
            blob = path.read_text(encoding='utf-8', errors='ignore')
            return blob[:cfg.title_limit + cfg.abstract_limit + cfg.methods_limit]
    log.warning(f'[{pxd}] No pub text found in {pub_text_dir}')
    return ''


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _parse_json_safe(text: str):
    if not text:
        return None
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        opens_b = text.count('{') - text.count('}')
        opens_k = text.count('[') - text.count(']')
        try:
            return json.loads(text + '}' * max(opens_b, 0) + ']' * max(opens_k, 0))
        except json.JSONDecodeError as e:
            log.debug(f'JSON parse failed: {e}. Snippet: {text[:200]}')
            return None


# ---------------------------------------------------------------------------
# Structure resolution helpers
# ---------------------------------------------------------------------------

def _mod_to_sdrf(mod) -> str:
    """Convert a mod dict/string from LLM to NT=;AC=... SDRF format."""
    try:
        from resolve import resolve_modification
        r = resolve_modification(mod)
        return ';'.join(f'{k}={v}' for k, v in r.items() if v and k != 'MT_default')
    except ImportError:
        if isinstance(mod, dict):
            name = mod.get('name') or mod.get('NT', '')
            res  = mod.get('residue') or mod.get('TA', '')
            mt   = mod.get('type') or mod.get('MT', 'Variable')
            parts = []
            if name: parts.append(f'NT={name}')
            if res:  parts.append(f'TA={res}')
            if mt:   parts.append(f'MT={mt}')
            return ';'.join(parts)
        return str(mod)


def _resolve_value(col: str, val: str) -> str:
    """Apply structure resolution for NT=;AC= fields."""
    try:
        from resolve import resolve_instrument, resolve_cleavage_agent
        if col == 'Comment[Instrument]':
            r = resolve_instrument(val)
            return f'NT={r["NT"]};AC={r["AC"]}' if r.get('AC') else val
        if col == 'Characteristics[CleavageAgent]':
            return resolve_cleavage_agent(val) or val
    except ImportError:
        pass
    return val


# ---------------------------------------------------------------------------
# Core: determine which columns to fix for a PXD
# ---------------------------------------------------------------------------

def _cols_to_fix(rows: pd.DataFrame, cfg: RefineConfig) -> list[str]:
    """
    Return the list of submission column names that need LLM correction
    for this PXD's rows.

    Rules:
    - GLOBALS_COLS: include if empty on the first row (they are uniform by
      definition, so first row == all rows).
    - SAMPLE_UNIFORM_COLS: include ONLY if the column is empty on ALL rows
      (if any row has a value, the rules already handled it correctly and
      the LLM cannot improve per-row assignments without filename reasoning).
    - If allow_overwrite=True, include all columns regardless of fill state.
    - Filter to cfg.cols_filter if set.
    """
    present = set(rows.columns) - SKIP_COLS
    first   = rows.iloc[0]

    result = []

    all_candidate_cols = (
        (cfg.cols_filter or GLOBALS_COLS + SAMPLE_UNIFORM_COLS)
    )

    for col in all_candidate_cols:
        if col not in present:
            continue
        if cfg.allow_overwrite:
            result.append(col)
            continue

        if col in GLOBALS_COLS:
            # Fix if empty on first row
            if _is_empty(first.get(col)):
                result.append(col)
        else:
            # Sample col: fix only if uniformly empty on ALL rows
            if _all_empty(rows[col]):
                result.append(col)

    return result


# ---------------------------------------------------------------------------
# Core: apply corrections dict to rows
# ---------------------------------------------------------------------------

def _apply_corrections(
    rows: pd.DataFrame,
    corrections_snake: dict,
    cols_to_fix: list[str],
    allow_overwrite: bool,
) -> pd.DataFrame:
    """
    Map snake_case corrections back to submission columns and broadcast
    the value to every row.  Modifications are spread across the
    Modification, Modification.1, ... column slots.
    """
    mod_cols = [
        'Characteristics[Modification]',
        'Characteristics[Modification].1',
        'Characteristics[Modification].2',
        'Characteristics[Modification].3',
        'Characteristics[Modification].4',
        'Characteristics[Modification].5',
        'Characteristics[Modification].6',
    ]

    for snake, val in corrections_snake.items():
        if val is None:
            continue

        # Modification list: spread across mod slot columns
        if snake == 'modification' and isinstance(val, list):
            for i, mod in enumerate(val[:7]):
                if i >= len(mod_cols):
                    break
                col = mod_cols[i]
                if col not in cols_to_fix and not allow_overwrite:
                    continue
                if col not in rows.columns:
                    continue
                sdrf_val = _mod_to_sdrf(mod)
                if not sdrf_val:
                    continue
                if allow_overwrite:
                    rows[col] = sdrf_val
                else:
                    mask = rows[col].apply(_is_empty)
                    rows.loc[mask, col] = sdrf_val
            continue

        # Single modification returned as dict instead of list
        if snake == 'modification' and isinstance(val, dict):
            col = 'Characteristics[Modification]'
            if col in rows.columns and (allow_overwrite or _all_empty(rows[col])):
                rows[col] = _mod_to_sdrf(val)
            continue

        # Numbered modification slots (modification_1, modification_2, ...)
        if re.match(r'^modification_\d+$', snake):
            idx = int(snake.split('_')[1])
            if idx < len(mod_cols):
                col = mod_cols[idx]
                if col in rows.columns and (allow_overwrite or _all_empty(rows[col])):
                    rows[col] = _mod_to_sdrf(val)
            continue

        # All other fields
        col = _SNAKE_TO_COL.get(snake)
        if not col or col not in rows.columns:
            continue
        if col not in cols_to_fix and not allow_overwrite:
            continue

        # Apply structural resolution (instrument → NT=;AC=, cleavage → NT=;AC=)
        sdrf_val = _resolve_value(col, str(val).strip())
        if not sdrf_val:
            continue

        if allow_overwrite:
            rows[col] = sdrf_val
        else:
            mask = rows[col].apply(_is_empty)
            rows.loc[mask, col] = sdrf_val

    return rows


# ---------------------------------------------------------------------------
# Main refiner class
# ---------------------------------------------------------------------------

class SDRFRefiner:
    """
    Refines a rules-produced submission.csv using one LLM call per PXD.

    Parameters
    ----------
    pub_text_dir : Path
        Directory with *_PubText.json or *_PubText.txt files.
    config : RefineConfig
        Tunable settings. Defaults are sensible for the HDD-2026 dataset.

    Notes
    -----
    Call dspy.configure(lm=...) before using this class.
    If resolve.py is on sys.path, NT=/AC= structure is added automatically.
    """

    def __init__(self, pub_text_dir: Path, config: RefineConfig = None):
        self.pub_text_dir = Path(pub_text_dir)
        self.cfg          = config or RefineConfig()
        self._refiner     = None

    def _ensure_module(self):
        if self._refiner is None:
            ExtractionRefiner = _build_module()
            self._refiner = ExtractionRefiner()

    def load_optimised(self, program_path: Path = None):
        """Load BFRS-optimised demo weights from a JSON file."""
        self._ensure_module()
        p = Path(program_path or self.cfg.program_path or '')
        if p.exists():
            self._refiner.load(str(p))
            log.info(f'Loaded optimised refiner program ← {p}')
        else:
            log.warning(f'Program file not found: {p}')

    # ── Public API ───────────────────────────────────────────────────────────

    def refine(self, submission: pd.DataFrame) -> pd.DataFrame:
        """
        Refine an entire submission DataFrame.
        One LLM call per PXD.  Returns a new DataFrame.
        """
        self._ensure_module()
        parts = []
        groups = list(submission.groupby('PXD', sort=False))
        log.info(f'Refining {len(groups)} PXDs ({len(submission)} rows total)…')
        for pxd, group in groups:
            try:
                refined = self.refine_pxd(pxd, group.copy())
            except Exception as e:
                log.error(f'[{pxd}] failed: {e} — keeping original rows')
                refined = group.copy()
            parts.append(refined)
        return pd.concat(parts, ignore_index=True)

    def refine_pxd(self, pxd: str, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Refine all rows for a single PXD with one LLM call.

        The result is always broadcast to every row — the LLM never sees
        or assigns per-row values.  Row-varying columns that the rules
        already filled are left untouched.

        Parameters
        ----------
        pxd  : ProteomeXchange accession string
        rows : All submission rows for this PXD (DataFrame slice, already copied)

        Returns
        -------
        Updated rows DataFrame (same shape, same index).
        """
        self._ensure_module()

        # 1. Which columns need fixing?
        cols_to_fix = _cols_to_fix(rows, self.cfg)
        if not cols_to_fix:
            log.info(f'[{pxd}] ({len(rows)} rows) — nothing to fix, skipping')
            return rows

        snake_to_fix = [_COL_TO_SNAKE.get(c, c) for c in cols_to_fix
                        if not re.match(r'^Characteristics\[Modification\]\.\d+$', c)]
        # Deduplicate (multiple Modification.N → single 'modification' snake key)
        snake_to_fix_dedup = list(dict.fromkeys(
            'modification' if s.startswith('modification') else s
            for s in snake_to_fix
        ))

        # 2. Build current-values dict (first row is enough — all uniform here)
        first = rows.iloc[0]
        current = {}
        for col in GLOBALS_COLS + SAMPLE_UNIFORM_COLS:
            if col not in rows.columns:
                continue
            snake = _COL_TO_SNAKE.get(col, col)
            val   = first.get(col)
            current[snake] = None if _is_empty(val) else str(val).strip()

        # 3. Load pub text
        pub_text = _load_pub_text(pxd, self.pub_text_dir, self.cfg)
        if not pub_text:
            log.warning(f'[{pxd}] no pub text — skipping')
            return rows

        fields_str = ', '.join(snake_to_fix_dedup)
        log.info(f'[{pxd}] ({len(rows)} rows) fixing: {fields_str}')

        # 4. LLM call with retries
        for attempt in range(self.cfg.max_retries):
            try:
                result = self._refiner(
                    pub_text       = pub_text,
                    current_values = json.dumps(current, ensure_ascii=False),
                    fields_to_fix  = fields_str,
                    schema_hint    = SCHEMA_HINT,
                )
                corrections = _parse_json_safe(result.corrections)
                if not isinstance(corrections, dict):
                    log.warning(f'[{pxd}] attempt {attempt+1}: bad JSON — retrying')
                    continue

                # 5. Apply resolve.py structure, then broadcast to all rows
                try:
                    from resolve import resolve_globals_structure
                    corrections = resolve_globals_structure(corrections)
                except ImportError:
                    pass

                rows = _apply_corrections(rows, corrections, cols_to_fix,
                                          self.cfg.allow_overwrite)
                n_applied = sum(1 for v in corrections.values() if v is not None)
                log.info(f'[{pxd}] {n_applied} field(s) corrected → broadcast to '
                         f'{len(rows)} rows')
                return rows

            except Exception as e:
                log.warning(f'[{pxd}] attempt {attempt+1} error: {e}')

        log.error(f'[{pxd}] all retries failed — keeping original rows')
        return rows


# ---------------------------------------------------------------------------
# BFRS optimiser
# ---------------------------------------------------------------------------

class SDRFRefineOptimiser:
    """
    Optimises SDRFRefiner using BootstrapFewShotWithRandomSearch.

    Builds training examples from (rules submission, ground-truth TSV) pairs:
    gold = fields that rules missed but ground truth has.

    Usage
    -----
        opt = SDRFRefineOptimiser(
            pub_text_dir   = TRAIN_TEXT_DIR,
            train_sdrf_dir = TRAIN_SDRF_DIR,
            rules_sub_path = Path('submission_rules.csv'),
            output_dir     = Path('optim_output'),
        )
        opt.run(n_train=30, n_val=10, num_candidates=8)
        # saves: optim_output/refine_optimized.json
    """

    def __init__(
        self,
        pub_text_dir  : Path,
        train_sdrf_dir: Path,
        rules_sub_path: Path,
        output_dir    : Path,
        config        : RefineConfig = None,
    ):
        self.pub_text_dir   = Path(pub_text_dir)
        self.train_sdrf_dir = Path(train_sdrf_dir)
        self.rules_sub_path = Path(rules_sub_path)
        self.output_dir     = Path(output_dir)
        self.cfg            = config or RefineConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_gt_tsv(self, pxd: str) -> Optional[pd.DataFrame]:
        for pat, sep in [(f'{pxd}_cleaned.sdrf.tsv', '\t'),
                          (f'Harmonized_{pxd}.csv',   ',')]:
            p = self.train_sdrf_dir / pat
            if p.exists():
                return pd.read_csv(p, sep=sep, low_memory=False)
        return None

    def _tsv_inner(self, col: str) -> str:
        m = re.match(r'^(?:Characteristics|Comment)\[(.+?)\](\.\d+)?$', col)
        if m:
            return m.group(1) + (m.group(2) or '')
        return col

    def _build_examples(self, pxd_list: list) -> list:
        """Build dspy.Example list for optimiser."""
        import dspy
        rules_sub = pd.read_csv(self.rules_sub_path)
        examples  = []

        for pxd in pxd_list:
            pub_text = _load_pub_text(pxd, self.pub_text_dir, self.cfg)
            if not pub_text:
                continue
            gt_df = self._load_gt_tsv(pxd)
            if gt_df is None or gt_df.empty:
                continue

            pxd_rows = rules_sub[rules_sub['PXD'] == pxd]
            if pxd_rows.empty:
                continue

            cols_to_fix = _cols_to_fix(pxd_rows, self.cfg)
            if not cols_to_fix:
                continue

            # Build current values from first rules row
            first = pxd_rows.iloc[0]
            current = {
                _COL_TO_SNAKE.get(col, col): (
                    None if _is_empty(first.get(col)) else str(first[col]).strip()
                )
                for col in GLOBALS_COLS + SAMPLE_UNIFORM_COLS
                if col in pxd_rows.columns
            }

            # Build gold corrections: fields rules missed but GT has
            gt_first  = gt_df.iloc[0]
            gold_corrections = {}
            for col in cols_to_fix:
                snake   = _COL_TO_SNAKE.get(col, col)
                tsv_col = self._tsv_inner(col)
                actual  = next(
                    (c for c in gt_df.columns
                     if c.lower().replace(' ','') == tsv_col.lower().replace(' ','')),
                    None)
                if not actual:
                    continue
                gt_val = gt_first.get(actual)
                if _is_empty(gt_val):
                    continue
                # Simplify: extract NT= name for structured fields
                gt_str = str(gt_val).strip()
                if 'NT=' in gt_str:
                    parts = [p for p in gt_str.split(';') if p.startswith('NT=')]
                    gt_str = parts[0].replace('NT=', '').strip() if parts else gt_str
                gold_corrections[snake] = gt_str

            if not gold_corrections:
                continue

            snake_to_fix = list(dict.fromkeys(
                'modification' if s.startswith('modification') else s
                for s in [_COL_TO_SNAKE.get(c, c) for c in cols_to_fix]
            ))

            ex = dspy.Example(
                pub_text       = pub_text,
                current_values = json.dumps(current, ensure_ascii=False),
                fields_to_fix  = ', '.join(snake_to_fix),
                schema_hint    = SCHEMA_HINT,
                corrections    = json.dumps(gold_corrections, ensure_ascii=False),
            ).with_inputs('pub_text', 'current_values', 'fields_to_fix', 'schema_hint')
            examples.append(ex)
            log.debug(f'[{pxd}] example: fixing {list(gold_corrections.keys())}')

        log.info(f'Built {len(examples)} training examples from {len(pxd_list)} PXDs')
        return examples

    @staticmethod
    def _metric(gold_example, prediction, trace=None) -> float:
        """Field-level F1 between gold corrections and predicted corrections."""
        gold = json.loads(gold_example.corrections)
        pred = _parse_json_safe(prediction.corrections)
        if not isinstance(pred, dict):
            return 0.0
        if not gold:
            return 1.0
        tp = sum(
            1 for k, v in gold.items()
            if k in pred and str(pred[k]).lower().strip() == str(v).lower().strip()
        )
        precision = tp / len(pred) if pred else 0.0
        recall    = tp / len(gold)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def run(
        self,
        n_train            : int = 30,
        n_val              : int = 10,
        num_candidates     : int = 8,
        max_bootstrapped_demos: int = 2,
        max_labeled_demos  : int = 1,
        num_threads        : int = 4,
        seed               : int = 42,
    ) -> Path:
        """Run BFRS and save the optimised program. Returns the save path."""
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch
        import random

        rules_sub = pd.read_csv(self.rules_sub_path)
        available = sorted(rules_sub['PXD'].unique().tolist())
        rng = random.Random(seed)
        rng.shuffle(available)
        train_pxds = available[:n_train]
        val_pxds   = available[n_train:n_train + n_val]

        trainset = self._build_examples(train_pxds)
        valset   = self._build_examples(val_pxds)

        if not trainset:
            log.error('No training examples — aborting')
            raise RuntimeError('No training examples could be built')

        ExtractionRefiner = _build_module()
        program = ExtractionRefiner()

        optimizer = BootstrapFewShotWithRandomSearch(
            metric                 = self._metric,
            max_bootstrapped_demos = max_bootstrapped_demos,
            max_labeled_demos      = max_labeled_demos,
            num_candidate_programs = num_candidates,
            num_threads            = num_threads,
            max_errors             = 5,
        )
        log.info(f'BFRS: {len(trainset)} train, {len(valset)} val, '
                 f'{num_candidates} candidates')
        optimised = optimizer.compile(program, trainset=trainset)

        save_path = self.output_dir / 'refine_optimized.json'
        optimised.save(str(save_path))
        log.info(f'Saved → {save_path}')

        if valset:
            scores = []
            for ex in valset:
                try:
                    pred = optimised(**ex.inputs())
                    scores.append(self._metric(ex, pred))
                except Exception as e:
                    log.warning(f'Val error: {e}')
                    scores.append(0.0)
            log.info(f'Val F1 = {sum(scores)/len(scores):.4f}  '
                     f'({len(scores)} examples)')

        return save_path