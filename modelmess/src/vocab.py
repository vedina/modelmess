"""
vocab.py  —  Training vocabulary builder and submission value snapper.

Reads all training SDRF TSV files, builds a per-submission-column Counter
of observed values, and provides snap_value() to normalise LLM outputs to
the nearest training-observed canonical form.

Usage in notebook cell (after importing):

    from vocab import build_vocab, snap_submission

    # Build once per session — point at your local training SDRF directory
    TRAIN_SDRF_DIR = Path('/kaggle/input/.../Training_SDRFs/HarmonizedFiles')
    vocab = build_vocab(TRAIN_SDRF_DIR)

    # Post-process a submission DataFrame
    submission_df = snap_submission(submission_df, vocab)

Where to insert in the notebook:
    After the submission builder cell (the one that writes submission.csv),
    call snap_submission() on the resulting DataFrame before saving.
"""

from __future__ import annotations

import re
import difflib
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger('sdrf.vocab')

NA = 'Not Applicable'

# ── Column name normalisation ─────────────────────────────────────────────────

def _sub_col_to_tsv(col: str) -> str:
    """Map a submission column name to its TSV field name (inner part)."""
    m = re.match(r'^(Characteristics|Comment)\[(.+?)\](\.\d+)?$', col)
    if m:
        return m.group(2) + (m.group(3) or '')
    m2 = re.match(r'^FactorValue\[(.+?)\]$', col)
    if m2:
        return f'FactorValue[{m2.group(1)}]'
    return col


def _tsv_col_norm(col: str) -> str:
    """Normalise a TSV column name for case-insensitive matching."""
    return re.sub(r'[\s\[\]().]', '', col).lower()


# ── NT= extraction (same logic as scorer) ────────────────────────────────────

def _extract_nt(value: str) -> str:
    """Extract the NT= component from a kv-string, or return value as-is."""
    if 'NT=' in value:
        parts = [p for p in value.split(';') if 'NT=' in p]
        if parts:
            return parts[0].replace('NT=', '').strip()
    return value.strip()


# ── Vocabulary builder ────────────────────────────────────────────────────────

# Columns where we track full string (not NT= extracted),
# because the exact format matters for clustering
_KEEP_FULL = {
    'Characteristics[CleavageAgent]',
    'Characteristics[Modification]',
    'Characteristics[Modification].1',
    'Characteristics[Modification].2',
    'Characteristics[Modification].3',
    'Characteristics[Modification].4',
    'Characteristics[Modification].5',
    'Characteristics[Modification].6',
    'Comment[Instrument]',
    'Comment[FragmentationMethod]',
    'Comment[FractionationMethod]',
    'Comment[Separation]',
    'Comment[EnrichmentMethod]',
    'Comment[AcquisitionMethod]',
    'Comment[MS2MassAnalyzer]',
    'Characteristics[Label]',
}

# Columns to skip entirely (too study-specific or numeric)
_SKIP_SNAP = {
    'Characteristics[Age]',
    'Characteristics[BMI]',
    'Characteristics[BiologicalReplicate]',
    'Comment[FractionIdentifier]',
    'Comment[CollisionEnergy]',
    'Comment[GradientTime]',
    'Comment[FlowRateChromatogram]',
    'Comment[NumberOfMissedCleavages]',
    'Comment[PrecursorMassTolerance]',
    'Comment[FragmentMassTolerance]',
    'Characteristics[NumberOfBiologicalReplicates]',
    'Characteristics[NumberOfSamples]',
    'Characteristics[NumberOfTechnicalReplicates]',
}


def build_vocab(train_sdrf_dir: Path,
                sample_sub_path: Path,
                min_count: int = 1) -> Dict[str, Counter]:
    """
    Read all training SDRF TSV (or CSV) files and build a Counter of
    observed values per submission column.

    Args:
        train_sdrf_dir  : directory containing *_cleaned.sdrf.tsv or Harmonized_*.csv
        sample_sub_path : explicit path to SampleSubmission.csv
        min_count       : minimum occurrences to include a value in vocab

    Returns:
        dict mapping submission_col_name → Counter{value: count}
    """
    train_sdrf_dir  = Path(train_sdrf_dir)
    sample_sub_path = Path(sample_sub_path)

    # Discover files — support both local (*.tsv) and Kaggle (*.csv) formats
    tsv_files = list(train_sdrf_dir.glob('*_cleaned.sdrf.tsv'))
    csv_files = list(train_sdrf_dir.glob('Harmonized_*.csv'))
    files     = tsv_files or csv_files

    if not files:
        log.warning(f'No training SDRF files found in {train_sdrf_dir}')
        return {}

    log.info(f'Building vocab from {len(files)} training files in {train_sdrf_dir}')

    # Load submission column list
    sub_template    = pd.read_csv(sample_sub_path, nrows=0)
    SUBMISSION_COLS = [c for c in sub_template.columns
                       if c not in ('ID', 'PXD', 'Raw Data File', 'Usage')]

    # Map normalised TSV col name → submission col name(s)
    tsv_norm_to_sub: dict[str, list[str]] = defaultdict(list)
    for sc in SUBMISSION_COLS:
        tc    = _sub_col_to_tsv(sc)
        # Also map the base (strip .N suffix) for Modification.1 etc.
        base  = tc.split('.')[0] if re.search(r'\.\d+$', tc) else tc
        tsv_norm_to_sub[_tsv_col_norm(tc)].append(sc)
        if base != tc:
            tsv_norm_to_sub[_tsv_col_norm(base)].append(sc)

    #print(tsv_norm_to_sub)
    vocab: Dict[str, Counter] = defaultdict(Counter)

    for fpath in files:
        sep = '\t' if fpath.suffix == '.tsv' else ','
        try:
            df = pd.read_csv(fpath, sep=sep, low_memory=False)
        except Exception as e:
            log.warning(f'Failed to read {fpath.name}: {e}')
            continue

        # Build normalised col lookup for this file
        file_col_norm = {
            _tsv_col_norm(c).replace("characteristics", "").replace("comment", ""):
            c
            for c in df.columns
        }
        #print("file_col_norm",file_col_norm)
        for tsv_norm, sub_cols in tsv_norm_to_sub.items():
            if tsv_norm not in file_col_norm:
                #print(tsv_norm, file_col_norm.keys())
                continue
            actual_col = file_col_norm[tsv_norm]
            vals = (df[actual_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist())
            for v in vals:
                v = v.strip()
                if v in ('', 'nan', 'NA', 'not applicable', 'Not Applicable',
                          'N/A', 'n/a', 'NA', 'TextSpan', 'Text Span'):
                    continue
                for sc in sub_cols:
                    vocab[sc][v] += 1

    # Filter by min_count
    if min_count > 1:
        vocab = {col: Counter({v: c for v, c in ctr.items() if c >= min_count})
                 for col, ctr in vocab.items()}

    log.info(f'Vocab built: {len(vocab)} columns, '
             f'{sum(len(c) for c in vocab.values())} total values')
    return dict(vocab)


# ── Value snapper ─────────────────────────────────────────────────────────────

def snap_value(value: str,
               candidates: list[str],
               threshold: float = 0.80,
               keep_full: bool = False) -> str:
    """
    Snap a single value to the nearest candidate using the same
    string-similarity logic as the scorer.

    For NT= fields (Instrument, CleavageAgent etc.) compares the NT= part,
    then returns the full candidate string so the AC= prefix is preserved.

    Args:
        value      : the value to normalise
        candidates : list of canonical values from training vocab
        threshold  : minimum similarity to snap (default 0.80, same as scorer)
        keep_full  : if True, compare full strings (not NT= extracted)

    Returns:
        best matching candidate, or original value if no match above threshold
    """
    if not candidates or not value or value == NA:
        return value

    v_cmp = value if keep_full else _extract_nt(value)

    best_val  = value
    best_sim  = 0.0

    for cand in candidates:
        c_cmp = cand if keep_full else _extract_nt(cand)
        sim   = difflib.SequenceMatcher(None,
                                         v_cmp.lower(),
                                         c_cmp.lower()).ratio()
        if sim > best_sim:
            best_sim = sim
            best_val = cand

    if best_sim >= threshold:
        if best_val != value:
            log.debug(f'Snapped {value!r} → {best_val!r} (sim={best_sim:.2f})')
        return best_val
    return value


def snap_submission(submission: pd.DataFrame,
                    vocab: Dict[str, Counter],
                    threshold: float = 0.80,
                    na_value: str = NA) -> pd.DataFrame:
    """
    Post-process a submission DataFrame by snapping each value to the
    nearest training-observed canonical form.

    Rules:
    - Columns in _SKIP_SNAP are left untouched (numeric / study-specific).
    - Values equal to na_value are left untouched.
    - For columns with NT= format (Instrument, CleavageAgent etc.),
      the NT= part is used for matching but the full canonical string
      (with AC=) is returned.
    - If no training candidate is within threshold, original value is kept.

    Args:
        submission : the submission DataFrame (has ID, PXD, Raw Data File, ...)
        vocab      : output of build_vocab()
        threshold  : string-similarity threshold (default 0.80)
        na_value   : the NA sentinel string (default 'Not Applicable')

    Returns:
        Modified copy of submission (original is not mutated).
    """
    out = submission.copy()
    data_cols = [c for c in submission.columns
                 if c not in ('ID', 'PXD', 'Raw Data File', 'Usage')]

    for col in data_cols:
        if col in _SKIP_SNAP:
            continue
        if col not in vocab or not vocab[col]:
            continue

        candidates  = list(vocab[col].keys())
        keep_full   = col in _KEEP_FULL

        def _snap(v):
            sv = str(v).strip()
            if sv in (na_value, '', 'nan', 'NaN'):
                return sv
            return snap_value(sv, candidates, threshold, keep_full)

        out[col] = out[col].apply(_snap)

    return out


# ── Standalone test / summary ─────────────────────────────────────────────────

def print_vocab_summary(vocab: Dict[str, Counter],
                        top_n: int = 5) -> None:
    """Print a summary of the vocabulary for inspection."""
    print(f"Vocabulary: {len(vocab)} columns")
    for col, ctr in sorted(vocab.items()):
        top = ctr.most_common(top_n)
        vals_str = ', '.join(f'{v!r}({c})' for v, c in top)
        print(f"  {col:50} [{len(ctr)} unique] {vals_str}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vocab.py <train_sdrf_dir> [sample_value col]")
        sys.exit(1)
    train_dir = Path(sys.argv[1])
    v = build_vocab(train_dir)
    print_vocab_summary(v)

    if len(sys.argv) == 4:
        val = sys.argv[2]
        col = sys.argv[3]
        if col in v:
            snapped = snap_value(val, list(v[col].keys()))
            print(f"\nSnap test: {val!r} in {col!r} → {snapped!r}")