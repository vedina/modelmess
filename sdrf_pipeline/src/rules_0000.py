"""
rules_0000.py  —  Rule-based SDRF initial-guess extractor
==========================================================
Input : structured paper JSON with keys:
          TITLE, ABSTRACT, METHODS, "Raw Data Files"   (others ignored)
Output: SDRFDocument with one row per channel per file
        (e.g. 16 rows per file for TMT16, 1 row for label-free)

Philosophy
----------
Do as much as possible without the LLM:
  - Raw files come directly from "Raw Data Files" list
  - Regex + keyword rules fill every field that can be inferred reliably
  - Fields the rules cannot determine are left as "not applicable"
    so the LLM pass only needs to fill those gaps

Usage
-----
    from src.rules_0000 import PaperJSON, extract_initial_sdrf

    paper = PaperJSON.from_file("PXD004010_PubText.json")
    doc   = extract_initial_sdrf(paper)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict

from src.models import SDRFDocument, SDRFRow


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Paper JSON loader
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PaperJSON:
    title:     str
    abstract:  str
    methods:   str
    raw_files: list[str]
    pxd:       str = "not available"

    # Combined searchable text (title + abstract + methods only — no results bloat)
    @property
    def searchable(self) -> str:
        return "\n".join([self.title, self.abstract, self.methods])

    @classmethod
    def from_file(cls, path: str | Path) -> "PaperJSON":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        title    = data.get("TITLE", "")
        abstract = data.get("ABSTRACT", "")
        methods  = data.get("METHODS", "")
        raw_files = data.get("Raw Data Files", [])
        # Retain unique values, preserve original order
        raw_files = list(dict.fromkeys(raw_files))
        # Try to find PXD id anywhere in the JSON values
        full_text = json.dumps(data)
        pxd_match = re.search(r'PXD\d{6}', full_text)
        pxd = pxd_match.group(0) if pxd_match else "not available"

        return cls(
            title=title, abstract=abstract, methods=methods,
            raw_files=raw_files, pxd=pxd,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "PaperJSON":
        """Convenience constructor from an already-parsed dict."""
        tmp_path = Path("/tmp/_paper_tmp.json")
        tmp_path.write_text(json.dumps(data), encoding="utf-8")
        return cls.from_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Individual rule functions
#     Each returns a value string or None (= could not determine)
# ══════════════════════════════════════════════════════════════════════════════

def _first(pattern: str, text: str, flags: int = re.IGNORECASE) -> Optional[str]:
    """Return the first non-empty capture group, or None."""
    m = re.search(pattern, text, flags)
    if not m:
        return None
    for g in m.groups():
        if g:
            return g.strip()
    return m.group(0).strip()


def _find_all(pattern: str, text: str, flags: int = re.IGNORECASE) -> list[str]:
    matches = re.findall(pattern, text, flags)
    flat = []
    for m in matches:
        if isinstance(m, tuple):
            flat.extend(v.strip() for v in m if v.strip())
        else:
            if m.strip():
                flat.append(m.strip())
    return list(dict.fromkeys(flat))  # deduplicate, preserve order


# ── Label / quantification ────────────────────────────────────────────────────

def rule_label_and_channels(text: str, raw_files: list | None = None) -> tuple[str, int]:
    """
    Detect isobaric label type and channel count from paper text and/or filenames.

    Priority order:
      1. Filenames  — TMT16/TMTpro16 in raw file names is a hard signal
      2. Explicit 'Xplex' number in text  — e.g. "TMTpro 16plex", "TMT 10plex"
      3. Implicit number without 'plex'   — e.g. "TMT 16", "TMT-16"
      4. Generic brand only               — e.g. "TMTpro" (16), "TMT" (6)
      5. SILAC / dimethyl
      6. Label-free (default)

    Root cause of old bug: \bTMT16\b and \bTMTpro16\b both fail on
    "TMTpro 16plex" because the space breaks the word-boundary match.
    Fix: separate brand detection from plex-number extraction.
    """
    # ── 1. Filename signal (strongest) ───────────────────────────────────────
    if raw_files:
        fn_text = " ".join(raw_files)
        m = re.search(r'TMT(?:pro)?[\s\-]?(\d+)', fn_text, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return f"TMT{n}", n
        m = re.search(r'iTRAQ[\s\-]?(\d+)', fn_text, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return f"iTRAQ{n}", n

    # ── 2. Explicit plex number in text ──────────────────────────────────────
    # matches: "TMTpro 16plex", "TMT16plex", "TMT 10plex"
    m = re.search(r'\bTMT(?:pro)?[\s\-]?(\d+)[\s\-]?plex\b', text, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return f"TMT{n}", n

    m = re.search(r'\biTRAQ[\s\-]?(\d+)[\s\-]?plex\b', text, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return f"iTRAQ{n}", n

    # ── 3. Number without 'plex' — "TMT 16", "TMTpro-16", "TMT16" ───────────
    m = re.search(r'\bTMT(?:pro)?[\s\-]?(\d+)\b', text, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return f"TMT{n}", n

    m = re.search(r'\biTRAQ[\s\-]?(\d+)\b', text, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return f"iTRAQ{n}", n

    # ── 7. Multi-dataset papers: 'first dataset ... label-free ... second ... TMT' ──
    # Split on ordinal dataset markers and check which block applies to these files.
    blocks = re.split(
        r'\b(?:first|second|third|1st|2nd|3rd)\s+dataset\b',
        text, flags=re.IGNORECASE
    )
    if len(blocks) > 1:
        # Default to first block (most papers: dataset 1 = the files we have)
        target = blocks[1]
        # Override if any filename appears verbatim in a different block
        if raw_files:
            for fn in raw_files[:3]:
                for blk in blocks[1:]:
                    if fn in blk:
                        target = blk
                        break
        if re.search(r'\blabel.?free\b', target, re.IGNORECASE):
            return "label free sample", 1
        # Re-run steps 2-4 on the narrowed block
        m = re.search(r'\bTMT(?:pro)?[\s\-]?(\d+)[\s\-]?plex\b', target, re.IGNORECASE)
        if m:
            n = int(m.group(1)); return f"TMT{n}", n
        m = re.search(r'\bTMT(?:pro)?[\s\-]?(\d+)\b', target, re.IGNORECASE)
        if m:
            n = int(m.group(1)); return f"TMT{n}", n
        if re.search(r'\bTMTpro\b', target, re.IGNORECASE): return "TMTpro16", 16
        if re.search(r'\bTMT\b',    target, re.IGNORECASE): return "TMT6", 6

    # ── 4. Generic brand only (no explicit channel count) ────────────────────
    if re.search(r'\bTMTpro\b', text, re.IGNORECASE):
        return "TMTpro16", 16   # TMTpro is always 16-plex
    if re.search(r'\bTMT\b', text, re.IGNORECASE):
        return "TMT6", 6        # plain TMT defaults to 6-plex
    if re.search(r'\biTRAQ\b', text, re.IGNORECASE):
        return "iTRAQ4", 4      # plain iTRAQ defaults to 4-plex

    # ── 5. SILAC ──────────────────────────────────────────────────────────────
    if re.search(r'\bSILAC\b', text, re.IGNORECASE):
        if re.search(r'light.*medium|medium.*heavy|triple[\s\-]?SILAC', text, re.IGNORECASE):
            return "SILAC", 3
        return "SILAC", 2

    # ── 6. Dimethyl ───────────────────────────────────────────────────────────
    if re.search(r'\bdimethyl\s+label', text, re.IGNORECASE):
        return "dimethyl", 2

    # ── 7. Multi-dataset papers: 'first dataset ... label-free ... second ... TMT' ──
    # Split on ordinal dataset markers and check which block applies to these files.
    blocks = re.split(
        r'\b(?:first|second|third|1st|2nd|3rd)\s+dataset\b',
        text, flags=re.IGNORECASE
    )
    if len(blocks) > 1:
        # Default to first block (most papers: dataset 1 = the files we have)
        target = blocks[1]
        # Override if any filename appears verbatim in a different block
        if raw_files:
            for fn in raw_files[:3]:
                for blk in blocks[1:]:
                    if fn in blk:
                        target = blk
                        break
        if re.search(r'\blabel.?free\b', target, re.IGNORECASE):
            return "label free sample", 1
        # Re-run steps 2-4 on the narrowed block
        m = re.search(r'\bTMT(?:pro)?[\s\-]?(\d+)[\s\-]?plex\b', target, re.IGNORECASE)
        if m:
            n = int(m.group(1)); return f"TMT{n}", n
        m = re.search(r'\bTMT(?:pro)?[\s\-]?(\d+)\b', target, re.IGNORECASE)
        if m:
            n = int(m.group(1)); return f"TMT{n}", n
        if re.search(r'\bTMTpro\b', target, re.IGNORECASE): return "TMTpro16", 16
        if re.search(r'\bTMT\b',    target, re.IGNORECASE): return "TMT6", 6

    # ── 8. Default: label-free ────────────────────────────────────────────────
    return "label free sample", 1


def rule_organism(text: str) -> str:
    mapping = {
        r'\bhuman\b': "Homo sapiens",
        r'\bHomo sapiens\b': "Homo sapiens",
        r'\bmouse\b|\bmurine\b': "Mus musculus",
        r'\bMus musculus\b': "Mus musculus",
        r'\brat\b|\bRattus\b': "Rattus norvegicus",
        r'\bbovine\b|\bcow\b|\bBos taurus\b': "Bos taurus",
        r'\byeast\b|\bS\.\s*cerevisiae\b': "Saccharomyces cerevisiae",
        r'\bE\.?\s*coli\b': "Escherichia coli",
        r'\bzebrafish\b|\bDanio rerio\b': "Danio rerio",
        r'\bfly\b|\bDrosophila\b': "Drosophila melanogaster",
        r'\bArabidopsis\b': "Arabidopsis thaliana",
    }
    for pattern, latin in mapping.items():
        if re.search(pattern, text, re.IGNORECASE):
            return latin
    return "not applicable"


def rule_organism_part(text: str) -> str:
    parts = {
        r'\bbrain\b': "brain",
        r'\bliver\b': "liver",
        r'\bkidney\b': "kidney",
        r'\blung\b': "lung",
        r'\bheart\b': "heart",
        r'\bplasma\b': "plasma",
        r'\bserum\b': "serum",
        r'\burine\b': "urine",
        r'\bmilk\b': "milk",
        r'\bblood\b': "blood",
        r'\bmuscle\b': "muscle",
        r'\bspleen\b': "spleen",
        r'\bcolon\b|\bcolorectal\b': "colon",
        r'\bpancrea': "pancreas",
        r'\bcell\s+line\b|\bcell\s+culture\b': "cell culture",
    }
    for pat, val in parts.items():
        if re.search(pat, text, re.IGNORECASE):
            return val
    return "not applicable"


def rule_disease(text: str) -> str:
    diseases = {
        r"Alzheimer'?s?\s+disease|AD\s+brain": "Alzheimer's disease",
        r'\bmultiple\s+myeloma\b': "multiple myeloma",
        r"\bParkinson'?s?\b": "Parkinson's disease",
        r'\bcancer\b|\bcarcinoma\b|\btumou?r\b|\bmalignant\b': "cancer",
        r'\bdiabetes\b': "diabetes",
        r'\bhealthy\b|\bnormal\b|\bcontrol\b': "normal",
    }
    for pat, val in diseases.items():
        if re.search(pat, text, re.IGNORECASE):
            return val
    return "not applicable"


def rule_cleavage_agent(text: str) -> str:
    agents = {
        r'\btrypsin\b': "Trypsin",
        r'\bLys-?C\b': "LysC",
        r'\bGlu-?C\b': "GluC",
        r'\bchymotrypsin\b': "Chymotrypsin",
        r'\bAsp-?N\b': "Asp-N",
        r'\bLys-?N\b': "LysN",
        r'\belastase\b': "Elastase",
    }
    found = []
    for pat, val in agents.items():
        if re.search(pat, text, re.IGNORECASE):
            found.append(val)
    return "/".join(found) if found else "not applicable"


def rule_modifications(text: str) -> list[str]:
    """Return up to 7 modification strings (fixed first, then variable)."""
    mods: list[str] = []

    # Fixed mods
    if re.search(r'\bcarbamidomethyl|\bIAA\b|iodoacetamide', text, re.IGNORECASE):
        mods.append("Carbamidomethyl")
    if re.search(r'\bTMT\b', text, re.IGNORECASE):
        # TMT fixed on K and N-term
        mods.insert(0, "TMT6plex")  # goes first as fixed

    # Variable mods
    if re.search(r'\boxidati', text, re.IGNORECASE):
        mods.append("Oxidation")
    if re.search(r'\bacetylat', text, re.IGNORECASE):
        mods.append("Acetyl")
    if re.search(r'\bphospho', text, re.IGNORECASE):
        mods.append("Phospho")
    if re.search(r'\bdeamidat', text, re.IGNORECASE):
        mods.append("Deamidation")
    if re.search(r'\bubiquitin|\bGlyGly\b', text, re.IGNORECASE):
        mods.append("GlyGly")

    # Pad to 7 slots
    while len(mods) < 7:
        mods.append("not applicable")
    return mods[:7]


def rule_instrument(text: str) -> str:
    instruments = [
        (r'Orbitrap\s+Astral', "Orbitrap Astral"),
        (r'Orbitrap\s+Eclipse', "Orbitrap Eclipse"),
        (r'Orbitrap\s+Exploris\s+480', "Orbitrap Exploris 480"),
        (r'Orbitrap\s+Exploris', "Orbitrap Exploris"),
        (r'Orbitrap\s+Lumos', "Orbitrap Lumos"),
        (r'Orbitrap\s+Fusion', "Orbitrap Fusion"),
        (r'Q\s?Exactive\s+HF-X', "Q Exactive HF-X"),
        (r'Q\s?Exactive\s+HF', "Q Exactive HF"),
        (r'Q\s?Exactive\s+Plus', "Q Exactive Plus"),
        (r'Q\s?Exactive', "Q Exactive"),
        (r'LTQ[\s\-]Orbitrap\s+Elite', "LTQ Orbitrap Elite"),
        (r'LTQ[\s\-]Orbitrap\s+Velos', "LTQ Orbitrap Velos"),
        (r'LTQ[\s\-]Orbitrap\s+XL', "LTQ Orbitrap XL"),
        (r'LTQ[\s\-]Orbitrap', "LTQ Orbitrap"),
        (r'timsTOF\s+Pro', "timsTOF Pro"),
        (r'timsTOF', "timsTOF"),
        (r'TripleTOF\s+6600', "TripleTOF 6600"),
        (r'TripleTOF\s+5600', "TripleTOF 5600"),
        (r'TripleTOF', "TripleTOF"),
        (r'QTOF', "QTOF"),
        (r'Fusion\s+Lumos', "Fusion Lumos"),
        (r'Fusion', "Orbitrap Fusion"),
    ]
    for pat, val in instruments:
        if re.search(pat, text, re.IGNORECASE):
            return val
    return "not applicable"


def rule_fragmentation(text: str) -> str:
    methods = {
        r'\bHCD\b': "HCD",
        r'\bCID\b': "CID",
        r'\bETD\b': "ETD",
        r'\bECD\b': "ECD",
        r'\bEThcD\b': "EThcD",
        r'\bUVPD\b': "UVPD",
    }
    found = [v for pat, v in methods.items() if re.search(pat, text, re.IGNORECASE)]
    return found[0] if found else "not applicable"


def rule_acquisition_method(text: str) -> str:
    if re.search(r'\bDIA\b|data[\s\-]independent', text, re.IGNORECASE):
        return "data-independent acquisition"
    if re.search(r'\bDDA\b|data[\s\-]dependent', text, re.IGNORECASE):
        return "data-dependent acquisition"
    if re.search(r'\bPRM\b|parallel\s+reaction\s+monitoring', text, re.IGNORECASE):
        return "PRM"
    if re.search(r'\bSRM\b|\bMRM\b', text, re.IGNORECASE):
        return "SRM"
    return "not applicable"


def rule_separation(text: str) -> str:
    if re.search(r'reverse[\s\-]phase|RPLC|RP-?HPLC|\bC18\b|\bC8\b', text, re.IGNORECASE):
        return "reverse phase"
    if re.search(r'\bHILIC\b', text, re.IGNORECASE):
        return "HILIC"
    if re.search(r'\bSCX\b|strong\s+cation', text, re.IGNORECASE):
        return "SCX"
    return "not applicable"


def rule_precursor_tolerance(text: str) -> str:
    # Look for "X ppm" near precursor/search/tolerance keywords
    m = re.search(
        r'(\d+\.?\d*)\s*ppm[^.]*?(precursor|MS1|tolerance|search)',
        text, re.IGNORECASE
    )
    if not m:
        # Also accept reverse word order
        m = re.search(
            r'(precursor|MS1|tolerance)[^.]*?(\d+\.?\d*)\s*ppm',
            text, re.IGNORECASE
        )
        if m:
            return m.group(2) + " ppm"
    if m:
        return m.group(1) + " ppm"
    return "not applicable"


def rule_missed_cleavages(text: str) -> str:
    m = re.search(r'(\d)\s*missed\s*cleav', text, re.IGNORECASE)
    if m:
        return m.group(1)
    return "not applicable"


def rule_reduction_reagent(text: str) -> str:
    reagents = {
        r'\bDTT\b|\bdithiothreitol\b': "Dithiothreitol",
        r'\bTCEP\b': "TCEP",
        r'\bbeta[\s\-]mercaptoethanol\b': "beta-mercaptoethanol",
        r'\bMMTS\b': "MMTS",
    }
    for pat, val in reagents.items():
        if re.search(pat, text, re.IGNORECASE):
            return val
    return "not applicable"


def rule_alkylation_reagent(text: str) -> str:
    reagents = {
        r'\biodoacetamide\b|\bIAA\b': "Iodoacetamide",
        r'\bchloroacetamide\b|\bCAA\b': "Chloroacetamide",
        r'\bN-ethylmaleimide\b|\bNEM\b': "N-ethylmaleimide",
    }
    for pat, val in reagents.items():
        if re.search(pat, text, re.IGNORECASE):
            return val
    return "not applicable"


def rule_material_type(text: str) -> str:
    if re.search(r'\bcell\s+line\b|\bcultur', text, re.IGNORECASE):
        return "cell culture"
    if re.search(r'\btissue\b|\bbiopsy\b|\bpost.?mortem\b', text, re.IGNORECASE):
        return "tissue"
    if re.search(r'\bplasma\b', text, re.IGNORECASE):
        return "plasma"
    if re.search(r'\bserum\b', text, re.IGNORECASE):
        return "serum"
    if re.search(r'\burine\b', text, re.IGNORECASE):
        return "urine"
    return "not applicable"


def rule_specimen(text: str) -> str:
    if re.search(r'\bFFPE\b', text, re.IGNORECASE):
        return "FFPE"
    if re.search(r'\bfresh[\s\-]frozen\b', text, re.IGNORECASE):
        return "fresh-frozen"
    if re.search(r'\bpost.?mortem\b', text, re.IGNORECASE):
        return "postmortem tissue"
    if re.search(r'\bcell\s+pellet\b', text, re.IGNORECASE):
        return "cell pellet"
    return "not applicable"


def rule_cell_line(text: str) -> str:
    known = [
        "HeLa", "HEK293T", "HEK293", "U2OS", "MCF-7", "A549",
        "Jurkat", "K562", "Huh7", "PC-3", "LNCaP", "ANBL6",
        "THP-1", "SH-SY5Y", "PC12",
    ]
    for cl in known:
        if re.search(rf'\b{re.escape(cl)}\b', text, re.IGNORECASE):
            return cl
    return "not applicable"


def rule_fractionation(text: str) -> Optional[tuple[str, int]]:
    """Returns (method, n_fractions) or None."""
    methods = {
        r'\bSCX\b|strong\s+cation': "SCX",
        r'\bHpRP\b|high[\s\-]pH\s+reverse': "HpRP",
        r'\bSAX\b|strong\s+anion': "SAX",
        r'\bIEF\b|isoelectric\s+focusing': "IEF",
        r'\bOFFGEL\b': "OFFGEL",
        r'\bgel[\s\-]based\b|SDS-PAGE|1D[\s\-]?gel': "gel-based",
    }
    for pat, method in methods.items():
        if re.search(pat, text, re.IGNORECASE):
            # Try to find fraction count
            n_m = re.search(r'(\d+)\s*fraction', text, re.IGNORECASE)
            n = int(n_m.group(1)) if n_m else 1
            return method, n
    return None


def rule_flow_rate(text: str) -> str:
    m = re.search(r'(\d+\.?\d*)\s*(nL|µL|uL)/min', text, re.IGNORECASE)
    if m:
        unit = "nL/min" if m.group(2).lower() in ("nl", "µl", "ul") else m.group(2) + "/min"
        return f"{m.group(1)} {unit}"
    return "not applicable"


def rule_gradient_time(text: str) -> str:
    # "X-min gradient" or "gradient of X min"
    m = re.search(r'(\d+)[\s\-]min(?:ute)?\s+gradient', text, re.IGNORECASE)
    if not m:
        m = re.search(r'gradient\s+(?:of\s+)?(\d+)\s*min', text, re.IGNORECASE)
    if m:
        return f"{m.group(1)} min"
    return "not applicable"


def rule_ms2_analyzer(text: str) -> str:
    if re.search(r'\borbitrap\b', text, re.IGNORECASE):
        return "Orbitrap"
    if re.search(r'\bion\s+trap\b|\blinear\s+trap\b|\bLTQ\b', text, re.IGNORECASE):
        return "linear ion trap"
    if re.search(r'\bTOF\b', text, re.IGNORECASE):
        return "TOF"
    return "not applicable"


def rule_sex(text: str) -> str:
    m_male = re.search(r'\bmale\b', text, re.IGNORECASE)
    m_female = re.search(r'\bfemale\b', text, re.IGNORECASE)
    if m_male and m_female:
        return "mixed"
    if m_female:
        return "female"
    if m_male:
        return "male"
    return "not applicable"


def rule_number_of_samples(raw_files: list[str], channels: int) -> str:
    return str(len(raw_files) * channels)


def rule_biological_replicate_from_filename(filename: str) -> str:
    """
    Try to extract replicate number from filename conventions:
    _R1_, _rep1_, -1-, _01., etc.
    """
    m = re.search(
        r'[_\-](?:rep|r|bio)?0*(\d{1,2})[_\-\.]', filename, re.IGNORECASE
    )
    if m:
        return m.group(1)
    # trailing number before extension: file01.raw → 1
    m = re.search(r'0*(\d+)\.(?:raw|mzML|wiff|d)$', filename, re.IGNORECASE)
    if m:
        return m.group(1)
    return "not applicable"


def rule_channel_label(label_base: str, channel_idx: int, total_channels: int) -> str:
    """Generate per-channel label string, e.g. TMT10-126."""
    TMT6_TAGS  = ["126", "127N", "127C", "128N", "128C", "129N"]
    TMT10_TAGS = TMT6_TAGS + ["129C", "130N", "130C", "131"]
    TMT11_TAGS = TMT10_TAGS + ["131C"]
    TMT16_TAGS = [
        "126", "127N", "127C", "128N", "128C",
        "129N", "129C", "130N", "130C", "131N",
        "131C", "132N", "132C", "133N", "133C", "134N",
    ]
    iTRAQ4_TAGS = ["114", "115", "116", "117"]
    iTRAQ8_TAGS = ["113", "114", "115", "116", "117", "118", "119", "121"]

    if "TMT16" in label_base or total_channels == 16:
        tags = TMT16_TAGS
    elif "TMT11" in label_base or total_channels == 11:
        tags = TMT11_TAGS
    elif "TMT10" in label_base or total_channels == 10:
        tags = TMT10_TAGS
    elif "TMT6" in label_base or total_channels == 6:
        tags = TMT6_TAGS
    elif "iTRAQ8" in label_base or total_channels == 8:
        tags = iTRAQ8_TAGS
    elif "iTRAQ4" in label_base or total_channels == 4:
        tags = iTRAQ4_TAGS
    else:
        tags = [str(i + 1) for i in range(total_channels)]

    tag = tags[channel_idx] if channel_idx < len(tags) else str(channel_idx + 1)
    prefix = label_base.split()[0]  # "TMT10" → "TMT10"
    return f"{prefix}-{tag}"


def _group_files_by_biorep(raw_files: list[str]) -> dict[str, list[str]]:
    """
    Group files by biological replicate inferred from filename.
    Returns dict: replicate_id -> list of files
    """
    groups = defaultdict(list)
    for f in raw_files:
        bio = rule_biological_replicate_from_filename(f)
        groups[bio].append(f)
    return groups


def rule_is_fractionated(filenames: list[str]) -> bool:
    """Check if the file list contains common fractionation patterns."""
    frac_pattern = re.compile(r'[_\-](?:F|frac|part|fraction|slice)[\s\-]?\d+', re.IGNORECASE)
    return any(frac_pattern.search(f) for f in filenames)


def get_sample_root(filename: str) -> str:
    """
    Removes fraction/part suffixes to find the base sample identity.
    Example: 'SampleA_F01.raw' -> 'SampleA'
    """
    stem = Path(filename).stem
    # Removes trailing _F01, _part1, -frac2, etc.
    root = re.sub(r'[_\-](?:F|frac|part|fraction|slice)[\s\-]?\d+$', '', stem, flags=re.IGNORECASE)
    return root


def get_canonical_root(filename: str, all_filenames: list[str]) -> str:
    """
    Identifies if a file is a 'retry' or 'timestamped' version of another.
    Example: 'data_1.raw' and 'data_1_20240101.raw' -> 'data_1'
    """
    stem = Path(filename).stem
    
    # Sort all stems by length (shortest first)
    all_stems = sorted([Path(f).stem for f in all_filenames], key=len)
    
    for s in all_stems:
        # If our current filename starts with a shorter existing filename 
        # AND is followed by a common separator (like _ or -)
        if stem.startswith(s) and stem != s:
            if stem[len(s)] in ("_", "-", "."):
                return s
                
    return stem

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Main extraction function
# ══════════════════════════════════════════════════════════════════════════════

def extract_initial_sdrf(paper: PaperJSON) -> SDRFDocument:
    """
    Apply all rules. Groups files by 'root' name to handle fractionation generically.
    One SDRFRow is produced per (Sample Group × channel).
    """
    text = paper.searchable
    
    # ── 1. Filter Logic: DIA vs DDA ──
    # If both are present, DIA are usually the samples, DDA are the library fractions.
    all_files = paper.raw_files
    dia_files = [f for f in all_files if "DIA" in f.upper()]
    dda_files = [f for f in all_files if "DDA" in f.upper()]
    
    primary_files = dia_files if (dia_files and dda_files) else all_files

    # ── 2. Grouping Logic (Collapse Fractions) ──
    sample_groups = defaultdict(list)
    for f in primary_files:
        #root = get_sample_root(f)
        root = get_canonical_root(f, primary_files)
        sample_groups[root].append(f)
    
    # ── 3. Shared Field Extraction ──
    organism = rule_organism(text)
    organism_part = rule_organism_part(text)
    disease = rule_disease(text)
    cleavage = rule_cleavage_agent(text)
    mods = rule_modifications(text)
    instrument = rule_instrument(text)
    fragmentation = rule_fragmentation(text)
    acquisition = rule_acquisition_method(text)
    separation = rule_separation(text)
    precursor_tol = rule_precursor_tolerance(text)
    missed_cleav = rule_missed_cleavages(text)
    reduction = rule_reduction_reagent(text)
    alkylation = rule_alkylation_reagent(text)
    material = rule_material_type(text)
    specimen = rule_specimen(text)
    cell_line = rule_cell_line(text)
    sex = rule_sex(text)
    flow_rate = rule_flow_rate(text)
    gradient = rule_gradient_time(text)
    ms2_analyzer = rule_ms2_analyzer(text)
    
    label_base, channels = rule_label_and_channels(text, raw_files=primary_files)
    
    # Total samples is (Unique Groups * channels)
    n_samples = str(len(sample_groups) * channels)

    frac_result = rule_fractionation(text)
    frac_method = frac_result[0] if frac_result else "not applicable"

    # ── 4. Per-Group Row Generation ──
    rows: list[SDRFRow] = []
    
    for group_idx, (root_name, files) in enumerate(sample_groups.items()):
        # Sort files to ensure F01, F02 order
        files.sort()
        num_fracs_in_group = len(files)
        
        # We pick the first file as the representative for metadata extraction
        rep_file = files[0]
        bio_rep = rule_biological_replicate_from_filename(rep_file)

        for ch_idx in range(channels):
            if channels > 1:
                ch_label = rule_channel_label(label_base, ch_idx, channels)
                assay = f"{root_name}_ch{ch_idx + 1}"
            else:
                ch_label = label_base
                assay = root_name

            row = SDRFRow(
                sample_source=f"source {group_idx * channels + ch_idx + 1}",
                assay_name=assay,
                # For fractionated samples, standard SDRF often joins files with ';' 
                # or lists the representative. Here we provide the representative.
                raw_data_file=";".join(files) if num_fracs_in_group > 1 else rep_file,
                PXD=paper.pxd,

                alkylation_reagent=alkylation,
                biological_replicate=bio_rep,
                cell_line=cell_line,
                cleavage_agent=cleavage,
                disease=disease,
                label=ch_label,
                material_type=material,
                modification=mods[0],
                modification_1=mods[1],
                modification_2=mods[2],
                modification_3=mods[3],
                modification_4=mods[4],
                modification_5=mods[5],
                modification_6=mods[6],
                number_of_samples=n_samples,
                number_of_technical_replicates="1",
                organism=organism,
                organism_part=organism_part,
                reduction_reagent=reduction,
                sex=sex,
                specimen=specimen,

                acquisition_method=acquisition,
                flow_rate_chromatogram=flow_rate,
                # If grouped, identifier is '1' for the merged row
                fraction_identifier="1", 
                fractionation_method=frac_method,
                fragmentation_method=fragmentation,
                gradient_time=gradient,
                instrument=instrument,
                ionization_type="ESI",
                ms2_mass_analyzer=ms2_analyzer,
                number_of_fractions=str(num_fracs_in_group),
                number_of_missed_cleavages=missed_cleav,
                precursor_mass_tolerance=precursor_tol,
                separation=separation,
            )
            rows.append(row)

    notes = (
        f"Grouped {len(primary_files)} files into {len(sample_groups)} sample(s). "
        f"Detected {channels} channel(s) per sample. "
        f"PXD: {paper.pxd}."
    )

    return SDRFDocument(rows=rows, extraction_notes=notes)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CLI convenience
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, csv

    if len(sys.argv) < 2:
        print("Usage: python -m src.rules_0000 <paper.json> [output.sdrf.csv]")
        sys.exit(1)

    paper = PaperJSON.from_file(sys.argv[1])
    doc   = extract_initial_sdrf(paper)

    out_path = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".json", ".sdrf.csv")

    # Import column order from pipeline
    from src.pipeline import SDRF_HEADERS, HEADER_TO_ATTR

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SDRF_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for row in doc.rows:
            d = row.model_dump()
            writer.writerow({h: d.get(a, "not applicable") for h, a in HEADER_TO_ATTR.items()})

    print(f"Written {len(doc.rows)} rows → {out_path}")
    if doc.extraction_notes:
        print("Notes:", doc.extraction_notes)

# (end of file — no patch needed, rule_label_and_channels is fully fixed above)