"""
Full SDRF-Proteomics 1.0.0 Pydantic models.

Maps directly to the columns in SampleSubmission.csv.
Modification fields use the NT=;AC=;TA=;MT= semicolon-separated format
required by sdrf-pipelines and the scoring function.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ── Controlled vocabulary helpers ────────────────────────────────────────────

LABEL_FREE = 'label free sample'
VALID_MT = {'fixed', 'variable', 'Fixed', 'Variable'}


def _norm_organism(v: str) -> str:
    """Lower-case and expand common abbreviations."""
    mapping = {
        'human': 'Homo sapiens',
        'mouse': 'Mus musculus',
        'rat'  : 'Rattus norvegicus',
        'yeast': 'Saccharomyces cerevisiae',
        'e.coli': 'Escherichia coli',
    }
    return mapping.get(v.strip().lower(), v.strip())


# ── Sub-models ────────────────────────────────────────────────────────────────

class ProteinModification(BaseModel):
    """
    A protein modification in SDRF key=value format.
    Serialises as: NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed
    """
    NT: str  = Field(description="Modification name, e.g. Carbamidomethyl")
    AC: Optional[str] = Field(None, description="UNIMOD accession, e.g. UNIMOD:4")
    MT: Optional[str] = Field(None, description="fixed or variable")
    PP: Optional[str] = Field(None, description="Position: Anywhere / Protein N-term / etc.")
    TA: Optional[str] = Field(None, description="Target amino acids, e.g. C or S,T,Y")
    CF: Optional[str] = Field(None, description="Chemical formula, e.g. H3C2NO")
    MM: Optional[float] = Field(None, description="Monoisotopic mass shift")

    def to_sdrf_string(self) -> str:
        parts = [f"NT={self.NT}"]
        if self.AC: parts.append(f"AC={self.AC}")
        if self.PP: parts.append(f"PP={self.PP}")
        if self.TA: parts.append(f"TA={self.TA}")
        if self.MT: parts.append(f"MT={self.MT}")
        if self.CF: parts.append(f"CF={self.CF}")
        if self.MM is not None: parts.append(f"MM={self.MM}")
        return ';'.join(parts)


class CleavageAgent(BaseModel):
    """Enzyme details. Serialises as: AC=MS:1001251;NT=Trypsin"""
    NT: str  = Field(description="Enzyme name, e.g. Trypsin")
    AC: Optional[str] = Field(None, description="PSI-MS CV accession, e.g. MS:1001251")
    CS: Optional[str] = Field(None, description="Cleavage site regex")

    def to_sdrf_string(self) -> str:
        parts = [f"NT={self.NT}"]
        if self.AC: parts.append(f"AC={self.AC}")
        if self.CS: parts.append(f"CS={self.CS}")
        return ';'.join(parts)


class InstrumentRef(BaseModel):
    """MS instrument. Serialises as: AC=MS:1001742;NT=LTQ Orbitrap Velos"""
    NT: str  = Field(description="Instrument name")
    AC: Optional[str] = Field(None, description="PSI-MS CV accession")

    def to_sdrf_string(self) -> str:
        parts = [f"NT={self.NT}"]
        if self.AC: parts.append(f"AC={self.AC}")
        return ';'.join(parts)


# ── Main SDRF row model ───────────────────────────────────────────────────────

class SDRFRow(BaseModel):
    """
    One row of an SDRF-Proteomics 1.0.0 file.
    Field names match SampleSubmission.csv column headers.

    Only the most commonly populated fields are mandatory;
    everything else is Optional to accommodate sparse papers.
    """

    # ── Identifiers ──────────────────────────────────────────────────────────
    source_name       : str  = Field(description="Unique sample name, e.g. 'Sample 1'")
    raw_data_file     : str  = Field(description="Raw file name, e.g. file01.raw")
    assay_name        : str  = Field(description="Run identifier, e.g. 'run 1'")
    usage             : str  = Field(default='Raw Data File')

    # ── Characteristics ───────────────────────────────────────────────────────
    organism              : str           = Field(description="Latin name, e.g. Homo sapiens")
    organism_part         : Optional[str] = None
    cell_type             : Optional[str] = None
    cell_line             : Optional[str] = None
    disease               : Optional[str] = None
    biological_replicate  : Optional[int] = None
    technical_replicate   : Optional[int] = None
    sex                   : Optional[str] = None
    age                   : Optional[str] = None
    developmental_stage   : Optional[str] = None
    strain                : Optional[str] = None
    genotype              : Optional[str] = None
    genetic_modification  : Optional[str] = None
    phenotype             : Optional[str] = None
    compound              : Optional[str] = None
    concentration_compound: Optional[str] = None
    treatment             : Optional[str] = None
    disease_treatment     : Optional[str] = None
    material_type         : Optional[str] = None
    pooled_sample         : Optional[str] = None
    specimen              : Optional[str] = None

    # ── Comment / MS Technical ────────────────────────────────────────────────
    label                   : str                       = Field(default=LABEL_FREE)
    fraction_identifier     : Optional[int]             = 1
    instrument              : Optional[InstrumentRef]   = None
    fragmentation_method    : Optional[str]             = None   # e.g. CID, ETD, HCD
    ms2_mass_analyzer       : Optional[str]             = None
    acquisition_method      : Optional[str]             = None   # DDA or DIA
    ionization_type         : Optional[str]             = None   # ESI, MALDI
    enrichment_method       : Optional[str]             = None
    fractionation_method    : Optional[str]             = None
    separation              : Optional[str]             = None
    gradient_time           : Optional[str]             = None
    flow_rate               : Optional[str]             = None
    precursor_mass_tolerance: Optional[str]             = None   # e.g. '10 ppm'
    fragment_mass_tolerance : Optional[str]             = None   # e.g. '0.02 Da'
    number_of_missed_cleavages: Optional[int]           = None
    number_of_fractions     : Optional[int]             = None
    collision_energy        : Optional[str]             = None

    # ── Modifications + Cleavage ──────────────────────────────────────────────
    modifications  : List[ProteinModification] = Field(default_factory=list,
                       description="Up to 7 modification entries")
    cleavage_agent : Optional[CleavageAgent]   = None

    # ── Factor values ─────────────────────────────────────────────────────────
    factor_disease    : Optional[str] = None
    factor_treatment  : Optional[str] = None
    factor_compound   : Optional[str] = None

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator('organism')
    @classmethod
    def normalise_organism(cls, v):
        return _norm_organism(v)

    @field_validator('label')
    @classmethod
    def normalise_label(cls, v):
        if re.search(r'label.?free', v, re.I):
            return LABEL_FREE
        return v

    @field_validator('usage')
    @classmethod
    def normalise_usage(cls, v):
        if v not in ('Raw Data File', 'Spectrum Library'):
            return 'Raw Data File'
        return v


class SDRFExperiment(BaseModel):
    """All rows extracted from one publication."""
    pxd  : str            = Field(description="ProteomeXchange ID, e.g. PXD000070")
    rows : List[SDRFRow]  = Field(default_factory=list)


print("Pydantic models loaded.")