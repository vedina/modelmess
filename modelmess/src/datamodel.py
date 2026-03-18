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
    SourceName       : str  = Field(description="Unique sample name, e.g. 'Sample 1'")
    raw_data_file     : str  = Field(description="Raw file name, e.g. file01.raw")
    AssayName        : str  = Field(description="Run identifier, e.g. 'run 1'")
    Usage  : str  = Field(default='Raw Data File')

    # ── Characteristics ───────────────────────────────────────────────────────
    Age: Optional[str] = None
    AlkylationReagent: Optional[str] = None
    AnatomicSiteTumor: Optional[str] = None
    AncestryCategory: Optional[str] = None
    BMI: Optional[str] = None
    Bait: Optional[str] = None
    BiologicalReplicate: Optional[str] = None
    CellLine: Optional[str] = None
    CellPart: Optional[str] = None
    CellType: Optional[str] = None
    CleavageAgent: Optional[str] = None
    Compound: Optional[str] = None
    ConcentrationOfCompound: Optional[str] = None
    Depletion: Optional[str] = None
    DevelopmentalStage: Optional[str] = None
    DiseaseTreatment: Optional[str] = None
    Disease: Optional[str] = None
    GeneticModification: Optional[str] = None
    Genotype: Optional[str] = None
    GrowthRate: Optional[str] = None
    Label: Optional[str] = None
    MaterialType: Optional[str] = None
    NumberOfBiologicalReplicates: Optional[str] = None
    NumberOfSamples: Optional[str] = None
    NumberOfTechnicalReplicates: Optional[str] = None
    OrganismPart: Optional[str] = None
    Organism: Optional[str] = None
    OriginSiteDisease: Optional[str] = None
    PooledSample: Optional[str] = None
    ReductionReagent: Optional[str] = None
    SamplingTime: Optional[str] = None
    Sex: Optional[str] = None
    Specimen: Optional[str] = None
    SpikedCompound: Optional[str] = None
    Staining: Optional[str] = None
    Strain: Optional[str] = None
    SyntheticPeptide: Optional[str] = None
    Temperature: Optional[str] = None
    Time: Optional[str] = None
    Treatment: Optional[str] = None
    TumorCellularity: Optional[str] = None
    TumorGrade: Optional[str] = None
    TumorSite: Optional[str] = None
    TumorSize: Optional[str] = None
    TumorStage: Optional[str] = None


    # ── Comment / MS Technical ────────────────────────────────────────────────
    comment_Label : str   = Field(default=LABEL_FREE)
    comment_AcquisitionMethod: Optional[str]  = None
    comment_CollisionEnergy: Optional[str]  = None
    comment_EnrichmentMethod: Optional[str]  = None
    comment_FlowRateChromatogram: Optional[str]  = None
    comment_FractionIdentifier: Optional[int]  = 1
    comment_FractionationMethod: Optional[str]  = None
    comment_FragmentMassTolerance: Optional[str]  = None
    comment_FragmentationMethod: Optional[str]  = None
    comment_GradientTime: Optional[str]  = None
    comment_Instrument: Optional[InstrumentRef]   = None
    comment_IonizationType: Optional[str]  = None
    comment_MS2MassAnalyzer: Optional[str]  = None
    comment_NumberOfFractions: Optional[str]  = None
    comment_NumberOfMissedCleavages: Optional[str]  = None
    comment_PrecursorMassTolerance: Optional[str]  = None
    comment_Separation: Optional[str]  = None

    # ── Modifications + Cleavage ──────────────────────────────────────────────
    Modifications : List[ProteinModification] = Field(default_factory=list,
            description="Up to 7 modification entries")
    CleavageAgent: Optional[CleavageAgent]   = None

    # ── Factor values ─────────────────────────────────────────────────────────
    factor_disease    : Optional[str] = None
    factor_treatment  : Optional[str] = None
    factor_compound   : Optional[str] = None

    factor_Bait: Optional[str] = None
    factor_CellPart: Optional[str] = None
    factor_Compound: Optional[str] = None
    factor_ConcentrationOfCompound: Optional[str] = None
    factor_Disease: Optional[str] = None
    factor_FractionIdentifier: Optional[str] = None
    factor_GeneticModification: Optional[str] = None
    factor_Temperature: Optional[str] = None
    factor_Treatment: Optional[str] = None


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
    
    def serialize_row(self: SDRFRow) -> dict:
        d = self.model_dump()

        # Convert list of objects → string
        if d.get("Modifications", None):
            d["Modifications"] = ";".join(
                str(m) for m in d["Modifications"]
            )
        else:
            d["Modifications"] = None

        # Convert instrument object
        if d.get("comment_Instrument"):
            d["comment_Instrument"] = str(d["comment_Instrument"])

        # Convert CleavageAgent if object
        if d.get("CleavageAgent"):
            d["CleavageAgent"] = str(d["CleavageAgent"])

        return d    


class SDRFExperiment(BaseModel):
    """All rows extracted from one publication."""
    pxd  : str = Field(description="ProteomeXchange ID, e.g. PXD000070")
    rows : List[SDRFRow]  = Field(default_factory=list)



