"""
Pydantic models for SDRF metadata fields.
All fields map 1:1 to the SDRF CSV column specification.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class SDRFRow(BaseModel):
    """- One SDRF row represents one sample–data file relationship."""

    # --- Row prefix ---
    sample_source: str = Field(default="sample source", description="Unique ID for the starting material (e.g. organism)")
    assay_name: str = Field(default="run 1", description="Unique ID for the assay (sample + file)")
    raw_data_file: str = Field(description="File name or path of the raw spectrum file")

    # --- Sample Characteristics ---
    age: str = Field(default="not applicable", description="Sample level - Age of the donor or developmental stage of the organism (e.g. “45 years”, “E14.5 embryo”).")
    alkylation_reagent: str = Field(default="not applicable", description="Sample level:  A chemical (like Iodoacetamide (IAA) or N-ethylmaleimide (NEM)) that irreversibly adds an alkyl group to the free sulfhydryl (-SH) of cysteine residues, blocking disulfide bonds and preventing protein re-folding.")
    anatomic_site_tumor: str = Field(default="not applicable")
    ancestry_category: str = Field(default="not applicable")
    bmi: str = Field(default="not applicable")
    bait: str = Field(default="not applicable")
    biological_replicate: str = Field(default="not applicable")
    cell_line: str = Field(default="not applicable")
    cell_part: str = Field(default="not applicable")
    cell_type: str = Field(default="not applicable")
    cleavage_agent: str = Field(default="not applicable")
    compound: str = Field(default="not applicable")
    concentration_of_compound: str = Field(default="not applicable")
    depletion: str = Field(default="not applicable")
    developmental_stage: str = Field(default="not applicable")
    disease_treatment: str = Field(default="not applicable")
    disease: str = Field(default="not applicable")
    genetic_modification: str = Field(default="not applicable")
    genotype: str = Field(default="not applicable")
    growth_rate: str = Field(default="not applicable")
    label: str = Field(default="not applicable")
    material_type: str = Field(default="not applicable")
    modification: str = Field(default="not applicable")
    modification_1: str = Field(default="not applicable")
    modification_2: str = Field(default="not applicable")
    modification_3: str = Field(default="not applicable")
    modification_4: str = Field(default="not applicable")
    modification_5: str = Field(default="not applicable")
    modification_6: str = Field(default="not applicable")
    number_of_biological_replicates: str = Field(default="not applicable")
    number_of_samples: str = Field(default="not applicable")
    number_of_technical_replicates: str = Field(default="not applicable")
    organism_part: str = Field(default="not applicable")
    organism: str = Field(default="not applicable")
    origin_site_disease: str = Field(default="not applicable")
    pooled_sample: str = Field(default="not applicable")
    reduction_reagent: str = Field(default="not applicable")
    sampling_time: str = Field(default="not applicable")
    sex: str = Field(default="not applicable")
    specimen: str = Field(default="not applicable")
    spiked_compound: str = Field(default="not applicable")
    staining: str = Field(default="not applicable")
    strain: str = Field(default="not applicable")
    synthetic_peptide: str = Field(default="not applicable")
    temperature: str = Field(default="not applicable")
    time: str = Field(default="not applicable")
    treatment: str = Field(default="not applicable")
    tumor_cellularity: str = Field(default="not applicable")
    tumor_grade: str = Field(default="not applicable")
    tumor_site: str = Field(default="not applicable")
    tumor_size: str = Field(default="not applicable")
    tumor_stage: str = Field(default="not applicable")

    # --- Comments (instrument / acquisition) ---
    acquisition_method: str = Field(default="not applicable")
    collision_energy: str = Field(default="not applicable")
    enrichment_method: str = Field(default="not applicable")
    flow_rate_chromatogram: str = Field(default="not applicable")
    fraction_identifier: str = Field(default="not applicable")
    fractionation_method: str = Field(default="not applicable")
    fragment_mass_tolerance: str = Field(default="not applicable")
    fragmentation_method: str = Field(default="not applicable")
    gradient_time: str = Field(default="not applicable")
    instrument: str = Field(default="not applicable")
    ionization_type: str = Field(default="not applicable")
    ms2_mass_analyzer: str = Field(default="not applicable")
    number_of_fractions: str = Field(default="not applicable")
    number_of_missed_cleavages: str = Field(default="not applicable")
    precursor_mass_tolerance: str = Field(default="not applicable")
    separation: str = Field(default="not applicable")

    # --- Factor Values ---
    fv_bait: str = Field(default="not applicable")
    fv_cell_part: str = Field(default="not applicable")
    fv_compound: str = Field(default="not applicable")
    fv_concentration_of_compound: str = Field(default="not applicable")
    fv_disease: str = Field(default="not applicable")
    fv_fraction_identifier: str = Field(default="not applicable")
    fv_genetic_modification: str = Field(default="not applicable")
    fv_temperature: str = Field(default="not applicable")
    fv_treatment: str = Field(default="not applicable")

    usage: str = Field(default="raw")


class SDRFDocument(BaseModel):
    """Full extraction result for a paper."""
    rows: List[SDRFRow] = Field(description="One SDRF row represents one sample-data file relationship")
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Any caveats or assumptions made during extraction"
    )
