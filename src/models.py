"""
Pydantic models for SDRF metadata fields.
All fields map 1:1 to the SDRF CSV column specification.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class SDRFRow(BaseModel):
    """- One SDRF row represents one sample–data file relationship."""

    # --- Row prefix ---
    sample_source: str = Field(default="Sample 1", description="Unique ID for the starting material.")
    assay_name: str = Field(default="run 1", description="Unique ID for the assay")
    raw_data_file: str = Field(description="File name or path of the raw spectrum file")

    # --- Sample Characteristics ---
    age: str = Field(default=None, 
                     description="Age of the donor or developmental stage of the organism",
                      examples="45 years; E14.5 embryo")
    alkylation_reagent: str = Field(default=None,
                                    description="Chemical that blocks cysteine -SH groups, preventing disulfide formation and protein refolding.",
                                     examples="IAA")
    
    anatomic_site_tumor: str = Field(default=None,
                                     description="Anatomical location from which a tumor sample was taken.",
                                     examples="left lung lobe")
    ancestry_category: str = Field(default=None,
                                   description="Donor ancestry or ethnicity category.",
                                   examples="Caucasian")
    bmi: str = Field(default=None,
                     description="Body‐Mass Index of the donor (kg/m²).")
    bait: str = Field(default=None,
                      description="The protein or molecule used as bait in an affinity‐purification experiment.")
    biological_replicate: str = Field(default=None,
                                      description="Identifier for biological replicates.",
                                      examples="BR1")
    cell_line: str = Field(default=None,
                           description="Name of the immortalized cell line.",
                           examples="HEK293T")
    cell_part: str = Field(default=None,
                           description="Subcellular compartment or fraction.",
                           examples="nucleus")
    cell_type: str = Field(default=None,
                           description="Primary cell type or lineage.",
                           examples="“neurons")
    cleavage_agent: str = Field(default=None,
                                description="Protease or chemical used to digest proteins.",
                                examples="trypsin")
    compound: str = Field(default=None,
                          description="Chemical added to the sample",
                           examples="drug, inhibitor")
    concentration_of_compound: str = Field(default=None,
                                           description="Concentration of the Compound used",
                                           examples="10 µM")
    depletion: str = Field(default=None,
                           description="Method used to remove high‐abundance proteins.",
                           examples="albumin depletion kit")
    developmental_stage: str = Field(default=None,
                                     description="Stage of development for the sample source.",
                                     examples="adult, P7 pup")
    disease_treatment: str = Field(default=None,
                                   description="Pre‐treatment applied to diseased samples", 
                                   examples="chemotherapy")
    disease: str = Field(default=None,
                         description="Disease state or diagnosis.", 
                         examples="breast cancer")
    genetic_modification: str = Field(default=None,
                                      description="Any genetic alteration in the source organism/cells",
                                       examples="knockout of gene X")
    genotype: str = Field(default=None,
                          description="Genotypic background.", examples="C57BL/6J")
    growth_rate: str = Field(default=None,
                             description="Doubling time or growth rate of cell cultures", 
                             examples="24 h doubling")
    label: str = Field(default=None,
                       description="Isobaric or metabolic label applied", examples="TMT-126, SILAC heavy")
    material_type: str = Field(default=None,
                               description="Broad class of material", 
                               examples="tissue, cell line, biofluid")
    modification: str = Field(default=None,
                              description="Post‐translational modification enrichment or tagging", 
                              examples="phosphorylation, ubiquitination")
    modification_1: str = Field(default=None,
                                description="", examples="")
    modification_2: str = Field(default=None,
                                description="", examples="")
    modification_3: str = Field(default=None,
                                description="", examples="")
    modification_4: str = Field(default=None,
                                description="", examples="")
    modification_5: str = Field(default=None,
                                description="", examples="")
    modification_6: str = Field(default=None,
                                description="", examples="")
    number_of_biological_replicates: str = Field(default=None,
                                                 description="Total number of biological replicates in the study.")
    number_of_samples: str = Field(default=None,
                                   description="Total number of samples processed.")
    number_of_technical_replicates: str = Field(default=None,
                                                description="Total number of technical replicates per sample.")
    organism_part: str = Field(default=None,
                               description="Tissue or organ of origin (Uberon term)", examples="UBERON:0002107 (liver)")
    organism: str = Field(default=None,
                          description="Source species (NCBI Taxonomy ID and name)", examples="9606 (Homo sapiens)")
    origin_site_disease: str = Field(default=None,
                                     description="Anatomical site of disease origin", examples="colon, prostate")
    pooled_sample: str = Field(default=None,
                               description="Indicates if multiple samples were pooled.", examples="pool1 of reps1–3")
    reduction_reagent: str = Field(default=None,
                                   description="Chemical used to reduce disulfide bonds.", examples="DTT, TCEP")
    sampling_time: str = Field(default=None,
                               description="Time point of sample collection.", examples="24 h post‐treatment")
    sex: str = Field(default=None,
                     description="Donor sex.", examples="male, female")
    specimen: str = Field(default=None,
                          description="Description of biological specimen.", examples="biopsy, plasma")
    spiked_compound: str = Field(default=None,
                                 description="Exogenous standard or spike‐in added.", examples="iRT peptides")
    staining: str = Field(default=None,
                          description="Any staining applied to the sample prior to mass spec that may still be present in the sample.",
                           )
    strain: str = Field(default=None,
                        description="Animal strain.", examples="BALB/c, FVB/N")
    synthetic_peptide: str = Field(default=None,
                                   description="Indicates a synthetic peptide sample.", examples="synthetic phosphopeptide")
    temperature: str = Field(default=None,
                             description="Growth temperature of the samples or perturbation temperature if a differential study."
                             )
    time: str = Field(default=None,
                      description="Broad time parameter.", examples="day 5, week 2")
    treatment: str = Field(default=None,
                           description="Experimental treatment.", examples="drug X 5 µM 24h")
    tumor_cellularity: str = Field(default=None,
                                   description="Percentage of tumor cells in the sample.", examples="80%")
    tumor_grade: str = Field(default=None,
                             description="Histological grade.", examples="Grade II")
    tumor_site: str = Field(default=None,
                            description="Physical size of the tumor.", examples="3 cm diameter")
    tumor_size: str = Field(default=None,
                            description="natomical site of tumor.", examples="breast, pancreas")
    tumor_stage: str = Field(default=None,
                             description="Clinical staging", examples="Stage III")

    # --- Comments (instrument / acquisition) ---
    acquisition_method: str = Field(default=None,
                                    description="MS acquisition scheme.", examples="DDA, DIA, PRM")
    collision_energy: str = Field(default=None,
                                  description="Collision energy applied in MS/MS", examples="27 eV")
    enrichment_method: str = Field(default=None,
                                   description="Peptide/enrichment protocol used", examples="TiO₂ phosphopeptide enrichment")
    flow_rate_chromatogram: str = Field(default=None,
                                        description="LC flow rate.", examples="300 nL/min")
    fraction_identifier: str = Field(default=None,
                                     description="Numeric or text ID of each fraction.", examples="F1,F2")
    fractionation_method: str = Field(default=None,
                                      description="Any off-line method used to fraction bulk sample into the primary samples used in the MS or LC/MS.")
    fragment_mass_tolerance: str = Field(default=None,
                                         description="Mass tolerance for fragment matching.", examples="0.02 Da")
    fragmentation_method: str = Field(default=None,
                                      description="Ion‐fragmentation technique.", examples="HCD, CID, ETD")
    gradient_time: str = Field(default=None,
                               description="Total LC gradient length.", examples="120 min")
    instrument: str = Field(default=None,
                            description="Mass spec make/model.", examples="Thermo Q-Exactive Plus")
    ionization_type: str = Field(default=None,
                                 description="Ionization source.", examples="nanoESI, MALDI")
    ms2_mass_analyzer: str = Field(default=None,
                                   description="Analyzer used for MS2.", examples="orbitrap, ion trap")
    number_of_fractions: str = Field(default=None,
                                     description="Total number of fractions generated from each sample.")
    number_of_missed_cleavages: str = Field(default=None,
                                            description="Max missed cleavages allowed in database search.", examples="2")
    precursor_mass_tolerance: str = Field(default=None,
                                          description="Mass tolerance for precursor matching.", examples="10 ppm")
    separation: str = Field(default=None,
                            description="Any on-line method used to separate the samples into fractions right before MS.")

    fv_bait: str = Field(default="not applicable",
                      description="Experimental factor: The protein or molecule used as bait in an affinity‐purification experiment.")
    fv_cell_part: str =  Field(default="not applicable",
                           description="Subcellular compartment or fraction.",
                           examples="nucleus, mitochondria")
    fv_compound: str = Field(default="not applicable",
                          description="Chemical or small molecule added to the sample",
                           examples="drug, inhibitor")
    fv_concentration_of_compound: str =  Field(default="not applicable",
                                           description="Concentration of the Compound used",
                                           examples="10 µM")
    fv_disease: str = Field(default="not applicable",
                         description="Disease state or diagnosis.", 
                         examples="breast cancer, Type 2 diabetes")
    fv_fraction_identifier: str = Field(default="not applicable",
                                     description="Numeric or text ID of each fraction.", examples="F1,F2")
    fv_genetic_modification: str = Field(default="not applicable",
                                      description="Any genetic alteration in the source organism/cells",
                                       examples="GFP‐tagged, knockout of gene X")
    fv_temperature: str = Field(default="not applicable",
                             description="Growth temperature of the samples or perturbation temperature if a differential study."
                             )
    fv_treatment: str = Field(default="not applicable",
                           description="Experimental treatment.", examples="drug X 5 µM 24h")
                
    factors: str = Field(
        default="not applicable",
        description="Semicolon-delimited names of experimental variables compared in this study (e.g. 'treatment;time')."
    )
    usage: str = Field(default="raw",
                       description="", examples="")


class SDRFDocument(BaseModel):
    """Full extraction result for a paper."""
    rows: List[SDRFRow] = Field(description="One SDRF row represents one sample-data file relationship")
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Any caveats or assumptions made during extraction"
    )
