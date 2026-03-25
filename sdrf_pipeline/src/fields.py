"""
SDRF Field Registry
====================
Single source of truth for every SDRF field:
  - canonical header name
  - Pydantic model attribute name
  - description of what to look for in the paper
  - canonical example values
  - regex hints for pre-extraction
  - whether it is a "high-value" field (almost always non-empty in proteomics)

The pipeline builds the extraction prompt directly from this registry,
so updating a field description here updates the prompt automatically.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SDRFField:
    header: str                          # exact SDRF CSV column name
    attr: str                            # Pydantic model attribute name
    description: str                     # what to look for in text
    examples: list[str]                  # canonical values
    regex: Optional[str] = None          # optional pre-extraction pattern
    high_value: bool = False             # True = almost always non-empty
    is_factor: bool = False              # True = experimental variable


# ── Registry ──────────────────────────────────────────────────────────────────
# Fields are listed in SDRF column order.
FIELDS: list[SDRFField] = [

    # ── Sample Characteristics ─────────────────────────────────────────────

    SDRFField(
        header="Characteristics[Age]",
        attr="age",
        description="Age of donor/subject at time of sampling. Look for: years, months, weeks, 'adult', 'neonatal', age ranges.",
        examples=["45 years", "8 weeks", "adult", "not applicable"],
        regex=r'\b(\d+)\s*(year|month|week|day)s?\s*old\b',
    ),
    SDRFField(
        header="Characteristics[AlkylationReagent]",
        attr="alkylation_reagent",
        description="Reagent used to alkylate cysteines. Look for: iodoacetamide, IAA, IAM, chloroacetamide, CAA, N-ethylmaleimide, NEM. Usually in sample prep / Methods.",
        examples=["Iodoacetamide", "Chloroacetamide", "not applicable"],
        regex=r'\b(iodoacetamide|IAA\b|IAM\b|chloroacetamide|CAA\b|N-ethylmaleimide|\bNEM\b)',
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[AnatomicSiteTumor]",
        attr="anatomic_site_tumor",
        description="Primary anatomic tumor site. Only relevant for cancer studies.",
        examples=["breast", "lung", "colon", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[AncestryCategory]",
        attr="ancestry_category",
        description="Genetic ancestry / ethnicity of human subjects if reported.",
        examples=["European", "East Asian", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[BMI]",
        attr="bmi",
        description="Body mass index of human subjects if reported.",
        examples=["22.5", "obese", "not applicable"],
        regex=r'\bBMI\s*[=:of]?\s*(\d+\.?\d*)',
    ),
    SDRFField(
        header="Characteristics[Bait]",
        attr="bait",
        description="Bait protein or molecule used in pulldown / AP-MS experiments.",
        examples=["FLAG-EGFP", "HA-BRCA1", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[BiologicalReplicate]",
        attr="biological_replicate",
        description="Biological replicate number for this sample. Infer from: replicate numbering in filenames (_rep1, -1-, _R1), 'triplicate', 'duplicate', or explicit replicate counts in Methods. Assign 1/2/3/… per condition group.",
        examples=["1", "2", "3"],
        regex=r'[_\-](rep)?(\d)[_\-\.]',
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[CellLine]",
        attr="cell_line",
        description="Name of cell line used. Look for: HeLa, HEK293, Jurkat, U2OS, MCF-7, A549, etc.",
        examples=["HeLa", "HEK293T", "Jurkat", "not applicable"],
        regex=r'\b(HeLa|HEK293|Jurkat|U2OS|MCF-7|A549|K562|Huh7|CaCo-2|PC-3|LNCaP)\b',
    ),
    SDRFField(
        header="Characteristics[CellPart]",
        attr="cell_part",
        description="Subcellular fraction or compartment. Look for: nucleus, cytoplasm, mitochondria, membrane, chromatin, cytosol.",
        examples=["nucleus", "cytoplasm", "mitochondria", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[CellType]",
        attr="cell_type",
        description="Primary cell type or sorted population. Look for: T cell, B cell, monocyte, neuron, hepatocyte, PBMC.",
        examples=["T cell", "PBMC", "hepatocyte", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[CleavageAgent]",
        attr="cleavage_agent",
        description="Protease(s) used for digestion. Look for: trypsin, LysC, GluC, chymotrypsin, Asp-N, LysN, elastase. Almost always stated in Methods.",
        examples=["Trypsin", "LysC", "Trypsin/LysC", "GluC"],
        regex=r'\b(trypsin|Lys-?C|Glu-?C|chymotrypsin|Asp-?N|Lys-?N|elastase)\b',
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[Compound]",
        attr="compound",
        description="Drug, chemical, or stimulus applied to cells/organism. Look for: drug names, inhibitors, cytokines, hormones.",
        examples=["rapamycin", "EGF", "doxorubicin", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[ConcentrationOfCompound]",
        attr="concentration_of_compound",
        description="Concentration of the compound/drug applied. Include units.",
        examples=["10 nM", "1 µM", "100 ng/mL", "not applicable"],
        regex=r'(\d+\.?\d*)\s*(nM|µM|uM|mM|ng/mL|µg/mL)',
    ),
    SDRFField(
        header="Characteristics[Depletion]",
        attr="depletion",
        description="Whether abundant proteins were depleted (e.g. albumin/IgG depletion from plasma/serum).",
        examples=["MARS14 depletion", "IgG depletion", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[DevelopmentalStage]",
        attr="developmental_stage",
        description="Developmental stage of organism/tissue.",
        examples=["embryonic", "neonatal", "adult", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[DiseaseTreatment]",
        attr="disease_treatment",
        description="Medical treatment the patient/subject received.",
        examples=["chemotherapy", "untreated", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Disease]",
        attr="disease",
        description="Disease state of sample. Use 'normal' or 'healthy' for controls. Look for disease names, cancer types, condition names.",
        examples=["breast cancer", "Alzheimer's disease", "normal", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[GeneticModification]",
        attr="genetic_modification",
        description="Genetic modification introduced: knockout, knockdown, overexpression, CRISPR edit.",
        examples=["BRCA1 knockout", "TP53 siRNA knockdown", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Genotype]",
        attr="genotype",
        description="Genotype of organism or strain, particularly for model organisms.",
        examples=["wild-type", "p53-/-", "APP/PS1", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[GrowthRate]",
        attr="growth_rate",
        description="Cell growth rate or doubling time if reported.",
        examples=["exponential phase", "24 h doubling time", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Label]",
        attr="label",
        description="Quantification labeling strategy. Look for: 'label-free', 'TMT', 'iTRAQ', 'SILAC', 'dimethyl labeling', 'tandem mass tag'. Default to 'label free sample' if no labeling is described.",
        examples=["label free sample", "TMT10", "iTRAQ4", "SILAC", "dimethyl"],
        regex=r'\b(label.free|TMT|iTRAQ|SILAC|dimethyl|tandem mass tag)\b',
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[MaterialType]",
        attr="material_type",
        description="Type of biological material. Look for: tissue, cell culture, plasma, serum, urine, CSF, milk, cell line, primary cells.",
        examples=["tissue", "cell culture", "plasma", "serum", "milk", "urine"],
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[Modification]",
        attr="modification",
        description="First/primary PTM or chemical modification. Fixed modifications first (e.g. carbamidomethylation). Look for: 'fixed modification', 'variable modification', modification names in Methods/search parameters.",
        examples=["Carbamidomethyl", "TMT6plex", "not applicable"],
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[Modification].1",
        attr="modification_1",
        description="Second modification (variable mods). Look for: oxidation of methionine, phosphorylation, acetylation.",
        examples=["Oxidation", "Phospho", "not applicable"],
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[Modification].2",
        attr="modification_2",
        description="Third modification. Common: N-terminal acetylation, deamidation.",
        examples=["Acetyl (N-term)", "Deamidation", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Modification].3",
        attr="modification_3",
        description="Fourth modification if present.",
        examples=["Deamidation (NQ)", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Modification].4",
        attr="modification_4",
        description="Fifth modification if present.",
        examples=["not applicable"],
    ),
    SDRFField(
        header="Characteristics[Modification].5",
        attr="modification_5",
        description="Sixth modification if present.",
        examples=["not applicable"],
    ),
    SDRFField(
        header="Characteristics[Modification].6",
        attr="modification_6",
        description="Seventh modification if present.",
        examples=["not applicable"],
    ),
    SDRFField(
        header="Characteristics[NumberOfBiologicalReplicates]",
        attr="number_of_biological_replicates",
        description="Total number of biological replicates per condition. Look for: 'triplicate' (3), 'duplicate' (2), 'n=3', 'three independent'. Count unique conditions × replicates.",
        examples=["3", "2", "4"],
        regex=r'\b(triplicate|duplicate|n\s*=\s*(\d+)|(\d+)\s*(biological\s*)?replicates?)\b',
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[NumberOfSamples]",
        attr="number_of_samples",
        description="Total number of MS runs / raw files in the dataset.",
        examples=["18", "24", "6"],
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[NumberOfTechnicalReplicates]",
        attr="number_of_technical_replicates",
        description="Number of technical replicates per sample. Often 1 (no technical replication). Look for: 'injected in duplicate', 'technical replicate'.",
        examples=["1", "2", "3"],
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[OrganismPart]",
        attr="organism_part",
        description="Tissue, organ, or biofluid of origin. Look for anatomical terms: liver, brain, kidney, plasma, serum, urine, milk, whole blood, muscle, lung.",
        examples=["liver", "plasma", "brain", "milk", "urine", "whole blood"],
        regex=r'\b(liver|brain|kidney|plasma|serum|urine|milk|blood|muscle|lung|heart|spleen|colon|skin)\b',
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[Organism]",
        attr="organism",
        description="Latin binomial species name. Map common names: human→Homo sapiens, mouse→Mus musculus, rat→Rattus norvegicus, bovine/cow→Bos taurus, yeast→Saccharomyces cerevisiae, fly→Drosophila melanogaster.",
        examples=["Homo sapiens", "Mus musculus", "Rattus norvegicus", "Bos taurus"],
        regex=r'\b(human|mouse|rat|bovine|yeast|E\.?\s*coli|zebrafish|fly|Arabidopsis)\b',
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[OriginSiteDisease]",
        attr="origin_site_disease",
        description="Tissue of disease origin (for metastatic/cancer samples).",
        examples=["primary tumor", "metastasis", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[PooledSample]",
        attr="pooled_sample",
        description="Whether samples were pooled before analysis.",
        examples=["yes", "no", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[ReductionReagent]",
        attr="reduction_reagent",
        description="Reagent used to reduce disulfide bonds. Look for: DTT, dithiothreitol, TCEP, beta-mercaptoethanol, MMTS.",
        examples=["Dithiothreitol", "TCEP", "not applicable"],
        regex=r'\b(DTT|dithiothreitol|TCEP|beta-mercaptoethanol|MMTS)\b',
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[SamplingTime]",
        attr="sampling_time",
        description="Time point at which sample was collected in a time-course experiment.",
        examples=["0 h", "24 h", "day 7", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Sex]",
        attr="sex",
        description="Biological sex of donor/animal. Look for: male, female, mixed.",
        examples=["male", "female", "mixed", "not applicable"],
        regex=r'\b(male|female)\b',
    ),
    SDRFField(
        header="Characteristics[Specimen]",
        attr="specimen",
        description="Type of specimen more specifically than organism part. E.g. FFPE tissue, fresh-frozen biopsy, cell pellet.",
        examples=["FFPE", "fresh-frozen", "cell pellet", "raw milk whey"],
        high_value=True,
    ),
    SDRFField(
        header="Characteristics[SpikedCompound]",
        attr="spiked_compound",
        description="Any standard spiked into sample for calibration (e.g. iRT peptides, UPS2 standard).",
        examples=["iRT peptides", "UPS2", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Staining]",
        attr="staining",
        description="Staining method used (relevant for tissue or gel-based proteomics).",
        examples=["Coomassie", "silver stain", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Strain]",
        attr="strain",
        description="Strain of model organism or microbial isolate.",
        examples=["C57BL/6", "BALB/c", "BY4742", "not applicable"],
        regex=r'\b(C57BL/6|BALB/c|129/Sv|Sprague.Dawley|Wistar|BY4742)\b',
    ),
    SDRFField(
        header="Characteristics[SyntheticPeptide]",
        attr="synthetic_peptide",
        description="Whether the sample contains synthetic peptides (targeted proteomics standards).",
        examples=["yes", "no", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[Temperature]",
        attr="temperature",
        description="Temperature of experimental condition (e.g. heat treatment, incubation). Extract numeric value only without units. For raw/untreated controls use 'not applicable'.",
        examples=["37", "65", "80", "not applicable"],
        regex=r'(\d+)\s*°C',
        high_value=False,
        is_factor=True,
    ),
    SDRFField(
        header="Characteristics[Time]",
        attr="time",
        description="Duration of treatment or incubation. Include units. Look for: min, h, hours, days after treatment/incubation verbs.",
        examples=["30 min", "24 h", "72 h", "not applicable"],
        regex=r'(\d+\.?\d*)\s*(min|minutes?|h\b|hours?|days?)',
    ),
    SDRFField(
        header="Characteristics[Treatment]",
        attr="treatment",
        description="Experimental treatment applied to sample. 'none' for untreated controls. Look for: heat treatment, drug treatment, stimulation, infection.",
        examples=["heat treatment", "none", "EGF stimulation", "doxorubicin treatment"],
        high_value=True,
        is_factor=True,
    ),
    SDRFField(
        header="Characteristics[TumorCellularity]",
        attr="tumor_cellularity",
        description="Percentage of tumor cells in tissue sample.",
        examples=["70%", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[TumorGrade]",
        attr="tumor_grade",
        description="Histological tumor grade.",
        examples=["Grade II", "G3", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[TumorSite]",
        attr="tumor_site",
        description="Specific anatomic location of tumor.",
        examples=["right breast", "sigmoid colon", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[TumorSize]",
        attr="tumor_size",
        description="Measured tumor size.",
        examples=["2.3 cm", "T2", "not applicable"],
    ),
    SDRFField(
        header="Characteristics[TumorStage]",
        attr="tumor_stage",
        description="Clinical tumor stage (TNM or Roman numeral).",
        examples=["Stage II", "T2N1M0", "not applicable"],
    ),

    # ── Comments (instrument / acquisition) ───────────────────────────────

    SDRFField(
        header="Comment[AcquisitionMethod]",
        attr="acquisition_method",
        description="MS acquisition strategy. Look for: 'data-dependent acquisition', 'DDA', 'data-independent acquisition', 'DIA', 'PRM', 'SRM', 'MRM', 'targeted'.",
        examples=["data-dependent acquisition", "data-independent acquisition", "PRM"],
        regex=r'\b(DDA|DIA|data.dependent|data.independent|PRM|SRM|MRM)\b',
        high_value=True,
    ),
    SDRFField(
        header="Comment[CollisionEnergy]",
        attr="collision_energy",
        description="Collision energy used for fragmentation, if explicitly stated.",
        examples=["28 NCE", "35%", "not applicable"],
    ),
    SDRFField(
        header="Comment[EnrichmentMethod]",
        attr="enrichment_method",
        description="Enrichment method for PTMs or specific protein classes. Look for: IMAC, TiO2, antibody enrichment, phospho-enrichment.",
        examples=["IMAC", "TiO2", "anti-phosphotyrosine", "not applicable"],
    ),
    SDRFField(
        header="Comment[FlowRateChromatogram]",
        attr="flow_rate_chromatogram",
        description="LC flow rate. Look for: µL/min, nL/min values near column/chromatography descriptions.",
        examples=["300 nL/min", "0.5 µL/min", "200 nL/min"],
        regex=r'(\d+\.?\d*)\s*(nL|µL|uL)/min',
        high_value=True,
    ),
    SDRFField(
        header="Comment[FractionIdentifier]",
        attr="fraction_identifier",
        description="Fraction number for pre-fractionated samples (SCX, HpRP). If no fractionation, use 1.",
        examples=["1", "2", "12", "not applicable"],
        high_value=True,
    ),
    SDRFField(
        header="Comment[FractionationMethod]",
        attr="fractionation_method",
        description="Pre-fractionation method used before LC-MS. Look for: SCX, HpRP, IEF, gel slices, OFFGEL.",
        examples=["SCX", "HpRP", "gel-based", "not applicable"],
    ),
    SDRFField(
        header="Comment[FragmentMassTolerance]",
        attr="fragment_mass_tolerance",
        description="MS2 mass tolerance used in database search. Include units (Da or ppm). Look for in search parameters / Methods.",
        examples=["0.5 Da", "20 ppm", "0.02 Da"],
        regex=r'(\d+\.?\d*)\s*(Da|ppm)\s*(fragment|MS2|MSMS)',
        high_value=True,
    ),
    SDRFField(
        header="Comment[FragmentationMethod]",
        attr="fragmentation_method",
        description="Ion fragmentation method. Look for: CID, HCD, ETD, ECD, EThcD, UVPD.",
        examples=["HCD", "CID", "ETD", "EThcD"],
        regex=r'\b(HCD|CID|ETD|ECD|EThcD|UVPD)\b',
        high_value=True,
    ),
    SDRFField(
        header="Comment[GradientTime]",
        attr="gradient_time",
        description="Total LC gradient duration. Include units. Look for: 'gradient', 'elution' followed by time.",
        examples=["90 min", "120 min", "50 min"],
        regex=r'(\d+)\s*min\s*(gradient|elution|LC)',
        high_value=True,
    ),
    SDRFField(
        header="Comment[Instrument]",
        attr="instrument",
        description="Mass spectrometer model. Look for: Q Exactive, Orbitrap Exploris, Fusion, Lumos, timsTOF, TripleTOF, QTOF, LTQ-Orbitrap. Use full model name.",
        examples=["Q Exactive HF", "Orbitrap Exploris 480", "timsTOF Pro", "LTQ-Orbitrap XL"],
        regex=r'(Q\s?Exactive[\s\w]*|Orbitrap[\s\w]+|Fusion[\s\w]*|timsTOF[\s\w]*|TripleTOF[\s\w]*|LTQ[\-\s]\w+)',
        high_value=True,
    ),
    SDRFField(
        header="Comment[IonizationType]",
        attr="ionization_type",
        description="Ion source type. Almost always ESI for LC-MS. Look for: ESI, nano-ESI, nanoESI, MALDI.",
        examples=["ESI", "nano-ESI", "MALDI"],
        regex=r'\b(ESI|nano.?ESI|MALDI|APCI)\b',
        high_value=True,
    ),
    SDRFField(
        header="Comment[MS2MassAnalyzer]",
        attr="ms2_mass_analyzer",
        description="Mass analyzer used for MS2 scans. Look for: Orbitrap, linear ion trap, TOF, quadrupole. Often distinct from MS1 analyzer.",
        examples=["Orbitrap", "linear ion trap", "TOF"],
        regex=r'(Orbitrap|linear\s+ion\s+trap|TOF|quadrupole)',
        high_value=True,
    ),
    SDRFField(
        header="Comment[NumberOfFractions]",
        attr="number_of_fractions",
        description="Number of fractions collected from pre-fractionation. 1 if no fractionation.",
        examples=["1", "12", "24", "not applicable"],
        high_value=True,
    ),
    SDRFField(
        header="Comment[NumberOfMissedCleavages]",
        attr="number_of_missed_cleavages",
        description="Maximum missed cleavages allowed in database search. Almost always stated in Methods.",
        examples=["2", "1", "0"],
        regex=r'(\d+)\s*missed\s*cleav',
        high_value=True,
    ),
    SDRFField(
        header="Comment[PrecursorMassTolerance]",
        attr="precursor_mass_tolerance",
        description="MS1 precursor mass tolerance for database search. Include units. Look for ppm values near 'first search', 'main search', 'precursor'.",
        examples=["10 ppm", "20 ppm", "4.5 ppm"],
        regex=r'(\d+\.?\d*)\s*ppm\s*(precursor|MS1|first|main)',
        high_value=True,
    ),
    SDRFField(
        header="Comment[Separation]",
        attr="separation",
        description="Chromatographic separation type. C18, C4, C8 columns → 'reverse phase'. Look for: RPLC, RP-HPLC, reverse phase, HILIC, SCX.",
        examples=["reverse phase", "HILIC", "SCX"],
        regex=r'(reverse\s+phase|RPLC|RP-?HPLC|C18|C8|HILIC|SCX)',
        high_value=True,
    ),

    # ── Factor Values ──────────────────────────────────────────────────────

    SDRFField(
        header="FactorValue[Bait]",
        attr="fv_bait",
        description="Factor value: bait protein (for AP-MS studies).",
        examples=["FLAG-EGFP", "not applicable"],
        is_factor=True,
    ),
    SDRFField(
        header="FactorValue[CellPart]",
        attr="fv_cell_part",
        description="Factor value: subcellular fraction if it is the experimental variable.",
        examples=["nucleus", "cytoplasm", "not applicable"],
        is_factor=True,
    ),
    SDRFField(
        header="FactorValue[Compound]",
        attr="fv_compound",
        description="Factor value: compound/drug if it is the experimental variable.",
        examples=["rapamycin", "EGF", "not applicable"],
        is_factor=True,
    ),
    SDRFField(
        header="FactorValue[ConcentrationOfCompound].1",
        attr="fv_concentration_of_compound",
        description="Factor value: compound concentration if it is the experimental variable.",
        examples=["10 nM", "1 µM", "not applicable"],
        is_factor=True,
    ),
    SDRFField(
        header="FactorValue[Disease]",
        attr="fv_disease",
        description="Factor value: disease state if it is the experimental variable (e.g. cancer vs normal).",
        examples=["breast cancer", "normal", "not applicable"],
        is_factor=True,
    ),
    SDRFField(
        header="FactorValue[FractionIdentifier]",
        attr="fv_fraction_identifier",
        description="Factor value: fraction number if fractionation is the variable.",
        examples=["1", "2", "not applicable"],
        is_factor=True,
    ),
    SDRFField(
        header="FactorValue[GeneticModification]",
        attr="fv_genetic_modification",
        description="Factor value: genetic modification if it is the experimental variable.",
        examples=["BRCA1 knockout", "not applicable"],
        is_factor=True,
    ),
    SDRFField(
        header="FactorValue[Temperature]",
        attr="fv_temperature",
        description="Factor value: temperature if heat/cold treatment is the primary experimental variable. Copy numeric value from Characteristics[Temperature]. Use 'not applicable' for untreated controls.",
        examples=["65", "70", "80", "not applicable"],
        is_factor=True,
    ),
    SDRFField(
        header="FactorValue[Treatment]",
        attr="fv_treatment",
        description="Factor value: treatment if treatment type is the primary experimental variable.",
        examples=["heat treatment", "none", "not applicable"],
        is_factor=True,
    ),
]

# ── Convenience accessors ─────────────────────────────────────────────────────

FIELD_BY_HEADER: dict[str, SDRFField] = {f.header: f for f in FIELDS}
FIELD_BY_ATTR: dict[str, SDRFField] = {f.attr: f for f in FIELDS}
HIGH_VALUE_ATTRS: set[str] = {f.attr for f in FIELDS if f.high_value}


def build_prompt_field_guide() -> str:
    """
    Render a compact field-by-field extraction guide from the registry.
    Injected directly into the LLM system prompt.
    """
    lines = [
        "FIELD-BY-FIELD EXTRACTION GUIDE",
        "================================",
        "For EVERY field below, actively search the paper before writing 'not applicable'.",
        "",
    ]
    for f in FIELDS:
        ex = " | ".join(f.examples[:3])
        lines.append(f"► {f.header}")
        lines.append(f"  What to look for: {f.description}")
        lines.append(f"  Example values:   {ex}")
        lines.append("")
    return "\n".join(lines)


def build_regex_hints(text: str) -> str:
    """
    Run all registry regexes against the paper text and return a
    formatted hint block to prepend to the paper when calling the LLM.
    Only includes fields where something was actually found.
    """
    found: list[str] = []
    for f in FIELDS:
        if not f.regex:
            continue
        matches = re.findall(f.regex, text, re.IGNORECASE)
        if matches:
            # Flatten tuple groups if any
            flat = []
            for m in matches:
                if isinstance(m, tuple):
                    flat.extend(v for v in m if v)
                else:
                    flat.append(m)
            unique = list(dict.fromkeys(flat))[:5]  # deduplicate, cap at 5
            found.append(f"  {f.header}: {', '.join(unique)}")

    if not found:
        return ""

    lines = [
        "━━━ AUTO-EXTRACTED HINTS (regex pre-scan) ━━━",
        "These values were found in the paper text by pattern matching.",
        "Use them to fill the corresponding SDRF fields:",
        *found,
        "━━━ END HINTS ━━━",
        "",
    ]
    return "\n".join(lines)
