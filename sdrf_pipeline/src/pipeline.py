"""
SDRF Extraction Pipeline
Uses LangChain with any OpenAI-compatible API to extract proteomics
metadata from paper text and output a valid SDRF CSV.
"""
import os
import re
import csv
import json
import logging
from pathlib import Path
from typing import Optional
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.models import SDRFDocument, SDRFRow

logger = logging.getLogger(__name__)

# ── SDRF column header order (must match spec exactly) ────────────────────────
SDRF_HEADERS = [
    "SampleSource", "AssayName", "Raw Data File",
    "Characteristics[Age]", "Characteristics[AlkylationReagent]",
    "Characteristics[AnatomicSiteTumor]", "Characteristics[AncestryCategory]",
    "Characteristics[BMI]", "Characteristics[Bait]",
    "Characteristics[BiologicalReplicate]", "Characteristics[CellLine]",
    "Characteristics[CellPart]", "Characteristics[CellType]",
    "Characteristics[CleavageAgent]", "Characteristics[Compound]",
    "Characteristics[ConcentrationOfCompound]", "Characteristics[Depletion]",
    "Characteristics[DevelopmentalStage]", "Characteristics[DiseaseTreatment]",
    "Characteristics[Disease]", "Characteristics[GeneticModification]",
    "Characteristics[Genotype]", "Characteristics[GrowthRate]",
    "Characteristics[Label]", "Characteristics[MaterialType]",
    "Characteristics[Modification]", "Characteristics[Modification].1",
    "Characteristics[Modification].2", "Characteristics[Modification].3",
    "Characteristics[Modification].4", "Characteristics[Modification].5",
    "Characteristics[Modification].6",
    "Characteristics[NumberOfBiologicalReplicates]",
    "Characteristics[NumberOfSamples]",
    "Characteristics[NumberOfTechnicalReplicates]",
    "Characteristics[OrganismPart]", "Characteristics[Organism]",
    "Characteristics[OriginSiteDisease]", "Characteristics[PooledSample]",
    "Characteristics[ReductionReagent]", "Characteristics[SamplingTime]",
    "Characteristics[Sex]", "Characteristics[Specimen]",
    "Characteristics[SpikedCompound]", "Characteristics[Staining]",
    "Characteristics[Strain]", "Characteristics[SyntheticPeptide]",
    "Characteristics[Temperature]", "Characteristics[Time]",
    "Characteristics[Treatment]", "Characteristics[TumorCellularity]",
    "Characteristics[TumorGrade]", "Characteristics[TumorSite]",
    "Characteristics[TumorSize]", "Characteristics[TumorStage]",
    "Comment[AcquisitionMethod]", "Comment[CollisionEnergy]",
    "Comment[EnrichmentMethod]", "Comment[FlowRateChromatogram]",
    "Comment[FractionIdentifier]", "Comment[FractionationMethod]",
    "Comment[FragmentMassTolerance]", "Comment[FragmentationMethod]",
    "Comment[GradientTime]", "Comment[Instrument]",
    "Comment[IonizationType]", "Comment[MS2MassAnalyzer]",
    "Comment[NumberOfFractions]", "Comment[NumberOfMissedCleavages]",
    "Comment[PrecursorMassTolerance]", "Comment[Separation]",
    "FactorValue[Bait]", "FactorValue[CellPart]",
    "FactorValue[Compound]", "FactorValue[ConcentrationOfCompound].1",
    "FactorValue[Disease]", "FactorValue[FractionIdentifier]",
    "FactorValue[GeneticModification]", "FactorValue[Temperature]",
    "FactorValue[Treatment]", "Usage",
]

# Maps SDRF header → SDRFRow field name
HEADER_TO_FIELD = {
    "sample_source": "SampleSource",
    "AssayName": "assay_name",
    "Raw Data File": "raw_data_file",
    "Characteristics[Age]": "age",
    "Characteristics[AlkylationReagent]": "alkylation_reagent",
    "Characteristics[AnatomicSiteTumor]": "anatomic_site_tumor",
    "Characteristics[AncestryCategory]": "ancestry_category",
    "Characteristics[BMI]": "bmi",
    "Characteristics[Bait]": "bait",
    "Characteristics[BiologicalReplicate]": "biological_replicate",
    "Characteristics[CellLine]": "cell_line",
    "Characteristics[CellPart]": "cell_part",
    "Characteristics[CellType]": "cell_type",
    "Characteristics[CleavageAgent]": "cleavage_agent",
    "Characteristics[Compound]": "compound",
    "Characteristics[ConcentrationOfCompound]": "concentration_of_compound",
    "Characteristics[Depletion]": "depletion",
    "Characteristics[DevelopmentalStage]": "developmental_stage",
    "Characteristics[DiseaseTreatment]": "disease_treatment",
    "Characteristics[Disease]": "disease",
    "Characteristics[GeneticModification]": "genetic_modification",
    "Characteristics[Genotype]": "genotype",
    "Characteristics[GrowthRate]": "growth_rate",
    "Characteristics[Label]": "label",
    "Characteristics[MaterialType]": "material_type",
    "Characteristics[Modification]": "modification",
    "Characteristics[Modification].1": "modification_1",
    "Characteristics[Modification].2": "modification_2",
    "Characteristics[Modification].3": "modification_3",
    "Characteristics[Modification].4": "modification_4",
    "Characteristics[Modification].5": "modification_5",
    "Characteristics[Modification].6": "modification_6",
    "Characteristics[NumberOfBiologicalReplicates]": "number_of_biological_replicates",
    "Characteristics[NumberOfSamples]": "number_of_samples",
    "Characteristics[NumberOfTechnicalReplicates]": "number_of_technical_replicates",
    "Characteristics[OrganismPart]": "organism_part",
    "Characteristics[Organism]": "organism",
    "Characteristics[OriginSiteDisease]": "origin_site_disease",
    "Characteristics[PooledSample]": "pooled_sample",
    "Characteristics[ReductionReagent]": "reduction_reagent",
    "Characteristics[SamplingTime]": "sampling_time",
    "Characteristics[Sex]": "sex",
    "Characteristics[Specimen]": "specimen",
    "Characteristics[SpikedCompound]": "spiked_compound",
    "Characteristics[Staining]": "staining",
    "Characteristics[Strain]": "strain",
    "Characteristics[SyntheticPeptide]": "synthetic_peptide",
    "Characteristics[Temperature]": "temperature",
    "Characteristics[Time]": "time",
    "Characteristics[Treatment]": "treatment",
    "Characteristics[TumorCellularity]": "tumor_cellularity",
    "Characteristics[TumorGrade]": "tumor_grade",
    "Characteristics[TumorSite]": "tumor_site",
    "Characteristics[TumorSize]": "tumor_size",
    "Characteristics[TumorStage]": "tumor_stage",
    "Comment[AcquisitionMethod]": "acquisition_method",
    "Comment[CollisionEnergy]": "collision_energy",
    "Comment[EnrichmentMethod]": "enrichment_method",
    "Comment[FlowRateChromatogram]": "flow_rate_chromatogram",
    "Comment[FractionIdentifier]": "fraction_identifier",
    "Comment[FractionationMethod]": "fractionation_method",
    "Comment[FragmentMassTolerance]": "fragment_mass_tolerance",
    "Comment[FragmentationMethod]": "fragmentation_method",
    "Comment[GradientTime]": "gradient_time",
    "Comment[Instrument]": "instrument",
    "Comment[IonizationType]": "ionization_type",
    "Comment[MS2MassAnalyzer]": "ms2_mass_analyzer",
    "Comment[NumberOfFractions]": "number_of_fractions",
    "Comment[NumberOfMissedCleavages]": "number_of_missed_cleavages",
    "Comment[PrecursorMassTolerance]": "precursor_mass_tolerance",
    "Comment[Separation]": "separation",
    "FactorValue[Bait]": "fv_bait",
    "FactorValue[CellPart]": "fv_cell_part",
    "FactorValue[Compound]": "fv_compound",
    "FactorValue[ConcentrationOfCompound].1": "fv_concentration_of_compound",
    "FactorValue[Disease]": "fv_disease",
    "FactorValue[FractionIdentifier]": "fv_fraction_identifier",
    "FactorValue[GeneticModification]": "fv_genetic_modification",
    "FactorValue[Temperature]": "fv_temperature",
    "FactorValue[Treatment]": "fv_treatment",
    "Usage": "usage",
}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Read the provided proteomics publication and list of mass spectrometry .raw filenames and compile metadata for metadata in the SDRF (Sample and Data Relationship Format) for the proteomics experiment.
1) experiment-level metadata shared across ALL samples.
2) sample level metadata for each biological object or condition

SDRF STRUCTURE (definitions)
1. One SDRF row represents one sample–data file relationship.
2. For each row, populate every field you can find or INFER evidence for in the text.
3. Use "Not Applicable" for fields with no relevant data (e.g. tumor fields for non-cancer studies).
4. Biological replicate: infer from replicate numbering in filenames or text (e.g. _rep1, -1-, triplicate).
5. Modifications: list fixed mods first, then variable mods across Modification columns.
6. Factor values should reflect the PRIMARY experimental variable(s) that differ between samples.

FILENAME-AWARE MAPPING (RAW_FILES)
You MUST use RAW_FILES to help assign metadata to each sample/file:
   - fractions (e.g., “F1”, “Frac03”, “fraction7”) -> fraction_identifier
   - technical replicate markers (e.g., “TR1”, “TechRep2”, “inj3”) -> technical_replicate
   - biological replicate markers (e.g., “BR1”, “BioRep3”, “rep2”) -> biological_replicate ONLY if manuscript supports that token meaning
   - timepoints (e.g., “0h”, “24h”, “day5”) -> FactorValue[Time] or time depending on study-variable rule above
   - condition markers (e.g., “CTRL”, “WT”, “KO”, “drugX”) -> treatment ONLY if token is self-descriptive AND matches manuscript terminology
   - labels (e.g., “TMT126”, “TMT-131N”, “SILAC”) -> label
Use filename cues to DISAMBIGUATE which samples get which conditions when the manuscript describes multiple conditions.

Return exactly one JSON object.
Return ONLY valid JSON matching the schema. No markdown, no explanation outside the JSON."""

HUMAN_PROMPT = """Extract SDRF metadata from the following paper text.

PAPER TEXT:
{paper_text}

Return JSON with this exact structure:
{{
  "rows": [
    {{
      "source_name": "<sample identifier>",
      "assay_name": "<assay identifier>",
      "raw_data_file": "<filename>",
      "age": "<value or not applicable>",
      "alkylation_reagent": "<value or not applicable>",
      "anatomic_site_tumor": "<value or not applicable>",
      "ancestry_category": "<value or not applicable>",
      "bmi": "<value or not applicable>",
      "bait": "<value or not applicable>",
      "biological_replicate": "<1|2|3|...>",
      "cell_line": "<value or not applicable>",
      "cell_part": "<value or not applicable>",
      "cell_type": "<value or not applicable>",
      "cleavage_agent": "<value>",
      "compound": "<value or not applicable>",
      "concentration_of_compound": "<value or not applicable>",
      "depletion": "<value or not applicable>",
      "developmental_stage": ""<value or not applicable>",
      "disease_treatment": "<value or not applicable>",
      "disease": "<value or not applicable>",
      "genetic_modification": "<value or not applicable>",
      "genotype": "<value or not applicable>",
      "growth_rate": "<value or not applicable>",
      "label": "<label free sample | TMT | iTRAQ | SILAC | ...>",
      "material_type": "<tissue | cell culture | plasma | ...>",
      "modification": "<first mod>",
      "modification_1": "<second mod or not applicable>",
      "modification_2": "<third mod or not applicable>",
      "modification_3": "<fourth mod or not applicable>",
      "modification_4": "<value or not applicable>",
      "modification_5": "<value or not applicable>",
      "modification_6": "<value or not applicable>",
      "number_of_biological_replicates": "<number>",
      "number_of_samples": "<number>",
      "number_of_technical_replicates": "<number>",
      "organism_part": "<tissue/organ>",
      "organism": "<species binomial>",
      "origin_site_disease": "<value or not applicable>",
      "pooled_sample": "<value or not applicable>",
      "reduction_reagent": "<value or not applicable>",
      "sampling_time": "<value or not applicable>",
      "sex": "<value or not applicable>",
      "specimen": "<value>",
      "spiked_compound": "<value or not applicable>",
      "staining": "<value or not applicable>",
      "strain": "<value or not applicable>",
      "synthetic_peptide": "<value or not applicable>",
      "temperature": "<numeric or not applicable>",
      "time": "<value or not applicable>",
      "treatment": "<value or none>",
      "tumor_cellularity": "<value or not applicable>",
      "tumor_grade": "<value or not applicable>",
      "tumor_site": "<value or not applicable>",
      "tumor_size": "<value or not applicable>",
      "tumor_stage": "<value or not applicable>",
      "acquisition_method": "<data-dependent acquisition | data-independent acquisition | ...>",
      "collision_energy": "<value or not applicable>",
      "enrichment_method": "<value or not applicable>",
      "flow_rate_chromatogram": "<value with units>",
      "fraction_identifier": "<number or not applicable>",
      "fractionation_method": "<value or not applicable>",
      "fragment_mass_tolerance": "<value with units>",
      "fragmentation_method": "<CID | HCD | ETD | ...>",
      "gradient_time": "<value with units>",
      "instrument": "<instrument name>",
      "ionization_type": "<ESI | MALDI | ...>",
      "ms2_mass_analyzer": "<Orbitrap | linear ion trap | ...>",
      "number_of_fractions": "<number or not applicable>",
      "number_of_missed_cleavages": "<number>",
      "precursor_mass_tolerance": "<value with units>",
      "separation": "<reverse phase | HILIC | ...>",
      "fv_bait": "<value or not applicable>",
      "fv_cell_part": "<value or not applicable>",
      "fv_compound": "<value or not applicable>",
      "fv_concentration_of_compound": "<value or not applicable>",
      "fv_disease": "<value or not applicable>",
      "fv_fraction_identifier": "<value or not applicable>",
      "fv_genetic_modification": "<value or not applicable>",
      "fv_temperature": "<numeric value if temperature is the factor, else not applicable>",
      "fv_treatment": "<treatment name if treatment is the factor, else not applicable>",
      "usage": "raw"
    }}
  ],
  "extraction_notes": "<optional notes about assumptions or gaps>"
}}"""


class SDRFPipeline:
    """
    Extracts SDRF metadata from proteomics paper text using an
    OpenAI-compatible LLM and writes a valid SDRF CSV.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,          # None = OpenAI default; set for local/proxy APIs
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def extract(self, paper_text: str) -> SDRFDocument:
        """Run extraction and return a validated SDRFDocument."""
        logger.info("Sending paper text to LLM (%d chars)…", len(paper_text))
        raw = self.chain.invoke({"paper_text": paper_text})

        # Validate through Pydantic
        doc = SDRFDocument(**raw)
        logger.info("Extracted %d row(s). Notes: %s", len(doc.rows), doc.extraction_notes)
        return doc

    def to_csv(self, doc: SDRFDocument, output_path: str | Path) -> Path:
        """Write SDRFDocument to a CSV file in the standard column order."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SDRF_HEADERS, extrasaction="ignore")
            writer.writeheader()
            for row in doc.rows:
                row_dict = row.model_dump()
                # Map field names → SDRF header names
                csv_row = {
                    header: row_dict.get(field, "not applicable")
                    for header, field in HEADER_TO_FIELD.items()
                }
                writer.writerow(csv_row)

        logger.info("SDRF written to %s", output_path)
        return output_path

    def process_file(self, input_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
        """Read a text file, extract metadata, write CSV. Returns output path."""
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".sdrf.csv")

        paper_text = input_path.read_text(encoding="utf-8")
        doc = self.extract(paper_text)
        return self.to_csv(doc, output_path)

    def prepare_text_for_submission(self, json_path, max_length=None):
        """
        Extracts TITLE, METHODS, and ABSTRACT from a JSON file and concatenates them.
        Optionally truncates the text to max_length characters.
        
        Args:
            json_path (str): Path to the JSON file.
            max_length (int, optional): Maximum length of the output text.
            
        Returns:
            str: Concatenated text.
        """
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract fields safely
        title = data.get("TITLE", "").strip()
        methods = data.get("METHODS", "").strip()
        abstract = data.get("ABSTRACT", "").strip()
        
        # Combine fields
        text = f"Title: {title}\n\nAbstract: {abstract}\n\nMethods: {methods}"
        
        # Apply length limit if specified
        if max_length is not None and len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'  # avoid cutting in the middle of a word
        
        return text

    # Example usage:
    # text_to_submit = prepare_text_for_submission("path/to/file.json", max_length=4000)
    # print(text_to_submit)

    def process_batch(self, input_dir: str | Path, output_dir: str | Path) -> list[Path]:
        """Process all .txt files in a directory. Returns list of output paths."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        txt_files = list(input_dir.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found in %s", input_dir)
            return results

        for i, txt_file in enumerate(txt_files, 1):
            logger.info("[%d/%d] Processing %s", i, len(txt_files), txt_file.name)
            try:
                out = self.process_file(
                    txt_file,
                    output_dir / txt_file.with_suffix(".sdrf.csv").name,
                )
                results.append(out)
            except Exception as e:
                logger.error("Failed on %s: %s", txt_file.name, e)

        return results
