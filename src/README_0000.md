Proteomics SDRF Metadata Extractor

This module automates the generation of SDRF (Sample and Data Relationship Format) documents. It is designed to navigate the "Proteomics Naming Chaos"—reconciling inconsistent file naming, experimental re-runs, and complex multiplexing (TMT) into a structured format for public repositories.

🚀 Core Philosophy: Dataset-Aware Extraction
Unlike a simple text scraper, this tool uses Dataset-Aware Logic. It compares the entire file list to understand the experimental design before generating a single row.

1. The Grouping Engine (get_canonical_root)
To prevent "Metadata Bloat" (where every file becomes a unique sample), the extractor collapses raw files into Logical Sample Groups:

Technical Re-runs: Automatically detects "retry" or "timestamped" files (e.g., data_1.raw and data_1_20220530.raw) and merges them into one biological identity.

Fractionation: Identifies files that are parts of a whole (e.g., _F01 through _F24) using the rule_is_fractionated logic.

DIA vs. DDA Filtering: If a dataset contains both DIA (experimental samples) and DDA (spectral library) files, it prioritizes DIA files to prevent library bloat in the SDRF.

2. Intelligent Replicate Detection
The rule_biological_replicate_from_filename uses Frequency Analysis to find the truth:

Stable Numbers (Project IDs): If a number like 2022 or a catalog ID like 11814460001 appears in every filename, it is flagged as "Project Metadata" and ignored.

Variable Numbers (Replicates): The tool looks for the number that changes across the dataset (e.g., _1, _2, _3) to identify the actual replicate.

Unit Protection: It automatically ignores numbers attached to experimental factors like 80C (temperature) or 30min (incubation time).

🛠 Extraction Heuristics
The extractor applies a hierarchy of rules to the paper's Title, Abstract, and Method

Category,Logic Description
Quantification,"Detects TMT (6/10/11/16-plex), iTRAQ, SILAC, or Label-Free."
TMT Expansion,"If TMT16 is found, 1 raw file group is expanded into 16 SDRF rows with correct channel tags (126, 127N, etc.)."
Organism,"Maps common names (e.g., ""murine"") to scientific names (Mus musculus)."
Instrument,"Identifies specific MS models (e.g., Orbitrap Exploris 480, timsTOF Pro)."
Modifications,"Prioritizes Fixed (Carbamidomethyl) followed by Variable (Oxidation, Phospho)."

📊 Logic Flow: From Files to Rows
Ingest: Load Paper JSON (Searchable text + Raw file list).

Filter: Isolate "Primary Files" (e.g., removing DDA if DIA is present).

Group: Identify unique biological samples by stripping fraction/timestamp suffixes.

Analyze Group Structure:

If a group contains 24 files with _F markers: number_of_fractions = 24.

If a group contains 2 files with no markers: number_of_technical_replicates = 2.

Multiply: Expand groups by the number of detected channels (e.g., 16 for TMTpro).

Populate: Create SDRFRow objects with extracted metadata.

📝 Code Usage

```python
from src.extractor import PaperJSON, extract_initial_sdrf

# 1. Load the paper data
paper = PaperJSON.from_file("pxd_data.json")

# 2. Run the extraction logic
sdrf_doc = extract_initial_sdrf(paper)

# 3. Access generated rows
for row in sdrf_doc.rows:
    print(f"Sample: {row.assay_name} | Files: {row.comment_data_file_}")
```