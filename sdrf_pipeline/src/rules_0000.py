import os
import re
import json
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

# paper agnostic rules from 
# https://www.kaggle.com/code/mawramusawwar/harmonizing-the-data-of-your-data-0-27
# PAPER overrides intentionally not used

# ============================================================
# DEFAULT MODIFICATION VALUES
# Most common across ALL training data — apply universally
# ============================================================
DEFAULT_MOD_CARBAMIDOMETHYL = "NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed"
# GT top value: "NT=Oxidation;AC=UNIMOD:35;MT=Variable;TA=M" [7,777 rows] — NOT the old field order
DEFAULT_MOD_OXIDATION = "NT=Oxidation;AC=UNIMOD:35;MT=Variable;TA=M"
DEFAULT_MOD_ACETYL = "NT=Acetyl;AC=UNIMOD:1;PP=Protein N-term;MT=variable"
TMT_MOD_K = "NT=TMT6plex;AC=UNIMOD:737;TA=K;MT=Fixed"
TMT_MOD_NTERM = "NT=TMT6plex;AC=UNIMOD:737;PP=Any N-term;MT=Fixed"

# ============================================================
# LOAD TEST PAPERS
# ============================================================
METHODS_KEYS = ["method", "material", "experimental", "procedure"]


def load_pub_json(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"full": str(data)}
    return data


def get_methods_text(paper, max_chars=15000):
    methods, others = [], []
    for k, v in paper.items():
        t = str(v)
        if any(mk in k.lower() for mk in METHODS_KEYS):
            methods.append(t)
        elif k.upper() in ("ABSTRACT", "TITLE"):
            others.append(t)
    return "\n\n".join(methods + others)[:max_chars]


def get_full_text(paper, max_chars=20000):
    return "\n\n".join(str(v) for v in paper.values())[:max_chars]

# ============================================================
# RULE-BASED TEXT EXTRACTORS
# ============================================================


def extract_organism(text):
    t = text.lower()
    if "homo sapiens" in t or re.search(r'\bhuman\b', t):
        return "Homo sapiens"
    if "mus musculus" in t or re.search(r'\bmice?\b|\bmouse\b', t):
        return "Mus musculus"
    if "rattus norvegicus" in t or re.search(r'\brats?\b', t):
        return "Rattus norvegicus"
    if "saccharomyces cerevisiae" in t or re.search(r'\byeast\b', t):
        return "Saccharomyces cerevisiae"
    if "bos taurus" in t or re.search(r'\bbovine\b|\bcow\b|\bcattle\b', t):
        return "Bos taurus"
    if "escherichia coli" in t or re.search(r'e\.\s*coli', t):
        return "Escherichia coli"
    if "arabidopsis" in t:
        return "Arabidopsis thaliana"
    if re.search(r'caenorhabditis|c\.\s*elegans', t):
        return "Caenorhabditis elegans"
    if "drosophila" in t:
        return "Drosophila melanogaster"
    if "plasmodium" in t:
        return "Plasmodium falciparum"
    return None


def extract_sex(text):
    t = text.lower()
    t = re.sub(r'fetal\s+bovine\s+serum|foetal\s+bovine\s+serum|\bfbs\b', '', t)
    has_male = bool(re.search(r'\bmale\b', t))
    has_female = bool(re.search(r'\bfemale\b', t))
    if has_male and has_female:
        return "male|female"
    if has_male:
        return "male"
    if has_female:
        return "female"
    return None


def extract_developmental_stage(text):
    t = text.lower()
    t_clean = re.sub(r'fetal\s+bovine\s+serum|foetal\s+bovine\s+serum|\bfbs\b|\bfcs\b', '', t)
    if re.search(r'\bembryo(?:nic)?\s+(?:stem|cell|tissue|development|proteom)', t_clean):
        return "embryo"
    if re.search(r'\bfetal\b|\bfoetal\b', t_clean):
        return "fetal"
    if re.search(r'\badult\b', t_clean):
        return "adult"
    if re.search(r'\bneonat|\bnewborn\b', t_clean):
        return "neonatal"
    return None


def extract_material_type(text):
    t = text.lower()
    t = re.sub(r'fetal\s+bovine\s+serum|fbs|foetal\s+bovine', '', t)
    if re.search(r'\bcell\s+line\b|\bcultured\s+cell|\bin\s+vitro\b', t):
        return "cell"
    if re.search(r'\btissue\b|\bbiopsy\b|\bsurgical\b', t):
        return "tissue"
    if re.search(r'\bblood\s+plasma\b|\bplasma\b(?!\s+membrane)', t):
        return "blood plasma"
    if re.search(r'\bserum\b(?!\s+albumin|\s+free|\s+starv)', t):
        return "serum"
    if re.search(r'\burine\b', t):
        return "urine"
    if re.search(r'\blysate\b', t):
        return "lysate"
    return None


def extract_cleavage_agent(text):
    t = text.lower()
    if re.search(r'lys.?c.*trypsin|trypsin.*lys.?c', t):
        return "AC=MS:1001309;NT=Lys-C|AC=MS:1001251;NT=Trypsin"
    if re.search(r'\blys.?c\b', t):
        return "AC=MS:1001309;NT=Lys-C"
    if re.search(r'\btrypsin\b', t):
        return "AC=MS:1001251;NT=Trypsin"
    if re.search(r'\bchymotrypsin\b', t):
        return "AC=MS:1001306;NT=Chymotrypsin"
    if re.search(r'\bglu.?c\b', t):
        return "AC=MS:1001917;NT=Glu-C"
    if re.search(r'\basp.?n\b', t):
        return "AC=MS:1001305;NT=Asp-N"
    return None


def extract_reduction_reagent(text):
    t = text.lower()
    if re.search(r'\btcep\b', t):
        return "TCEP"
    if re.search(r'\bdtt\b|dithiothreitol', t):
        return "DTT"
    if re.search(r'\bbme\b|beta.mercaptoethanol|2-mercaptoethanol', t):
        return "beta-mercaptoethanol"
    return None


def extract_alkylation_reagent(text):
    t = text.lower()
    if re.search(r'\biaa\b|iodoacetamide', t):
        return "IAA"
    if re.search(r'\bcaa\b|chloroacetamide', t):
        return "CAA"
    if re.search(r'\bnem\b|n-ethylmaleimide', t):
        return "NEM"
    return None


def extract_label(text):
    """
    Returns plain text label format matching GT training data.
    e.g. "AC=MS:1002038;NT=label free sample", "TMT126", "SILAC heavy"
    """
    t = text.lower()
    if re.search(r'\blabel.free\b|label\s+free\b', t):
        return "AC=MS:1002038;NT=label free sample"
    if re.search(r'\bdia\b|data.independent', t) and not re.search(r'\btmt\b|\bsilac\b|\bitraq\b', t):
        return "AC=MS:1002038;NT=label free sample"
    if re.search(r'label(?:ed|ing)?\s+with\s+tmt|tmt\s+label|tmt\s+(?:6|10|11|16|18|plex)', t):
        m = re.search(r'tmt.?(6|10|11|16|18).?plex', t)
        plex = m.group(1) if m else "10"
        return f"TMT{plex}plex"
    if re.search(r'\bitraq\s+label|label.*itraq', t):
        return "iTRAQ"
    if re.search(r'\bsilac\b', t):
        if "heavy" in t:
            return "SILAC heavy"
        if "medium" in t:
            return "SILAC medium"
        if "light" in t:
            return "SILAC light"
        return "SILAC"
    return None


def extract_instrument(text):
    """
    Returns instrument in most-common training GT format.
    Ordering (NT= first vs AC= first) matters — 80% similarity won't bridge the gap.
    Formats derived from training data top values.
    """
    t = text.lower()
    patterns = [
        # --- Formats from training top values (exact ordering) ---
        (r'q\s*exactive\s*hf.?x',            "NT=Q Exactive HF-X;AC=MS:1002877"),
        (r'q\s*exactive\s*hf\b',             "AC=MS:1002523;NT=Q Exactive HF"),       # AC first [2,994]
        (r'q\s*exactive\s*plus',             "NT=Q Exactive Plus;AC=MS:1001911"),      # NT first [4,424]
        (r'q\s*exactive\b',                  "NT=Q Exactive;AC=MS:1001911"),           # NT first [4,855]
        (r'orbitrap\s*fusion\s*lumos',       "NT=Orbitrap Fusion Lumos;AC=MS:1002731"),# NT first [1,460]
        (r'orbitrap\s*fusion\b',             "AC=MS:1000639;NT=Orbitrap Fusion"),      # AC first [1,687]
        (r'orbitrap\s*exploris\s*480',       "NT=Orbitrap Exploris 480;AC=MS:1003028"),
        (r'astral\b',                        "NT=Orbitrap Astral;AC=MS:1003378"),
        (r'orbitrap\s*eclipse',              "NT=Orbitrap Eclipse;AC=MS:1003029"),
        (r'orbitrap\s*elite',                "AC=MS:1001910;NT=LTQ Orbitrap Elite"),   # AC first [2,938]
        (r'orbitrap\s*velos',                "AC=MS:1001742;NT=LTQ Orbitrap Velos"),   # AC first [3,976]
        (r'ltq\s*orbitrap\s*xl',              "AC=MS:1000447;NT=LTQ Orbitrap XL"),
        (r'ltq\s*orbitrap\b',                "NT=LTQ Orbitrap;AC=MS:1000449"),
        (r'timstof\s*pro',                   "NT=timsTOF Pro;AC=MS:1003005"),
        (r'zeno\s*tof\s*7600|zeno.*7600',    "NT=Zeno TOF 7600;AC=MS:1003027"),
        (r'triple\s*tof\s*5600',             "NT=TripleTOF 5600+;AC=MS:1000931"),
        (r'timstof',                         "NT=timsTOF;AC=MS:1002817"),
        (r'sciex\s*6600|triple\s*tof\s*6600',"NT=TripleTOF 6600;AC=MS:1002533"),
        (r'synapt',                          "NT=Synapt MS;AC=MS:1001490"),
    ]
    for pattern, sdrf_val in patterns:
        if re.search(pattern, t):
            return sdrf_val
    return None


def extract_fragmentation(text):
    """Returns AC=MS format matching GT training data (top values use AC= format)."""
    t = text.lower()
    if re.search(r'\bethcd\b|ethcdhcd|electron\s+transfer.*higher.energy', t):
        return "AC=MS:1002631;NT=EThcd/ETD"
    if re.search(r'\bhcd\b|higher.energy\s+collision', t):
        return "AC=MS:1000422;NT=HCD"
    if re.search(r'\bcid\b|collision.induced', t):
        return "AC=MS:1000133;NT=CID"
    if re.search(r'\betd\b|electron\s+transfer\s+dissociation', t):
        return "AC=MS:1000598;NT=ETD"
    if re.search(r'\becd\b|electron\s+capture\s+dissociation', t):
        return "AC=MS:1000250;NT=ECD"
    return None


def extract_acquisition_method(text):
    t = text.lower()
    if re.search(r'\bdia\b|data.independent', t):
        return "DIA"
    if re.search(r'\bdda\b|data.dependent', t):
        return "DDA"
    if re.search(r'\bprm\b|parallel\s+reaction\s+monitoring', t):
        return "PRM"
    if re.search(r'\bsrm\b|\bmrm\b|multiple\s+reaction\s+monitoring', t):
        return "MRM"
    return None


def extract_enrichment(text):
    t = text.lower()
    if re.search(r'phospho(?:peptide|protein)?\s+enrich|enrich.*phospho|tio2|imac\s+enrich', t):
        return "enrichment of phosphorylated Protein"
    if re.search(r'ubiquitin.*enrich|diglycine\s+enrich|k.?gg\s+enrich', t):
        return "ubiquitination enrichment"
    if re.search(r'glyco(?:peptide|protein)?\s+enrich|enrich.*glyco', t):
        return "glycopeptide enrichment"
    if re.search(r'no\s+enrich|without\s+enrich', t):
        return "no enrichment"
    return None


def extract_fractionation(text):
    t = text.lower()
    if re.search(r'no\s+fraction|without\s+fraction|single\s+shot|not\s+fraction', t):
        return "no fractionation"
    if re.search(r'high.?ph\s+rp|basic\s+rp|high\s+ph\s+reverse', t):
        return "NT=high pH RPLC;AC=PRIDE:0000564"
    if re.search(r'\bscx\s+fraction|\bscx\s+chrom|strong\s+cation\s+exchange', t):
        return "NT=Strong cation-exchange chromatography (SCX);AC=PRIDE:0000561"
    if re.search(r'sds.?page.*fraction|gel.*fraction|in.gel\s+digest|bn.?page', t):
        return "NT=SDS-PAGE;AC=PRIDE:0000568"
    return None


def extract_organism_part(text):
    t = text.lower()
    t = re.sub(
        r'fetal\s+bovine\s+serum|foetal\s+bovine\s+serum|\bfbs\b'
        r'|foetal\s+calf\s+serum|\bfcs\b|bovine\s+serum\s+albumin|\bbsa\b'
        r'|horse\s+serum|goat\s+serum|blocking\s+serum|normal\s+serum',
        '', t)
    parts = [
        (r'\bblood\s+plasma\b|\bplasma\b(?!\s+membrane|\s+cell|\s+protein|\s+level)', "blood plasma"),
        (r'\bserum\b(?!\s+albumin|\s+free|\s+starv|\s+level)', "serum"),
        (r'\burine\b', "urine"),
        (r'\bbrain\b', "brain"),
        (r'\bmilk\b', "milk"),
        (r'\bliver\b', "liver"),
        (r'\blung\b', "lung"),
        (r'\bheart\b', "heart"),
        (r'\bkidney\b', "kidney"),
        (r'\bcolon\b|\bcolorectal\b', "colon"),
        (r'\bovary\b|\bovarian\b', "ovary"),
        (r'\bpancrea', "pancreas"),
        (r'\bprostate\b', "prostate gland"),
        (r'\bcerebrospinal\b|\bcsf\b', "cerebrospinal fluid"),
        (r'\bcell\s+line\b|\bin\s+vitro\b', "cell culture"),
        (r'\bskin\b', "skin"),
        (r'\bmuscle\b', "muscle"),
        (r'\bbone\s+marrow\b', "bone marrow"),
        (r'\blymph\s+node\b', "lymph node"),
        (r'\bsynovial\b', "synovial membrane"),
        (r'\bcartilage\b', "cartilage"),
    ]
    found, seen = [], set()
    for pattern, label in parts:
        if label not in seen and re.search(pattern, t):
            found.append(label)
            seen.add(label)
    return "|".join(found[:3]) if found else None


def extract_cell_type(text):
    t = text.lower()
    types = [
        (r'\bfibroblast', "fibroblast"),
        (r'\bneuron|\bneuronal\b', "neuron"),
        (r'\bmacrophage', "macrophage"),
        (r'\bt.cell\b|\bt\s+cell\b', "T cell"),
        (r'\bb.cell\b|\bb\s+cell\b', "B cell"),
        (r'\bmonocyte', "monocyte"),
        (r'\bplatelet', "platelet"),
        (r'\bepithelial', "epithelial cell"),
        (r'\bhepatocyte', "hepatocyte"),
        (r'\bendothelial', "endothelial cell"),
    ]
    for pattern, label in types:
        if re.search(pattern, t):
            return label
    return None


def extract_cell_line(text):
    cell_lines = [
        "HEK293T", "HEK293", "HeLa", "Jurkat", "MCF-7", "MCF7",
        "U2OS", "A549", "HCT116", "PC-3", "LNCaP", "SK-OV-3",
        "K562", "THP-1", "RAW264", "SH-SY5Y", "Neuro2a",
        "NIH3T3", "CHO", "Vero", "Caco-2", "SW480", "HT-29",
        "MDA-MB-231", "MDA-MB-468", "DU145", "MRC5", "MelJuSo",
    ]
    for cl in cell_lines:
        if cl.lower() in text.lower():
            return cl
    return None


def extract_specimen(text):
    t = text.lower()
    t = re.sub(r'fetal\s+bovine\s+serum|fbs|bovine\s+serum\s+albumin|bsa|blocking\s+serum', '', t)
    specs = [
        (r'\bbiopsy\b', "biopsy"),
        (r'\bblood\s+plasma\b|\bplasma\b(?!\s+membrane)', "plasma"),
        (r'\bserum\b(?!\s+albumin|\s+free)', "serum"),
        (r'\burine\b', "urine"),
        (r'\bcsf\b|cerebrospinal\s+fluid', "cerebrospinal fluid"),
        (r'\bmilk\b', "milk"),
    ]
    for pattern, label in specs:
        if re.search(pattern, t):
            return label
    return None


def extract_separation(text):
    t = text.lower()
    if re.search(r'reverse.phase|rp.?lc|c18|c-18', t):
        return "AC=PRIDE:0000563;NT=Reversed-phase chromatography"
    if re.search(r'hilic', t):
        return "AC=PRIDE:0000083;NT=HILIC"
    return None


def extract_ms2_analyzer(text):
    t = text.lower()
    if re.search(r'orbitrap', t):
        return "AC=MS:1000484; NT=Orbitrap"
    if re.search(r'ion\s*trap|linear\s*trap', t):
        return "AC=MS:1000264; NT=ion trap"
    if re.search(r'tof\b|time.of.flight', t):
        return "AC=MS:1000084; NT=TOF"
    return None


def extract_precursor_tol(text):
    m = re.search(r'precursor[^.\n]{0,60}?(\d+(?:\.\d+)?)\s*ppm', text.lower())
    if m:
        return f"{m.group(1)} ppm"
    m = re.search(r'(\d+(?:\.\d+)?)\s*ppm[^.\n]{0,40}?precursor', text.lower())
    if m:
        return f"{m.group(1)} ppm"
    m = re.search(r'(\d+(?:\.\d+)?)\s*ppm', text.lower())
    if m and float(m.group(1)) <= 50:
        return f"{m.group(1)} ppm"
    return None


def extract_fragment_tol(text):
    t = text.lower()
    m = re.search(r'(?:fragment|ms2|ms/ms)[^.\n]{0,80}?(\d+(?:\.\d+)?)\s*(da)\b', t)
    if m:
        return f"{m.group(1)} Da"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(da)\b[^.\n]{0,80}?(?:fragment|ms2|ms/ms)', t)
    if m:
        return f"{m.group(1)} Da"
    m = re.search(r'(?:fragment|ms2)[^.\n]{0,80}?(\d+(?:\.\d+)?)\s*ppm', t)
    if m:
        return f"{m.group(1)} ppm"
    return None


def extract_missed_cleavages(text):
    t = text.lower()
    m = re.search(r'(?:maximum\s+of\s+|up\s+to\s+|allow(?:ing|ed)?\s+)([012])\s+missed\s+cleav', t)
    if m:
        return m.group(1)
    m = re.search(r'\b([012])\s+missed\s+cleav', t)
    if m:
        return m.group(1)
    m = re.search(r'missed\s+cleav[^.\n]{0,20}?([012])\b', t)
    if m:
        return m.group(1)
    return None


def extract_gradient_time(text):
    t = text.lower()
    m = re.search(r'gradient[^.\n]{0,60}?(\d{2,3})\s*min\b', t)
    if m:
        return f"{m.group(1)} min"
    m = re.search(r'(\d{2,3})\s*min\b[^.\n]{0,40}?gradient', t)
    if m:
        return f"{m.group(1)} min"
    return None


def extract_flow_rate(text):
    t = text.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*nl.?/\s*min', t)
    if m:
        return f"{m.group(1)} nL/min"
    m = re.search(r'(\d+(?:\.\d+)?)\s*µl.?/\s*min', t)
    if m:
        return f"{m.group(1)} µL/min"
    return None


def extract_disease(text):
    t = text.lower()
    disease_map = [
        (r'osteoarthritis',                   "osteoarthritis"),
        (r'alzheimer',                         "Alzheimer disease"),
        (r'parkinson',                         "Parkinson disease"),
        (r'glioblastoma|glioma',               "glioblastoma"),
        (r'lung\s+cancer|nsclc',               "lung cancer"),
        (r'breast\s+cancer',                   "breast cancer"),
        (r'colorectal|colon\s+cancer',         "colorectal cancer"),
        (r'ovarian\s+cancer',                  "ovarian cancer"),
        (r'prostate\s+cancer',                 "prostate cancer"),
        (r'hepatocellular|liver\s+cancer',     "hepatocellular carcinoma"),
        (r'melanoma',                          "melanoma"),
        (r'leukemia|lymphoma',                 "leukemia"),
        (r'diabetes|diabetic',                 "diabetes mellitus"),
        (r'healthy|normal|non.diseased|control', "normal"),
    ]
    for pattern, label in disease_map:
        if re.search(pattern, t):
            return label
    return None


# ============================================================
# EXTRACTOR DISPATCH TABLE
# ============================================================
EXTRACTORS = {
    "Characteristics[Organism]":           extract_organism,
    "Characteristics[Sex]":                extract_sex,
    "Characteristics[DevelopmentalStage]": extract_developmental_stage,
    "Characteristics[MaterialType]":       extract_material_type,
    "Characteristics[CleavageAgent]":      extract_cleavage_agent,
    "Characteristics[ReductionReagent]":   extract_reduction_reagent,
    "Characteristics[AlkylationReagent]":  extract_alkylation_reagent,
    "Characteristics[Label]":              extract_label,
    "Characteristics[Disease]":            extract_disease,
    "Characteristics[OrganismPart]":       extract_organism_part,
    "Characteristics[CellType]":           extract_cell_type,
    "Characteristics[CellLine]":           extract_cell_line,
    "Characteristics[Specimen]":           extract_specimen,
    "Comment[Instrument]":                 extract_instrument,
    "Comment[FragmentationMethod]":        extract_fragmentation,
    "Comment[AcquisitionMethod]":          extract_acquisition_method,
    "Comment[EnrichmentMethod]":           extract_enrichment,
    "Comment[FractionationMethod]":        extract_fractionation,
    "Comment[Separation]":                 extract_separation,
    "Comment[MS2MassAnalyzer]":            extract_ms2_analyzer,
    "Comment[PrecursorMassTolerance]":     extract_precursor_tol,
    "Comment[FragmentMassTolerance]":      extract_fragment_tol,
    "Comment[NumberOfMissedCleavages]":    extract_missed_cleavages,
    "Comment[GradientTime]":               extract_gradient_time,
    "Comment[FlowRateChromatogram]":       extract_flow_rate,
}


def get_test_papers(TEST_TEXT_DIR):
    test_papers = {}
    for fname in os.listdir(TEST_TEXT_DIR):
        if fname.endswith(".json") and fname != "PubText.json":
            pxd = fname.split("_")[0]
            test_papers[pxd] = load_pub_json(os.path.join(TEST_TEXT_DIR, fname))
    return test_papers


def get_text_pxd(SAMPLE_SUB):
    # ============================================================
    # LOAD SUBMISSION TEMPLATE
    # ============================================================
    sample_sub = pd.read_csv(SAMPLE_SUB)
    id_cols = ["ID", "PXD", "Raw Data File", "Usage"]
    target_cols = [c for c in sample_sub.columns if c not in id_cols]

    pxd_rawfiles = defaultdict(list)
    for _, row in sample_sub.iterrows():
        pxd_rawfiles[row["PXD"]].append(str(row["Raw Data File"]).strip())

    test_pxds = sorted(sample_sub["PXD"].unique())
    print(f"Target columns: {len(target_cols)}")
    print(f"Test PXDs: {len(test_pxds)}")
    return test_pxds, target_cols, sample_sub


def extract_per_file_metadata(pxd, raw_file):
    """
    Filename per-file metadata.
    Returns dict of {col: value}
    """
    result = {}
    name = os.path.splitext(raw_file)[0]
    tokens = re.split(r'[_\-\.]', name)
    tokens_lower = [t.lower() for t in tokens]
    name_lower = name.lower()

    # --- Universal: BiologicalReplicate from filename ---
    for i, t in enumerate(tokens_lower):
        # BR1, BR2, BioRep1, rep1, r1, R1, s1, S1 (sample)
        m = re.match(r'^(?:br|biorep|rep|r|s|sample)(\d+)$', t)
        if m:
            result["Characteristics[BiologicalReplicate]"] = m.group(1)
            break
        # biorepX or repX at end
        if t in ("rep", "biorep", "br", "r", "s") and i + 1 < len(tokens):
            nxt = tokens[i+1]
            if nxt.isdigit():
                result["Characteristics[BiologicalReplicate]"] = nxt
                break
    # If still not found, try trailing digit in filename (e.g. PXD_name_01, SampleName_3)
    if "Characteristics[BiologicalReplicate]" not in result:
        m = re.search(r'[_\-](\d{1,2})(?:\.raw)?$', raw_file, re.IGNORECASE)
        if m and int(m.group(1)) <= 20:  # Reasonable replicate range
            result["Characteristics[BiologicalReplicate]"] = m.group(1).lstrip("0") or "1"

    # --- Universal: FractionIdentifier ---
    for i, t in enumerate(tokens_lower):
        m = re.match(r'^f(\d+)$', t)
        if m:
            result["Comment[FractionIdentifier]"] = m.group(1)
            break
        if t in ("frac", "fraction") and i + 1 < len(tokens):
            nxt = tokens[i+1]
            if nxt.isdigit():
                result["Comment[FractionIdentifier]"] = nxt
                break


def build_vocab(TRAIN_CSV, target_cols):
    gt_counter = defaultdict(Counter)

    print("Loading training.csv for vocab...")
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    print(f"training.csv shape: {train_df.shape}")

    for col in target_cols:
        if col in train_df.columns:
            vals = train_df[col].dropna().astype(str)
            vals = vals[~vals.isin(["Not Applicable", "not applicable", "NA", "nan", "TextSpan", ""])]
            gt_counter[col].update(vals.tolist())

    n_train_pxds = train_df["PXD"].nunique() if "PXD" in train_df.columns else 103

    global_modes = {}
    non_na_ratio = {}
    for col in target_cols:
        total = sum(gt_counter[col].values())
        if total > 0:
            global_modes[col] = gt_counter[col].most_common(1)[0][0]
            non_na_ratio[col] = total / n_train_pxds
        else:
            global_modes[col] = "Not Applicable"
            non_na_ratio[col] = 0

    print(f"Vocab built from {n_train_pxds} training PXDs")
    del train_df
    return gt_counter, global_modes, non_na_ratio


NEVER_GLOBAL = {
    "Characteristics[Age]", "Characteristics[AncestryCategory]",
    "Characteristics[Bait]", "Characteristics[CellLine]",
    "Characteristics[CellPart]", "Characteristics[Compound]",
    "Characteristics[Depletion]", "Characteristics[GrowthRate]",
    "Characteristics[PooledSample]", "Characteristics[SamplingTime]",
    "Characteristics[SpikedCompound]", "Characteristics[Staining]",
    "Characteristics[Strain]", "Characteristics[SyntheticPeptide]",
    "Characteristics[Temperature]", "Characteristics[Time]",
    "Characteristics[Treatment]", "Characteristics[TumorSize]",
    "Characteristics[TumorGrade]", "Characteristics[TumorStage]",
    "Characteristics[TumorCellularity]", "Characteristics[TumorSite]",
    "Characteristics[AnatomicSiteTumor]", "Characteristics[BMI]",
    "Characteristics[GeneticModification]", "Characteristics[Genotype]",
    "Characteristics[NumberOfBiologicalReplicates]",
    "Characteristics[NumberOfSamples]",
    "Characteristics[NumberOfTechnicalReplicates]",
    "Characteristics[OriginSiteDisease]", "Characteristics[DiseaseTreatment]",
    "Comment[CollisionEnergy]", "Comment[NumberOfFractions]",
    "Characteristics[Sex]", "Characteristics[DevelopmentalStage]",
    # FactorValue columns — only set when explicitly known
    "FactorValue[Bait]", "FactorValue[CellPart]", "FactorValue[Compound]",
    "FactorValue[ConcentrationOfCompound].1", "FactorValue[Disease]",
    "FactorValue[FractionIdentifier]", "FactorValue[GeneticModification]",
    "FactorValue[Temperature]", "FactorValue[Treatment]",
}


def rules_pipeline(SAMPLE_SUB, TEST_TEXT_DIR, TRAIN_CSV):
    """
    Heuristic pipeline for the "Harmonizing the Data of Your Data" Kaggle competition.

    This function orchestrates the end-to-end prediction workflow for extracting
    structured metadata from scientific documents. The goal is to transform
    unstructured text (test papers) into standardized tabular outputs aligned
    with SDRF-like target columns.

    Steps:
    -------
    1. Load submission template and extract PXD identifiers.
    2. Parse test documents from the provided directory and identify target columns.
    3. Build a vocabulary/statistical representation from the training dataset.
    4. Generate predictions for each PXD entry based on learned patterns.

    Parameters:
    ----------
    SAMPLE_SUB : str or Path
        Path to the sample submission file. Used to extract test identifiers (e.g., PXD IDs)
        and enforce the expected submission format.

    TEST_TEXT_DIR : str or Path
        Directory containing raw text files of test scientific papers. These documents
        are the input source for metadata extraction.

    TRAIN_CSV : str or Path
        Path to the training CSV file containing labeled metadata. Used to construct
        vocabularies, frequency statistics, or heuristics for prediction.

    Returns:
    -------
    dict
        Dictionary mapping PXD identifiers to predicted metadata fields
        (column-value mappings aligned with submission requirements).

    Notes:
    ------
    - This implementation relies on simple vocabulary-based heuristics (e.g., global modes,
      token matching) rather than full NLP pipelines.
    - The competition task focuses on harmonizing heterogeneous textual metadata into
      consistent structured formats, a common problem in bioinformatics and data integration.
    - Output should later be converted into a submission DataFrame matching SAMPLE_SUB.

    """
    test_pxds = get_text_pxd(SAMPLE_SUB)
    test_papers, target_cols, sample_sub = get_test_papers(TEST_TEXT_DIR)
    gt_counter, global_modes, non_na_ratio = build_vocab(TRAIN_CSV, target_cols)
    pxd_predictions = {}

    for pxd in tqdm(test_pxds, desc="Extracting"):
        if pxd not in test_papers:
            pxd_predictions[pxd] = {}
            continue

        paper = test_papers[pxd]
        methods = get_methods_text(paper)
        fulltext = get_full_text(paper)

        pred = {}

        # Run text extractors on methods, fall back to full text
        for col, extractor in EXTRACTORS.items():
            val = extractor(methods)
            if not val:
                val = extractor(fulltext)
            if val:
                pred[col] = val

        # Modification vocab matching
        mod_cols = [c for c in target_cols if c.startswith("Characteristics[Modification]")]
        for col in mod_cols:
            col_sfx = col.replace("Characteristics[Modification]", "").replace(".", "")
            col_num = int(col_sfx) if col_sfx.isdigit() else 0
            for val, count in gt_counter[col].most_common():
                m = re.search(r'NT=([^;]+)', val)
                if not m:
                    continue
                mod_name = m.group(1).strip().lower()
                if mod_name in ("oxidation", "acetyl", "carbamidomethyl") and col_num >= 3:
                    continue
                if mod_name in methods.lower() or mod_name in fulltext.lower():
                    pred[col] = val
                    break

        # we do not want  paper-specific global overrides (highest priority)
        #if pxd in PAPER_OVERRIDES:
        #    for col, val in PAPER_OVERRIDES[pxd].items():
        #        pred[col] = val

        pxd_predictions[pxd] = pred

    # Preview results
    print("\n=== Extraction results per PXD ===")
    for pxd, pred in pxd_predictions.items():
        n = len([v for v in pred.values() if v and v != "Not Applicable"])
        print(f"  {pxd}: {n} columns extracted")

    # Submission
    final_sub = sample_sub.copy()       
    src_counts = {"extractor": 0, "per_file": 0, "default_mod": 0, "global": 0, "na": 0}

    for idx, row in final_sub.iterrows():
        pxd = row["PXD"]
        raw_file = str(row["Raw Data File"]).strip()
        pred = pxd_predictions.get(pxd, {})

        # Get per-file metadata from filename
        per_file = extract_per_file_metadata(pxd, raw_file)

        for col in target_cols:
            # Priority 1: Per-file filename extraction
            val = per_file.get(col)
            if val:
                src_counts["per_file"] += 1
                final_sub.at[idx, col] = val
                continue

            # Priority 2: Global paper-level extraction / text extractors
            val = pred.get(col)
            if val:
                src_counts["extractor"] += 1
                final_sub.at[idx, col] = val
                continue

            # Priority 3: Default Carbamidomethyl modification (near-universal)
            if col == "Characteristics[Modification]":
                final_sub.at[idx, col] = DEFAULT_MOD_CARBAMIDOMETHYL
                src_counts["default_mod"] += 1
                continue

            # Priority 4: Default Oxidation as Modification.1 (near-universal)
            if col == "Characteristics[Modification].1":
                final_sub.at[idx, col] = DEFAULT_MOD_OXIDATION
                src_counts["default_mod"] += 1
                continue

            # Priority 5: TMT modifications or Acetyl as Modification.2/.3
            if col == "Characteristics[Modification].2":
                lbl = pred.get("Characteristics[Label]", "")
                if "TMT" in str(lbl):
                    final_sub.at[idx, col] = TMT_MOD_K
                    src_counts["default_mod"] += 1
                    continue
                else:
                    # Acetyl is very common Mod.2 for non-TMT papers (4,214 rows GT)
                    final_sub.at[idx, col] = DEFAULT_MOD_ACETYL
                    src_counts["default_mod"] += 1
                    continue

            if col == "Characteristics[Modification].3":
                lbl = pred.get("Characteristics[Label]", "")
                if "TMT" in str(lbl):
                    final_sub.at[idx, col] = TMT_MOD_NTERM
                    src_counts["default_mod"] += 1
                    continue
                else:
                    # Deamidation is very common Mod.3 for non-TMT papers (4,176 rows GT)
                    final_sub.at[idx, col] = "NT=Gln->pyro-Glu;AC=UNIMOD:28;PP=Any N-term;MT=variable"
                    src_counts["default_mod"] += 1
                    continue

            # Priority 6: Global mode fallback (only for well-populated columns)
            if col not in NEVER_GLOBAL and non_na_ratio[col] > 0.75:
                final_sub.at[idx, col] = global_modes[col]
                src_counts["global"] += 1
                continue

            # Default: Not Applicable
            final_sub.at[idx, col] = "Not Applicable"
            src_counts["na"] += 1     
    final_sub = final_sub.fillna("Not Applicable")
    if "Unnamed: 0" in final_sub.columns:
        final_sub = final_sub.drop(columns=["Unnamed: 0"])

    print(f"\nFinal shape: {final_sub.shape}")
    print(f"Null count:  {final_sub.isnull().sum().sum()}")
    print(f"\nPrediction sources:")
    print(f"  Rule-based / overrides: {src_counts['extractor']:,}")
    print(f"  Per-file (filenames):   {src_counts['per_file']:,}")
    print(f"  Default modifications:  {src_counts['default_mod']:,}")
    print(f"  Global mode fallback:   {src_counts['global']:,}")
    print(f"  Not Applicable:         {src_counts['na']:,}")

    return final_sub