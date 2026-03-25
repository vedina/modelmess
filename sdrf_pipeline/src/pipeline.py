"""
SDRF Extraction Pipeline — v2
Prompts are built automatically from the field registry in fields.py.
Adds:
  - Regex pre-scan hints prepended to paper text
  - Two-pass chain-of-thought (reason → extract)
  - Post-extraction audit for high-value fields still marked "not applicable"
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from src.models import SDRFDocument, SDRFRow
from src.fields import (
    FIELDS,
    HIGH_VALUE_ATTRS,
    build_prompt_field_guide,
    build_regex_hints,
    FIELD_BY_ATTR,
)

logger = logging.getLogger(__name__)

# ── SDRF column order ─────────────────────────────────────────────────────────
SDRF_HEADERS = ["D", "PXD", "Raw Data File"] + [f.header for f in FIELDS]
HEADER_TO_ATTR = {"D": "D", "PXD": "PXD", "Raw Data File": "raw_data_file"}
HEADER_TO_ATTR.update({f.header: f.attr for f in FIELDS})

# ── Prompt templates ──────────────────────────────────────────────────────────
_FIELD_GUIDE = build_prompt_field_guide()

SYSTEM_PROMPT = f"""You are a proteomics metadata extraction expert specialising in SDRF (Sample and Data Relationship Format).

Your task: read a scientific paper and extract every available metadata value into a structured JSON object.

CORE RULES:
1. One JSON row per raw MS file. If filenames are listed, create exactly one row per file.
2. Assign "not applicable" ONLY when a field is genuinely absent from the paper.
3. Before writing "not applicable" for any field, re-read the relevant section.
4. Biological replicate numbers: infer from filename patterns (_1_, -rep2-, _R3_) or from "triplicate"/"duplicate" in text.
5. Organism: always use latin binomial (Homo sapiens, Mus musculus, Bos taurus, ...).
6. Temperature: numeric only, no units (e.g. "65" not "65 degC").
7. Label: default to "label free sample" if no labeling chemistry is described.
8. Modifications: fixed mods in first slot, variable mods in subsequent slots.
9. Factor values: reflect the PRIMARY experimental variable(s) that differ between samples.
10. Usage: always "raw" for raw MS data files.

{_FIELD_GUIDE}

Return ONLY valid JSON -- no markdown, no preamble."""


REASONING_PROMPT = """Read the following paper text and produce an exhaustive metadata inventory.

For each piece of information you find, write one line:
  FIELD: <SDRF field name> -> VALUE: <what you found> (SOURCE: "<quoted phrase from text>")

Be thorough. Cover: organisms, tissues, instruments, reagents, modifications,
acquisition parameters, experimental conditions, file names, replicate structure.
If you find nothing for a standard field, skip it (do not write "not found").

PAPER TEXT:
{text}"""


EXTRACTION_PROMPT = """Using the metadata inventory below AND the original paper text,
fill the SDRF JSON. Every field in the inventory MUST appear as a non-"not applicable" value.
Cross-check the paper text for anything the inventory may have missed.

METADATA INVENTORY:
{inventory}

PAPER TEXT (for cross-checking):
{text}

Return JSON with this structure -- one object per raw file in the "rows" array:
{schema}"""


AUDIT_PROMPT = """You extracted SDRF metadata but these high-value fields are still "not applicable":

{na_list}

Re-read the paper text below and try harder to find values for each field.
For each field either:
  - Provide the corrected value and the source sentence, OR
  - Confirm it is truly absent with a brief reason

Then return the COMPLETE corrected JSON (all rows, all fields).

PAPER TEXT:
{text}

CURRENT JSON:
{current_json}"""


def _row_schema_comment() -> str:
    lines = ["{", '  "rows": [', "    {",
             '      "D": "S",',
             '      "PXD": "<PXDxxxxxx or not available>",',
             '      "raw_data_file": "<filename.raw>",']
    for f in FIELDS:
        ex = f.examples[0] if f.examples else "not applicable"
        lines.append(f'      "{f.attr}": "{ex}",  // {f.header}')
    lines += ["    }", "  ],", '  "extraction_notes": "<optional>"', "}"]
    return "\n".join(lines)


_ROW_SCHEMA = _row_schema_comment()


class SDRFPipeline:
    """
    Extracts SDRF metadata from proteomics paper text.

    Configuration:
        api_key         – API key (or set OPENAI_API_KEY env var)
        model           – model name (default: gpt-4o)
        base_url        – OpenAI-compatible endpoint; None = official OpenAI
        two_pass        – reasoning -> extraction chain (better, 2x tokens)
        audit           – self-audit pass if high-value fields still N/A
        audit_threshold – minimum N/A count to trigger audit
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        two_pass: bool = True,
        audit: bool = True,
        audit_threshold: int = 5,
    ):
        self.two_pass = two_pass
        self.audit = audit
        self.audit_threshold = audit_threshold

        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def extract(self, paper_text: str) -> SDRFDocument:
        hints = build_regex_hints(paper_text)
        augmented = hints + paper_text if hints else paper_text
        logger.debug("Regex hints:\n%s", hints or "(none)")

        raw = self._two_pass_extract(augmented) if self.two_pass else self._single_pass_extract(augmented)
        doc = SDRFDocument(**raw)
        logger.info("Extracted %d row(s)", len(doc.rows))

        if self.audit:
            doc = self._audit_pass(doc, paper_text)

        if doc.extraction_notes:
            logger.info("Notes: %s", doc.extraction_notes)

        return doc

    def to_csv(self, doc: SDRFDocument, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SDRF_HEADERS, extrasaction="ignore")
            writer.writeheader()
            for row in doc.rows:
                row_dict = row.model_dump()
                csv_row = {h: row_dict.get(a, "not applicable") for h, a in HEADER_TO_ATTR.items()}
                writer.writerow(csv_row)
        logger.info("SDRF -> %s", output_path)
        return output_path

    def process_file(self, input_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".sdrf.csv")
        doc = self.extract(input_path.read_text(encoding="utf-8"))
        return self.to_csv(doc, output_path)

    def process_batch(self, input_dir: str | Path, output_dir: str | Path) -> list[Path]:
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, txt in enumerate(sorted(input_dir.glob("*.txt")), 1):
            logger.info("[%d] %s", i, txt.name)
            try:
                results.append(self.process_file(txt, output_dir / txt.with_suffix(".sdrf.csv").name))
            except Exception as e:
                logger.error("Failed %s: %s", txt.name, e)
        return results

    # ── Internal passes ───────────────────────────────────────────────────

    def _call_llm(self, system: str, human: str) -> str:
        return self.llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content

    def _single_pass_extract(self, text: str) -> dict:
        logger.info("Single-pass extraction...")
        human = f"Extract SDRF metadata from the paper text below.\n\nPAPER TEXT:\n{text}\n\nReturn JSON:\n{_ROW_SCHEMA}"
        return self._parse_json(self._call_llm(SYSTEM_PROMPT, human))

    def _two_pass_extract(self, text: str) -> dict:
        logger.info("Pass 1: reasoning inventory...")
        inventory = self._call_llm(
            "You are a proteomics metadata extraction expert.",
            REASONING_PROMPT.format(text=text),
        )
        logger.debug("Inventory (first 600 chars):\n%s", inventory[:600])

        logger.info("Pass 2: structured extraction...")
        human = EXTRACTION_PROMPT.format(inventory=inventory, text=text, schema=_ROW_SCHEMA)
        return self._parse_json(self._call_llm(SYSTEM_PROMPT, human))

    def _audit_pass(self, doc: SDRFDocument, paper_text: str) -> SDRFDocument:
        na_headers = []
        for row in doc.rows:
            row_dict = row.model_dump()
            for attr in HIGH_VALUE_ATTRS:
                if row_dict.get(attr, "not applicable") == "not applicable":
                    field = FIELD_BY_ATTR.get(attr)
                    h = field.header if field else attr
                    if h not in na_headers:
                        na_headers.append(h)

        if len(na_headers) < self.audit_threshold:
            logger.info("Audit skipped: %d N/A high-value fields (threshold=%d)", len(na_headers), self.audit_threshold)
            return doc

        logger.info("Audit: %d high-value fields still N/A, re-querying...", len(na_headers))
        current_json = json.dumps({"rows": [r.model_dump() for r in doc.rows]}, indent=2)
        human = AUDIT_PROMPT.format(
            na_list="\n".join(f"  - {h}" for h in na_headers),
            text=paper_text,
            current_json=current_json,
        )
        try:
            audited = self._parse_json(self._call_llm(SYSTEM_PROMPT, human))
            return SDRFDocument(**audited)
        except Exception as e:
            logger.warning("Audit parse failed (%s) -- keeping original.", e)
            return doc

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
        return json.loads(text)
