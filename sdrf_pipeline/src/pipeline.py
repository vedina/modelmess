"""
SDRF Extraction Pipeline — v3
Fixes context-window overflow that caused truncated JSON responses.

Key changes vs v2:
  - System prompt contains field guide ONCE; paper text never repeated across passes
  - Two-pass uses conversation history (assistant + user turns) so paper is sent once
  - Compact JSON schema (attr names only, no inline comments)
  - Robust JSON repair: attempt to recover truncated responses before giving up
  - Audit sends only the diff (N/A field names) not the full current JSON again
  - Configurable paper_max_chars to truncate very long inputs with a warning
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.models import SDRFDocument
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

# ── Compact JSON schema (sent once, attr names only) ──────────────────────────
def _build_compact_schema() -> str:
    """~800 chars instead of ~3 kB annotated version."""
    attrs = ["D", "PXD", "raw_data_file"] + [f.attr for f in FIELDS]
    placeholder = {a: "..." for a in attrs}
    placeholder["D"] = "S"
    placeholder["usage"] = "raw"
    return (
        '{"rows": [' + json.dumps(placeholder) + ', ...], '
        '"extraction_notes": "..."}'
    )

_COMPACT_SCHEMA = _build_compact_schema()

# ── Prompts ───────────────────────────────────────────────────────────────────
# Field guide lives in the system prompt ONCE — never repeated in human turns.
_FIELD_GUIDE = build_prompt_field_guide()

SYSTEM_PROMPT = f"""You are a proteomics metadata extraction expert specialising in SDRF \
(Sample and Data Relationship Format).

TASK: extract metadata from a scientific paper into a JSON object with a "rows" array \
(one row per raw MS file) and an optional "extraction_notes" string.

CORE RULES:
1. One row per raw MS file. Use all listed filenames.
2. Use "not applicable" ONLY when a field is genuinely absent — not as a default.
3. Biological replicate: infer from filename suffixes (_1_, -rep2-, _R3_) or text cues \
(triplicate, n=3).
4. Organism: latin binomial always (Homo sapiens, Mus musculus, Bos taurus …).
5. Temperature: numeric only, no units ("65" not "65°C").
6. Label: default "label free sample" when no labeling is mentioned.
7. Modifications: fixed mods first slot, variable mods in subsequent slots.
8. Factor values: the PRIMARY variable(s) that differ between conditions.
9. Usage: always "raw".
10. Return ONLY valid JSON — no markdown fences, no preamble.

JSON SCHEMA (attr names; fill every key for every row):
{_COMPACT_SCHEMA}

{_FIELD_GUIDE}"""


# Pass 1 — ask for free-text inventory (no schema, no JSON — keeps response short)
_PASS1_HUMAN = """Carefully read the paper below and list every piece of metadata you find.
One line per finding:
  FIELD: <SDRF field name> -> VALUE: <value> (SOURCE: "<exact phrase>")

Cover: organism, tissue/material, instrument, LC parameters, reagents, modifications, \
acquisition mode, experimental conditions, filenames, replicate structure.
Skip fields with no evidence. Do NOT output JSON yet.

PAPER:
{text}"""

# Pass 2 — use the inventory (already in context) to fill JSON
_PASS2_HUMAN = """Using your inventory above, now output the complete SDRF JSON.
Every value you listed MUST appear in the JSON — do not revert to "not applicable" \
for fields you already found.
Return only the JSON object, no commentary."""

# Audit — sent as a follow-up in the same conversation; paper already in context
_AUDIT_HUMAN = """These high-value fields are still "not applicable" in your JSON:
{na_list}

Re-examine the paper (already provided above) for each field.
Then return the COMPLETE corrected JSON with all rows."""


class SDRFPipeline:
    """
    Extracts SDRF metadata from proteomics paper text.

    Parameters
    ----------
    api_key         : API key (or set OPENAI_API_KEY env var)
    model           : model name, default gpt-4o
    base_url        : OpenAI-compatible endpoint; None = official OpenAI
    temperature     : LLM temperature (default 0.0)
    max_tokens      : max tokens per LLM response (default 4096)
    two_pass        : use reasoning→extraction chain (better quality)
    audit           : re-query if high-value fields remain N/A
    audit_threshold : minimum N/A count before audit fires
    paper_max_chars : truncate paper text above this length (default 12000)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        two_pass: bool = True,
        audit: bool = True,
        audit_threshold: int = 5,
        paper_max_chars: int = 12_000,
    ):
        self.two_pass = two_pass
        self.audit = audit
        self.audit_threshold = audit_threshold
        self.paper_max_chars = paper_max_chars

        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def extract(self, paper_text: str) -> SDRFDocument:
        """Full pipeline: regex hints → (reason →) extract → (audit)."""
        paper_text = self._maybe_truncate(paper_text)

        hints = build_regex_hints(paper_text)
        if hints:
            logger.debug("Regex hints found:\n%s", hints)
            # Prepend hints as a brief preamble so the model sees them first
            paper_text = hints + "\nORIGINAL PAPER TEXT:\n" + paper_text

        raw = self._two_pass_extract(paper_text) if self.two_pass \
            else self._single_pass_extract(paper_text)

        doc = SDRFDocument(**raw)
        logger.info("Extracted %d row(s)", len(doc.rows))

        if self.audit:
            doc = self._audit_pass(doc, paper_text)

        if doc.extraction_notes:
            logger.info("Extraction notes: %s", doc.extraction_notes)

        return doc

    def to_csv(self, doc: SDRFDocument, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SDRF_HEADERS, extrasaction="ignore")
            writer.writeheader()
            for row in doc.rows:
                d = row.model_dump()
                writer.writerow({h: d.get(a, "not applicable") for h, a in HEADER_TO_ATTR.items()})
        logger.info("SDRF written → %s", output_path)
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
                results.append(self.process_file(
                    txt, output_dir / txt.with_suffix(".sdrf.csv").name
                ))
            except Exception as e:
                logger.error("Failed %s: %s", txt.name, e)
        return results

    # ── Internal extraction passes ────────────────────────────────────────

    def _call_llm(self, messages: list) -> str:
        """Invoke with a pre-built message list; return text content."""
        return self.llm.invoke(messages).content

    def _single_pass_extract(self, text: str) -> dict:
        logger.info("Single-pass extraction…")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Extract SDRF metadata from this paper:\n\n{text}"),
        ]
        return self._parse_json(self._call_llm(messages))

    def _two_pass_extract(self, text: str) -> dict:
        """
        Uses a multi-turn conversation so the paper is sent ONCE.
        Turn 1: inventory (free text, short response)
        Turn 2: JSON extraction (uses inventory already in context)
        """
        logger.info("Pass 1: building metadata inventory…")
        messages: list = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_PASS1_HUMAN.format(text=text)),
        ]
        inventory = self._call_llm(messages)
        logger.debug("Inventory (%d chars):\n%s…", len(inventory), inventory[:500])

        logger.info("Pass 2: structured JSON extraction…")
        messages += [
            AIMessage(content=inventory),
            HumanMessage(content=_PASS2_HUMAN),
        ]
        raw_json = self._call_llm(messages)
        return self._parse_json(raw_json)

    def _audit_pass(self, doc: SDRFDocument, paper_text: str) -> SDRFDocument:
        """
        If too many high-value fields are N/A, send a targeted follow-up.
        We re-open the conversation with paper in context to avoid resending it.
        """
        na_headers = sorted({
            (FIELD_BY_ATTR[a].header if a in FIELD_BY_ATTR else a)
            for row in doc.rows
            for a in HIGH_VALUE_ATTRS
            if row.model_dump().get(a, "not applicable") == "not applicable"
        })

        if len(na_headers) < self.audit_threshold:
            logger.info("Audit skipped (%d N/A < threshold %d)", len(na_headers), self.audit_threshold)
            return doc

        logger.info("Audit: %d high-value fields still N/A → re-querying…", len(na_headers))

        # Reconstruct a minimal conversation: system + paper + current JSON as context
        current_json = json.dumps({"rows": [r.model_dump() for r in doc.rows]}, indent=2)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Here is the paper:\n\n{paper_text}"),
            AIMessage(content=current_json),
            HumanMessage(content=_AUDIT_HUMAN.format(
                na_list="\n".join(f"  - {h}" for h in na_headers)
            )),
        ]
        try:
            raw = self._call_llm(messages)
            return SDRFDocument(**self._parse_json(raw))
        except Exception as e:
            logger.warning("Audit failed (%s) — keeping original extraction.", e)
            return doc

    # ── Utilities ─────────────────────────────────────────────────────────

    def _maybe_truncate(self, text: str) -> str:
        if len(text) <= self.paper_max_chars:
            return text
        logger.warning(
            "Paper text truncated from %d to %d chars. "
            "Increase paper_max_chars or split the paper.",
            len(text), self.paper_max_chars,
        )
        return text[: self.paper_max_chars]

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Strip markdown fences, then try to parse.
        If truncated (common when max_tokens is hit), attempt repair by
        closing any open string and array/object brackets.
        """
        # Strip code fences
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text.strip(), flags=re.MULTILINE)
        text = text.strip()

        # Fast path — valid JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Repair path — try to close truncated JSON
        logger.warning("JSON truncated or malformed — attempting repair…")
        repaired = _repair_json(text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            # Last resort: log the tail so the user can inspect
            logger.error(
                "JSON repair failed. Last 200 chars of response:\n%s",
                text[-200:],
            )
            raise ValueError(
                "Could not parse LLM response as JSON. "
                "Try increasing max_tokens or reducing paper_max_chars."
            ) from e


def _repair_json(text: str) -> str:
    """
    Heuristic repair for truncated JSON.
    1. Close any open string literal.
    2. Strip any trailing incomplete key or comma.
    3. Use a bracket stack to close in the correct nested order.
    """
    text = text.rstrip().rstrip(',').rstrip()

    # Walk char-by-char to detect open string (handles escape sequences)
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and in_string:
            i += 2
            continue
        if c == '"':
            in_string = not in_string
        i += 1

    if in_string:
        text += '"'

    # Strip trailing comma again (may have appeared after the closed quote)
    text = text.rstrip().rstrip(',').rstrip()

    # Build a bracket stack, respecting string contents
    stack: list[str] = []
    in_str = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and in_str:
            i += 2
            continue
        if c == '"':
            in_str = not in_str
        elif not in_str:
            if c in ('{', '['):
                stack.append(c)
            elif c == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif c == ']' and stack and stack[-1] == '[':
                stack.pop()
        i += 1

    # Close openers in reverse order (innermost first)
    closing = ''.join('}' if ch == '{' else ']' for ch in reversed(stack))
    return text + closing
