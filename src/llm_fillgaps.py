"""
llm_fillgaps.py  —  LLM gap-fill pass for SDRF
================================================
Takes:
  - A PaperJSON (TITLE + ABSTRACT + METHODS sections)
  - An SDRFDocument already partially filled by rules_0000.py

Fills only the fields that are still "not applicable", leaving
rule-derived values untouched.

Key design decisions vs old pipeline.py
----------------------------------------
1. The LLM never sees all 77 fields — only the ones that are N/A.
   This keeps the prompt small and focused.
2. Rules output is shown to the LLM as context ("already known values")
   so it doesn't contradict what the rules found.
3. Two-pass: free-text inventory → targeted JSON patch (not full row rewrite).
   The patch only contains the N/A fields, not all 77 per row.
4. Field descriptions come from fields.py registry — single source of truth.
5. No full-paper repeat across turns; paper sent once in turn 1.

Usage
-----
    from rules_0000 import PaperJSON, extract_initial_sdrf
    from llm_fillgaps import LLMFillGaps

    paper   = PaperJSON.from_file("PXD004010_PubText.json")
    initial = extract_initial_sdrf(paper)

    filler  = LLMFillGaps(api_key="sk-...", model="gpt-4o")
    final   = filler.fill(paper, initial)
    filler.to_csv(final, "output.sdrf.csv")
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from models import SDRFDocument, SDRFRow
from fields import FIELDS, FIELD_BY_ATTR
from prompts import PromptConfig
from rules_0000 import PaperJSON
from pipeline import SDRF_HEADERS, HEADER_TO_ATTR, _repair_json

logger = logging.getLogger(__name__)
NA = "not applicable"


def _is_empty(value) -> bool:
    """True when a field should be treated as unfilled.

    Handles all forms an empty/default field can take:
      - Python None     (Pydantic default when no value was set by rules)
      - "not applicable" (explicit N/A written by rules or a prior pass)
      - "None"          (None leaked to CSV via str() then read back as string)
      - empty string
    """
    if value is None:
        return True
    return str(value).strip().lower() in ("", "not applicable", "none", "na", "n/a")


# ── Field lookup: attr → SDRFField ────────────────────────────────────────────
_ALL_ATTRS: list[str] = [f.attr for f in FIELDS]


def _na_attrs(row: SDRFRow) -> list[str]:
    """Return attribute names that are still 'not applicable' in a row.

    'factors' is not in FIELDS (and therefore not in _ALL_ATTRS) but is
    a plain str field on SDRFRow; we add it manually so the LLM is asked
    to fill it whenever it is still 'not applicable'.
    """
    d = row.model_dump()
    na = [a for a in _ALL_ATTRS if _is_empty(d.get(a))]
    if _is_empty(d.get("factors")):
        na.append("factors")
    return na


def _known_summary(row: SDRFRow) -> str:
    """One-line summary of already-known values for a row (for LLM context)."""
    d = row.model_dump()
    pairs = [
        f"{a}={v!r}"
        for a in _ALL_ATTRS
        if not _is_empty(v := d.get(a))
    ]
    return ", ".join(pairs) if pairs else "(nothing known yet)"


# ── Field guide — only for the fields we're asking about ─────────────────────

def _mini_guide(attrs: list[str]) -> str:
    """Compact guide covering only the requested attributes."""
    lines = []
    for attr in attrs:
        f = FIELD_BY_ATTR.get(attr)
        if not f:
            continue
        ex = " | ".join(f.examples[:3]) if f.examples else ""
        lines.append(f"  {f.header}")
        lines.append(f"    {f.description}")
        if ex:
            lines.append(f"    Examples: {ex}")
    return "\n".join(lines)


# ── Token / char budget ──────────────────────────────────────────────────────
# 1 token ≈ 4 chars for English prose — good enough for a trim budget.
_CHARS_PER_TOKEN = 4


def _count_tokens_approx(text: str) -> int:
    """Fast, model-agnostic token estimate (chars / 4)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _trim_paper_to_budget(
    paper: PaperJSON,
    na_attrs: list[str],
    context_limit: Optional[int],
    max_output_tokens: int,
    system_prompt: str = "",
) -> tuple[str, str, str]:
    """
    Return (title, abstract, methods) trimmed so that pass-1 input tokens
    fit within context_limit.

    Budget
    ------
    available_for_paper = context_limit
                          - max_output_tokens        # space the model needs to reply
                          - fixed_overhead            # system + field list + guide
                          - SAFETY_MARGIN             # extra buffer for tokeniser
                                                      # approximation errors

    Title is always kept whole. Abstract and methods are trimmed
    proportionally if needed.
    """

    paper_len = len(paper.methods) + len(paper.abstract) + len(paper.title)
    logger.info(f"Chars: Paper{paper_len} = M{len(paper.methods)}+A{len(paper.abstract)}+T{len(paper.title)}")

    title    = paper.title
    abstract = paper.abstract
    methods  = paper.methods

    title_tokens    = _count_tokens_approx(title)
    abstract_tokens = _count_tokens_approx(abstract)
    methods_tokens  = _count_tokens_approx(methods)
    paper_tokens    = title_tokens + abstract_tokens + methods_tokens
    logger.info(f"Tokens: Paper{paper_tokens} = M{methods_tokens}+A{abstract_tokens}+T{title_tokens}")

    if context_limit == None:
         available = float('inf')
         return title, abstract, methods   

 # Extra headroom beyond the estimate to absorb chars/4 approximation error.
    # Set to 5% of context_limit (minimum 500 tokens).
    SAFETY_MARGIN = max(750, context_limit // 20)
    
    na_text = "\n".join(
        f"- {FIELD_BY_ATTR[a].header if a in FIELD_BY_ATTR else a}" for a in na_attrs
    )
    guide_text = _mini_guide(na_attrs)
    # Use the actual system prompt text passed in (falls back to empty string
    # which underestimates slightly, but the safety margin compensates).
    fixed_tokens = (
        _count_tokens_approx(system_prompt + na_text + guide_text)
        + 200           # template scaffolding (headers, separators, etc.)
        + SAFETY_MARGIN
    )

    available = context_limit - max_output_tokens - fixed_tokens
    logger.info(f"Context limit {context_limit} Available {available} = CT{context_limit}-MT{max_output_tokens}-FT{fixed_tokens}")

    if available <= 0:
        logger.warning(
            "Context budget too tight (limit=%d, output=%d, fixed=%d) "
            "-- sending title only.",
            context_limit, max_output_tokens, fixed_tokens,
        )
        return paper.title, "", ""
    
    if paper_tokens <= available:
        return title, abstract, methods   # fits -- no trimming needed

    # Title is always kept whole; trim abstract + methods proportionally.
    remaining = available - title_tokens
    if remaining <= 0:
        return title[:available * _CHARS_PER_TOKEN], "", ""

    ratio = remaining / (abstract_tokens + methods_tokens)
    abstract_chars = int(len(abstract) * ratio)
    methods_chars  = int(len(methods)  * ratio)

    logger.info(
        "Paper text trimmed to fit context (limit=%d tok, margin=%d): "
        "abstract %d->%d chars, methods %d->%d chars.",
        context_limit, SAFETY_MARGIN,
        len(abstract), abstract_chars,
        len(methods),  methods_chars,
    )
    return title, abstract[:abstract_chars], methods[:methods_chars]

# ── Default prompts singleton ─────────────────────────────────────────────────
# Used when LLMFillGaps is constructed without an explicit PromptConfig.
_DEFAULT_PROMPTS: PromptConfig = PromptConfig.defaults()


# ── Main class ────────────────────────────────────────────────────────────────

class LLMFillGaps:
    """
    Fills "not applicable" SDRF fields using an LLM.

    Parameters
    ----------
    api_key       : API key (or OPENAI_API_KEY env var)
    model         : model name (default gpt-4o)
    base_url      : OpenAI-compatible endpoint; None = official OpenAI
    temperature   : default 0.0
    max_tokens    : per-response limit (default 2048 — patches are small)
    deduplicate   : if True, send ONE LLM call per unique N/A pattern across
                    all rows (saves tokens when rows share the same gaps)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        context_limit: Optional[int] = None,
        deduplicate: bool = True,
        prompts: Optional[PromptConfig] = None,
        debug: bool = False,
    ):
        self.deduplicate = deduplicate
        self.max_tokens = max_tokens
        self.context_limit = context_limit
        self.prompts = prompts if prompts is not None else _DEFAULT_PROMPTS
        if base_url == "https://api.anthropic.com":
            self.llm = ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens_to_sample=max_tokens  # <--- note the rename
            )            
        else:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )        
        self.debug = debug

    # ── Public API ────────────────────────────────────────────────────────

    def fill(self, paper: PaperJSON, doc: SDRFDocument) -> SDRFDocument:
        """
        Fill N/A fields in doc using the paper. Returns a new SDRFDocument.
        """
        if self.deduplicate:
            filled_rows = self._fill_deduplicated(paper, doc)
        else:
            filled_rows = [self._fill_row(paper, row) for row in doc.rows]

        return SDRFDocument(
            rows=filled_rows,
            extraction_notes=(doc.extraction_notes or "") + " | LLM gap-fill applied.",
        )

    def to_csv(self, doc: SDRFDocument, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SDRF_HEADERS, extrasaction="ignore")
            writer.writeheader()
            for row in doc.rows:
                d = row.model_dump()
                writer.writerow({
                    h: (v if (v := d.get(a)) is not None else NA)
                    for h, a in HEADER_TO_ATTR.items()
                })
        logger.info("SDRF → %s", output_path)
        return output_path

    # ── Deduplication ─────────────────────────────────────────────────────

    def _fill_deduplicated(self, paper: PaperJSON, doc: SDRFDocument) -> list[SDRFRow]:
        """
        Group rows by their set of N/A fields. Make ONE LLM call per unique
        group, then apply the same patch to all rows in that group.

        In most datasets all rows share identical N/A fields (same instrument,
        same methods), so this reduces N API calls to 1.
        """
        # Group row indices by frozenset of N/A attrs
        groups: dict[frozenset, list[int]] = {}
        for i, row in enumerate(doc.rows):
            key = frozenset(_na_attrs(row))
            groups.setdefault(key, []).append(i)

        logger.info(
            "Deduplicated: %d row(s) → %d unique N/A pattern(s)",
            len(doc.rows), len(groups),
        )

        patched_rows = list(doc.rows)  # copy

        for na_set, indices in groups.items():
            if not na_set:
                logger.debug("Group %s: no N/A fields, skipping.", indices)
                continue

            representative = doc.rows[indices[0]]
            logger.info(
                "Filling %d field(s) for group of %d row(s)…",
                len(na_set), len(indices),
            )
            patch = self._get_patch(paper, representative, sorted(na_set))

            for idx in indices:
                patched_rows[idx] = _apply_patch(patched_rows[idx], patch)

        return patched_rows

    # ── Single-row fill ───────────────────────────────────────────────────

    def _fill_row(self, paper: PaperJSON, row: SDRFRow) -> SDRFRow:
        na = _na_attrs(row)
        if not na:
            return row
        patch = self._get_patch(paper, row, na)
        return _apply_patch(row, patch)

    # ── LLM interaction ───────────────────────────────────────────────────

    def _get_patch(
        self, paper: PaperJSON, row: SDRFRow, na_attrs: list[str]
    ) -> dict[str, str]:
        """Two-pass LLM call → returns {attr: value} patch dict."""
        known = _known_summary(row)

        # Build pass-1 message: dynamic parts (field list, guide, paper) injected here
        guide = _mini_guide(na_attrs)
        field_names = "\n".join(
            f"  - {FIELD_BY_ATTR[a].header if a in FIELD_BY_ATTR else a}"
            for a in na_attrs
        )
        title, abstract, methods = _trim_paper_to_budget(
            paper, na_attrs, self.context_limit, self.max_tokens,
            system_prompt=self.prompts.system,
        )

        pass1_text = self.prompts.render_pass1(
            field_names=field_names,
            guide=guide,
            title=title,
            abstract=abstract,
            methods=methods,
        )

        # Pass 1: free-text inventory
        messages: list = [
            SystemMessage(content=self.prompts.system),
            HumanMessage(content=pass1_text),
        ]
        inventory = self._call_llm(messages)
        logger.debug("Inventory (%d chars): %s…", len(inventory), inventory)

        # Build pass-2 message: attr list + already-known values
        attr_list = "\n".join(f"- {a}" for a in na_attrs)
        pass2_text = self.prompts.render_pass2(
            attr_list=attr_list,
            known_summary=known,
        )

        # Pass 2: JSON patch (inventory already in context, no paper re-send)
        messages += [
            AIMessage(content=inventory),
            HumanMessage(content=pass2_text),
        ]
        raw = self._call_llm(messages)
        if self.debug:
            logger.debug("RAW LLM OUTPUT:\n", raw)
        return self._parse_patch(raw, na_attrs)

    def _call_llm(self, messages: list) -> str:
        return self.llm.invoke(messages).content

    def _parse_patch(self, text: str, expected_attrs: list[str]) -> dict[str, str]:
        """Parse the JSON patch response; fall back to empty dict on failure."""
        logger.info(f"=============== Response received len={len(text)}")
        logger.debug(text)
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text.strip(), flags=re.MULTILINE)
        text = text.strip()
        try:
            patch = json.loads(text)
        except json.JSONDecodeError:
            #logger.debug(text)
            logger.warning("JSON parse failed, trying repair…")
            try:
                patch = json.loads(_repair_json(text))
            except Exception as e:
                logger.error("Patch parse failed (%s) — skipping this group.", e)
                return {}

        if not isinstance(patch, dict):
            logger.warning("LLM returned non-dict patch: %r", patch)
            return {}

        # Keep only attrs we asked for; ignore extras

        # Always include 'factors' even if not in the N/A list
        # (it is not in FIELDS so _na_attrs never adds it automatically)
        if "factors" not in expected_attrs:
            expected_attrs.append("factors")

        result: dict[str, str] = {}
        for k, v in patch.items():
            if k not in expected_attrs:
                continue
            # Normalise lists -> semicolon-delimited string
            # (the LLM sometimes returns lists for multi-value fields like
            #  factors or temperature; every SDRFRow field is now a plain str)
            if isinstance(v, list):
                v = ";".join(str(i) for i in v if i is not None and str(i).strip())
            else:
                v = str(v) if v is not None else ""
            # Skip empty / not-applicable values
            if not v or v.strip().lower() in ("not applicable", "na", "n/a"):
                continue
            result[k] = v
        logger.debug("Patch factors: %s", result.get("factors"))
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_patch(row: SDRFRow, patch: dict[str, str]) -> SDRFRow:
    """Return a new SDRFRow with patch values applied over existing N/A fields.

    All values in patch are plain strings at this point (lists were already
    joined to semicolon-delimited strings in _parse_patch).
    Only writes a value if the field is currently 'not applicable'.
    """
    if not patch:
        return row
    data = row.model_dump()
    for attr, value in patch.items():
        if _is_empty(data.get(attr)) and not _is_empty(value):
            data[attr] = value
    return SDRFRow(**data)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.llm_fillgaps <paper.json> [output.sdrf.csv]\n"
            "       API key from OPENAI_API_KEY env var.\n"
            "       Set BASE_URL env var for non-OpenAI endpoints."
        )
        sys.exit(1)

    from rules_0000 import extract_initial_sdrf

    paper   = PaperJSON.from_file(sys.argv[1])
    initial = extract_initial_sdrf(paper)

    print(f"Rules filled {sum(1 for r in initial.rows[0].model_dump().values() if r != NA)} "
          f"fields in row 0. Sending gaps to LLM…")

    api_key  = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("BASE_URL", None)
    model    = os.environ.get("LLM_MODEL", "gpt-4o")

    filler = LLMFillGaps(api_key=api_key, base_url=base_url, model=model)
    final  = filler.fill(paper, initial)

    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".json", ".sdrf.csv")
    filler.to_csv(final, out)
    print(f"Written {len(final.rows)} rows → {out}")
