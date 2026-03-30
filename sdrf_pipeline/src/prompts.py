"""
prompts.py  —  Prompt configuration for LLMFillGaps
=====================================================
Holds the three prompt templates used in the two-pass LLM extraction:

  system    — behaviour rules sent as the system message
  pass1     — "find these fields in this paper" (human turn 1)
  pass2     — "now output the JSON patch"       (human turn 2)

Pass1 and pass2 are Python format strings.  The following placeholders
are injected automatically at call time — do not remove them:

  pass1 placeholders:
    {field_names}   — bullet list of SDRF headers being requested
    {guide}         — field-by-field descriptions and examples
    {title}         — paper title (may be trimmed to fit context)
    {abstract}      — paper abstract (may be trimmed)
    {methods}       — paper methods section (may be trimmed)

  pass2 placeholders:
    {attr_list}     — snake_case attribute names being requested
    {known_summary} — already-filled values (for context / no-overwrite)

Loading
-------
  from src.prompts import PromptConfig

  # Built-in defaults (no file needed)
  cfg = PromptConfig.defaults()

  # From a TOML file (only sections present in the file override defaults)
  cfg = PromptConfig.from_toml("my_prompts.toml")

  # Dump current defaults to a TOML file as a starting point
  PromptConfig.defaults().to_toml("prompts.toml")
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ── Default prompt text ───────────────────────────────────────────────────────

_DEFAULT_SYSTEM = """\
You are a proteomics SDRF metadata extraction assistant.

You are given ONE experiment with multiple samples (rows).

Your task is to fill missing fields ('not applicable') across ALL rows.

You must:
- maintain consistency across the experiment
- assign sample-specific values correctly
- avoid contradictions

RULES:

ATTRIBUTE RULES:
- Use ONLY provided snake_case attribute names
- Do NOT invent fields

VALUE RULES:
- If value is "not applicable" → fill it if possible
- If value exists → DO NOT overwrite

EVIDENCE RULES:
- Primary source: manuscript text
- Secondary: filenames (ONLY for per-sample mapping)

FACTOR VARIABLE RULES:

A factor is a variable intentionally varied to test its effect.
Ask: Did researchers deliberately change this across conditions?

Set "factors" to semicolon-separated true factors.
Set fv_* fields only if that variable is the tested factor.

True factors (manipulated):

Temperature series → fv_temperature
Drug/dose variation → fv_compound, fv_concentration_of_compound
Knockout vs wildtype → fv_genetic_modification
Disease vs healthy (explicit comparison) → fv_disease
Bait swap (AP-MS) → fv_bait

Not factors (descriptive, not manipulated):

Demographics (age, sex, BMI)
Cell line (unless compared)
Organism/sample type
Instruments, settings, reagents
Disease status used only for cohort selection

Key test: Could the study be titled "Effect of X on…" or "X vs Y"?
→ Yes = factor; No = characteristics.

If unsure: set fv_* = "not applicable" and exclude from "factors".
Do not infer factors from filenames alone.

FILENAME RULES:
- Use parsed filenames ONLY to:
    • distinguish samples
    • assign replicate numbers
    • map known conditions to correct rows

Return JSON only.\
"""

_DEFAULT_PASS1 = """\
Read the paper sections below and find values for these MISSING fields:

{field_names}

FIELD GUIDE (what each field means and example values):
{guide}

For each field you find evidence for, write ONE line:
  FIELD: <field name> -> VALUE: <value>  (SOURCE: "<exact phrase>")

SPECIAL RULE for FactorValue[*] fields and "factors":
Only fill these if the paper explicitly frames the variable as the thing being compared
or tested. If in doubt, skip them — a missing factor is better than a wrong one

Skip fields with no evidence. Do NOT output JSON yet.

--- TITLE ---
{title}

--- ABSTRACT ---
{abstract}

--- METHODS ---
{methods}\
"""

_DEFAULT_PASS2 = """\
Based on your findings above, output a JSON object.

You MUST use EXACT attribute keys (snake_case) listed below:

{attr_list}

Rules:
- Use ONLY these keys (do not invent new ones)
- If no evidence was found for a field, OMIT it from the JSON entirely
- Do NOT include keys with "not applicable" or null values.
- Already-known values — DO NOT OVERWRITE. You may APPEND maximum 5 UNIQUE valid values using ";".
  {known_summary}

- For "factors" and all "fv_*" keys: only include if the paper explicitly
states this variable was deliberately varied to test its effect.
When in doubt, omit rather than guess.

Return JSON only, no commentary.

Example:
{{
  "instrument": "Orbitrap Fusion",
  "fragmentation_method": "HCD",
  "factors": ["time"]
}}\
"""


# ── PromptConfig ──────────────────────────────────────────────────────────────

@dataclass
class PromptConfig:
    """
    The three prompt templates used by LLMFillGaps.

    All three fields are plain strings. Pass1 and pass2 are Python
    format strings with named placeholders (see module docstring).
    """
    system: str = field(default_factory=lambda: _DEFAULT_SYSTEM)
    pass1:  str = field(default_factory=lambda: _DEFAULT_PASS1)
    pass2:  str = field(default_factory=lambda: _DEFAULT_PASS2)

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def defaults(cls) -> "PromptConfig":
        """Return a config populated with the built-in default prompts."""
        return cls(
            system=_DEFAULT_SYSTEM,
            pass1=_DEFAULT_PASS1,
            pass2=_DEFAULT_PASS2,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "PromptConfig":
        """
        Load prompt overrides from a TOML file.

        Only sections present in the file override defaults — missing
        sections keep the built-in text.  This means you can write a
        TOML with just [pass1] to change only the first-pass prompt.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        defaults = cls.defaults()
        return cls(
            system=data.get("system", {}).get("text", defaults.system),
            pass1 =data.get("pass1",  {}).get("template", defaults.pass1),
            pass2 =data.get("pass2",  {}).get("template", defaults.pass2),
        )

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_toml(self, path: str | Path) -> Path:
        """
        Write this config to a TOML file.

        Running this on the defaults produces a fully-annotated starting
        point that can be edited freely.
        """
        path = Path(path)

        # TOML multiline strings use triple-quotes; escape any that appear
        # in the content (unlikely, but safe).
        def _toml_str(s: str) -> str:
            return s.replace('"""', '""\\"')

        lines = [
            "# SDRF pipeline prompt configuration",
            "# Generated by PromptConfig.to_toml()",
            "#",
            "# pass1 placeholders: {field_names} {guide} {title} {abstract} {methods}",
            "# pass2 placeholders: {attr_list} {known_summary}",
            "",
            "[system]",
            f'text = """',
            _toml_str(self.system),
            '"""',
            "",
            "[pass1]",
            f'template = """',
            _toml_str(self.pass1),
            '"""',
            "",
            "[pass2]",
            f'template = """',
            _toml_str(self.pass2),
            '"""',
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ── Rendering helpers (used by llm_fillgaps) ──────────────────────────

    def render_pass1(
        self,
        field_names: str,
        guide: str,
        title: str,
        abstract: str,
        methods: str,
    ) -> str:
        return self.pass1.format(
            field_names=field_names,
            guide=guide,
            title=title,
            abstract=abstract,
            methods=methods,
        )

    def render_pass2(self, attr_list: str, known_summary: str) -> str:
        return self.pass2.format(
            attr_list=attr_list,
            known_summary=known_summary,
        )
