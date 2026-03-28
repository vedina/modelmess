#!/usr/bin/env python3
"""
SDRF Extraction — two-stage pipeline
=====================================

Three stages, selectable via --stage:

  rules  — rule-based extraction only; writes CSVs to --rules-dir
  llm    — LLM gap-fill only; reads from --fill-from (or --rules-dir),
           writes to --llm-dir.  Requires paper JSONs for context.
  both   — runs rules then llm in sequence (default)

Typical iterative workflow
--------------------------
  # 1. Bootstrap rule output once (fast, free)
  python main_fill.py papers/ --stage rules --rules-dir output/rules

  # 2. Fill gaps with model A
  python main_fill.py papers/ --stage llm \\
      --fill-from output/rules --llm-dir output/llm_gpt4o \\
      --model gpt-4o --api-key $OPENAI_API_KEY

  # 3. Fill gaps with model B (same rules, different output folder)
  python main_fill.py papers/ --stage llm \\
      --fill-from output/rules --llm-dir output/llm_llama \\
      --model llama3.1:70b --base-url http://localhost:11434/v1 --api-key ollama

  # 4. Re-fill a subset (pass a folder of specific rule CSVs)
  python main_fill.py papers/ --stage llm \\
      --fill-from output/rules_subset --llm-dir output/llm_retry

Single-file mode (batch inferred from directory input)
------------------------------------------------------
  python main_fill.py PXD004010_PubText.json --stage both
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

NA = "not applicable"

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ── CSV ↔ SDRFDocument round-trip ─────────────────────────────────────────────

def _csv_to_sdrf_doc(csv_path: Path):
    """
    Reload a rules (or any stage) CSV back into an SDRFDocument.
    Needed so the LLM stage can consume previously written rule output.
    """
    from src.models import SDRFDocument, SDRFRow
    from src.pipeline import HEADER_TO_ATTR

    attr_to_header = {v: k for k, v in HEADER_TO_ATTR.items()}
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            # Map SDRF header names → SDRFRow attribute names
            kwargs: dict[str, str] = {}
            for header, value in csv_row.items():
                attr = HEADER_TO_ATTR.get(header)
                if attr:
                    kwargs[attr] = value if value else NA
            # SDRFRow requires raw_data_file; guard against malformed CSVs
            if "raw_data_file" not in kwargs:
                continue
            try:
                rows.append(SDRFRow(**kwargs))
            except Exception as e:
                logging.warning("Skipping malformed CSV row: %s", e)
    return SDRFDocument(rows=rows, extraction_notes="Loaded from rules CSV.")


def _write_csv(doc, output_path: Path) -> None:
    from src.pipeline import SDRF_HEADERS, HEADER_TO_ATTR

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SDRF_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for row in doc.rows:
            d = row.model_dump()
            writer.writerow({h: d.get(a, NA) for h, a in HEADER_TO_ATTR.items()})
    logging.info("Written %d rows → %s", len(doc.rows), output_path)


# ── Stage 1: rules ────────────────────────────────────────────────────────────

def run_rules(json_path: Path, rules_dir: Path):
    """
    Run rule-based extraction for one paper JSON.
    Returns the SDRFDocument (also writes it to rules_dir).
    """
    from src.rules_0000 import PaperJSON, extract_initial_sdrf

    paper = PaperJSON.from_file(json_path)
    doc = extract_initial_sdrf(paper)

    row0 = doc.rows[0].model_dump() if doc.rows else {}
    n_filled = sum(1 for v in row0.values() if v != NA)
    logging.info(
        "[rules] %s → %d rows, %d/%d fields filled in row 0",
        json_path.name, len(doc.rows), n_filled, len(row0),
    )

    out_csv = rules_dir / json_path.with_suffix(".sdrf.csv").name
    if len(doc.rows) > 0:
        _write_csv(doc, out_csv)
    else:
        logging.warning(f"Empty document skipping {out_csv}")
    return doc


# ── Stage 2: llm ─────────────────────────────────────────────────────────────

def run_llm(
    json_path: Path,
    rules_doc,           # SDRFDocument from stage 1 (or loaded from CSV)
    llm_dir: Path,
    api_key: str,
    base_url: Optional[str],
    model: str,
    max_tokens: int,
    deduplicate: bool,
) -> None:
    """
    Run LLM gap-fill for one paper JSON, given a partially-filled SDRFDocument.
    Writes result to llm_dir.
    """
    from src.rules_0000 import PaperJSON
    from src.llm_fillgaps import LLMFillGaps

    paper = PaperJSON.from_file(json_path)
    filler = LLMFillGaps(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        deduplicate=deduplicate,
    )
    final = filler.fill(paper, rules_doc)

    row0_before = rules_doc.rows[0].model_dump() if rules_doc.rows else {}
    row0_after  = final.rows[0].model_dump()     if final.rows else {}
    n_before = sum(1 for v in row0_before.values() if v != NA)
    n_after  = sum(1 for v in row0_after.values()  if v != NA)
    logging.info(
        "[llm]   %s → %d/%d fields filled in row 0 (+%d via LLM)",
        json_path.name, n_after, len(row0_after), n_after - n_before,
    )

    out_csv = llm_dir / json_path.with_suffix(".sdrf.csv").name
    _write_csv(final, out_csv)


# ── Per-file dispatcher ───────────────────────────────────────────────────────

def process_one(
    json_path: Path,
    stage: str,
    rules_dir: Path,
    llm_dir: Path,
    fill_from: Optional[Path],
    api_key: str,
    base_url: Optional[str],
    model: str,
    max_tokens: int,
    deduplicate: bool,
) -> None:
    stem_csv = json_path.with_suffix(".sdrf.csv").name

    if stage in ("rules", "both"):
        # Always run rules and write to rules_dir
        rules_doc = run_rules(json_path, rules_dir)
    else:
        # stage == "llm": load rules CSV from fill_from (or rules_dir fallback)
        search_dir = fill_from if fill_from else rules_dir
        candidate = search_dir / stem_csv
        if candidate.exists():
            logging.info("[llm] Loading rules CSV from %s", candidate)
            rules_doc = _csv_to_sdrf_doc(candidate)
        else:
            logging.warning(
                "[llm] Rules CSV not found at %s — running rules first.", candidate
            )
            rules_dir.mkdir(parents=True, exist_ok=True)
            rules_doc = run_rules(json_path, rules_dir)

    if stage in ("llm", "both"):
        if not api_key:
            logging.warning("No API key — skipping LLM stage for %s.", json_path.name)
            return
        run_llm(
            json_path, rules_doc, llm_dir,
            api_key, base_url, model, max_tokens, deduplicate,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SDRF extraction — rules + LLM gap-fill pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ──────────────────────────────────────────────────────────────
    p.add_argument(
        "input",
        help=(
            "Paper JSON source. Three forms accepted:\n"
            "  directory/          → all *.json files in that folder\n"
            "  directory/PXD*.json → glob pattern inside a folder\n"
            "  path/to/file.json   → single file"
        ),
    )

    # ── Stage ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--stage",
        choices=["rules", "llm", "both"],
        default="both",
        help="Which stage(s) to run (default: both).",
    )
    # Backwards-compat alias
    p.add_argument(
        "--rules-only",
        action="store_true",
        help="Alias for --stage rules.",
    )

    # ── Output folders ─────────────────────────────────────────────────────
    p.add_argument(
        "--rules-dir",
        default="output/rules",
        help="Folder for rule-based CSVs (default: output/rules).",
    )
    p.add_argument(
        "--llm-dir",
        default="output/llm",
        help="Folder for LLM-filled CSVs (default: output/llm).",
    )
    p.add_argument(
        "--fill-from",
        default=None,
        metavar="DIR",
        help=(
            "For --stage llm: folder of pre-computed rule CSVs to use as input. "
            "If a file is missing, rules are re-run automatically. "
            "Defaults to --rules-dir."
        ),
    )

    # ── LLM config ─────────────────────────────────────────────────────────
    p.add_argument("--api-key",    default=None,          help="LLM API key (default: OPENAI_API_KEY env).")
    p.add_argument("--base-url",   default=None,          help="OpenAI-compatible base URL.")
    p.add_argument("--model",      default="gpt-4o-mini", help="Model name (default: gpt-4o-mini).")
    p.add_argument("--max-tokens", type=int, default=8192, help="Max LLM response tokens (default: 8192).")
    p.add_argument("--no-dedup",   action="store_true",   help="Disable row deduplication (one LLM call/row).")

    # ── Misc ───────────────────────────────────────────────────────────────
    p.add_argument("--verbose", "-v", action="store_true")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    # Backwards compat: --rules-only → --stage rules
    stage = "rules" if args.rules_only else args.stage

    api_key   = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    rules_dir = Path(args.rules_dir)
    llm_dir   = Path(args.llm_dir)
    fill_from = Path(args.fill_from) if args.fill_from else None
    deduplicate = not args.no_dedup

    # Ensure output dirs exist
    if stage in ("rules", "both"):
        rules_dir.mkdir(parents=True, exist_ok=True)
    if stage in ("llm", "both"):
        llm_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)

    # ── Collect files ──────────────────────────────────────────────────────
    inp_str = args.input
    if any(c in inp_str for c in ("*", "?", "[")):
        # Glob pattern: split into parent dir + pattern
        # e.g. "papers/PXD*.json" → parent="papers/", pattern="PXD*.json"
        inp_path = Path(inp_str)
        parent = inp_path.parent
        pattern = inp_path.name
        if not parent.is_dir():
            parser.error(f"Directory not found: {parent}")
        json_files = sorted(parent.glob(pattern))
        if not json_files:
            parser.error(f"No files matched pattern '{pattern}' in {parent}")
        logging.info(
            "Glob '%s' in %s → %d file(s)", pattern, parent, len(json_files)
        )
    elif inp.is_dir():
        json_files = sorted(inp.glob("*.json"))
        if not json_files:
            parser.error(f"No .json files found in {inp}")
        logging.info("Found %d paper JSON file(s) in %s", len(json_files), inp)
    elif inp.is_file():
        json_files = [inp]
    else:
        parser.error(f"Input not found: {inp}")

    # ── Process ────────────────────────────────────────────────────────────
    ok = 0
    for i, jf in enumerate(json_files, 1):
        logging.info("── [%d/%d] %s ──", i, len(json_files), jf.name)
        try:
            process_one(
                json_path=jf,
                stage=stage,
                rules_dir=rules_dir,
                llm_dir=llm_dir,
                fill_from=fill_from,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                deduplicate=deduplicate,
            )
            ok += 1
        except Exception as e:
            logging.error("FAILED %s: %s", jf.name, e, exc_info=args.verbose)

    # ── Summary ────────────────────────────────────────────────────────────
    total = len(json_files)
    print(f"\n✓ {ok}/{total} file(s) processed.")
    if stage in ("rules", "both"):
        print(f"  Rules CSVs → {rules_dir}/")
    if stage in ("llm", "both"):
        print(f"  LLM CSVs   → {llm_dir}/")
    if ok < total:
        print(f"  {total - ok} file(s) failed — check logs above.")


if __name__ == "__main__":
    main()
