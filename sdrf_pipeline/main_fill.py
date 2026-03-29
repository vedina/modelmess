#!/usr/bin/env python3
"""
SDRF Extraction -- two-stage pipeline
=====================================

Three stages, selectable via --stage:

  rules  -- rule-based extraction only; writes CSVs to --rules-dir
  llm    -- LLM gap-fill only; reads from --fill-from (or --rules-dir),
            writes to --llm-dir.  Requires paper JSONs for context.
  both   -- runs rules then llm in sequence (default)

Typical iterative workflow
--------------------------
  # 1. Bootstrap rule output once (fast, free)
  python main_fill.py papers/ --stage rules --rules-dir output/rules

  # 2. Dump default prompts to a file, then edit them
  python main_fill.py papers/ --dump-prompts my_prompts.toml

  # 3. Fill gaps with model A, custom prompts
  python main_fill.py papers/ --stage llm \\
      --fill-from output/rules --llm-dir output/llm_gpt4o \\
      --model gpt-4o --api-key $OPENAI_API_KEY \\
      --prompts my_prompts.toml

  # 4. Fill gaps with model B (same rules, different output folder)
  python main_fill.py papers/ --stage llm \\
      --fill-from output/rules --llm-dir output/llm_llama \\
      --model llama3.1:70b --base-url http://localhost:11434/v1 --api-key ollama

  # 5. Re-fill a subset (pass a folder of specific rule CSVs)
  python main_fill.py papers/ --stage llm \\
      --fill-from output/rules_subset --llm-dir output/llm_retry
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import json


NA = "not applicable"


# -- Logging ------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# -- CSV <-> SDRFDocument round-trip ------------------------------------------

def _csv_to_sdrf_doc(csv_path: Path):
    """
    Reload a rules (or any stage) CSV back into an SDRFDocument.
    Needed so the LLM stage can consume previously written rule output.
    """
    from src.models import SDRFDocument, SDRFRow
    from src.pipeline import HEADER_TO_ATTR

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            kwargs: dict[str, str] = {}
            for header, value in csv_row.items():
                attr = HEADER_TO_ATTR.get(header)
                if attr:
                    kwargs[attr] = value if value else NA
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
    logging.info("Written %d rows -> %s", len(doc.rows), output_path)


# -- Stage 1: rules -----------------------------------------------------------

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
        "[rules] %s -> %d rows, %d/%d fields filled in row 0",
        json_path.name, len(doc.rows), n_filled, len(row0),
    )

    out_csv = rules_dir / json_path.with_suffix(".sdrf.csv").name
    if doc.rows:
        _write_csv(doc, out_csv)
    else:
        logging.warning("Empty document, skipping %s", out_csv)
    return doc


# -- Stage 2: llm -------------------------------------------------------------

def run_llm(
    json_path: Path,
    rules_doc,
    llm_dir: Path,
    api_key: str,
    base_url: Optional[str],
    model: str,
    max_tokens: int,
    context_limit: int,
    deduplicate: bool,
    prompts,            # PromptConfig or None
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
        context_limit=context_limit,
        deduplicate=deduplicate,
        prompts=prompts,
    )
    final = filler.fill(paper, rules_doc)

    row0_before = rules_doc.rows[0].model_dump() if rules_doc.rows else {}
    row0_after  = final.rows[0].model_dump()     if final.rows else {}
    n_before = sum(1 for v in row0_before.values() if v != NA)
    n_after  = sum(1 for v in row0_after.values()  if v != NA)
    logging.info(
        "[llm]   %s -> %d/%d fields filled in row 0 (+%d via LLM)",
        json_path.name, n_after, len(row0_after), n_after - n_before,
    )

    out_csv = llm_dir / json_path.with_suffix(".sdrf.csv").name
    _write_csv(final, out_csv)

    final_dict = final.model_dump(mode="python") 
    with open(str(out_csv).replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)
    


# -- Per-file dispatcher ------------------------------------------------------

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
    context_limit: int,
    deduplicate: bool,
    prompts,            # PromptConfig or None
) -> None:
    stem_csv = json_path.with_suffix(".sdrf.csv").name

    if stage in ("rules", "both"):
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
                "[llm] Rules CSV not found at %s -- running rules first.", candidate
            )
            rules_dir.mkdir(parents=True, exist_ok=True)
            rules_doc = run_rules(json_path, rules_dir)

    if stage in ("llm", "both"):
        if not api_key:
            logging.warning("No API key -- skipping LLM stage for %s.", json_path.name)
            return
        run_llm(
            json_path, rules_doc, llm_dir,
            api_key, base_url, model, max_tokens, context_limit, deduplicate,
            prompts,
        )


# -- CLI ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SDRF extraction -- rules + LLM gap-fill pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Paper JSON source. Two forms:\n"
            "  directory/        -> all *.json files in that folder\n"
            "  path/to/file.json -> single file\n"
            "Use --pattern to filter files inside a directory.\n"
            "Not required when using --dump-prompts."
        ),
    )
    p.add_argument(
        "--pattern",
        default="*.json",
        metavar="GLOB",
        help="Glob pattern inside the input directory (default: *.json). "
             "Example: --pattern 'PXD06*.json'",
    )

    # Stage
    p.add_argument(
        "--stage",
        choices=["rules", "llm", "both"],
        default="both",
        help="Which stage(s) to run (default: both).",
    )
    p.add_argument(
        "--rules-only",
        action="store_true",
        help="Alias for --stage rules (backwards compat).",
    )

    # Output folders
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
        help="For --stage llm: folder of pre-computed rule CSVs to use as input. "
             "Falls back to --rules-dir if a file is missing.",
    )

    # LLM config
    p.add_argument("--api-key",    default=None,           help="LLM API key (default: OPENAI_API_KEY env).")
    p.add_argument("--base-url",   default=None,           help="OpenAI-compatible base URL.")
    p.add_argument("--model",      default="gpt-4o-mini",  help="Model name (default: gpt-4o-mini).")
    p.add_argument("--max-tokens", type=int, default=8192, help="Max LLM response tokens (default: 8192).")
    p.add_argument(
        "--context-limit",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Context window size in tokens for trimming paper text "
             "(default: None). Examples: 8192 for older models, 128000 for gpt-4o.",
    )
    p.add_argument("--no-dedup", action="store_true", help="One LLM call per row (disables deduplication).")

    # Prompts
    p.add_argument(
        "--prompts",
        default=None,
        metavar="TOML",
        help="Path to a TOML file overriding prompt templates. "
             "Missing sections fall back to built-in defaults. "
             "Use --dump-prompts to generate a starting file.",
    )
    p.add_argument(
        "--dump-prompts",
        default=None,
        metavar="TOML",
        help="Write the current default prompts to TOML and exit. "
             "Example: --dump-prompts prompts.toml",
    )

    # Misc
    p.add_argument("--verbose", "-v", action="store_true")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    # --dump-prompts: write defaults to file and exit
    if args.dump_prompts:
        from src.prompts import PromptConfig
        out = PromptConfig.defaults().to_toml(args.dump_prompts)
        print(f"Default prompts written to: {out}")
        sys.exit(0)

    # Backwards compat
    stage = "rules" if args.rules_only else args.stage

    # Load prompt config (None -> LLMFillGaps uses defaults internally)
    prompts = None
    if args.prompts:
        from src.prompts import PromptConfig
        prompts = PromptConfig.from_toml(args.prompts)
        logging.info("Loaded prompts from %s", args.prompts)

    api_key     = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    rules_dir   = Path(args.rules_dir)
    llm_dir     = Path(args.llm_dir)
    fill_from   = Path(args.fill_from) if args.fill_from else None
    deduplicate = not args.no_dedup

    if args.input is None:
        parser.error("input is required (unless using --dump-prompts)")

    # Ensure output dirs exist
    if stage in ("rules", "both"):
        rules_dir.mkdir(parents=True, exist_ok=True)
    if stage in ("llm", "both"):
        llm_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)

    # Collect files
    if inp.is_dir():
        pattern = args.pattern
        json_files = sorted(inp.glob(pattern))
        if not json_files:
            parser.error(f"No files matched '{pattern}' in {inp}")
        logging.info("Pattern '%s' in %s -> %d file(s)", pattern, inp, len(json_files))
    elif inp.is_file():
        json_files = [inp]
    else:
        parser.error(f"Input not found: {inp}")

    # Process
    ok = 0
    for i, jf in enumerate(json_files, 1):
        logging.info("-- [%d/%d] %s --", i, len(json_files), jf.name)
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
                context_limit=args.context_limit,
                deduplicate=deduplicate,
                prompts=prompts,
            )
            ok += 1
        except Exception as e:
            logging.error("FAILED %s: %s", jf.name, e, exc_info=args.verbose)

    # Summary
    total = len(json_files)
    print(f"\n{ok}/{total} file(s) processed.")
    if stage in ("rules", "both"):
        print(f"  Rules CSVs -> {rules_dir}/")
    if stage in ("llm", "both"):
        print(f"  LLM CSVs   -> {llm_dir}/")
    if ok < total:
        print(f"  {total - ok} file(s) failed -- check logs above.")


if __name__ == "__main__":
    main()
