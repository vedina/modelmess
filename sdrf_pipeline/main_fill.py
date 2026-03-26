#!/usr/bin/env python3
"""
SDRF Extraction — two-stage pipeline
=====================================
Stage 1 (rules_0000):  deterministic regex/keyword extraction — no API cost
Stage 2 (llm_fillgaps): LLM fills only the remaining "not applicable" fields

Usage examples
--------------
  # Single paper JSON, API key from env
  python main.py paper.json

  # Rules only (no LLM)
  python main.py paper.json --rules-only

  # Batch: all .json files in a folder
  python main.py papers/ --batch --output-dir results/

  # Custom model / base URL
  python main.py paper.json --base-url http://localhost:11434/v1 --model llama3.1:70b --api-key ollama
"""

import argparse, csv, logging, os, sys
from pathlib import Path
from typing import Optional


def build_parser():
    p = argparse.ArgumentParser(description="Extract SDRF from structured paper JSON.")
    p.add_argument("input")
    p.add_argument("--batch",       action="store_true")
    p.add_argument("--output",      default=None)
    p.add_argument("--output-dir",  default="output")
    p.add_argument("--rules-only",  action="store_true", help="Skip LLM stage")
    p.add_argument("--legacy",      action="store_true", help="Use old plain-text pipeline")
    p.add_argument("--api-key",     default=None)
    p.add_argument("--base-url",    default=None)
    p.add_argument("--model",       default="gpt-4o")
    p.add_argument("--max-tokens",  type=int, default=2048)
    p.add_argument("--no-dedup",    action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def process_one(input_path, output_path, rules_only, api_key, base_url, model, max_tokens, dedup):
    from src.rules_0000 import PaperJSON, extract_initial_sdrf
    from src.pipeline import SDRF_HEADERS, HEADER_TO_ATTR

    NA = "not applicable"
    paper   = PaperJSON.from_file(input_path)
    initial = extract_initial_sdrf(paper)

    row0     = initial.rows[0].model_dump()
    n_filled = sum(1 for v in row0.values() if v != NA)
    logging.info("Rules: %d/%d fields filled in row 0", n_filled, len(row0))

    if rules_only or not api_key:
        if not rules_only:
            logging.warning("No API key — running rules-only.")
        doc = initial
    else:
        from src.llm_fillgaps import LLMFillGaps
        filler = LLMFillGaps(api_key=api_key, base_url=base_url, model=model,
                             max_tokens=max_tokens, deduplicate=dedup)
        doc = filler.fill(paper, initial)
        n_after = sum(1 for v in doc.rows[0].model_dump().values() if v != NA)
        logging.info("After LLM: %d/%d fields filled (+%d)", n_after, len(row0), n_after - n_filled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SDRF_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for row in doc.rows:
            d = row.model_dump()
            writer.writerow({h: d.get(a, NA) for h, a in HEADER_TO_ATTR.items()})
    logging.info("Written %d rows -> %s", len(doc.rows), output_path)


def main():
    parser = build_parser()
    args   = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    if args.legacy:
        from src.pipeline import SDRFPipeline
        pl = SDRFPipeline(api_key=api_key, model=args.model,
                          base_url=args.base_url, max_tokens=args.max_tokens)
        inp = Path(args.input)
        pl.process_batch(inp, args.output_dir) if args.batch else pl.process_file(inp, args.output)
        return

    inp = Path(args.input)
    dedup = not args.no_dedup

    if args.batch:
        if not inp.is_dir():
            parser.error(f"--batch needs a directory, got: {inp}")
        files = sorted(inp.glob("*.json"))
        out_dir = Path(args.output_dir)
        for i, jf in enumerate(files, 1):
            logging.info("[%d/%d] %s", i, len(files), jf.name)
            try:
                process_one(jf, out_dir / jf.with_suffix(".sdrf.csv").name,
                            args.rules_only, api_key, args.base_url,
                            args.model, args.max_tokens, dedup)
            except Exception as e:
                logging.error("Failed %s: %s", jf.name, e)
        print(f"\n✓ {len(files)} file(s) -> {args.output_dir}/")
    else:
        if not inp.is_file():
            parser.error(f"File not found: {inp}")
        out = Path(args.output) if args.output else inp.with_suffix(".sdrf.csv")
        process_one(inp, out, args.rules_only, api_key,
                    args.base_url, args.model, args.max_tokens, dedup)
        print(f"\n✓ SDRF -> {out}")


if __name__ == "__main__":
    main()
