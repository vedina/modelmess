#!/usr/bin/env python3
"""
SDRF Extraction CLI
====================
Usage examples:

  # Single file, API key from env
  python main.py paper.txt

  # Single file with explicit output path
  python main.py paper.txt --output results/my_dataset.sdrf.csv

  # Batch: all .txt files in a folder
  python main.py papers/ --batch --output-dir results/

  # Custom model / base URL (e.g. Ollama, Together AI, Azure, local vLLM)
  python main.py paper.txt \\
    --base-url http://localhost:11434/v1 \\
    --model llama3.1:70b \\
    --api-key ollama

  # OpenRouter example
  python main.py paper.txt \\
    --base-url https://openrouter.ai/api/v1 \\
    --model anthropic/claude-3.5-sonnet \\
    --api-key $OPENROUTER_API_KEY
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.pipeline import SDRFPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract proteomics SDRF metadata from paper text using an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    p.add_argument(
        "input",
        help="Path to a .txt file (single mode) or directory of .txt files (--batch mode).",
    )

    # Mode
    p.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Process all .txt files in the input directory.",
    )

    # Output
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (single mode). Defaults to <input>.sdrf.csv.",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for batch mode (default: ./output/).",
    )

    # API config
    p.add_argument(
        "--api-key",
        default=None,
        help="API key. Falls back to OPENAI_API_KEY env var.",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help=(
            "Base URL for an OpenAI-compatible API. "
            "Examples: http://localhost:11434/v1 (Ollama), "
            "https://openrouter.ai/api/v1 (OpenRouter), "
            "https://<resource>.openai.azure.com/ (Azure). "
            "Leave unset for the official OpenAI API."
        ),
    )
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name (default: gpt-4o).",
    )

    # LLM tuning
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0 for deterministic extraction).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens in LLM response (default: 8192).",
    )

    # Logging
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error(
            "No API key provided. Use --api-key or set OPENAI_API_KEY env var."
        )

    pipeline = SDRFPipeline(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    input_path = Path(args.input)

    if args.batch:
        if not input_path.is_dir():
            parser.error(f"--batch requires a directory, got: {input_path}")
        outputs = pipeline.process_batch(input_path, args.output_dir)
        print(f"\n✓ Processed {len(outputs)} file(s) → {args.output_dir}/")
        for o in outputs:
            print(f"  {o}")

    else:
        if not input_path.is_file():
            parser.error(f"File not found: {input_path}")
        out = pipeline.process_file(input_path, args.output)
        print(f"\n✓ SDRF written to: {out}")


if __name__ == "__main__":
    main()
