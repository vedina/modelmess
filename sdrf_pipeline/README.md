# SDRF Extraction Pipeline

The pipeline converts scientific proteomics publications into **SDRF (Sample and Data Relationship Format)** metadata files, which describe the experimental samples, instruments, and conditions behind mass spectrometry datasets in a standardised, machine-readable way. It works in two stages: 

  1. A fast rule-based pass reads the structured paper JSON (title, abstract, and methods section) and uses regex patterns and keyword lookups to deterministically fill fields it can extract with high confidence — organism, tissue, labelling strategy, instrument model, cleavage agent, modifications, and others — writing one row per raw data file (with one row per TMT/iTRAQ channel when multiplexed).
  2. Second, an LLM gap-fill pass takes the partially completed SDRF, identifies only the fields still marked "not applicable", and sends those fields together with the relevant paper text to a configurable language model, which returns a targeted patch without touching values the rules already set.

The two stages write to separate output folders so rule output is never overwritten, and the LLM stage can be re-run independently against the same rule baseline using different models or prompts — making the pipeline both deterministic at its core and iteratively improvable at the edges.

---

## Architecture

```
paper_text (.txt)
      │
      ▼
┌─────────────────────────────────────────┐
│  LangChain ChatPromptTemplate           │
│  (system + human extraction prompts)    │
└────────────────────┬────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  ChatOpenAI           │  ← any OpenAI-compatible API
         │  (configurable model, │    (OpenAI, Ollama, vLLM,
         │   base_url, api_key)  │     Azure, OpenRouter …)
         └───────────┬───────────┘
                     │  JSON response
                     ▼
         ┌───────────────────────┐
         │  JsonOutputParser     │
         │  + Pydantic           │  ← typed validation / defaults
         │  SDRFDocument model   │
         └───────────┬───────────┘
                     │
                     ▼
            SDRF CSV output
```

**Why LangChain?**
- `ChatOpenAI` supports `base_url` out of the box → works with Ollama, vLLM,
  Azure OpenAI, OpenRouter, Together AI, and any other OpenAI-compatible server
  without any code changes.
- `ChatPromptTemplate` cleanly separates prompt logic from pipeline code.
- `JsonOutputParser` + Pydantic gives structured, validated output with
  meaningful defaults (`"not applicable"`) so the CSV is always well-formed.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```
# One-time bootstrap (fast, no API cost)
python main_fill.py papers/ --stage rules --rules-dir output/rules

# Fill with gpt-4o
python main_fill.py papers/ --stage llm \
    --fill-from output/rules --llm-dir output/llm_gpt4o \
    --model gpt-4o --api-key $OPENAI_API_KEY

# Fill same rules with a different model — rules unchanged
python main_fill.py papers/ --stage llm \
    --fill-from output/rules --llm-dir output/llm_llama \
    --model llama3.1:70b --base-url http://localhost:11434/v1 --api-key ollama

# Default (both stages, legacy behaviour)
python main_fill.py papers/ --api-key $OPENAI_API_KEY

# Single file
python main_fill.py PXD004010_PubText.json --stage rules --rules-dir output/rules
```

### 4. Use as a Python library

TBD update

```python

```

---

## CLI Options

```
usage: main_fill.py [-h] [--stage {rules,llm,both}] [--rules-only] [--rules-dir RULES_DIR] [--llm-dir LLM_DIR] [--fill-from DIR] [--api-key API_KEY]
                    [--base-url BASE_URL] [--model MODEL] [--max-tokens MAX_TOKENS] [--no-dedup] [--verbose]
                    input

SDRF extraction — rules + LLM gap-fill pipeline.

positional arguments:
  input                 Path to a single paper JSON or a directory of paper JSONs.

options:
  -h, --help            show this help message and exit
  --stage {rules,llm,both}
                        Which stage(s) to run (default: both).
  --rules-only          Alias for --stage rules.
  --rules-dir RULES_DIR
                        Folder for rule-based CSVs (default: output/rules).
  --llm-dir LLM_DIR     Folder for LLM-filled CSVs (default: output/llm).
  --fill-from DIR       For --stage llm: folder of pre-computed rule CSVs to use as input. If a file is missing, rules are re-run automatically. Defaults to   
                        --rules-dir.
  --api-key API_KEY     LLM API key (default: OPENAI_API_KEY env).
  --base-url BASE_URL   OpenAI-compatible base URL.
  --model MODEL         Model name (default: gpt-4o-mini).
  --max-tokens MAX_TOKENS
                        Max LLM response tokens (default: 8192).
  --no-dedup            Disable row deduplication (one LLM call/row).
  --verbose, -v

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
  python main_fill.py papers/ --stage llm \
      --fill-from output/rules --llm-dir output/llm_gpt4o \
      --model gpt-4o --api-key $OPENAI_API_KEY

  # 3. Fill gaps with model B (same rules, different output folder)
  python main_fill.py papers/ --stage llm \
      --fill-from output/rules --llm-dir output/llm_llama \
      --model llama3.1:70b --base-url http://localhost:11434/v1 --api-key ollama

  # 4. Re-fill a subset (pass a folder of specific rule CSVs)
  python main_fill.py papers/ --stage llm \
      --fill-from output/rules_subset --llm-dir output/llm_retry

Single-file mode (batch inferred from directory input)
------------------------------------------------------
  python main_fill.py PXD004010_PubText.json --stage both
---

```

## Project Structure

```text
sdrf_pipeline/
├── main_fill.py              # CLI entrypoint
├── requirements.txt
├── src/
│   ├── models.py        # Pydantic SDRFRow + SDRFDocument models
│   └── pipeline.py      # LangChain extraction chain + CSV writer
├── examples/
│   └── paper.txt        # Example input
└── output/              # Default output directory
```

---

## Output format

The output CSV matches the SDRF format in this [Kaggle competition](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data)  with all required columns. Fields not extractable from the paper text are filled with `"not applicable"`. 

One row is written per sample. The "rules" phase infers the samples based on file names parsing.

```csv
ID,PXD,Raw Data File,Characteristics[Age],...,Usage
1,PXD016436,file1.raw,not applicable,...,raw
2,PXD016436,file2.raw,not applicable,...,raw
```

The pipeline does not merge the SDRF files (as required for Kaggle submission).  There is a separate Jupyter notebook which merges, normalizes (using keyword maps and OLS) and tries to evaluate the submission.
This is intentional, as the pipeline emerges as a generic solution, and submission format, normalisation and scoring are competition specific.
