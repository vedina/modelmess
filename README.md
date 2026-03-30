# kaggle_sdrfmess — SDRF Extraction Pipeline

Rule-based and LLM gap-fill pipeline for extracting proteomics
[SDRF](https://github.com/bigbio/proteomics-sample-metadata) metadata from
structured paper JSON. Used as the extraction backend for the
[Harmonizing the Data of Your Data](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data)
Kaggle competition.

---

## Method

### Stage 1 — Rule-based extraction (fast, free)

A deterministic pass over the paper's title, abstract, and methods section.
Regex patterns and keyword lookups fill every field that can be extracted
reliably without ambiguity: organism, organism part, labelling strategy
(TMT/iTRAQ/SILAC/label-free, with correct channel count), instrument model,
cleavage agent, fragmentation method, acquisition mode, LC parameters,
missed cleavages, precursor/fragment tolerances, reduction and alkylation
reagents, and modifications. One SDRF row is produced per raw data file,
with one row per isobaric channel for multiplexed experiments.

Rule output is written to a dedicated folder and never overwritten by
subsequent stages.

### Stage 2 — LLM gap-fill

An LLM fills only the fields still marked `Not Applicable` after the rule
pass. The paper text is trimmed to fit the model's context window, and the
fill uses a two-turn conversation:

1. **Pass 1** — free-text inventory: the model reads the paper and lists
   evidence for each missing field.
2. **Pass 2** — JSON patch: the model converts its inventory into a targeted
   key-value patch, constrained to the exact set of N/A attributes and
   prohibited from overwriting rule-derived values.

Rows sharing identical N/A patterns are deduplicated into a single LLM call,
so most datasets incur one API call regardless of file count.

Prompts are fully configurable via TOML (`--prompts`); defaults are dumped
with `--dump-prompts`.

### Post-processing

The submission script (`make_submission.py`) combines per-PXD SDRF CSVs,
aligns columns to the sample submission schema, assigns sequential IDs,
and applies CV/ontology normalisation via the
[`sdrf-pipelines`](https://github.com/bigbio/sdrf-pipelines) OLS lookup.
A separate merge utility (`comparison.py`) reconciles outputs from multiple
models using per-column consensus strategies.

---

## Key design decisions

- **Rules-first**: deterministic extraction is free and fast; the LLM only
  sees what rules could not resolve, keeping prompts focused and token costs low.
- **Separate output folders per stage**: rules and LLM outputs are never
  mixed, enabling independent re-runs with different models or prompts.
- **Model-agnostic**: any OpenAI-compatible endpoint works — OpenAI, Anthropic,
  Ollama, vLLM, OpenRouter — via `--base-url`.
- **Context safety**: paper text is trimmed proportionally (title preserved,
  abstract and methods trimmed to a configurable token budget) with a 5%
  safety margin to absorb tokeniser approximation errors.

---

## Installation

Requires Python ≥ 3.13.

```bash
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv sync
```

To use this as a dependency from another project:

```
kaggle_sdrfmess @ git+https://github.com/vedina/modelmess.git@main#subdirectory=sdrf_pipeline
```

---

## Usage

```bash
# Stage 1: rule-based extraction
python -m main_fill papers/ --stage rules --rules-dir output/rules

# Stage 2: LLM gap-fill (OpenAI)
python -m main_fill papers/ --stage llm \
    --fill-from output/rules --llm-dir output/llm \
    --model gpt-4o --api-key $OPENAI_API_KEY

# Both stages in one pass
python -m main_fill papers/ --model gpt-4o --api-key $OPENAI_API_KEY

# Local model via Ollama
python -m main_fill papers/ --stage llm \
    --fill-from output/rules --llm-dir output/llm_llama \
    --model llama3.1:70b \
    --base-url http://localhost:11434/v1 --api-key ollama

# Dump default prompts for editing
python -m main_fill --dump-prompts prompts.toml
```

Full CLI reference: `python -m main_fill --help`

---

## References

- Deutsch EW et al. *A proteomics sample metadata representation for multiomics
  integration and big data analysis.* Nature Communications, 2020.
- [SDRF-Proteomics specification](https://github.com/bigbio/proteomics-sample-metadata)
- [sdrf-pipelines](https://github.com/bigbio/sdrf-pipelines) — OLS ontology
  normalisation used in post-processing
- [LangChain](https://github.com/langchain-ai/langchain) — LLM orchestration
