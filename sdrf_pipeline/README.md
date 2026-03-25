# SDRF Extraction Pipeline

Automatically extracts proteomics metadata from scientific paper text and
outputs a valid **SDRF (Sample and Data Relationship Format)** CSV using any
OpenAI-compatible LLM.

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
- No need for vector DBs, indexing, or retrieval — LangChain's lightweight
  chain API is exactly the right level of abstraction here.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Single file

```bash
# Official OpenAI (key from env)
export OPENAI_API_KEY=sk-...
python main.py paper.txt

# Explicit output path
python main.py paper.txt --output results/PXD016436.sdrf.csv
```

### 2. Batch (directory of .txt files)

```bash
python main.py papers/ --batch --output-dir results/
```

### 3. Custom API endpoint

```bash
# Ollama (local)
python main.py paper.txt \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:70b \
  --api-key ollama

# vLLM (local server)
python main.py paper.txt \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --api-key token-abc123

# OpenRouter
python main.py paper.txt \
  --base-url https://openrouter.ai/api/v1 \
  --model anthropic/claude-3.5-sonnet \
  --api-key $OPENROUTER_API_KEY

# Azure OpenAI
python main.py paper.txt \
  --base-url https://<resource>.openai.azure.com/ \
  --model gpt-4o \
  --api-key $AZURE_OPENAI_KEY
```

### 4. Use as a Python library

```python
from src.pipeline import SDRFPipeline

pipeline = SDRFPipeline(
    api_key="sk-...",
    model="gpt-4o",
    base_url=None,          # None = official OpenAI
)

# From a string
doc = pipeline.extract(paper_text)
print(f"Extracted {len(doc.rows)} samples")
path = pipeline.to_csv(doc, "output/result.sdrf.csv")

# From a file
path = pipeline.process_file("paper.txt", "output/result.sdrf.csv")

# Batch
paths = pipeline.process_batch("papers/", "output/")
```

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `input` | *(required)* | `.txt` file or directory |
| `--batch` | off | Process all `.txt` files in directory |
| `--output` | `<input>.sdrf.csv` | Output path (single mode) |
| `--output-dir` | `./output/` | Output directory (batch mode) |
| `--api-key` | `$OPENAI_API_KEY` | LLM API key |
| `--base-url` | OpenAI default | Base URL for OpenAI-compatible API |
| `--model` | `gpt-4o` | Model name |
| `--temperature` | `0.0` | LLM temperature |
| `--max-tokens` | `8192` | Max response tokens |
| `--verbose` / `-v` | off | Debug logging |

---

## Project Structure

```
sdrf_pipeline/
├── main.py              # CLI entrypoint
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

The output CSV matches the SDRF specification with all required columns.
Fields not extractable from the paper text are filled with `"not applicable"`.
One row is written per raw MS data file found in the paper.

```
D,PXD,Raw Data File,Characteristics[Age],...,Usage
S,PXD016436,file1.raw,not applicable,...,raw
S,PXD016436,file2.raw,not applicable,...,raw
```
