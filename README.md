# PDF RAG with Ollama and NVIDIA

🌐 *Read this in [Português (Brasil)](README.pt-br.md).*

A Python application for querying **PDF documents on any subject** with **Retrieval-Augmented Generation (RAG)**. It supports two execution modes — **Ollama (local)** and the **NVIDIA API (cloud)** — combining semantic search with answers grounded in retrieved passages.

## How it works

**Question → Rewrite → Retrieve → Read → Answer**

1. PDF text is split into chunks and indexed in **ChromaDB**, with separate indexes for each provider.
2. An LLM rewrites the question for semantic search; the **4 most relevant chunks** are retrieved.
3. The Reader receives the **original question** and retrieved passages labeled by source document.
4. A **SHA-256 manifest** allows index reuse and rebuilding when PDFs change.

| Mode | LLM | Embeddings |
| --- | --- | --- |
| Ollama (local) | Qwen 3.5 9B | BGE-M3 |
| NVIDIA API | Nemotron 3.5 Lightning 30B A3B | Nemotron 3 Embed 1B |

**Stack:** Python 3.12, LangChain, ChromaDB, pypdf, Ollama, and NVIDIA API.

**Cost:** local Ollama inference has no per-request fee but requires your own computing resources. NVIDIA offers **free access for development and testing** through [NVIDIA Build](https://build.nvidia.com/), subject to usage limits and terms; this access is not intended for production.

## Installation and usage

From the repository root, with **Python 3.12** installed:

```bash
python -m venv .venv
```

Activate the environment: on **Windows (PowerShell)**, `.venv\Scripts\Activate.ps1`; on **Linux/macOS**, `source .venv/bin/activate`. Then:

```bash
python -m pip install -r requirements.txt
```

1. Copy `.env.example` to `.env`. For NVIDIA mode, get a key from [NVIDIA Build](https://build.nvidia.com/) and set `NVIDIA_API_KEY`. Ollama needs no NVIDIA key; LangSmith is optional and disabled by default.
2. Place PDFs in `data/documents/`. Documents, indexes, and credentials are not committed.
3. Set `PROVIDER` in `src/main.py` to `"ollama"` or `"nvidia"`.
4. To use the local mode or run the evaluator, install and start [Ollama](https://ollama.com/) and pull the models:

```bash
ollama pull qwen3.5:9b
ollama pull bge-m3:567m
```

Run the CLI from the repository root:

```bash
python src/main.py
```

Enter questions in the terminal; type `sair` to exit. PDFs are indexed on first use, and the index can be reused on subsequent runs.

## Experimental evaluation

Quality was investigated through a **15-question benchmark based on two public Vale manuals**, across three runs per provider. The `src/evaluate.py` and `src/evaluator.py` scripts record answers, rewritten queries, retrieved sources, stage timings, and local LLM judgments (**0–2 points per question**).

To evaluate other PDFs, use `evaluation/benchmark.example.json` as a reference and prepare your own benchmark with questions, expected answers, essential points, and supporting evidence. File paths are provided through command-line arguments. Responses are judged by a local LLM via **Ollama**, even when evaluating the NVIDIA RAG pipeline.

The scores below use the **same judging prompt**; responses from the first two runs were regraded without rerunning the RAG pipelines.

| Run | NVIDIA | Ollama |
| --- | ---: | ---: |
| 1 — initial | 26/30 (86.7%) | 20/30 (66.7%) |
| 2 — revision | 21/30 (70.0%) | 16/30 (53.3%) |
| 3 — further adjustments | 25/30 (83.3%) | 19/30 (63.3%) |

**Interpretation:** these are automated scores, **not manually verified accuracy rates**. Judge inconsistencies, retrieval failures, and confusion between procedures were observed. The two modes use different LLMs and embeddings, so this compares **complete pipelines**. Score changes cannot be attributed to prompts alone. The PDFs used in testing are not distributed.

### How to run the evaluation

The `evaluation/benchmark.example.json` file demonstrates the expected structure. To evaluate your own documents, create a benchmark based on the information contained in your PDFs.

Run the queries:

```bash
python src/evaluate.py --provider nvidia --benchmark evaluation/benchmark.local.json
```

Evaluate the generated responses:

```bash
python src/evaluator.py --benchmark evaluation/benchmark.local.json --results evaluation/results/GENERATED_FILE.json
```

Results are saved as JSON files for further analysis. The evaluator assigns scores from **0 to 2** based on the benchmark criteria.

## Project limitations

The project supports **PDFs with extractable text** (no OCR). Retrieving four chunks may miss evidence, and consequential answers require verification. In NVIDIA mode, data used in queries is sent to the provider. The current version is an **experimental CLI**, not a production-ready service.
