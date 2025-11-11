# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmark suite for comparing prompt injection detection models on Groq's API:
1. **GPT-OSS-Safeguard** (openai/gpt-oss-safeguard-20b) - Uses system prompt with detection policy
2. **Llama 3.1 8B Instant** (llama-3.1-8b-instant) - Uses user prompt with embedded instructions
3. **Llama Prompt Guard** (meta-llama/llama-prompt-guard-2-86m) - Classification model

## Development Commands

**Setup:**
```bash
uv sync
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

**Run benchmarks:**
```bash
# Benchmark with GPT-OSS-Safeguard (default model)
uv run python benchmark_groq.py --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json --max-concurrency 10

# Benchmark with Llama 3.1 8B Instant
uv run python benchmark_groq.py --model llama-3.1-8b-instant --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json --max-concurrency 10

# With verbose output to see request/response content
uv run python benchmark_groq.py --model llama-3.1-8b-instant --dataset datasets/jailbreaks_english.json --verbose --max-concurrency 5

# Full benchmark with custom output and raw data
uv run python benchmark_groq.py --dataset datasets/jailbreaks_english.json --num-prompts 50 --output results/my_benchmark.json --save-raw
```

**Generate sample datasets:**
```bash
uv run python generate_large_datasets.py
```

## Environment

Required: `GROQ_API_KEY` in .env file or via `--api-key` flag

## Architecture

### Core Files
- **benchmark_groq.py**: Main async benchmarking script
  - Model-specific prompt formatting (system vs user prompts)
  - High-precision timing: TTFT, E2EL, TPOT, ITL
  - Concurrent execution with semaphore control
  - Automatic logging for long requests (>1s) with full request/response content
  - Verbose mode for detailed per-request analysis
  - Suppresses HTTP 200 OK logs from httpx

- **benchmark_utils.py**: Metrics and statistics
  - Dataclasses: `BenchmarkConfig`, `RequestOutput`, `BenchmarkMetrics`
  - NumPy-based statistics with configurable percentiles
  - Per-language comparison tables with network overhead analysis
  - JSON export with optional raw request data

- **load_datasets.py**: Dataset management
  - JSON format: `[{"prompt": "...", "language": "en/ja", "category": "..."}]`
  - Multi-language support

- **generate_large_datasets.py**: Dataset generation
  - Creates jailbreak test datasets (100 prompts each)
  - Multiple attack categories with safe baseline prompts

## Prompt Formats

The benchmark uses different prompt formats optimized for each model:

**GPT-OSS-Safeguard:**
- System prompt with full detection policy
- User input embedded in system message
- User message: "Please analyze the content above."

**Llama 3.1 8B Instant:**
- Single user prompt with instructions embedded
- No system prompt (for model compatibility)
- Includes task description, examples, and input to analyze

**Llama Prompt Guard:**
- Direct prompt classification (no modifications)

## Benchmark Metrics

**Client-side:** TTFT, E2EL, TPOT, ITL (high-precision timing)

**Server-side:** Queue time, prompt time, completion time, total time, token counts, cache hits

**Network:** Client E2EL - Server total = Network overhead

**Statistics:** Mean, median, std, min, max, configurable percentiles (default: p1, p5, p10, p25, p50, p75, p90, p95, p99)

**Multi-language:** Automatic comparison tables with per-language breakdowns and differences
