# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spike project for benchmarking Rakuten's prompt injection detection using Groq's API. Tests two approaches:
1. Llama Prompt Guard model (meta-llama/llama-prompt-guard-2-86m)
2. Custom policy-based detection using GPT-OSS-Safeguard (openai/gpt-oss-safeguard-20b)

## Development Commands

**Setup:**
```bash
uv sync
```

**Run benchmarks:**
```bash
# Benchmark English prompts with GPT-OSS-Safeguard
uv run python benchmark_groq.py --dataset datasets/jailbreaks_english.json --max-concurrency 5

# Benchmark both languages with reasoning_effort=low
uv run python benchmark_groq.py --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json --reasoning-effort low --max-concurrency 10

# Benchmark with Llama 3.1 8B Instant
uv run python benchmark_groq.py --model llama-3.1-8b-instant --dataset datasets/jailbreaks_english.json --max-concurrency 10

# Full benchmark with custom output
uv run python benchmark_groq.py --dataset datasets/jailbreaks_english.json --num-prompts 50 --output results/my_benchmark.json --save-raw
```

**Generate sample datasets:**
```bash
uv run python load_datasets.py
```

## Environment

Required: `GROQ_API_KEY` in .env file or via `--api-key` flag

## Architecture

### Core Files
- **benchmark_groq.py**: Main async benchmarking script with CLI interface
  - Measures TTFT, E2EL, TPOT, ITL with high-precision timing
  - Supports concurrent requests with semaphore control
  - Generates percentile reports (p1, p5, p10, p25, p50, p75, p90, p95, p99)
  - Per-language metric tracking and comparison

- **benchmark_utils.py**: Dataclasses and utility functions
  - `BenchmarkConfig`, `RequestOutput`, `BenchmarkMetrics` dataclasses
  - Statistics calculation using NumPy
  - Result formatting and JSON export

- **load_datasets.py**: Dataset loading and management
  - Supports JSON format: `[{"prompt": "...", "language": "en/ja", "category": "..."}]`
  - Can generate sample jailbreak datasets

### Legacy Files
- **query_groq.py**: Original examples with Prompt Guard and policy-based detection
- **main.py**: Entry point placeholder

## Benchmark Metrics

Client-side: TTFT (Time to First Token), E2EL (End-to-End Latency), TPOT (Time Per Output Token), ITL (Inter-Token Latency)

Server-side: Queue time, token counts, cache hits (extracted from response headers when available)

Output includes mean, median, std, min, max, and configurable percentiles for all metrics

Multi-language benchmarks automatically generate comparison tables for both TTFT and E2EL across languages with difference calculations
