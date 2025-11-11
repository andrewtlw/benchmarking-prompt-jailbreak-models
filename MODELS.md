# Available Models for Benchmarking

This document lists the Groq models that can be used with the benchmark script.

## Prompt Injection Detection Models

### GPT-OSS-Safeguard (20B)
**Model ID:** `openai/gpt-oss-safeguard-20b`

Purpose-built for detecting prompt injection attacks and jailbreak attempts.

**Features:**
- Supports `reasoning_effort` parameter (`low`, `medium`, `high`)
- Optimized for security classification tasks
- Returns structured JSON responses for violation detection
- Uses system prompt with embedded detection policy

**Prompt Format (automatic):**
- System message: Full detection policy with input embedded
- User message: "Please analyze the content above."

**Example:**
```bash
uv run python benchmark_groq.py \
  --model openai/gpt-oss-safeguard-20b \
  --dataset datasets/jailbreaks_english.json \
  --reasoning-effort low \
  --max-concurrency 10
```

### Llama Prompt Guard (86M)
**Model ID:** `meta-llama/llama-prompt-guard-2-86m`

Lightweight model specifically designed for prompt injection detection.

**Features:**
- Very fast inference (small 86M parameter model)
- Direct classification of prompt safety
- Low latency for real-time filtering

**Example:**
```bash
uv run python benchmark_groq.py \
  --model meta-llama/llama-prompt-guard-2-86m \
  --dataset datasets/jailbreaks_english.json \
  --max-concurrency 10
```

## General-Purpose Language Models

### Llama 3.1 8B Instant
**Model ID:** `llama-3.1-8b-instant`

Fast, efficient general-purpose language model adapted for prompt injection detection.

**Features:**
- Excellent multilingual capabilities (English, Japanese, etc.)
- Optimized for low latency ("instant" variant)
- Can be prompted for security classification tasks
- 8B parameters - balanced size/performance
- Uses user prompt format (no system message)

**Prompt Format (automatic):**
- Single user message with embedded instructions, examples, and input
- No system prompt (for optimal model compatibility)

**Example:**
```bash
# English benchmark
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --dataset datasets/jailbreaks_english.json \
  --max-concurrency 10

# Japanese benchmark
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --dataset datasets/jailbreaks_japanese.json \
  --max-concurrency 10

# Multi-language comparison
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json \
  --max-concurrency 10
```

### Llama 3.1 70B Versatile
**Model ID:** `llama-3.1-70b-versatile`

Larger, more capable model with enhanced reasoning abilities.

**Features:**
- Superior performance on complex tasks
- Better multilingual understanding
- Higher latency than 8B model but more accurate
- 70B parameters - larger context understanding

**Example:**
```bash
uv run python benchmark_groq.py \
  --model llama-3.1-70b-versatile \
  --dataset datasets/jailbreaks_english.json \
  --max-concurrency 5
```