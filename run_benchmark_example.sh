#!/bin/bash
# Example benchmark runs for Groq API testing

echo "======================================"
echo "Groq API Benchmark Examples"
echo "======================================"

# 1. English-only benchmark with GPT-OSS-Safeguard
echo -e "\n[1] Running English benchmark with reasoning_effort=low..."
uv run python benchmark_groq.py \
  --model openai/gpt-oss-safeguard-20b \
  --dataset datasets/jailbreaks_english.json \
  --reasoning-effort low \
  --max-concurrency 10 \
  --output results/english_safeguard_low.json

# 2. Japanese-only benchmark
echo -e "\n[2] Running Japanese benchmark with reasoning_effort=low..."
uv run python benchmark_groq.py \
  --model openai/gpt-oss-safeguard-20b \
  --dataset datasets/jailbreaks_japanese.json \
  --reasoning-effort low \
  --max-concurrency 10 \
  --output results/japanese_safeguard_low.json

# 3. Combined multi-language benchmark for comparison
echo -e "\n[3] Running multi-language comparison benchmark..."
uv run python benchmark_groq.py \
  --model openai/gpt-oss-safeguard-20b \
  --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json \
  --reasoning-effort low \
  --max-concurrency 10 \
  --output results/multilang_comparison.json \
  --save-raw

# 4. Llama Prompt Guard benchmark
echo -e "\n[4] Running benchmark with Llama Prompt Guard model..."
uv run python benchmark_groq.py \
  --model meta-llama/llama-prompt-guard-2-86m \
  --dataset datasets/jailbreaks_english.json \
  --max-concurrency 10 \
  --output results/prompt_guard_english.json

# 5. Llama 3.1 8B Instant - English benchmark
echo -e "\n[5] Running English benchmark with Llama 3.1 8B Instant..."
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --dataset datasets/jailbreaks_english.json \
  --max-concurrency 10 \
  --output results/llama31_8b_english.json

# 6. Llama 3.1 8B Instant - Japanese benchmark
echo -e "\n[6] Running Japanese benchmark with Llama 3.1 8B Instant..."
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --dataset datasets/jailbreaks_japanese.json \
  --max-concurrency 10 \
  --output results/llama31_8b_japanese.json

# 7. Llama 3.1 8B Instant - Multi-language comparison
echo -e "\n[7] Running multi-language comparison with Llama 3.1 8B Instant..."
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json \
  --max-concurrency 10 \
  --output results/llama31_8b_multilang.json \
  --save-raw

echo -e "\n======================================"
echo "Benchmark runs completed!"
echo "Results saved in results/ directory"
echo "======================================"
