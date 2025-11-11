#!/bin/bash
# Example benchmark runs for Groq API testing
# Compares prompt injection detection across GPT-OSS-Safeguard and Llama 3.1 8B Instant

echo "======================================"
echo "Groq API Benchmark Examples"
echo "======================================"
echo ""
echo "Note: These benchmarks use different prompt formats per model:"
echo "  - GPT-OSS-Safeguard: System prompt with detection policy"
echo "  - Llama 3.1 8B: User prompt with embedded instructions"
echo ""

# 1. GPT-OSS-Safeguard - Multi-language comparison
echo -e "\n[1] GPT-OSS-Safeguard: Multi-language comparison..."
uv run python benchmark_groq.py \
  --model openai/gpt-oss-safeguard-20b \
  --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json \
  --reasoning-effort low \
  --max-concurrency 10 \
  --output results/safeguard_multilang.json \
  --save-raw

# 2. Llama 3.1 8B Instant - Multi-language comparison
echo -e "\n[2] Llama 3.1 8B Instant: Multi-language comparison..."
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json \
  --max-concurrency 10 \
  --output results/llama31_8b_multilang.json \
  --save-raw

# 3. Llama Prompt Guard - Fast classification baseline
echo -e "\n[3] Llama Prompt Guard: Fast classification baseline..."
uv run python benchmark_groq.py \
  --model meta-llama/llama-prompt-guard-2-86m \
  --dataset datasets/jailbreaks_english.json \
  --max-concurrency 10 \
  --output results/prompt_guard_english.json

# 4. Verbose mode example - see request/response content
echo -e "\n[4] Verbose mode example (shows all request/response content)..."
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --dataset datasets/jailbreaks_english.json \
  --num-prompts 5 \
  --verbose \
  --max-concurrency 3 \
  --output results/verbose_example.json

echo -e "\n======================================"
echo "Benchmark runs completed!"
echo "Results saved in results/ directory"
echo ""
echo "Review results:"
echo "  - Multi-language comparisons show per-language metrics"
echo "  - Network overhead separated from server-side timing"
echo "  - Long requests (>1s) automatically logged with full content"
echo "======================================"
