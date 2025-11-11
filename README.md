# Groq API Benchmarking

Benchmark suite for comparing prompt injection detection models on Groq's API. Features model-specific prompt formatting, multi-language support, comprehensive metrics, and automatic network overhead analysis.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Generate sample datasets (100 prompts each):**
   ```bash
   uv run python generate_large_datasets.py
   ```

4. **Run a quick benchmark:**
   ```bash
   # GPT-OSS-Safeguard (default) - uses system prompt
   uv run python benchmark_groq.py --dataset datasets/jailbreaks_english.json --max-concurrency 5

   # Llama 3.1 8B - uses user prompt with embedded instructions
   uv run python benchmark_groq.py --model llama-3.1-8b-instant --dataset datasets/jailbreaks_english.json --max-concurrency 5
   ```

## Usage Examples

### Basic Benchmark
```bash
uv run python benchmark_groq.py \
  --dataset datasets/jailbreaks_english.json \
  --max-concurrency 10
```

### Multi-Language Benchmark
```bash
uv run python benchmark_groq.py \
  --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json \
  --max-concurrency 10
```

### With Reasoning Effort (for GPT-OSS-Safeguard)
```bash
uv run python benchmark_groq.py \
  --model openai/gpt-oss-safeguard-20b \
  --dataset datasets/jailbreaks_english.json \
  --reasoning-effort low \
  --max-concurrency 10
```

### With Llama 3.1 8B Instant
```bash
# English only
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --dataset datasets/jailbreaks_english.json \
  --max-concurrency 10

# Multi-language comparison
uv run python benchmark_groq.py \
  --model llama-3.1-8b-instant \
  --datasets datasets/jailbreaks_english.json datasets/jailbreaks_japanese.json \
  --max-concurrency 10
```

### Custom Output and Percentiles
```bash
uv run python benchmark_groq.py \
  --dataset datasets/jailbreaks_english.json \
  --percentiles 50,90,95,99 \
  --output results/my_benchmark.json \
  --save-raw
```

### Disable Prompt Caching
```bash
# Add unique prefix to each prompt to prevent caching
uv run python benchmark_groq.py \
  --dataset datasets/jailbreaks_english.json \
  --disable-cache \
  --max-concurrency 10
```

## Model-Specific Prompt Formats

The benchmark automatically uses the optimal prompt format for each model:

### GPT-OSS-Safeguard (openai/gpt-oss-safeguard-20b)
- **System message**: Full detection policy with user input embedded
- **User message**: "Please analyze the content above."
- Best for: Leveraging system-level instructions

### Llama 3.1 8B Instant (llama-3.1-8b-instant)
- **Single user message**: Instructions, examples, and input combined
- **No system prompt**: Optimized for model compatibility
- Best for: Multilingual detection with embedded prompts

### Llama Prompt Guard (meta-llama/llama-prompt-guard-2-86m)
- **Direct prompt**: No modifications
- Best for: Fast classification baseline

## Metrics Captured

### Client-Side Timing
- **TTFT** (Time to First Token): Latency from request to first token
  - For streaming models: measures when first token arrives
  - For classification models (e.g., llama-prompt-guard): equals E2EL
- **E2EL** (End-to-End Latency): Total request latency
- **TPOT** (Time Per Output Token): Average time per token
- **ITL** (Inter-Token Latency): Time between consecutive tokens

### Server-Side Metrics (from Groq API)
- **Queue time** - Time spent waiting in server queue
- **Prompt time** - Time to process the input prompt
- **Completion time** - Time to generate the response
- **Server total time** - Total server-side processing time
- **Token counts** - Prompt, completion, and total tokens
- **Cache hits** - Prompt cache utilization

### Calculated Metrics
- **Network overhead** - Client E2EL minus server total time
  - Reveals network latency and streaming overhead
  - Helps identify if bottleneck is server or network

### Statistics
For each metric: mean, median, std, min, max, and configurable percentiles (default: p1, p5, p10, p25, p50, p75, p90, p95, p99)

See [METRICS_EXPLAINED.md](METRICS_EXPLAINED.md) for detailed explanation of metrics and how to interpret them.

## Dataset Format

JSON array with prompt objects:
```json
[
  {
    "prompt": "Your prompt text here",
    "language": "en",
    "category": "prompt_injection"
  }
]
```

### Included Datasets

Both `jailbreaks_english.json` and `jailbreaks_japanese.json` contain **100 prompts each** with the following category distribution:
- **30%** direct_override - Direct instruction override attempts
- **16%** system_exposure - Attempts to reveal system prompts
- **14%** role_play - Role-playing jailbreak attempts (DAN, etc.)
- **10%** mode_switching - Developer/admin mode activation
- **10%** encoding - Base64/ROT13 obfuscation attempts
- **20%** safe - Benign prompts for baseline comparison

## Output

Results are saved as JSON with full metrics breakdown. Per-language metrics are calculated automatically when multiple languages are present in the dataset.

Example output tables:
```
Time to First Token (TTFT) - in milliseconds
Percentile    English (ms)    Japanese (ms)    Difference (ms)
p1            327.9           2415.7           2087.7
p50           2478.7          3534.1           1055.4
p99           5117.7          5701.3           583.5

End-to-End Latency (E2EL) - in milliseconds
Percentile    English (ms)    Japanese (ms)    Difference (ms)
p1            350.2           2450.3           2100.1
p50           2478.7          3534.1           1055.4
p99           5117.7          5701.3           583.5
```

## CLI Options

```
--model MODEL                 Groq model to benchmark
--dataset PATH               Single dataset file
--datasets PATH [PATH ...]   Multiple dataset files
--num-prompts N              Limit number of prompts
--max-concurrency N          Max concurrent requests (default: 10)
--reasoning-effort LEVEL     low/medium/high for supported models
--percentiles P,P,...        Comma-separated percentiles (default: 1,5,10,25,50,75,90,95,99)
--disable-cache              Add unique prefix to prompts to prevent caching
--output PATH                Output JSON file
--save-raw                   Include raw request data in output
--verbose                    Print detailed per-request info including response content
```

### Logging Levels

- **Default**: Only warnings and errors
  - Long requests (>1s) with full response content
  - Timing anomalies and errors
  - No HTTP 200 OK logs
- **--verbose**: Detailed per-request information
  - TTFT, E2EL, token counts for every request
  - Full response content for all requests
  - Server-side timing breakdown
  - Network overhead metrics
