#!/usr/bin/env python3
"""
Async benchmarking script for Groq API.

Measures TTFT, E2EL, TPOT, and server-side metrics across multiple languages.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import List, Dict, Optional
from pathlib import Path

from groq import AsyncGroq
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from benchmark_utils import (
    BenchmarkConfig,
    RequestOutput,
    BenchmarkMetrics,
    calculate_metrics,
    print_results_table,
    print_language_comparison,
    save_results,
)
from load_datasets import load_dataset, load_multiple_datasets, get_language_counts


async def send_groq_request(
    client: AsyncGroq,
    prompt: str,
    model: str,
    language: str = "unknown",
    reasoning_effort: Optional[str] = None,
    disable_cache: bool = False,
    verbose: bool = False,
) -> RequestOutput:
    """
    Send a single request to Groq API and capture all metrics.

    Args:
        client: AsyncGroq client
        prompt: Input prompt text
        model: Model identifier
        language: Language code for the prompt
        reasoning_effort: Optional reasoning effort level ('low', 'medium', 'high')
        disable_cache: Add unique prefix to prevent caching
        verbose: Print detailed timing info

    Returns:
        RequestOutput with all captured metrics
    """
    request_start = time.perf_counter()

    # Add unique prefix to prevent caching if requested
    if disable_cache:
        cache_bust_prefix = f"[Request-{time.time_ns()}] "
        prompt = cache_bust_prefix + prompt

    try:
        # Detect if model is a text classification model (doesn't support streaming)
        is_classification_model = "prompt-guard" in model.lower()

        # Build request parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": not is_classification_model,  # Classification models don't support streaming
        }

        # Add reasoning_effort if supported and specified
        if reasoning_effort and "safeguard" in model.lower():
            params["extra_body"] = {"reasoning_effort": reasoning_effort}

        # Send request
        response = await client.chat.completions.create(**params)

        # Track timing
        first_token_time = None
        last_token_time = request_start
        inter_token_latencies = []
        content_chunks = []

        # Initialize metrics
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        queue_time = None
        prompt_time = None
        completion_time = None
        server_total_time = None
        cache_hit = False

        if is_classification_model:
            # Handle non-streaming response (classification models)
            end_time = time.perf_counter()
            total_latency = end_time - request_start

            # For classification models, TTFT = E2EL (no streaming)
            first_token_time = end_time

            # Extract content and metrics
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content or ""

            # Extract usage metrics
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # Extract server-side timing if available
                if hasattr(response.usage, 'queue_time'):
                    queue_time = response.usage.queue_time
                if hasattr(response.usage, 'prompt_time'):
                    prompt_time = response.usage.prompt_time
                if hasattr(response.usage, 'completion_time'):
                    completion_time = response.usage.completion_time
                if hasattr(response.usage, 'total_time'):
                    server_total_time = response.usage.total_time

            if verbose:
                print(f"  E2EL: {total_latency*1000:.2f}ms (non-streaming)")
        else:
            # Handle streaming response
            async for chunk in response:
                current_time = time.perf_counter()

                # Check if we have content
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    if delta.content:
                        if first_token_time is None:
                            # First token received
                            first_token_time = current_time
                            ttft = first_token_time - request_start

                            if verbose:
                                print(f"  TTFT: {ttft*1000:.2f}ms")
                        else:
                            # Subsequent tokens - track inter-token latency
                            itl = current_time - last_token_time
                            inter_token_latencies.append(itl)

                        content_chunks.append(delta.content)
                        last_token_time = current_time

            end_time = time.perf_counter()
            total_latency = end_time - request_start

            # Get final content
            content = ''.join(content_chunks)

            # Try to extract metrics from last chunk
            if chunk.choices and len(chunk.choices) > 0:
                # Groq provides usage in the last chunk with timing breakdown
                if hasattr(chunk, 'usage') and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

                    # Extract server-side timing
                    if hasattr(chunk.usage, 'queue_time'):
                        queue_time = chunk.usage.queue_time
                    if hasattr(chunk.usage, 'prompt_time'):
                        prompt_time = chunk.usage.prompt_time
                    if hasattr(chunk.usage, 'completion_time'):
                        completion_time = chunk.usage.completion_time
                    if hasattr(chunk.usage, 'total_time'):
                        server_total_time = chunk.usage.total_time

            # Estimate completion tokens if not provided
            if completion_tokens is None:
                completion_tokens = len(content_chunks)

        # Calculate final metrics
        ttft_value = first_token_time - request_start if first_token_time else None

        # Log long requests (>1 second)
        if total_latency > 1.0:
            logger.warning(f"Long request detected: {total_latency:.3f}s")
            logger.info(f"  Language: {language}")
            logger.info(f"  Model: {model}")
            logger.info(f"  TTFT: {ttft_value*1000:.2f}ms" if ttft_value else "  TTFT: N/A")
            logger.info(f"  E2EL: {total_latency*1000:.2f}ms")
            logger.info(f"  Prompt tokens: {prompt_tokens}")
            logger.info(f"  Completion tokens: {completion_tokens}")
            logger.info(f"  Content length: {len(content) if content else 0} chars")

            if server_total_time:
                network_overhead = total_latency - server_total_time
                logger.info(f"  Server-side breakdown:")
                logger.info(f"    Queue time: {queue_time*1000:.2f}ms" if queue_time else "    Queue time: N/A")
                logger.info(f"    Prompt time: {prompt_time*1000:.2f}ms" if prompt_time else "    Prompt time: N/A")
                logger.info(f"    Completion time: {completion_time*1000:.2f}ms" if completion_time else "    Completion time: N/A")
                logger.info(f"    Server total: {server_total_time*1000:.2f}ms")
                logger.info(f"    Network overhead: {network_overhead*1000:.2f}ms")

            # Log first 200 chars of content for debugging
            if content:
                preview = content[:200].replace('\n', ' ')
                logger.debug(f"  Content preview: {preview}...")

        # Validate timing consistency
        if ttft_value and total_latency and ttft_value > total_latency:
            logger.error(f"TIMING ERROR: TTFT ({ttft_value:.3f}s) > E2EL ({total_latency:.3f}s)")

        if server_total_time and total_latency and server_total_time > total_latency * 1.1:
            # Allow 10% tolerance for timing precision
            logger.warning(f"TIMING ANOMALY: Server time ({server_total_time:.3f}s) > Client E2EL ({total_latency:.3f}s)")
            logger.warning(f"  This may indicate server timing is in different units or includes network time")

        return RequestOutput(
            success=True,
            prompt=prompt,
            language=language,
            ttft=ttft_value,
            latency=total_latency,
            itl=inter_token_latencies,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            queue_time=queue_time,
            prompt_time=prompt_time,
            completion_time=completion_time,
            server_total_time=server_total_time,
            cache_hit=cache_hit,
            content=content,
        )

    except Exception as e:
        end_time = time.perf_counter()
        error_latency = end_time - request_start

        logger.error(f"Request failed after {error_latency:.3f}s: {str(e)}")
        logger.debug(f"  Model: {model}")
        logger.debug(f"  Language: {language}")
        logger.debug(f"  Prompt length: {len(prompt)} chars")

        if verbose:
            print(f"  Error: {str(e)}")

        return RequestOutput(
            success=False,
            prompt=prompt,
            language=language,
            latency=error_latency,
            error=str(e),
        )


async def run_benchmark(config: BenchmarkConfig, prompts: List[Dict]) -> List[RequestOutput]:
    """
    Run the benchmark with configured parameters.

    Args:
        config: Benchmark configuration
        prompts: List of prompt dictionaries

    Returns:
        List of RequestOutput objects
    """
    # Initialize Groq client
    client = AsyncGroq(api_key=config.api_key)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def bounded_request(prompt_dict: Dict) -> RequestOutput:
        """Execute request with semaphore-based concurrency control."""
        async with semaphore:
            return await send_groq_request(
                client=client,
                prompt=prompt_dict['prompt'],
                model=config.model,
                language=prompt_dict.get('language', 'unknown'),
                reasoning_effort=config.reasoning_effort,
                disable_cache=config.disable_cache,
                verbose=config.verbose,
            )

    print(f"\nStarting benchmark:")
    print(f"  Model: {config.model}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Max Concurrency: {config.max_concurrency}")
    if config.reasoning_effort:
        print(f"  Reasoning Effort: {config.reasoning_effort}")
    if config.disable_cache:
        print(f"  Cache: Disabled (unique prefix added to each prompt)")
    print()

    # Create tasks for all requests
    tasks = [bounded_request(p) for p in prompts]

    # Execute with progress bar
    outputs = []
    for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Running requests"):
        result = await coro
        outputs.append(result)

    return outputs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Groq API performance with latency percentiles"
    )

    # Model and API
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-safeguard-20b",
        help="Groq model to benchmark (default: openai/gpt-oss-safeguard-20b)"
    )
    parser.add_argument(
        "--api-key",
        help="Groq API key (default: from GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort level for supported models (e.g., gpt-oss-safeguard)"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Paths to multiple dataset JSON files (will be combined)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        help="Limit number of prompts to use (default: use all)"
    )

    # Benchmark parameters
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--percentiles",
        default="1,5,10,25,50,75,90,95,99",
        help="Comma-separated percentiles to calculate (default: 1,5,10,25,50,75,90,95,99)"
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Add unique prefix to each prompt to prevent caching"
    )

    # Output
    parser.add_argument(
        "--output",
        help="Output JSON file path (default: results/benchmark_TIMESTAMP.json)"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw request outputs in results file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-request information"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (includes content previews for long requests)"
    )

    args = parser.parse_args()

    # Set logging level based on flags
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found. Set via --api-key or GROQ_API_KEY env var.")
        sys.exit(1)

    # Load prompts
    if args.datasets:
        prompts = load_multiple_datasets(args.datasets, args.num_prompts)
    elif args.dataset:
        prompts = load_dataset(args.dataset, args.num_prompts)
    else:
        print("Error: Must specify --dataset or --datasets")
        sys.exit(1)

    if not prompts:
        print("Error: No prompts loaded from dataset(s)")
        sys.exit(1)

    # Show language distribution
    lang_counts = get_language_counts(prompts)
    print(f"\nDataset loaded: {len(prompts)} prompts")
    print(f"Languages: {dict(lang_counts)}")

    # Parse percentiles
    percentiles = [int(p.strip()) for p in args.percentiles.split(',')]

    # Create config
    config = BenchmarkConfig(
        model=args.model,
        api_key=api_key,
        num_prompts=len(prompts),
        max_concurrency=args.max_concurrency,
        percentiles=percentiles,
        reasoning_effort=args.reasoning_effort,
        disable_cache=args.disable_cache,
        verbose=args.verbose,
    )

    # Run benchmark
    benchmark_start = time.perf_counter()
    outputs = asyncio.run(run_benchmark(config, prompts))
    duration = time.perf_counter() - benchmark_start

    print(f"\nBenchmark completed in {duration:.2f}s")

    # Calculate overall metrics
    overall_metrics = calculate_metrics(
        outputs=outputs,
        duration=duration,
        model=config.model,
        percentiles=percentiles,
    )

    print_results_table(overall_metrics, "Overall Results")

    # Calculate per-language metrics
    languages = sorted(set(o.language for o in outputs if o.language != 'unknown'))
    metrics_by_lang = {}

    if len(languages) > 1:
        print("\nCalculating per-language metrics...")
        for lang in languages:
            lang_metrics = calculate_metrics(
                outputs=outputs,
                duration=duration,
                model=config.model,
                percentiles=percentiles,
                language=lang,
            )
            metrics_by_lang[lang] = lang_metrics
            print_results_table(lang_metrics, f"Results for {lang.upper()}")

        # Print comparison table
        print_language_comparison(metrics_by_lang)

    # Save results
    if args.output:
        output_file = args.output
    else:
        # Create default output path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"results/benchmark_{timestamp}.json"

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    save_results(
        metrics=overall_metrics,
        output_file=output_file,
        outputs=outputs if args.save_raw else None,
    )

    # Also save per-language results if available
    if metrics_by_lang:
        base_path = Path(output_file)
        for lang, metrics in metrics_by_lang.items():
            lang_output = base_path.parent / f"{base_path.stem}_{lang}{base_path.suffix}"
            save_results(metrics, str(lang_output))


if __name__ == "__main__":
    main()
