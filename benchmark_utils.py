"""Utility classes and functions for Groq API benchmarking."""

import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    model: str = "openai/gpt-oss-safeguard-20b"
    api_key: Optional[str] = None
    num_prompts: int = 100
    max_concurrency: int = 10
    request_rate: float = float('inf')  # requests per second, inf = burst mode
    dataset_path: Optional[str] = None
    output_file: Optional[str] = None
    percentiles: List[int] = field(default_factory=lambda: [1, 5, 10, 25, 50, 75, 90, 95, 99])
    reasoning_effort: Optional[str] = None  # 'low', 'medium', 'high' for supported models
    disable_cache: bool = False  # Add unique prefix to prompts to prevent caching
    verbose: bool = False


@dataclass
class RequestOutput:
    """Output from a single API request."""
    success: bool
    prompt: str
    language: str = "unknown"

    # Client-side timing
    ttft: Optional[float] = None  # Time to first token (seconds)
    latency: Optional[float] = None  # End-to-end latency (seconds)
    itl: List[float] = field(default_factory=list)  # Inter-token latencies (seconds)

    # Server-side metrics (from Groq API response)
    queue_time: Optional[float] = None  # Time spent in queue (seconds)
    prompt_time: Optional[float] = None  # Time to process prompt (seconds)
    completion_time: Optional[float] = None  # Time to generate completion (seconds)
    server_total_time: Optional[float] = None  # Total server-side time (seconds)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cache_hit: bool = False

    # Response data
    content: Optional[str] = None
    error: Optional[str] = None

    @property
    def tpot(self) -> Optional[float]:
        """Time per output token (seconds)."""
        if self.itl and len(self.itl) > 0:
            return sum(self.itl) / len(self.itl)
        return None


@dataclass
class MetricStats:
    """Statistical summary for a single metric."""
    mean: float
    median: float
    std: float
    min: float
    max: float
    percentiles: Dict[int, float]  # {percentile: value}

    def to_ms_dict(self) -> Dict[str, Any]:
        """Convert to milliseconds for display."""
        return {
            "mean_ms": self.mean * 1000,
            "median_ms": self.median * 1000,
            "std_ms": self.std * 1000,
            "min_ms": self.min * 1000,
            "max_ms": self.max * 1000,
            "percentiles_ms": {f"p{p}": v * 1000 for p, v in self.percentiles.items()}
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics from benchmark run."""
    # Metadata
    model: str
    language: Optional[str] = None  # None means all languages combined
    duration: float = 0.0  # Total benchmark duration (seconds)

    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Token counts
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    # Throughput
    request_throughput: float = 0.0  # requests/sec
    prompt_token_throughput: float = 0.0  # tokens/sec
    completion_token_throughput: float = 0.0  # tokens/sec
    total_token_throughput: float = 0.0  # tokens/sec

    # Latency statistics
    ttft_stats: Optional[MetricStats] = None  # Time to first token
    e2el_stats: Optional[MetricStats] = None  # End-to-end latency
    tpot_stats: Optional[MetricStats] = None  # Time per output token
    itl_stats: Optional[MetricStats] = None  # Inter-token latency

    # Server-side statistics
    queue_time_stats: Optional[MetricStats] = None
    prompt_time_stats: Optional[MetricStats] = None
    completion_time_stats: Optional[MetricStats] = None
    server_total_time_stats: Optional[MetricStats] = None

    # Network overhead (client E2E - server total)
    network_time_stats: Optional[MetricStats] = None

    # Cache statistics
    cache_hit_count: int = 0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert MetricStats to dict
        stats_keys = [
            'ttft_stats', 'e2el_stats', 'tpot_stats', 'itl_stats',
            'queue_time_stats', 'prompt_time_stats', 'completion_time_stats',
            'server_total_time_stats', 'network_time_stats'
        ]
        for key in stats_keys:
            if result[key] is not None:
                result[key] = result[key]  # asdict already handles this
        return result


def calculate_metric_stats(values: List[float], percentiles: List[int]) -> Optional[MetricStats]:
    """Calculate statistical summary for a list of values."""
    if not values:
        return None

    values_array = np.array(values)

    return MetricStats(
        mean=float(np.mean(values_array)),
        median=float(np.median(values_array)),
        std=float(np.std(values_array)),
        min=float(np.min(values_array)),
        max=float(np.max(values_array)),
        percentiles={p: float(np.percentile(values_array, p)) for p in percentiles}
    )


def calculate_metrics(
    outputs: List[RequestOutput],
    duration: float,
    model: str,
    percentiles: List[int],
    language: Optional[str] = None
) -> BenchmarkMetrics:
    """Calculate aggregate metrics from request outputs."""

    # Filter by language if specified
    if language:
        outputs = [o for o in outputs if o.language == language]

    successful = [o for o in outputs if o.success]

    # Extract timing arrays
    ttfts = [o.ttft for o in successful if o.ttft is not None]
    e2els = [o.latency for o in successful if o.latency is not None]
    tpots = [o.tpot for o in successful if o.tpot is not None]

    # Flatten all ITLs
    all_itls = []
    for o in successful:
        if o.itl:
            all_itls.extend(o.itl)

    # Server-side metrics
    queue_times = [o.queue_time for o in successful if o.queue_time is not None]
    prompt_times = [o.prompt_time for o in successful if o.prompt_time is not None]
    completion_times = [o.completion_time for o in successful if o.completion_time is not None]
    server_total_times = [o.server_total_time for o in successful if o.server_total_time is not None]

    # Token counts
    total_prompt_tokens = sum(o.prompt_tokens for o in successful if o.prompt_tokens)
    total_completion_tokens = sum(o.completion_tokens for o in successful if o.completion_tokens)
    total_tokens = sum(o.total_tokens for o in successful if o.total_tokens)

    # Cache statistics
    cache_hits = sum(1 for o in successful if o.cache_hit)
    cache_hit_rate = cache_hits / len(successful) if successful else 0.0

    # Calculate throughput
    request_throughput = len(successful) / duration if duration > 0 else 0.0
    prompt_token_throughput = total_prompt_tokens / duration if duration > 0 else 0.0
    completion_token_throughput = total_completion_tokens / duration if duration > 0 else 0.0
    total_token_throughput = total_tokens / duration if duration > 0 else 0.0

    # Calculate network time (client E2EL - server total time)
    network_times = []
    for o in successful:
        if o.latency is not None and o.server_total_time is not None:
            network_time = o.latency - o.server_total_time
            network_times.append(network_time)

    return BenchmarkMetrics(
        model=model,
        language=language,
        duration=duration,
        total_requests=len(outputs),
        successful_requests=len(successful),
        failed_requests=len(outputs) - len(successful),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        request_throughput=request_throughput,
        prompt_token_throughput=prompt_token_throughput,
        completion_token_throughput=completion_token_throughput,
        total_token_throughput=total_token_throughput,
        ttft_stats=calculate_metric_stats(ttfts, percentiles),
        e2el_stats=calculate_metric_stats(e2els, percentiles),
        tpot_stats=calculate_metric_stats(tpots, percentiles),
        itl_stats=calculate_metric_stats(all_itls, percentiles),
        queue_time_stats=calculate_metric_stats(queue_times, percentiles),
        prompt_time_stats=calculate_metric_stats(prompt_times, percentiles),
        completion_time_stats=calculate_metric_stats(completion_times, percentiles),
        server_total_time_stats=calculate_metric_stats(server_total_times, percentiles),
        network_time_stats=calculate_metric_stats(network_times, percentiles),
        cache_hit_count=cache_hits,
        cache_hit_rate=cache_hit_rate,
    )


def print_results_table(metrics: BenchmarkMetrics, title: str = "Benchmark Results"):
    """Print formatted results table using Rich."""
    console.print()

    # Header panel
    header_text = f"[bold cyan]{title}[/bold cyan]\n"
    header_text += f"Model: [yellow]{metrics.model}[/yellow]"
    if metrics.language:
        header_text += f" | Language: [yellow]{metrics.language}[/yellow]"
    header_text += f" | Duration: [yellow]{metrics.duration:.2f}s[/yellow]"
    console.print(Panel(header_text, box=box.DOUBLE))

    # Request Summary Table
    summary_table = Table(title="Request Summary", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Total Requests", str(metrics.total_requests))
    summary_table.add_row("Successful", str(metrics.successful_requests))
    summary_table.add_row("Failed", f"[red]{metrics.failed_requests}[/red]" if metrics.failed_requests > 0 else "0")
    success_rate = metrics.successful_requests/metrics.total_requests*100 if metrics.total_requests > 0 else 0
    summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
    console.print(summary_table)

    # Throughput Table
    throughput_table = Table(title="Throughput", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    throughput_table.add_column("Metric", style="cyan")
    throughput_table.add_column("Value", justify="right", style="green")

    throughput_table.add_row("Request Throughput", f"{metrics.request_throughput:.2f} req/s")
    throughput_table.add_row("Total Token Throughput", f"{metrics.total_token_throughput:.2f} tok/s")
    throughput_table.add_row("  Prompt Tokens", f"{metrics.prompt_token_throughput:.2f} tok/s")
    throughput_table.add_row("  Completion Tokens", f"{metrics.completion_token_throughput:.2f} tok/s")
    console.print(throughput_table)

    # Token Counts Table
    token_table = Table(title="Token Counts", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    token_table.add_column("Token Type", style="cyan")
    token_table.add_column("Count", justify="right", style="green")

    token_table.add_row("Prompt Tokens", f"{metrics.total_prompt_tokens:,}")
    token_table.add_row("Completion Tokens", f"{metrics.total_completion_tokens:,}")
    token_table.add_row("Total Tokens", f"{metrics.total_tokens:,}")
    console.print(token_table)

    # Cache Statistics (if applicable)
    if metrics.cache_hit_count > 0:
        cache_table = Table(title="Cache Statistics", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("Value", justify="right", style="green")

        cache_table.add_row("Cache Hits", str(metrics.cache_hit_count))
        cache_table.add_row("Cache Hit Rate", f"{metrics.cache_hit_rate*100:.1f}%")
        console.print(cache_table)

    # Latency metrics
    if metrics.ttft_stats:
        _print_metric_stats_table(metrics.ttft_stats, "Time to First Token (TTFT)")

    if metrics.e2el_stats:
        _print_metric_stats_table(metrics.e2el_stats, "End-to-End Latency (E2EL)")

    if metrics.tpot_stats:
        _print_metric_stats_table(metrics.tpot_stats, "Time Per Output Token (TPOT)")

    if metrics.itl_stats:
        _print_metric_stats_table(metrics.itl_stats, "Inter-Token Latency (ITL)")

    if metrics.queue_time_stats:
        _print_metric_stats_table(metrics.queue_time_stats, "Queue Time (Server-Side)")

    console.print()


def _print_metric_stats_table(stats: MetricStats, title: str):
    """Helper to print metric statistics as a Rich table."""
    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value (ms)", justify="right", style="green")

    table.add_row("Mean", f"{stats.mean*1000:.2f}")
    table.add_row("Median", f"{stats.median*1000:.2f}")
    table.add_row("Std Dev", f"{stats.std*1000:.2f}")
    table.add_row("Min", f"{stats.min*1000:.2f}")
    table.add_row("Max", f"{stats.max*1000:.2f}")

    for p, v in sorted(stats.percentiles.items()):
        table.add_row(f"p{p}", f"{v*1000:.2f}")

    console.print(table)


def print_language_comparison(metrics_by_lang: Dict[str, BenchmarkMetrics]):
    """Print comparison table across languages using Rich."""
    if len(metrics_by_lang) < 2:
        return

    languages = sorted(metrics_by_lang.keys())
    console.print()
    console.print(Panel("[bold cyan]Language Comparison[/bold cyan]", box=box.DOUBLE))

    # Extract percentiles from first metric (they should all be the same)
    first_metric = next(iter(metrics_by_lang.values()))
    if first_metric.ttft_stats:
        percentiles = sorted(first_metric.ttft_stats.percentiles.keys())

        # TTFT Comparison
        ttft_table = Table(
            title="Time to First Token (TTFT) - milliseconds",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        ttft_table.add_column("Percentile", style="cyan", justify="left")
        for lang in languages:
            ttft_table.add_column(lang.upper(), justify="right", style="green")
        if len(languages) == 2:
            ttft_table.add_column("Difference (%)", justify="right", style="yellow")

        for p in percentiles:
            row = [f"p{p}"]
            values = []
            for lang in languages:
                metric = metrics_by_lang[lang]
                if metric.ttft_stats and p in metric.ttft_stats.percentiles:
                    val = metric.ttft_stats.percentiles[p] * 1000
                    values.append(val)
                    row.append(f"{val:.1f}")
                else:
                    row.append("N/A")

            if len(values) == 2 and values[0] > 0:
                # Calculate percentage change: ((Japanese - English) / English) * 100
                pct_diff = ((values[1] - values[0]) / values[0]) * 100
                color = "red" if pct_diff > 0 else "green" if pct_diff < 0 else "white"
                row.append(f"[{color}]{pct_diff:+.1f}%[/{color}]")
            elif len(values) == 2:
                row.append("N/A")

            ttft_table.add_row(*row)

        console.print(ttft_table)

    # E2E Latency comparison
    if first_metric.e2el_stats:
        e2el_table = Table(
            title="End-to-End Latency (E2EL) - milliseconds",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        e2el_table.add_column("Percentile", style="cyan", justify="left")
        for lang in languages:
            e2el_table.add_column(lang.upper(), justify="right", style="green")
        if len(languages) == 2:
            e2el_table.add_column("Difference (%)", justify="right", style="yellow")

        for p in percentiles:
            row = [f"p{p}"]
            values = []
            for lang in languages:
                metric = metrics_by_lang[lang]
                if metric.e2el_stats and p in metric.e2el_stats.percentiles:
                    val = metric.e2el_stats.percentiles[p] * 1000
                    values.append(val)
                    row.append(f"{val:.1f}")
                else:
                    row.append("N/A")

            if len(values) == 2 and values[0] > 0:
                # Calculate percentage change: ((Japanese - English) / English) * 100
                pct_diff = ((values[1] - values[0]) / values[0]) * 100
                color = "red" if pct_diff > 0 else "green" if pct_diff < 0 else "white"
                row.append(f"[{color}]{pct_diff:+.1f}%[/{color}]")
            elif len(values) == 2:
                row.append("N/A")

            e2el_table.add_row(*row)

        console.print(e2el_table)

    # Server-Side Timing comparison
    if first_metric.server_total_time_stats:
        server_table = Table(
            title="Server-Side Breakdown - milliseconds",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        server_table.add_column("Metric", style="cyan", justify="left")
        server_table.add_column("Percentile", style="cyan", justify="left")
        for lang in languages:
            server_table.add_column(lang.upper(), justify="right", style="green")
        if len(languages) == 2:
            server_table.add_column("Difference (%)", justify="right", style="yellow")

        # Queue Time
        if first_metric.queue_time_stats:
            for p in percentiles:
                row = ["Queue Time", f"p{p}"]
                values = []
                for lang in languages:
                    metric = metrics_by_lang[lang]
                    if metric.queue_time_stats and p in metric.queue_time_stats.percentiles:
                        val = metric.queue_time_stats.percentiles[p] * 1000
                        values.append(val)
                        row.append(f"{val:.1f}")
                    else:
                        row.append("N/A")
                if len(values) == 2 and values[0] > 0:
                    pct_diff = ((values[1] - values[0]) / values[0]) * 100
                    color = "red" if pct_diff > 0 else "green" if pct_diff < 0 else "white"
                    row.append(f"[{color}]{pct_diff:+.1f}%[/{color}]")
                elif len(values) == 2:
                    row.append("N/A")
                server_table.add_row(*row)

        # Prompt Time
        if first_metric.prompt_time_stats:
            for p in percentiles:
                row = ["Prompt Time", f"p{p}"]
                values = []
                for lang in languages:
                    metric = metrics_by_lang[lang]
                    if metric.prompt_time_stats and p in metric.prompt_time_stats.percentiles:
                        val = metric.prompt_time_stats.percentiles[p] * 1000
                        values.append(val)
                        row.append(f"{val:.1f}")
                    else:
                        row.append("N/A")
                if len(values) == 2 and values[0] > 0:
                    pct_diff = ((values[1] - values[0]) / values[0]) * 100
                    color = "red" if pct_diff > 0 else "green" if pct_diff < 0 else "white"
                    row.append(f"[{color}]{pct_diff:+.1f}%[/{color}]")
                elif len(values) == 2:
                    row.append("N/A")
                server_table.add_row(*row)

        # Completion Time
        if first_metric.completion_time_stats:
            for p in percentiles:
                row = ["Completion Time", f"p{p}"]
                values = []
                for lang in languages:
                    metric = metrics_by_lang[lang]
                    if metric.completion_time_stats and p in metric.completion_time_stats.percentiles:
                        val = metric.completion_time_stats.percentiles[p] * 1000
                        values.append(val)
                        row.append(f"{val:.1f}")
                    else:
                        row.append("N/A")
                if len(values) == 2 and values[0] > 0:
                    pct_diff = ((values[1] - values[0]) / values[0]) * 100
                    color = "red" if pct_diff > 0 else "green" if pct_diff < 0 else "white"
                    row.append(f"[{color}]{pct_diff:+.1f}%[/{color}]")
                elif len(values) == 2:
                    row.append("N/A")
                server_table.add_row(*row)

        # Server Total Time
        for p in percentiles:
            row = ["Server Total", f"p{p}"]
            values = []
            for lang in languages:
                metric = metrics_by_lang[lang]
                if metric.server_total_time_stats and p in metric.server_total_time_stats.percentiles:
                    val = metric.server_total_time_stats.percentiles[p] * 1000
                    values.append(val)
                    row.append(f"{val:.1f}")
                else:
                    row.append("N/A")
            if len(values) == 2 and values[0] > 0:
                pct_diff = ((values[1] - values[0]) / values[0]) * 100
                color = "red" if pct_diff > 0 else "green" if pct_diff < 0 else "white"
                row.append(f"[{color}]{pct_diff:+.1f}%[/{color}]")
            elif len(values) == 2:
                row.append("N/A")
            server_table.add_row(*row)

        console.print(server_table)

    # Network Overhead - separate section
    if first_metric.network_time_stats:
        network_table = Table(
            title="Network Overhead - milliseconds",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        network_table.add_column("Percentile", style="cyan", justify="left")
        for lang in languages:
            network_table.add_column(lang.upper(), justify="right", style="green")
        if len(languages) == 2:
            network_table.add_column("Difference (%)", justify="right", style="yellow")

        for p in percentiles:
            row = [f"p{p}"]
            values = []
            for lang in languages:
                metric = metrics_by_lang[lang]
                if metric.network_time_stats and p in metric.network_time_stats.percentiles:
                    val = metric.network_time_stats.percentiles[p] * 1000
                    values.append(val)
                    row.append(f"{val:.1f}")
                else:
                    row.append("N/A")
            if len(values) == 2 and values[0] > 0:
                pct_diff = ((values[1] - values[0]) / values[0]) * 100
                color = "red" if pct_diff > 0 else "green" if pct_diff < 0 else "white"
                row.append(f"[{color}]{pct_diff:+.1f}%[/{color}]")
            elif len(values) == 2:
                row.append("N/A")
            network_table.add_row(*row)

        console.print(network_table)

    console.print()


def save_results(
    metrics: BenchmarkMetrics,
    output_file: str,
    outputs: Optional[List[RequestOutput]] = None
):
    """Save benchmark results to JSON file."""
    result = {
        "metrics": metrics.to_dict(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if outputs:
        # Save raw request data (optional, can be large)
        result["raw_outputs"] = [
            {
                "success": o.success,
                "language": o.language,
                "ttft": o.ttft,
                "latency": o.latency,
                "tpot": o.tpot,
                "prompt_tokens": o.prompt_tokens,
                "completion_tokens": o.completion_tokens,
                "queue_time": o.queue_time,
                "cache_hit": o.cache_hit,
                "error": o.error,
            }
            for o in outputs
        ]

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    console.print(f"[green]Results saved to: {output_file}[/green]")
