from __future__ import annotations

import concurrent.futures
import time
from typing import Any
import psutil
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from src.utils import (
    get_installed_models,
    get_hardware_snapshot,
    format_bytes,
    save_benchmark_log,
    export_markdown_report,
)

DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_PROMPT = "Write a quick Python script to reverse a string."
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
DEFAULT_TEMPERATURE = "0.2"

console = Console()


def neon_banner() -> Panel:
    """Build the startup banner for the benchmark session.

    Returns:
        A styled ``rich.panel.Panel`` that introduces the CLI and explains the
        user inputs required to run the benchmark.
    """
    # Use dense Rich text segments so the banner feels like a command center header.
    title = Text()
    title.append("Gaugr", style="bold bright_cyan")
    title.append("  |  ", style="dim")
    title.append("Concurrent Local LLM Benchmarking Suite", style="bold green")

    # Keep the usage guidance compact and explicit to reduce operator friction.
    subtitle = Text()
    subtitle.append("How to use:\n", style="bold bright_yellow")
    subtitle.append(
        " • Press Enter on any prompt to accept the default value shown in [brackets].\n",
        style="dim",
    )
    subtitle.append(" • Prompt: ", style="cyan")
    subtitle.append(
        "The instruction or question you want the local AI models to answer.\n",
        style="white",
    )
    subtitle.append(" • System Prompt: ", style="cyan")
    subtitle.append(
        "Background persona or rules for the AI (e.g., 'You are a helpful coding assistant').\n",
        style="white",
    )
    subtitle.append(" • Temperature: ", style="cyan")
    subtitle.append(
        "Creativity level from 0.0 (strict/deterministic) to 1.0 (creative/chaotic).\n",
        style="white",
    )
    subtitle.append(" • API URL: ", style="cyan")
    subtitle.append(
        "Your local Ollama endpoint (default: http://localhost:11434/api/generate).",
        style="white",
    )

    return Panel(
        Text.assemble(title, "\n\n", subtitle),
        border_style="bright_cyan",
        title="[bold bright_green]WELCOME & USAGE GUIDE[/bold bright_green]",
        subtitle="[bright_cyan]v1.0 CLI Arena[/bright_cyan]",
        padding=(1, 2),
    )


def hardware_panel() -> Panel:
    """Build the host telemetry snapshot panel.

    Returns:
        A ``rich.panel.Panel`` containing the current CPU, RAM, thermal, and
        memory baseline values for the local machine.
    """
    # Capture a single snapshot so the displayed values stay internally consistent.
    cpu, ram_percent, thermal, ram_used, ram_total = get_hardware_snapshot()
    rows = Table.grid(padding=(0, 1))
    rows.add_column(style="bold bright_cyan", justify="right")
    rows.add_column(style="white")
    rows.add_row("Host CPU Baseline", f"{cpu:.1f}%")
    rows.add_row("Host RAM Baseline", f"{ram_percent:.1f}%")
    rows.add_row("Host Thermals", thermal)
    rows.add_row(
        "System Memory", f"{format_bytes(ram_used)} / {format_bytes(ram_total)}"
    )
    return Panel(
        rows,
        border_style="bright_magenta",
        title="[bold bright_magenta]HOST SYSTEM TELEMETRY (BASELINE)[/bold bright_magenta]",
    )


def benchmark_single_model(
    model: str, prompt: str, system_prompt: str, temperature: float, url: str
) -> dict[str, Any]:
    """Benchmark one model against the configured local API endpoint.

    Args:
        model: The model name submitted to the inference endpoint.
        prompt: The user prompt to evaluate.
        system_prompt: The system instruction used to shape model behavior.
        temperature: The generation temperature passed to the endpoint.
        url: The local inference API URL.

    Returns:
        A dictionary of normalized benchmark results. Successful and failed
        requests share the same output shape so the caller can sort and render
        the results without branching on error types.
    """
    # Package the benchmark request in the format expected by Ollama-style APIs.
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "options": {"temperature": temperature},
        "stream": False,
    }

    # Measure resource deltas around the request so the report can include a rough cost profile.
    cpu_before = psutil.cpu_percent(interval=None)
    ram_before = psutil.virtual_memory().percent

    start_time = time.perf_counter()
    try:
        # Keep the request synchronous so latency math stays straightforward and reproducible.
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        total_time = time.perf_counter() - start_time

        # Convert nanosecond durations from the provider into seconds for human-readable output.
        prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1_000_000_000
        ttft = prompt_eval_duration if prompt_eval_duration > 0 else (total_time * 0.2)

        # Prefer provider token counts, but fall back to a local approximation when needed.
        response_text = data.get("response", "")
        eval_count = data.get("eval_count", len(response_text.split()))
        eval_duration = data.get("eval_duration", 0) / 1_000_000_000

        # Compute throughput from provider timings when available.
        tokens_per_sec = (
            (eval_count / eval_duration)
            if eval_duration > 0
            else (eval_count / max(total_time - ttft, 0.1))
        )

        # Capture the post-request snapshot to report a simple resource delta.
        cpu_after = psutil.cpu_percent(interval=None)
        ram_after = psutil.virtual_memory().percent

        # Rank faster, denser responses higher with a coarse composite score.
        efficiency_score = round(
            tokens_per_sec * (eval_count / max(total_time, 1.0)), 2
        )

        return {
            "model": model,
            "success": True,
            "response": response_text.strip(),
            "ttft_seconds": ttft,
            "total_time": total_time,
            "tokens_per_sec": tokens_per_sec,
            "tokens": eval_count,
            "score": efficiency_score,
            "cpu_delta": round(cpu_after - cpu_before, 1),
            "ram_delta": round(ram_after - ram_before, 1),
        }
    except Exception as exc:
        # Normalize failures into the same schema so the caller can keep the ranking pipeline simple.
        return {
            "model": model,
            "success": False,
            "error": str(exc),
            "ttft_seconds": 0.0,
            "total_time": time.perf_counter() - start_time,
            "tokens_per_sec": 0.0,
            "tokens": 0,
            "score": 0.0,
            "cpu_delta": 0.0,
            "ram_delta": 0.0,
            "response": "",
        }


def run_arena() -> None:
    """Run the full interactive benchmark workflow.

    The function gathers the benchmark inputs, discovers available local
    models, executes requests concurrently, and renders a ranked summary plus
    response previews.
    """
    # Start with a clean screen so the session feels like a dedicated CLI tool.
    console.clear()
    console.print(neon_banner())
    console.print(hardware_panel())

    # Collect all benchmark parameters before any network requests are issued.
    prompt = Prompt.ask(
        "[bold bright_cyan]Prompt[/bold bright_cyan]",
        default=DEFAULT_PROMPT,
        console=console,
    ).strip()
    system_prompt = Prompt.ask(
        "[bold bright_cyan]System Prompt[/bold bright_cyan]",
        default=DEFAULT_SYSTEM_PROMPT,
        console=console,
    ).strip()
    temperature = float(
        Prompt.ask(
            "[bold bright_cyan]Temperature (0.0 - 1.0)[/bold bright_cyan]",
            default=DEFAULT_TEMPERATURE,
            console=console,
        )
    )
    # Keep the local API endpoint configurable for different machines and environments.
    url = Prompt.ask(
        "[bold bright_cyan]Local API URL[/bold bright_cyan]",
        default=DEFAULT_URL,
        console=console,
    ).strip()

    # Discover the installed models at runtime so the benchmark list stays current.
    available_models = get_installed_models()
    console.print(
        f"[bold yellow]Detected local models:[/bold yellow] [dim]{', '.join(available_models)}[/dim]"
    )

    selected_input = Prompt.ask(
        "[bold bright_cyan]Models to benchmark (comma-separated or 'all')[/bold bright_cyan]",
        default="all",
        console=console,
    ).strip()
    models_to_test = (
        available_models
        if selected_input.lower() == "all"
        else [m.strip() for m in selected_input.split(",") if m.strip()]
    )

    # Announce the fan-out so the user sees that the benchmark is actively dispatching work.
    console.print(
        f"\n[bold yellow]Dispatching prompt concurrently to {len(models_to_test)} models...[/bold yellow]\n"
    )

    results = []
    # Threading is appropriate here because each benchmark is primarily I/O bound.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_model = {
            executor.submit(
                benchmark_single_model,
                model,
                prompt,
                system_prompt,
                temperature,
                url,
            ): model
            for model in models_to_test
        }
        for future in concurrent.futures.as_completed(future_to_model):
            results.append(future.result())

    # Sort by throughput so the most responsive model appears first in the final table.
    results.sort(key=lambda x: x["tokens_per_sec"], reverse=True)

    # Persist the run in both machine-readable and human-readable formats.
    save_benchmark_log(prompt, results)
    export_markdown_report(prompt, results)

    # Customized Table with Bright Cyan/Magenta borders
    table = Table(
        title="Multi-Model Performance Matrix (Ranked)",
        title_style="bold bright_magenta",
        border_style="dim bright_magenta",
        header_style="bold white",
        show_header=True,
    )
    table.add_column(
        "Rank", style="bold magenta", justify="center", header_style="bold white"
    )
    table.add_column("Model", style="cyan", no_wrap=True, header_style="bold white")
    table.add_column("Status", style="bold green", header_style="bold white")
    table.add_column("TTFT", justify="right", header_style="bold white")
    table.add_column(
        "Speed", justify="right", style="yellow", header_style="bold white"
    )
    table.add_column("Tokens", justify="right", header_style="bold white")
    table.add_column(
        "Score", justify="right", style="bold green", header_style="bold white"
    )

    # Render the ranking row-by-row so operators can scan the results quickly.
    for idx, res in enumerate(results, start=1):
        status = "[green]SUCCESS[/green]" if res["success"] else "[red]FAILED[/red]"
        table.add_row(
            f"#{idx}",
            res["model"],
            status,
            f"{res['ttft_seconds'] * 1000:.1f} ms" if res["success"] else "n/a",
            f"{res['tokens_per_sec']:.2f} tok/s" if res["success"] else "0.00",
            str(res["tokens"]),
            str(res["score"]),
        )

    console.print(table)
    console.print(
        "[dim]📂 Exported markdown report to benchmark_report.md & logged to gaugr_benchmarks.json[/dim]\n"
    )

    # Show a short preview of each successful completion without flooding the terminal.
    for res in results:
        if res["success"]:
            console.print(
                Panel(
                    res["response"][:400]
                    + ("..." if len(res["response"]) > 400 else ""),
                    title=f"[bold magenta]{res['model']} Output Preview[/bold magenta]",
                    border_style="dim bright_magenta",
                )
            )


if __name__ == "__main__":
    run_arena()
