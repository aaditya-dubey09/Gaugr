from __future__ import annotations

import json
import time
from typing import Any
import psutil
import requests

LOG_FILE = "gaugr_benchmarks.json"
REPORT_FILE = "benchmark_report.md"


def get_installed_models() -> list[str]:
    """Query local Ollama instance to retrieve all available models."""
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            if models:
                return models
    except Exception:
        pass
    return ["qwen2.5-coder:1.5b"]


def format_bytes(value: int) -> str:
    """Format raw byte counts into human-readable units."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def get_hardware_snapshot() -> tuple[float, float, str, int, int]:
    """Capture system CPU load, RAM utilization, thermal readings, and memory usage."""
    cpu = psutil.cpu_percent(interval=0.3)
    ram = psutil.virtual_memory()
    thermal = "n/a"

    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        sensors = {}

    hottest: tuple[str, float] | None = None
    for sensor_name, entries in sensors.items():
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is None:
                continue
            if hottest is None or current > hottest[1]:
                hottest = (sensor_name, float(current))

    if hottest is not None:
        thermal = f"{hottest[1]:.1f}°C ({hottest[0].replace('_', ' ')})"

    return cpu, ram.percent, thermal, ram.used, ram.total


def save_benchmark_log(prompt: str, results: list[dict[str, Any]]) -> None:
    """Append structured benchmark results to the local JSON log history file."""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "results": results
    }
    try:
        try:
            with open(LOG_FILE, "r") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        
        history.append(log_entry)
        with open(LOG_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def export_markdown_report(prompt: str, results: list[dict[str, Any]]) -> None:
    """Overwrite the latest markdown report with current benchmark performance metrics."""
    md_content = f"# Gaugr: Local LLMs Benchmark Report\n\n"
    md_content += f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Prompt:** `{prompt}`\n\n"
    md_content += "| Model | Status | TTFT | Total Time | Speed (tok/s) | Tokens | Score |\n"
    md_content += "| :--- | :---: | ---: | ---: | ---: | ---: | ---: |\n"
    
    for res in results:
        status = "PASS" if res["success"] else "FAIL"
        ttft_str = f"{res['ttft_seconds'] * 1000:.1f} ms" if res["success"] else "n/a"
        md_content += f"| `{res['model']}` | {status} | {ttft_str} | {res['total_time']:.2f}s | **{res['tokens_per_sec']:.2f}** | {res['tokens']} | {res['score']} |\n"
    
    md_content += "\n## Output Previews\n\n"
    for res in results:
        if res["success"]:
            md_content += f"### Model: `{res['model']}`\n```text\n{res['response']}\n```\n\n"

    try:
        with open(REPORT_FILE, "w") as f:
            f.write(md_content)
    except Exception:
        pass
