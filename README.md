# Gaugr TUI

Gaugr TUI is a terminal-based AI engine benchmark and local model performance monitor built with Python and Rich. It is designed for quick, repeatable benchmarking of local inference endpoints such as Ollama while keeping an eye on system health in real time.

Created for **MLH Global Hack Week** with **GitHub Copilot**.

## Features

- **Real-time hardware monitoring** for CPU usage, RAM usage, and thermal/memory status.
- **Prompt-driven benchmarking** with a clean form for the prompt, API URL, and model name.
- **TTFT tracking** to measure time to first token.
- **Tokens/sec measurement** for throughput analysis.
- **Total response timing** for end-to-end latency visibility.
- **Rich live log output** for streamed model responses and benchmark metrics.

## Requirements

- Python 3.13+
- `psutil`
- `requests`
- `rich`

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the application:

```bash
python main.py
```

By default, the benchmark panel targets:

```text
http://localhost:11434/api/generate
```

Enter your prompt, set the model name, and run the benchmark.

## Benchmark Workflow

1. Launch the script.
2. Enter a prompt when prompted.
3. Confirm or change the local API URL.
4. Enter the model name you want to evaluate.
5. Review the streamed output and the final metrics report.

## Notes

- The app is optimized for local inference endpoints that expose an Ollama-style `generate` API.
- Hardware metrics are captured at launch and shown in the summary banner.
- If your system does not expose thermal sensors through `psutil`, thermals may appear as `n/a`.
