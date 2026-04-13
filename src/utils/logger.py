"""Structured logging utility for the research pipeline.

Logs all experiment results, metrics, and metadata as JSON files
in results/logs/ for easy analysis and reproducibility.
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent.parent / "results" / "logs"


def get_device():
    """Get the best available device: CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_system_info():
    """Capture system metadata."""
    import torch
    device = get_device()
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "device": device,
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def create_log(script_name: str, config: dict = None):
    """Create a new log entry. Returns a log dict to be populated."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "script": script_name,
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
        "system": get_system_info(),
        "config": config or {},
        "steps": [],
        "results": {},
        "errors": [],
    }


def log_step(log: dict, step_name: str, details: dict = None):
    """Log a pipeline step with timing."""
    log["steps"].append({
        "step": step_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    })


def log_metrics(log: dict, model_name: str, horizon: str, season: str, metrics: dict):
    """Log evaluation metrics for a model/horizon/season combination."""
    key = f"{model_name}|{horizon}|{season}"
    log["results"][key] = {
        "model": model_name,
        "horizon": horizon,
        "season": season,
        **metrics,
    }


def log_error(log: dict, error_msg: str):
    """Log an error."""
    log["errors"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error_msg,
    })


def save_log(log: dict):
    """Save the log to a JSON file. Overwrites previous log for the same script."""
    log["timestamp_end"] = datetime.now(timezone.utc).isoformat()

    # Calculate duration
    start = datetime.fromisoformat(log["timestamp_start"])
    end = datetime.fromisoformat(log["timestamp_end"])
    log["duration_seconds"] = (end - start).total_seconds()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    filepath = LOG_DIR / f"{log['script']}.json"

    with open(filepath, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\nLog saved to {filepath}")
    return filepath


def save_summary(all_results: list, filename: str):
    """Save a summary of all results as both JSON and markdown."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = LOG_DIR / f"{filename}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": all_results,
        }, f, indent=2, default=str)

    # Markdown table
    md_path = LOG_DIR / f"{filename}.md"
    if all_results:
        headers = list(all_results[0].keys())
        with open(md_path, "w") as f:
            f.write(f"# {filename}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in all_results:
                values = []
                for h in headers:
                    v = row.get(h, "")
                    if isinstance(v, float):
                        values.append(f"{v:.4f}")
                    else:
                        values.append(str(v))
                f.write("| " + " | ".join(values) + " |\n")

    print(f"Summary saved to {json_path} and {md_path}")
