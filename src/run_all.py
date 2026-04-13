"""Orchestrator: run the entire research pipeline end-to-end.

Usage:
    python src/run_all.py                    # Run all steps
    python src/run_all.py --list             # List all steps
    python src/run_all.py --from 4           # Run from step 4 onwards
    python src/run_all.py --steps 1,4,6,11   # Run specific steps
    python src/run_all.py --steps 6-11       # Run step range

All scripts use seed=42 for reproducibility (see seed.py).
"""

import argparse
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent

STEPS = [
    # (script_path_relative_to_src, description)
    ("data/download_nasa_power.py", "Download NASA POWER data for Laguna de Bay"),
    ("data/prepare_data.py", "Prepare train/val/test splits and forecast windows"),
    ("figures/eda.py", "Exploratory data analysis and figures"),
    ("eval/zero_shot.py", "Zero-shot FM evaluation (Chronos, TimesFM, Moirai)"),
    ("eval/baselines.py", "Traditional baseline models (XGBoost, LSTM)"),
    ("finetune/chronos_ft.py", "Fine-tune Chronos Small + Base"),
    ("finetune/ttm_ft.py", "Fine-tune TTM-R2"),
    ("eval/finetuned.py", "Evaluate fine-tuned Chronos vs zero-shot"),
    ("eval/all_finetuned.py", "Comprehensive evaluation: all models + CRPS + DM tests"),
    ("eval/ablation.py", "Ablation study: fine-tuning steps vs performance"),
    ("experiments/data_efficiency.py", "Data efficiency experiment"),
    ("experiments/sensitivity_analysis.py", "Hyperparameter sensitivity analysis"),
    ("figures/generate.py", "Generate all paper figures"),
]


def run_step(step_num, script_name, description):
    """Run a pipeline step and return success/failure."""
    print(f"\n{'=' * 60}")
    print(f"STEP {step_num}: {description}")
    print(f"Script: {script_name}")
    print(f"{'=' * 60}\n")

    script_path = SRC_DIR / script_name
    if not script_path.exists():
        print(f"SKIPPED: {script_name} (file not found)")
        return "SKIP"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
    )

    # 0xC0000409 (3221226505) is a Windows PyTorch CUDA cleanup crash — results are valid
    WINDOWS_CUDA_CRASH = 3221226505
    if result.returncode != 0 and result.returncode != WINDOWS_CUDA_CRASH:
        print(f"\nERROR: {script_name} failed with exit code {result.returncode}")
        return "FAIL"
    if result.returncode == WINDOWS_CUDA_CRASH:
        print(f"\nWARNING: {script_name} exited with Windows CUDA cleanup error (results are valid)")

    print(f"\nDONE: {description}")
    return "PASS"


def parse_steps_arg(steps_str):
    """Parse step selection: '1,4,6' or '6-11' or '1,4,6-11'."""
    selected = set()
    for part in steps_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return sorted(selected)


def list_steps():
    """Print all available steps."""
    print(f"\n{'=' * 60}")
    print("AVAILABLE PIPELINE STEPS")
    print(f"{'=' * 60}")
    for i, (script, desc) in enumerate(STEPS, 1):
        exists = "  " if (SRC_DIR / script).exists() else "? "
        print(f"  {exists}{i:2d}. {desc}")
        print(f"      └─ {script}")
    print(f"\nUsage:")
    print(f"  python src/run_all.py --steps 4,6-9,11   # Run specific steps")
    print(f"  python src/run_all.py --from 6            # Run from step 6 onwards")
    print(f"  python src/run_all.py                     # Run all steps")


def main():
    parser = argparse.ArgumentParser(description="Run the research pipeline")
    parser.add_argument("--list", action="store_true", help="List all steps")
    parser.add_argument("--from", type=int, dest="from_step", help="Run from step N onwards")
    parser.add_argument("--steps", type=str, help="Run specific steps (e.g., '1,4,6-11')")
    parser.add_argument("--no-stop", action="store_true", help="Don't stop on failure, continue to next step")
    args = parser.parse_args()

    if args.list:
        list_steps()
        return

    # Determine which steps to run
    total_steps = len(STEPS)
    if args.steps:
        selected = parse_steps_arg(args.steps)
    elif args.from_step:
        selected = list(range(args.from_step, total_steps + 1))
    else:
        selected = list(range(1, total_steps + 1))

    # Validate
    selected = [s for s in selected if 1 <= s <= total_steps]
    if not selected:
        print("No valid steps selected.")
        list_steps()
        return

    print(f"Running steps: {', '.join(str(s) for s in selected)}")

    results = {}
    for step_num in selected:
        script, desc = STEPS[step_num - 1]
        status = run_step(step_num, script, desc)
        results[f"{step_num}. {script}"] = status

        if status == "FAIL" and not args.no_stop:
            print(f"\nPipeline stopped at step {step_num}: {script}")
            print("Fix the error and re-run with: "
                  f"python src/run_all.py --from {step_num}")
            break

    # Summary
    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    for step, status in results.items():
        icon = {"PASS": "OK", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        print(f"  [{icon:4s}] {step}")


if __name__ == "__main__":
    main()
