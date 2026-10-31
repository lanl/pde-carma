"""
Run infer_flowformer.py sequentially for multiple checkpoints and ODE-step counts.

For every configuration, this script:
  1. Prints inference output live.
  2. Saves the complete output to a separate log file.
  3. Extracts:
       - Mean NSP MSE
       - Mean rollout MSE
       - Cyclic inference time
  4. Updates one consolidated summary after every run.

The inference script itself controls evaluation coverage:
    max_T_rollout = T
    num_traj_rollout = N
    num_eval = N * (T - 1)

Run from the pde-flowformer project directory:

    python run_ode_inference_sweep.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

RUNS = [
    {
        "dataset": "ce-rpui",
        "model": "flowformer-mi",
        "checkpoint": "models/flowformer-mi_ce-rpui_2000.pth",
    },
    {
        "dataset": "ce-rpui",
        "model": "flowformer-s",
        "checkpoint": "models/flowformer-s_ce-rpui_2000.pth",
    },
    {
        "dataset": "ce-rpui",
        "model": "flowformer-m",
        "checkpoint": "models/flowformer-m_ce-rpui_2000.pth",
    },
    {
        "dataset": "ns-pwc",
        "model": "flowformer-mi",
        "checkpoint": "models/flowformer-mi_ns-pwc_1773.pth",
    },
    {
        "dataset": "ns-pwc",
        "model": "flowformer-s",
        "checkpoint": "models/flowformer-s_ns-pwc_1198.pth",
    },
    {
        "dataset": "ns-pwc",
        "model": "flowformer-m",
        "checkpoint": "models/flowformer-m_ns-pwc_1552.pth",
    },
]


ODE_STEPS = [1, 5, 20, 50, 100]

CYCLIC_RUNS = 1

# False means one failed configuration will not stop the remaining runs.
STOP_ON_ERROR = False


# -----------------------------------------------------------------------------
# Output patterns
# -----------------------------------------------------------------------------

NSP_PATTERN = re.compile(
    r"Mean\s+NSP\s+MSE\s*:\s*([-+0-9.eE]+)",
    re.IGNORECASE,
)

ROLLOUT_PATTERN = re.compile(
    r"Mean\s+rollout\s+MSE\s*:\s*([-+0-9.eE]+)",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"(?:K=\d+\s+)?total\s+inference\s+time\s*:"
    r"\s*([-+0-9.eE]+)\s*s",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Find the directory containing infer_flowformer.py.
    """

    script_dir = Path(__file__).resolve().parent

    candidates = [
        Path.cwd().resolve(),
        script_dir,
        script_dir.parent,
    ]

    for candidate in candidates:
        if (candidate / "infer_flowformer.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find infer_flowformer.py. Place this script in the "
        "pde-flowformer project directory or run it from that directory."
    )


def last_match(
    pattern: re.Pattern[str],
    text: str,
) -> Optional[float]:
    """
    Return the last matching numeric value.
    """

    matches = pattern.findall(text)

    if not matches:
        return None

    try:
        return float(matches[-1])
    except ValueError:
        return None


def format_value(
    value: Optional[float],
    decimals: int = 10,
) -> str:
    """
    Format a metric value for the summary.
    """

    if value is None:
        return "NOT FOUND"

    return f"{value:.{decimals}g}"


# -----------------------------------------------------------------------------
# Summary writer
# -----------------------------------------------------------------------------

def write_summary(
    summary_path: Path,
    results: list[dict],
    started_at: datetime,
    total_expected: int,
) -> None:
    """
    Rewrite the summary after every completed configuration.

    This ensures partial results remain available if the sweep is interrupted.
    """

    completed = len(results)

    succeeded = sum(
        result["status"] == "SUCCESS"
        for result in results
    )

    failed = completed - succeeded

    lines = [
        "FLOWFORMER ODE INFERENCE SWEEP SUMMARY",
        "=" * 112,
        f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Completed configurations: {completed}/{total_expected}",
        f"Successful: {succeeded}",
        f"Failed or incomplete: {failed}",
        "",
        (
            "Evaluation settings: full-test NSP, full-test rollout, "
            "all available time steps, "
            f"cyclic_runs={CYCLIC_RUNS}"
        ),
        "",
        (
            f"{'Dataset':<12} "
            f"{'Model':<16} "
            f"{'ODE':>5} "
            f"{'Mean NSP MSE':>16} "
            f"{'Mean rollout MSE':>20} "
            f"{'Cyclic time (s)':>18} "
            f"{'Status':>12}"
        ),
        "-" * 112,
    ]

    for result in results:
        lines.append(
            f"{result['dataset']:<12} "
            f"{result['model']:<16} "
            f"{result['ode_steps']:>5} "
            f"{format_value(result['nsp_mse']):>16} "
            f"{format_value(result['rollout_mse']):>20} "
            f"{format_value(result['inference_time'], decimals=8):>18} "
            f"{result['status']:>12}"
        )

    lines.extend(
        [
            "",
            "DETAILED RESULTS",
            "=" * 112,
        ]
    )

    for result in results:
        lines.extend(
            [
                f"Dataset: {result['dataset']}",
                f"Model: {result['model']}",
                f"Checkpoint: {result['checkpoint']}",
                f"ODE steps: {result['ode_steps']}",
                (
                    "Mean NSP MSE: "
                    f"{format_value(result['nsp_mse'])}"
                ),
                (
                    "Mean rollout MSE: "
                    f"{format_value(result['rollout_mse'])}"
                ),
                (
                    "Cyclic inference time: "
                    f"{format_value(result['inference_time'], decimals=8)} s"
                ),
                f"Status: {result['status']}",
                f"Return code: {result['return_code']}",
                f"Full log: {result['log_path']}",
                "-" * 112,
            ]
        )

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Run one checkpoint and ODE-step configuration
# -----------------------------------------------------------------------------

def run_one_configuration(
    project_root: Path,
    run_config: dict,
    ode_steps: int,
    log_dir: Path,
) -> dict:
    """
    Run infer_flowformer.py for one checkpoint and one ODE-step value.
    """

    checkpoint_relative = Path(run_config["checkpoint"])
    checkpoint_absolute = project_root / checkpoint_relative

    safe_model = run_config["model"].replace("/", "-")
    safe_dataset = run_config["dataset"].replace("/", "-")

    log_path = (
        log_dir
        / f"{safe_model}_{safe_dataset}_ode{ode_steps}.log"
    )

    base_result = {
        "dataset": run_config["dataset"],
        "model": run_config["model"],
        "checkpoint": checkpoint_relative.as_posix(),
        "ode_steps": ode_steps,
        "nsp_mse": None,
        "rollout_mse": None,
        "inference_time": None,
        "status": "FAILED",
        "return_code": -1,
        "log_path": str(log_path),
    }

    if not checkpoint_absolute.is_file():
        message = f"Checkpoint not found: {checkpoint_absolute}"

        print(f"ERROR: {message}", flush=True)

        log_path.write_text(
            message + "\n",
            encoding="utf-8",
        )

        return base_result

    # The inference script controls:
    #   - number of NSP samples
    #   - number of rollout trajectories
    #   - rollout horizon
    command = [
        sys.executable,
        "-u",
        "infer_flowformer.py",
        "--ckpt",
        str(checkpoint_relative),
        "--num_ode_steps",
        str(ode_steps),
        "--cyclic_runs",
        str(CYCLIC_RUNS),
    ]

    print("\n" + "=" * 92, flush=True)

    print(
        f"Dataset={run_config['dataset']} | "
        f"Model={run_config['model']} | "
        f"ODE steps={ode_steps}",
        flush=True,
    )

    print(
        "Command:",
        subprocess.list2cmdline(command),
        flush=True,
    )

    print(
        f"Log file: {log_path}",
        flush=True,
    )

    print("=" * 92, flush=True)

    output_lines: list[str] = []

    try:
        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:

            process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                print(
                    line,
                    end="",
                    flush=True,
                )

                log_file.write(line)
                log_file.flush()

                output_lines.append(line)

            return_code = process.wait()

    except Exception as exc:
        error_message = (
            f"Runner exception: "
            f"{type(exc).__name__}: {exc}\n"
        )

        print(
            error_message,
            end="",
            flush=True,
        )

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(error_message)

        base_result["return_code"] = -1

        return base_result

    output_text = "".join(output_lines)

    nsp_mse = last_match(
        NSP_PATTERN,
        output_text,
    )

    rollout_mse = last_match(
        ROLLOUT_PATTERN,
        output_text,
    )

    inference_time = last_match(
        TIME_PATTERN,
        output_text,
    )

    metrics_found = all(
        value is not None
        for value in (
            nsp_mse,
            rollout_mse,
            inference_time,
        )
    )

    if return_code == 0 and metrics_found:
        status = "SUCCESS"
    else:
        status = "INCOMPLETE"

    return {
        **base_result,
        "nsp_mse": nsp_mse,
        "rollout_mse": rollout_mse,
        "inference_time": inference_time,
        "status": status,
        "return_code": return_code,
    }


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------

def main() -> None:
    project_root = find_project_root()

    results_dir = project_root / "results"
    log_dir = results_dir / "ode_inference_logs"

    summary_path = (
        results_dir
        / "ode_inference_summary.txt"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_expected = len(RUNS) * len(ODE_STEPS)

    started_at = datetime.now()

    results: list[dict] = []

    print(
        f"Project root: {project_root}",
        flush=True,
    )

    print(
        f"Configurations to run: {total_expected}",
        flush=True,
    )

    print(
        f"Summary file: {summary_path}",
        flush=True,
    )

    print(
        f"Per-run logs: {log_dir}",
        flush=True,
    )

    # Create the summary immediately.
    write_summary(
        summary_path=summary_path,
        results=results,
        started_at=started_at,
        total_expected=total_expected,
    )

    run_number = 0

    for run_config in RUNS:
        for ode_steps in ODE_STEPS:
            run_number += 1

            print(
                f"\nStarting configuration "
                f"{run_number}/{total_expected}",
                flush=True,
            )

            result = run_one_configuration(
                project_root=project_root,
                run_config=run_config,
                ode_steps=ode_steps,
                log_dir=log_dir,
            )

            results.append(result)

            write_summary(
                summary_path=summary_path,
                results=results,
                started_at=started_at,
                total_expected=total_expected,
            )

            print(
                "\nExtracted metrics: "
                f"NSP MSE={format_value(result['nsp_mse'])}, "
                f"rollout MSE="
                f"{format_value(result['rollout_mse'])}, "
                f"cyclic inference time="
                f"{format_value(result['inference_time'], decimals=8)} s, "
                f"status={result['status']}",
                flush=True,
            )

            print(
                f"Updated summary: {summary_path}",
                flush=True,
            )

            if (
                STOP_ON_ERROR
                and result["status"] != "SUCCESS"
            ):
                print(
                    "Stopping because STOP_ON_ERROR=True.",
                    flush=True,
                )
                return

    print("\n" + "=" * 92, flush=True)

    print(
        "All requested inference configurations have finished.",
        flush=True,
    )

    print(
        f"Consolidated summary: {summary_path}",
        flush=True,
    )

    print(
        f"Individual logs: {log_dir}",
        flush=True,
    )

    print("=" * 92, flush=True)


if __name__ == "__main__":
    main()