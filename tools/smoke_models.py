#!/usr/bin/env python3
"""Run lightweight synthetic tensor checks against each preserved model.

Each implementation is loaded in its own Python process so that its top-level
``utils`` package cannot collide with another snapshot.  The checks stay on
CPU, work in a temporary directory, and disable bytecode/cache writes beneath
the immutable implementation trees.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


VARIANTS = {
    "dit-unidirectional": ("dit_unidirectional", "dit", False),
    "dit-bidirectional": ("dit_bidirectional", "dit", True),
    "flowformer-unidirectional": (
        "flowformer_unidirectional",
        "flowformer",
        False,
    ),
    "flowformer-bidirectional": (
        "flowformer_bidirectional",
        "flowformer",
        True,
    ),
}


def _repository_root(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend((Path(__file__).resolve().parents[1], Path.cwd()))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "implementations").is_dir():
            return root
    raise FileNotFoundError("Could not locate a checkout containing implementations/")


def _shape(value: Any) -> list[int]:
    return [int(part) for part in value.shape]


def _require_shape(value: Any, expected: tuple[int, ...], label: str) -> None:
    actual = tuple(int(part) for part in value.shape)
    if actual != expected:
        raise AssertionError(f"{label}: expected shape {expected}, got {actual}")


def _require_finite(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise AssertionError(f"{label} contains a NaN or infinity")


def _exercise_training_step(torch: Any, model: Any, output: Any) -> float:
    target = torch.randn_like(output)
    loss = torch.mean((output - target) ** 2)
    _require_finite(torch, loss, "training loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError("synthetic backward pass produced no parameter gradients")
    if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
        raise AssertionError("synthetic backward pass produced a non-finite gradient")
    model.zero_grad(set_to_none=True)
    return float(loss.detach().item())


def _smoke_dit(torch: Any, *, bidirectional: bool) -> dict[str, Any]:
    from utils.my_model import DiTInference
    from utils.vit_diffusion import ViTTime

    common = {
        "in_channels": 2,
        "embed_dim": 16,
        "time_emb_dim": 16,
        "img_size": (16, 16),
        "patch_size": (8, 8),
        "depth": 1,
        "num_heads": 2,
    }
    control = torch.tensor([[0.0, 1.0] if bidirectional else [1.0, 0.0]])
    pair = torch.randn(1, 4, 16, 16)
    time = torch.tensor([0.25])

    model = ViTTime(**common, dit_arch="legacy")
    model.train()
    output = model(pair, time, control)
    _require_shape(output, (1, 2, 16, 16), "DiT forward output")
    _require_finite(torch, output, "DiT forward output")
    loss = _exercise_training_step(torch, model, output)

    model.eval()
    inference = DiTInference(
        model=model,
        config={
            "num_diffusion_steps": 4,
            "samp_diff_steps": 1,
            "beta_start": 1e-4,
            "beta_end": 2e-2,
            "ddim_eta": 0.0,
        },
        device="cpu",
    )
    condition = torch.randn(1, 2, 16, 16)
    path = inference.sample(
        condition,
        samp_diff_steps=1,
        c=control,
        ddim_eta=0.0,
        return_path=True,
    )
    _require_shape(path, (1, 1, 2, 16, 16), "DiT one-step DDIM path")
    _require_finite(torch, path, "DiT one-step DDIM path")

    # Both source-selectable DiT backbones must at least construct and run.
    adaln_model = ViTTime(**common, dit_arch="adaln_zero").eval()
    with torch.inference_mode():
        adaln_output = adaln_model(pair, time, control)
    _require_shape(adaln_output, (1, 2, 16, 16), "adaLN-Zero forward output")
    _require_finite(torch, adaln_output, "adaLN-Zero forward output")

    return {
        "architectures": ["legacy", "adaln_zero"],
        "forward_shape": _shape(output),
        "adaln_forward_shape": _shape(adaln_output),
        "sample_shape": _shape(path),
        "loss": loss,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "control": [float(value) for value in control[0].tolist()],
    }


def _smoke_flowformer(torch: Any, *, bidirectional: bool) -> dict[str, Any]:
    from utils.my_model import FlowFormerInference
    from utils.vit_flow import ViTTime

    model = ViTTime(
        in_channels=2,
        embed_dim=16,
        time_emb_dim=16,
        img_size=(16, 16),
        patch_size=(8, 8),
        depth=1,
        num_heads=2,
    )
    pair = torch.randn(1, 4, 16, 16)
    time = torch.tensor([0.25])
    control = torch.tensor([[0.0, 1.0]]) if bidirectional else None

    model.train()
    output = model(pair, time, control) if bidirectional else model(pair, time)
    _require_shape(output, (1, 2, 16, 16), "FlowFormer forward output")
    _require_finite(torch, output, "FlowFormer forward output")
    loss = _exercise_training_step(torch, model, output)

    model.eval()
    inference = FlowFormerInference(
        model=model,
        config={"num_ode_steps": 1},
        device="cpu",
    )
    condition = torch.randn(1, 2, 16, 16)
    if bidirectional:
        path = inference.sample(condition, num_ode_steps=1, c=control)
    else:
        path = inference.sample(condition, num_ode_steps=1)
    _require_shape(path, (1, 1, 2, 16, 16), "FlowFormer one-step ODE path")
    _require_finite(torch, path, "FlowFormer one-step ODE path")

    return {
        "forward_shape": _shape(output),
        "sample_shape": _shape(path),
        "loss": loss,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "control": None if control is None else [float(value) for value in control[0].tolist()],
    }


def _run_child(variant: str, repository_root: Path) -> int:
    implementation_name, family, bidirectional = VARIANTS[variant]
    implementation = repository_root / "implementations" / implementation_name
    if not implementation.is_dir():
        raise FileNotFoundError(f"Implementation directory not found: {implementation}")

    # The snapshots use top-level imports such as ``utils.vit_flow``.
    sys.path.insert(0, str(implementation))
    import torch

    torch.manual_seed(2027)
    torch.set_num_threads(1)
    details = (
        _smoke_dit(torch, bidirectional=bidirectional)
        if family == "dit"
        else _smoke_flowformer(torch, bidirectional=bidirectional)
    )
    result = {
        "variant": variant,
        "status": "pass",
        "torch_version": str(torch.__version__),
        **details,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _torch_available(python_executable: str, environment: dict[str, str], cwd: str) -> bool:
    try:
        check = subprocess.run(
            [python_executable, "-B", "-c", "import torch"],
            cwd=cwd,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return check.returncode == 0


def _run_parent(args: argparse.Namespace) -> int:
    root = _repository_root(args.repo_root)
    selected = args.variant or list(VARIANTS)
    python_executable = args.python or sys.executable
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="pde-carma-smoke-") as temporary:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TORCH_HOME": temporary,
                "XDG_CACHE_HOME": temporary,
            }
        )
        if not _torch_available(python_executable, environment, temporary):
            message = (
                f"PyTorch is unavailable to {python_executable!r}; model smoke tests "
                "were not executed. Install a CPU/GPU build of torch or pass "
                "--python pointing to an environment that provides it."
            )
            if not args.allow_missing_torch:
                print(f"ERROR: {message}", file=sys.stderr)
                return 2
            results = [
                {"variant": variant, "status": "skip", "reason": message}
                for variant in selected
            ]
        else:
            for variant in selected:
                command = [
                    python_executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--_child",
                    variant,
                    "--repo-root",
                    str(root),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        cwd=temporary,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=args.timeout,
                    )
                except subprocess.TimeoutExpired as exc:
                    results.append(
                        {
                            "variant": variant,
                            "status": "fail",
                            "reason": f"timed out after {exc.timeout} seconds",
                        }
                    )
                    continue
                if completed.returncode != 0:
                    results.append(
                        {
                            "variant": variant,
                            "status": "fail",
                            "returncode": completed.returncode,
                            "stdout": completed.stdout.strip(),
                            "stderr": completed.stderr.strip(),
                        }
                    )
                    continue
                try:
                    results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
                except (IndexError, json.JSONDecodeError) as exc:
                    results.append(
                        {
                            "variant": variant,
                            "status": "fail",
                            "reason": f"child emitted invalid JSON: {exc}",
                            "stdout": completed.stdout.strip(),
                            "stderr": completed.stderr.strip(),
                        }
                    )

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            status = str(result["status"]).upper()
            if result["status"] == "pass":
                print(
                    f"{status} {result['variant']}: "
                    f"forward={result['forward_shape']} sample={result['sample_shape']} "
                    f"torch={result['torch_version']}"
                )
            else:
                reason = result.get("reason") or result.get("stderr") or "unknown error"
                print(f"{status} {result['variant']}: {reason}")
    return 1 if any(result["status"] == "fail" for result in results) else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise preserved CARMA model construction, forward/backward, and sampling",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(VARIANTS),
        help="variant to test; repeat as needed (default: all four)",
    )
    parser.add_argument("--python", help="Python interpreter containing PyTorch")
    parser.add_argument("--repo-root", help="pde-carma checkout root")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per variant")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument(
        "--allow-missing-torch",
        action="store_true",
        help="report skipped variants and exit successfully when PyTorch is absent",
    )
    parser.add_argument("--_child", choices=tuple(VARIANTS), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._child:
        if importlib.util.find_spec("torch") is None:
            print("PyTorch is unavailable in the child interpreter", file=sys.stderr)
            return 2
        return _run_child(args._child, _repository_root(args.repo_root))
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
