#!/usr/bin/env python3
"""Exercise the preserved NPY loaders and training-pair datasets synthetically.

Every implementation runs in an isolated subprocess whose input, normalized
validation/test outputs, working directory, and caches live in a temporary
directory.  Nothing is written beneath ``implementations/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


VARIANTS = {
    "dit-unidirectional": ("dit_unidirectional", "dit_unidirectional"),
    "dit-bidirectional": ("dit_bidirectional", "bidirectional"),
    "flowformer-unidirectional": (
        "flowformer_unidirectional",
        "flowformer_unidirectional",
    ),
    "flowformer-bidirectional": ("flowformer_bidirectional", "bidirectional"),
}

REQUIRED_IMPORTS = ("torch", "numpy", "xarray")


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


def _assert_close(torch: Any, actual: Any, expected: Any, label: str) -> None:
    if not bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)):
        difference = float(torch.max(torch.abs(actual - expected)).item())
        raise AssertionError(f"{label} differs (maximum absolute error {difference})")


def _synthetic_arrays(numpy: Any) -> tuple[Any, Any, Any]:
    def make(count: int, offset: float) -> Any:
        values = numpy.arange(count * 4 * 2 * 4 * 4, dtype=numpy.float32)
        return values.reshape(count, 4, 2, 4, 4) / 10.0 + offset

    return make(2, 0.0), make(1, 100.0), make(1, 200.0)


def _write_presplit_npy(numpy: Any, directory: Path) -> tuple[Path, Any, Any, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    train, validation, test = _synthetic_arrays(numpy)
    train_path = directory / "vids_train_synthetic.npy"
    numpy.save(train_path, train)
    numpy.save(directory / "vids_val_synthetic.npy", validation)
    numpy.save(directory / "vids_test_synthetic.npy", test)
    return train_path, train, validation, test


def _expected_normalized(torch: Any, raw_train: Any, raw_value: Any) -> Any:
    train = torch.from_numpy(raw_train).float()
    value = torch.from_numpy(raw_value).float()
    mean = train.mean(dim=(0, 1, 3, 4), keepdim=True)
    std = train.std(dim=(0, 1, 3, 4), keepdim=True) + 1e-6
    return (value - mean) / std


def _check_normalization(
    torch: Any,
    normalized_train: Any,
    normalized_validation: Any,
    raw_train: Any,
    raw_validation: Any,
) -> dict[str, list[float]]:
    expected_train = _expected_normalized(torch, raw_train, raw_train)
    expected_validation = _expected_normalized(torch, raw_train, raw_validation)
    _assert_close(torch, normalized_train, expected_train, "normalized train tensor")
    _assert_close(
        torch,
        normalized_validation,
        expected_validation,
        "validation tensor normalized with train statistics",
    )
    means = normalized_train.mean(dim=(0, 1, 3, 4))
    stds = normalized_train.std(dim=(0, 1, 3, 4))
    return {
        "train_channel_mean": [float(value) for value in means.tolist()],
        "train_channel_std": [float(value) for value in stds.tolist()],
    }


def _load_unidirectional_dit_trainer(implementation: Path) -> Any:
    """Load the exact designated pairing classes without plotting dependencies.

    ``train_dit_unidirectional.py`` imports ``TrainingManager`` at module import
    time, although that class is irrelevant to data pairing and pulls in
    Matplotlib.  A minimal temporary module keeps this data-only smoke check
    scoped to torch/numpy/xarray while executing the original class definitions.
    """

    module_name = "utils.training_manager"
    previous = sys.modules.get(module_name)
    shim = types.ModuleType(module_name)
    shim.TrainingManager = object
    sys.modules[module_name] = shim
    try:
        source = implementation / "train_dit_unidirectional.py"
        spec = importlib.util.spec_from_file_location("carma_unidirectional_dit_smoke", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def _loader(
    loader_class: Any,
    dataset_path: Path,
    output_directory: Path,
    *,
    direction: str | None = None,
) -> Any:
    arguments = SimpleNamespace(direction=direction) if direction is not None else None
    return loader_class(
        dataset_name="synthetic",
        dataset_loc=str(dataset_path),
        save_dir=str(output_directory),
        var_name="unused_for_npy",
        num_traj_use=None,
        channels=2,
        args=arguments,
    )


def _check_batch(batch: Any, *, controlled: bool) -> list[list[int]]:
    expected_length = 3 if controlled else 2
    if len(batch) != expected_length:
        raise AssertionError(f"expected {expected_length} batch tensors, got {len(batch)}")
    shapes = [_shape(value) for value in batch]
    expected = [[2, 2, 4, 4], [2, 2, 4, 4]]
    if controlled:
        expected.append([2, 2])
    if shapes != expected:
        raise AssertionError(f"expected batch shapes {expected}, got {shapes}")
    return shapes


def _check_forward_dataset(torch: Any, dataset: Any, videos: Any, *, controlled: bool) -> None:
    sample = dataset[0]
    _assert_close(torch, sample[0], videos[0, 0], "forward condition")
    _assert_close(torch, sample[1], videos[0, 1], "forward target")
    if controlled:
        _assert_close(torch, sample[2], torch.tensor([1.0, 0.0]), "forward control")


def _check_bidirectional_dataset(torch: Any, dataset: Any, videos: Any) -> None:
    forward = dataset[0]
    backward = dataset[1]
    _assert_close(torch, forward[0], videos[0, 0], "bidirectional forward condition")
    _assert_close(torch, forward[1], videos[0, 1], "bidirectional forward target")
    _assert_close(torch, forward[2], torch.tensor([1.0, 0.0]), "forward control")
    _assert_close(torch, backward[0], videos[0, 1], "backward condition")
    _assert_close(torch, backward[1], videos[0, 0], "backward target")
    _assert_close(torch, backward[2], torch.tensor([0.0, 1.0]), "backward control")


def _smoke_standard(
    torch: Any,
    loader_class: Any,
    dataset_path: Path,
    raw_train: Any,
    raw_validation: Any,
    output_directory: Path,
    *,
    unidirectional_dit: bool,
) -> dict[str, Any]:
    loader = _loader(loader_class, dataset_path, output_directory)
    train, validation, test = loader.import_and_split(n_train=2, n_val=1, n_test=1)
    normalization = _check_normalization(
        torch, train, validation, raw_train, raw_validation
    )
    train_dl, validation_dl = loader.dataloaders(
        train, validation, batch_size=2, num_workers=0
    )
    controlled = True
    if unidirectional_dit:
        _check_forward_dataset(torch, train_dl.dataset, train, controlled=True)
        expected_pairs = 2 * (4 - 1)
        controls = [[1.0, 0.0]]
    else:
        _check_bidirectional_dataset(torch, train_dl.dataset, train)
        expected_pairs = 2 * (4 - 1) * 2
        controls = [[1.0, 0.0], [0.0, 1.0]]
    if len(train_dl.dataset) != expected_pairs:
        raise AssertionError(
            f"expected {expected_pairs} training pairs, got {len(train_dl.dataset)}"
        )
    batch_shapes = _check_batch(next(iter(train_dl)), controlled=controlled)
    saved = sorted(path.name for path in output_directory.glob("*.npy"))
    if len(saved) != 2:
        raise AssertionError(f"expected two normalized output files, got {saved}")
    return {
        "train_shape": _shape(train),
        "validation_shape": _shape(validation),
        "test_shape": _shape(test),
        "train_pairs": len(train_dl.dataset),
        "validation_pairs": len(validation_dl.dataset),
        "batch_shapes": batch_shapes,
        "controls": controls,
        "saved_outputs": saved,
        **normalization,
    }


def _smoke_unidirectional_flowformer(
    torch: Any,
    loader_class: Any,
    dataset_path: Path,
    raw_train: Any,
    raw_validation: Any,
    output_directory: Path,
) -> dict[str, Any]:
    directions: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    for direction in ("forward", "backward"):
        loader = _loader(
            loader_class,
            dataset_path,
            output_directory,
            direction=direction,
        )
        train, validation, test = loader.import_and_split(n_train=2, n_val=1, n_test=1)
        expected_train = _expected_normalized(torch, raw_train, raw_train)
        expected_validation = _expected_normalized(torch, raw_train, raw_validation)
        if direction == "backward":
            expected_train = torch.flip(expected_train, dims=[1])
            expected_validation = torch.flip(expected_validation, dims=[1])
        _assert_close(torch, train, expected_train, f"{direction} normalized train")
        _assert_close(
            torch,
            validation,
            expected_validation,
            f"{direction} normalized validation",
        )
        train_dl, validation_dl = loader.dataloaders(
            train, validation, batch_size=2, num_workers=0
        )
        _check_forward_dataset(torch, train_dl.dataset, train, controlled=False)
        expected_pairs = 2 * (4 - 1)
        if len(train_dl.dataset) != expected_pairs:
            raise AssertionError(
                f"expected {expected_pairs} {direction} pairs, got {len(train_dl.dataset)}"
            )
        directions[direction] = {
            "train_shape": _shape(train),
            "validation_shape": _shape(validation),
            "test_shape": _shape(test),
            "train_pairs": len(train_dl.dataset),
            "validation_pairs": len(validation_dl.dataset),
            "batch_shapes": _check_batch(next(iter(train_dl)), controlled=False),
        }
        loaded[direction] = train

    _assert_close(
        torch,
        loaded["backward"],
        torch.flip(loaded["forward"], dims=[1]),
        "backward-direction time reversal",
    )
    saved = sorted(path.name for path in output_directory.glob("*.npy"))
    if len(saved) != 4:
        raise AssertionError(f"expected four direction-specific outputs, got {saved}")
    return {
        "directions": directions,
        "time_reversal_verified": True,
        "saved_outputs": saved,
    }


def _run_child(variant: str, repository_root: Path) -> int:
    implementation_name, behavior = VARIANTS[variant]
    implementation = repository_root / "implementations" / implementation_name
    if not implementation.is_dir():
        raise FileNotFoundError(f"Implementation directory not found: {implementation}")
    sys.path.insert(0, str(implementation))

    import numpy
    import torch
    from utils.load_data import LoadData

    torch.manual_seed(2027)
    torch.set_num_threads(1)
    working = Path.cwd()
    dataset_path, raw_train, raw_validation, _raw_test = _write_presplit_npy(
        numpy, working / "input"
    )
    outputs = working / "normalized_outputs"

    if behavior == "dit_unidirectional":
        trainer = _load_unidirectional_dit_trainer(implementation)
        details = _smoke_standard(
            torch,
            trainer.ForwardLoadData,
            dataset_path,
            raw_train,
            raw_validation,
            outputs,
            unidirectional_dit=True,
        )
    elif behavior == "flowformer_unidirectional":
        details = _smoke_unidirectional_flowformer(
            torch,
            LoadData,
            dataset_path,
            raw_train,
            raw_validation,
            outputs,
        )
    else:
        details = _smoke_standard(
            torch,
            LoadData,
            dataset_path,
            raw_train,
            raw_validation,
            outputs,
            unidirectional_dit=False,
        )

    result = {
        "variant": variant,
        "status": "pass",
        "torch_version": str(torch.__version__),
        "numpy_version": str(numpy.__version__),
        **details,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _dependencies_available(
    python_executable: str,
    environment: dict[str, str],
    cwd: str,
) -> tuple[bool, str]:
    statement = "; ".join(f"import {name}" for name in REQUIRED_IMPORTS)
    try:
        check = subprocess.run(
            [python_executable, "-B", "-c", statement],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except OSError as exc:
        return False, str(exc)
    except subprocess.TimeoutExpired:
        return False, "dependency import check timed out"
    detail = check.stderr.strip() or check.stdout.strip()
    return check.returncode == 0, detail


def _run_parent(args: argparse.Namespace) -> int:
    root = _repository_root(args.repo_root)
    selected = args.variant or list(VARIANTS)
    python_executable = args.python or sys.executable
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="pde-carma-data-smoke-") as temporary:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "XDG_CACHE_HOME": temporary,
                "MPLCONFIGDIR": temporary,
            }
        )
        available, dependency_detail = _dependencies_available(
            python_executable, environment, temporary
        )
        if not available:
            message = (
                f"Required imports {', '.join(REQUIRED_IMPORTS)} are unavailable to "
                f"{python_executable!r}; data-loader smoke tests were not executed."
            )
            if dependency_detail:
                message += f" Import check: {dependency_detail}"
            if not args.allow_missing_dependencies:
                print(f"ERROR: {message}", file=sys.stderr)
                return 2
            results = [
                {"variant": variant, "status": "skip", "reason": message}
                for variant in selected
            ]
        else:
            for variant in selected:
                variant_directory = Path(temporary) / variant
                variant_directory.mkdir()
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
                        cwd=variant_directory,
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
                if "directions" in result:
                    pairs = {
                        key: value["train_pairs"]
                        for key, value in result["directions"].items()
                    }
                else:
                    pairs = result["train_pairs"]
                print(f"{status} {result['variant']}: train_pairs={pairs}")
            else:
                reason = result.get("reason") or result.get("stderr") or "unknown error"
                print(f"{status} {result['variant']}: {reason}")
    return 1 if any(result["status"] == "fail" for result in results) else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise preserved NPY normalization, pairing, and DataLoader behavior",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(VARIANTS),
        help="variant to test; repeat as needed (default: all four)",
    )
    parser.add_argument("--python", help="Python interpreter with torch, numpy, and xarray")
    parser.add_argument("--repo-root", help="pde-carma checkout root")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per variant")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument(
        "--allow-missing-dependencies",
        action="store_true",
        help="report skips and exit successfully when required imports are absent",
    )
    parser.add_argument("--_child", choices=tuple(VARIANTS), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._child:
        missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
        if missing:
            print(f"Required imports unavailable: {', '.join(missing)}", file=sys.stderr)
            return 2
        return _run_child(args._child, _repository_root(args.repo_root))
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
