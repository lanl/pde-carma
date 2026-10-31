"""Resolve and launch original scripts without importing or modifying them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .registry import VARIANTS, VariantSpec, get_variant


@dataclass(frozen=True)
class Invocation:
    variant: str
    operation: str
    cwd: Path
    script: Path
    command: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "operation": self.operation,
            "cwd": str(self.cwd),
            "script": str(self.script),
            "command": list(self.command),
        }


def repository_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate the checkout containing ``implementations/``.

    A non-editable installation must provide ``--repo-root`` or set
    ``PDE_CARMA_ROOT`` because the preserved sources are not package data.
    """

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("PDE_CARMA_ROOT"):
        candidates.append(Path(os.environ["PDE_CARMA_ROOT"]))
    candidates.extend((Path(__file__).resolve().parents[2], Path.cwd()))

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        checked.append(str(resolved))
        if (resolved / "implementations").is_dir():
            return resolved
    raise FileNotFoundError(
        "Could not locate the pde-carma checkout. Set PDE_CARMA_ROOT or pass "
        f"--repo-root. Checked: {', '.join(checked)}"
    )


def _implementation_root(root: Path, spec: VariantSpec) -> Path:
    implementations = (root / "implementations").resolve()
    implementation = (implementations / spec.directory).resolve()
    if implementation.parent != implementations or not implementation.is_dir():
        raise FileNotFoundError(f"Implementation directory not found: {implementation}")
    return implementation


def _strip_separator(arguments: Sequence[str]) -> tuple[str, ...]:
    values = tuple(arguments)
    return values[1:] if values[:1] == ("--",) else values


def build_invocation(
    variant: str,
    operation: str,
    arguments: Sequence[str] = (),
    *,
    repo_root: str | os.PathLike[str] | None = None,
    python_executable: str | os.PathLike[str] | None = None,
    reproduction_script: str | None = None,
) -> Invocation:
    """Build a subprocess invocation while preserving source arguments exactly."""

    spec = get_variant(variant)
    root = repository_root(repo_root)
    cwd = _implementation_root(root, spec)

    normalized_operation = "infer" if operation == "evaluate" else operation
    if normalized_operation == "train":
        relative_script = spec.train_entry
    elif normalized_operation == "infer":
        relative_script = spec.infer_entry
    elif normalized_operation == "carma":
        if not spec.carma_supported:
            raise ValueError(
                f"{spec.label} has no source-backed CARMA cyclic selector. "
                "Use its model-specific inference/evaluation workflow instead."
            )
        relative_script = spec.infer_entry
    elif normalized_operation == "reproduce":
        if reproduction_script is None:
            raise ValueError("reproduce requires a source script path")
        if reproduction_script not in spec.reproduction_scripts:
            allowed = ", ".join(spec.reproduction_scripts)
            raise ValueError(
                f"Script {reproduction_script!r} is not allowlisted for {variant}. "
                f"Allowed scripts: {allowed}"
            )
        relative_script = reproduction_script
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    script = (cwd / relative_script).resolve()
    if cwd not in script.parents or not script.is_file():
        raise FileNotFoundError(f"Original entry point not found: {script}")

    interpreter = str(
        python_executable or os.environ.get("PDE_CARMA_PYTHON") or sys.executable
    )
    forwarded = _strip_separator(arguments)
    command = (interpreter, str(script), *forwarded)
    return Invocation(
        variant=variant,
        operation=operation,
        cwd=cwd,
        script=script,
        command=command,
    )


def run_invocation(invocation: Invocation, *, dry_run: bool = False) -> int:
    if dry_run:
        print(json.dumps(invocation.as_dict(), indent=2))
        return 0
    environment = os.environ.copy()
    # Preserve the copied trees as file sets while executing their original code.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    print(
        "pde-carma invocation: " + json.dumps(invocation.as_dict(), sort_keys=True),
        file=sys.stderr,
    )
    completed = subprocess.run(
        invocation.command,
        cwd=invocation.cwd,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def wrapper_main(variant: str, operation: str, argv: Sequence[str] | None = None) -> int:
    """Entry helper for the eight minimal variant-specific wrapper scripts."""

    values = list(sys.argv[1:] if argv is None else argv)
    dry_run = False
    if values[:1] == ["--wrapper-dry-run"]:
        dry_run = True
        values.pop(0)
    invocation = build_invocation(variant, operation, values)
    return run_invocation(invocation, dry_run=dry_run)


def registry_summary() -> list[dict[str, object]]:
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "directory": spec.directory,
            "train_entry": spec.train_entry,
            "infer_entry": spec.infer_entry,
            "carma_supported": spec.carma_supported,
            "reproduction_scripts": list(spec.reproduction_scripts),
        }
        for spec in VARIANTS.values()
    ]

