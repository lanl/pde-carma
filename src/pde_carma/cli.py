"""Unified, dependency-light command line for the preserved repositories."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .launcher import build_invocation, registry_summary, run_invocation
from .registry import VARIANTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pde-carma",
        description=(
            "Launch an unchanged implementation in its own working directory. "
            "Use `--` before arguments intended for the original script."
        ),
    )
    parser.add_argument("--repo-root", default=None, help="pde-carma checkout root")
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
        help="Interpreter for the original implementation (or set PDE_CARMA_PYTHON)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved cwd and exact subprocess command without executing it",
    )

    subparsers = parser.add_subparsers(dest="operation", required=True)
    list_parser = subparsers.add_parser("list", help="List registered variants")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    list_parser.add_argument(
        "--reproduction",
        action="store_true",
        help="Include every allowlisted source reproduction script",
    )

    for operation, help_text in (
        ("train", "Run the original training entry point"),
        ("infer", "Run the original inference entry point"),
        ("evaluate", "Alias for the original model-specific inference/evaluation entry point"),
        ("carma", "Run source-backed cyclic selection for a bidirectional variant"),
    ):
        subparser = subparsers.add_parser(operation, help=help_text)
        subparser.add_argument("variant", choices=tuple(VARIANTS))
        subparser.add_argument("source_args", nargs=argparse.REMAINDER)

    reproduce = subparsers.add_parser(
        "reproduce", help="Run an allowlisted paper/ablation script from a source tree"
    )
    reproduce.add_argument("variant", choices=tuple(VARIANTS))
    reproduce.add_argument("source_script", help="Relative source script path shown by `list --reproduction`")
    reproduce.add_argument("source_args", nargs=argparse.REMAINDER)
    return parser


def _print_registry(*, as_json: bool, include_reproduction: bool) -> None:
    rows = registry_summary()
    if not include_reproduction:
        rows = [
            {key: value for key, value in row.items() if key != "reproduction_scripts"}
            for row in rows
        ]
    if as_json:
        print(json.dumps(rows, indent=2))
        return

    for row in rows:
        selector = "yes" if row["carma_supported"] else "no"
        print(
            f"{row['key']}: train={row['train_entry']} "
            f"infer={row['infer_entry']} CARMA={selector}"
        )
        if include_reproduction:
            for script in row["reproduction_scripts"]:
                print(f"  - {script}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.operation == "list":
        _print_registry(
            as_json=args.json,
            include_reproduction=args.reproduction,
        )
        return 0

    try:
        invocation = build_invocation(
            args.variant,
            args.operation,
            args.source_args,
            repo_root=args.repo_root,
            python_executable=args.python_executable,
            reproduction_script=getattr(args, "source_script", None),
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"pde-carma: error: {exc}\n")
    return run_invocation(invocation, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

