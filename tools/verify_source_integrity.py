#!/usr/bin/env python3
"""Verify byte-for-byte integrity of the four preserved implementations.

The expected path, size, and SHA-256 values are stored in
``docs/source_manifest.json``.  This verifier uses only the Python standard
library and never imports or executes implementation code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VariantResult:
    variant: str
    expected_files: int
    actual_files: int
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    size_mismatches: tuple[str, ...]
    hash_mismatches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.missing
            or self.extra
            or self.size_mismatches
            or self.hash_mismatches
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported source manifest schema: {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("implementations"), dict):
        raise ValueError("Manifest is missing an 'implementations' object")
    return manifest


def verify_variant(
    repository_root: Path,
    variant: str,
    record: dict[str, Any],
) -> VariantResult:
    relative_root = record["destination"]
    implementation_root = (repository_root / relative_root).resolve()
    if not implementation_root.is_dir():
        expected = tuple(sorted(item["path"] for item in record["files"]))
        return VariantResult(
            variant=variant,
            expected_files=len(expected),
            actual_files=0,
            missing=expected,
            extra=(),
            size_mismatches=(),
            hash_mismatches=(),
        )

    expected_records = {item["path"]: item for item in record["files"]}
    actual_paths = _implementation_files(implementation_root)
    expected_names = set(expected_records)
    actual_names = set(actual_paths)

    missing = tuple(sorted(expected_names - actual_names))
    extra = tuple(sorted(actual_names - expected_names))
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []

    for relative_path in sorted(expected_names & actual_names):
        expected = expected_records[relative_path]
        actual = actual_paths[relative_path]
        if actual.stat().st_size != expected["size"]:
            size_mismatches.append(relative_path)
            continue
        if sha256_file(actual).lower() != expected["sha256"].lower():
            hash_mismatches.append(relative_path)

    return VariantResult(
        variant=variant,
        expected_files=len(expected_names),
        actual_files=len(actual_names),
        missing=missing,
        extra=extra,
        size_mismatches=tuple(size_mismatches),
        hash_mismatches=tuple(hash_mismatches),
    )


def verify_repository(
    repository_root: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[VariantResult]]:
    manifest = load_manifest(manifest_path)
    results = [
        verify_variant(repository_root, variant, record)
        for variant, record in manifest["implementations"].items()
    ]
    return manifest, results


def _print_text_report(manifest: dict[str, Any], results: list[VariantResult]) -> None:
    for result in results:
        state = "PASS" if result.passed else "FAIL"
        discrepancies = (
            len(result.missing)
            + len(result.extra)
            + len(result.size_mismatches)
            + len(result.hash_mismatches)
        )
        print(
            f"{state} {result.variant}: expected={result.expected_files} "
            f"actual={result.actual_files} discrepancies={discrepancies}"
        )
        for label, paths in (
            ("missing", result.missing),
            ("extra", result.extra),
            ("size mismatch", result.size_mismatches),
            ("SHA-256 mismatch", result.hash_mismatches),
        ):
            for path in paths:
                print(f"  {label}: {path}")

    expected_total = sum(result.expected_files for result in results)
    actual_total = sum(result.actual_files for result in results)
    discrepancy_total = sum(
        len(result.missing)
        + len(result.extra)
        + len(result.size_mismatches)
        + len(result.hash_mismatches)
        for result in results
    )
    state = "PASS" if all(result.passed for result in results) else "FAIL"
    print(
        f"{state} total: expected={expected_total} actual={actual_total} "
        f"discrepancies={discrepancy_total}"
    )
    declared_total = manifest.get("totals", {}).get("files")
    if declared_total != expected_total:
        print(
            f"FAIL manifest total: declared={declared_total!r} "
            f"enumerated={expected_total}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify preserved implementation files against SHA-256 manifest"
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT,
        help="pde-carma repository root",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest path (default: <repository>/docs/source_manifest.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the verification result as JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root = args.repository_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else repository_root / "docs" / "source_manifest.json"
    )
    try:
        manifest, results = verify_repository(repository_root, manifest_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Integrity verification could not run: {error}", file=sys.stderr)
        return 2

    manifest_total_matches = manifest.get("totals", {}).get("files") == sum(
        result.expected_files for result in results
    )
    passed = all(result.passed for result in results) and manifest_total_matches
    if args.json:
        payload = {
            "passed": passed,
            "manifest": str(manifest_path),
            "repository_root": str(repository_root),
            "results": [asdict(result) | {"passed": result.passed} for result in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text_report(manifest, results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

