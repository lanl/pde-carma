from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pde_carma_smoke_data_loaders",
    ROOT / "tools" / "smoke_data_loaders.py",
)
assert SPEC is not None and SPEC.loader is not None
SMOKE_DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE_DATA)


def _arguments(*, allow_missing: bool, as_json: bool) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(ROOT),
        variant=None,
        python="python-without-data-dependencies",
        timeout=1.0,
        json=as_json,
        allow_missing_dependencies=allow_missing,
    )


class DataSmokeToolTests(unittest.TestCase):
    def test_registry_covers_all_preserved_variants(self) -> None:
        self.assertEqual(
            set(SMOKE_DATA.VARIANTS),
            {
                "dit-unidirectional",
                "dit-bidirectional",
                "flowformer-unidirectional",
                "flowformer-bidirectional",
            },
        )
        self.assertEqual(SMOKE_DATA.REQUIRED_IMPORTS, ("torch", "numpy", "xarray"))

    def test_missing_dependencies_are_an_error_by_default(self) -> None:
        stderr = io.StringIO()
        unavailable = (False, "No module named 'torch'")
        with mock.patch.object(
            SMOKE_DATA, "_dependencies_available", return_value=unavailable
        ):
            with contextlib.redirect_stderr(stderr):
                returncode = SMOKE_DATA._run_parent(
                    _arguments(allow_missing=False, as_json=False)
                )
        self.assertEqual(returncode, 2)
        self.assertIn("data-loader smoke tests were not executed", stderr.getvalue())

    def test_missing_dependencies_can_be_explicit_skips(self) -> None:
        stdout = io.StringIO()
        unavailable = (False, "No module named 'torch'")
        with mock.patch.object(
            SMOKE_DATA, "_dependencies_available", return_value=unavailable
        ):
            with contextlib.redirect_stdout(stdout):
                returncode = SMOKE_DATA._run_parent(
                    _arguments(allow_missing=True, as_json=True)
                )
        self.assertEqual(returncode, 0)
        results = json.loads(stdout.getvalue())
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result["status"] == "skip" for result in results))


if __name__ == "__main__":
    unittest.main()
