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
    "pde_carma_smoke_models",
    ROOT / "tools" / "smoke_models.py",
)
assert SPEC is not None and SPEC.loader is not None
SMOKE_MODELS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE_MODELS)


def _arguments(*, allow_missing_torch: bool, as_json: bool) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(ROOT),
        variant=None,
        python="python-without-torch",
        timeout=1.0,
        json=as_json,
        allow_missing_torch=allow_missing_torch,
    )


class ModelSmokeToolTests(unittest.TestCase):
    def test_registry_covers_all_preserved_variants(self) -> None:
        self.assertEqual(
            set(SMOKE_MODELS.VARIANTS),
            {
                "dit-unidirectional",
                "dit-bidirectional",
                "flowformer-unidirectional",
                "flowformer-bidirectional",
            },
        )
        for implementation, _family, _bidirectional in SMOKE_MODELS.VARIANTS.values():
            self.assertTrue((ROOT / "implementations" / implementation).is_dir())

    def test_missing_torch_is_an_error_by_default(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(SMOKE_MODELS, "_torch_available", return_value=False):
            with contextlib.redirect_stderr(stderr):
                returncode = SMOKE_MODELS._run_parent(
                    _arguments(allow_missing_torch=False, as_json=False)
                )
        self.assertEqual(returncode, 2)
        self.assertIn("PyTorch is unavailable", stderr.getvalue())

    def test_missing_torch_can_be_reported_as_explicit_skips(self) -> None:
        stdout = io.StringIO()
        with mock.patch.object(SMOKE_MODELS, "_torch_available", return_value=False):
            with contextlib.redirect_stdout(stdout):
                returncode = SMOKE_MODELS._run_parent(
                    _arguments(allow_missing_torch=True, as_json=True)
                )
        self.assertEqual(returncode, 0)
        results = json.loads(stdout.getvalue())
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result["status"] == "skip" for result in results))


if __name__ == "__main__":
    unittest.main()
