"""Dependency-free test for the immutable implementation copies."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SourceIntegrityTests(unittest.TestCase):
    def test_complete_source_manifest(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "tools" / "verify_source_integrity.py"),
                "--repository-root",
                str(REPOSITORY_ROOT),
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("PASS total: expected=170 actual=170 discrepancies=0", completed.stdout)


if __name__ == "__main__":
    unittest.main()

