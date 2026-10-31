import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pde_carma.cli import main
from pde_carma.launcher import build_invocation, run_invocation
from pde_carma.registry import VARIANTS


class RegistryTests(unittest.TestCase):
    def test_every_registered_source_path_exists(self):
        for spec in VARIANTS.values():
            with self.subTest(variant=spec.key):
                root = REPOSITORY_ROOT / "implementations" / spec.directory
                self.assertTrue((root / spec.train_entry).is_file())
                self.assertTrue((root / spec.infer_entry).is_file())
                for relative in spec.reproduction_scripts:
                    self.assertTrue((root / relative).is_file(), relative)

    def test_required_original_entry_points(self):
        expected = {
            "dit-unidirectional": ("train_dit_unidirectional.py", "infer_dit.py"),
            "dit-bidirectional": ("train_dit.py", "infer_dit.py"),
            "flowformer-unidirectional": ("train_flowformer.py", "infer_flowformer.py"),
            "flowformer-bidirectional": ("train_flowformer.py", "infer_flowformer.py"),
        }
        self.assertEqual(
            {key: (spec.train_entry, spec.infer_entry) for key, spec in VARIANTS.items()},
            expected,
        )

    def test_unidirectional_allowlists_exclude_inherited_carma_scripts(self):
        self.assertEqual(
            VARIANTS["dit-unidirectional"].reproduction_scripts,
            ("scripts/evaluate_unidirectional_dit_multirun.py",),
        )
        self.assertEqual(
            VARIANTS["flowformer-unidirectional"].reproduction_scripts,
            ("scripts/generate_unifwd_table3.py",),
        )


class InvocationTests(unittest.TestCase):
    def test_arguments_are_forwarded_without_rewriting(self):
        forwarded = ["--ds", "ce-rm", "--tag", "value with spaces", "--cyclic_runs", "4", "8"]
        invocation = build_invocation(
            "dit-bidirectional",
            "train",
            forwarded,
            repo_root=REPOSITORY_ROOT,
            python_executable="python-for-test",
        )
        self.assertEqual(invocation.command[0], "python-for-test")
        self.assertEqual(list(invocation.command[2:]), forwarded)
        self.assertEqual(invocation.cwd, REPOSITORY_ROOT / "implementations" / "dit_bidirectional")
        self.assertEqual(invocation.script.name, "train_dit.py")

    def test_separator_is_not_forwarded(self):
        invocation = build_invocation(
            "flowformer-unidirectional",
            "train",
            ["--", "--direction", "backward"],
            repo_root=REPOSITORY_ROOT,
            python_executable="python-for-test",
        )
        self.assertEqual(invocation.command[2:], ("--direction", "backward"))

    def test_evaluate_uses_original_inference_entry(self):
        invocation = build_invocation(
            "flowformer-unidirectional",
            "evaluate",
            repo_root=REPOSITORY_ROOT,
        )
        self.assertEqual(invocation.script.name, "infer_flowformer.py")
        self.assertEqual(invocation.operation, "evaluate")

    def test_carma_is_restricted_to_source_backed_bidirectional_variants(self):
        for key in ("dit-bidirectional", "flowformer-bidirectional"):
            with self.subTest(variant=key):
                invocation = build_invocation(key, "carma", repo_root=REPOSITORY_ROOT)
                self.assertTrue(invocation.script.is_file())
        for key in ("dit-unidirectional", "flowformer-unidirectional"):
            with self.subTest(variant=key):
                with self.assertRaisesRegex(ValueError, "no source-backed CARMA"):
                    build_invocation(key, "carma", repo_root=REPOSITORY_ROOT)

    def test_reproduction_scripts_are_allowlisted(self):
        relative = "scripts/generate_unifwd_table3.py"
        invocation = build_invocation(
            "flowformer-unidirectional",
            "reproduce",
            ["--", "--help"],
            repo_root=REPOSITORY_ROOT,
            reproduction_script=relative,
        )
        self.assertEqual(invocation.script, invocation.cwd / relative)
        self.assertEqual(invocation.command[-1], "--help")
        with self.assertRaisesRegex(ValueError, "not allowlisted"):
            build_invocation(
                "flowformer-unidirectional",
                "reproduce",
                repo_root=REPOSITORY_ROOT,
                reproduction_script="../../outside.py",
            )

    def test_cli_dry_run_is_machine_readable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(
                [
                    "--repo-root",
                    str(REPOSITORY_ROOT),
                    "--python",
                    "python-for-test",
                    "--dry-run",
                    "train",
                    "dit-unidirectional",
                    "--",
                    "--ds",
                    "ce-rm",
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["variant"], "dit-unidirectional")
        self.assertEqual(payload["command"][-2:], ["--ds", "ce-rm"])

    def test_subprocess_receives_exact_command_and_implementation_cwd(self):
        invocation = build_invocation(
            "flowformer-bidirectional",
            "infer",
            ["--ckpt", "checkpoint.pth"],
            repo_root=REPOSITORY_ROOT,
            python_executable="python-for-test",
        )
        with patch("pde_carma.launcher.subprocess.run") as run:
            run.return_value.returncode = 17
            with contextlib.redirect_stderr(io.StringIO()):
                status = run_invocation(invocation)
        call = run.call_args
        self.assertEqual(call.args, (invocation.command,))
        self.assertEqual(call.kwargs["cwd"], invocation.cwd)
        self.assertFalse(call.kwargs["check"])
        self.assertEqual(call.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(status, 17)


if __name__ == "__main__":
    unittest.main()

