import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hive_train.initialization import export_fixture
from hive_train.model import build_stage
from hive_train.validation import require_device, run_equivalence


class DiagnosticTests(unittest.TestCase):
    def test_unavailable_backend_fails_without_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                require_device("cuda")
        with patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "fallback must be disabled"):
                require_device("mps")

    def test_empty_suite_cannot_report_pass(self):
        with self.assertRaises(ValueError):
            run_equivalence(splits=())
        with self.assertRaises(ValueError):
            run_equivalence(scenarios=())

    def test_fixture_export_bytes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixture"
            manifest = export_fixture(build_stage(), output)
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
            size = 0
            for tensor in manifest["tensors"]:
                payload = (output / tensor["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), tensor["sha256"])
                size += len(payload)
            self.assertEqual(size, 875264 * 4)
            with self.assertRaises(FileExistsError):
                export_fixture(build_stage(), output)

    def test_cli_help_and_failed_verification_exit_code(self):
        help_result = subprocess.run([sys.executable, "-m", "hive_train", "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failure.json"
            result = subprocess.run([sys.executable, "-m", "hive_train", "verify", "--steps", "0",
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "fail")
            self.assertIn("steps must be positive", report["error"])
