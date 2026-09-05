from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = WORKSPACE_ROOT / "router_v2" / "reports" / "baseline_hashes.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> tuple[int, int, str]:
    records: list[str] = []
    file_count = 0
    total_bytes = 0
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative_path = path.relative_to(WORKSPACE_ROOT).as_posix()
        size = path.stat().st_size
        records.append(f"{relative_path}\t{size}\t{sha256_file(path)}\n")
        file_count += 1
        total_bytes += size
    digest = hashlib.sha256("".join(records).encode("utf-8")).hexdigest()
    return file_count, total_bytes, digest


class V1RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with BASELINE_PATH.open("r", encoding="utf-8") as handle:
            cls.baseline = json.load(handle)

    def test_all_recorded_read_only_trees_match_baseline(self) -> None:
        for relative_root, expected in self.baseline["read_only_trees"].items():
            with self.subTest(root=relative_root):
                actual_count, actual_bytes, actual_hash = hash_tree(
                    WORKSPACE_ROOT / relative_root
                )
                self.assertEqual(actual_count, expected["file_count"])
                self.assertEqual(actual_bytes, expected["total_bytes"])
                self.assertEqual(actual_hash, expected["tree_sha256"])

    def test_critical_v1_files_match_baseline(self) -> None:
        for relative_path, expected_hash in self.baseline["critical_files"].items():
            with self.subTest(path=relative_path):
                self.assertEqual(
                    sha256_file(WORKSPACE_ROOT / relative_path),
                    expected_hash,
                )

    def test_importing_router_v2_does_not_change_v1_output(self) -> None:
        script_template = """
import json
import torch
{v2_import}
from router.model import RADTokenRouter

model = RADTokenRouter(beta_max=10.0, beta_init=None)
with torch.no_grad():
    for index, parameter in enumerate(model.parameters()):
        parameter.fill_((index + 1) * 0.01)
base = torch.arange(40, dtype=torch.float32).reshape(2, 20) / 7.0
reward = torch.flip(base, dims=[-1]) / 3.0
with torch.no_grad():
    beta, gate = model(base, reward)
print(json.dumps({{"beta": beta.tolist(), "gate": gate.tolist()}}))
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        outputs: list[str] = []
        for v2_import in ("", "import router_v2"):
            completed = subprocess.run(
                [sys.executable, "-c", script_template.format(v2_import=v2_import)],
                cwd=WORKSPACE_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])

    def test_method_router_remains_absent_as_recorded_at_baseline(self) -> None:
        self.assertFalse((WORKSPACE_ROOT / "Method" / "Router").exists())


if __name__ == "__main__":
    unittest.main()
