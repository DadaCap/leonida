"""scripts/bench_translate.py runs, reports the documented numbers, and fails a slowdown check.

    python -I tests/test_bench_translate.py

The measurement itself is only asserted for shape: a positive, finite per-instruction time and the same fields in
--json. The absolute numbers depend on the machine, which is why the script records them for a same-machine
comparison instead of testing them.
"""
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "bench_translate.py")


def run(*args):
    return subprocess.run([sys.executable, "-I", SCRIPT, *args], capture_output=True, text=True, timeout=600)


class BenchTranslateTest(unittest.TestCase):
    def test_json_reports_every_section(self):
        result = run("--json", "--runs", "1", "--size", "40")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["runs"], 1)
        self.assertEqual(report["synthetic"]["size"], 40)
        # Two synthetic programs, and the fixtures that translate without an FSSM header.
        self.assertEqual([row["name"] for row in report["synthetic"]["programs"]], ["alu", "nop"])
        self.assertGreater(report["corpus"]["summary"]["programs"], 0)
        for row in report["synthetic"]["programs"] + report["corpus"]["programs"]:
            self.assertGreater(row["instructions"], 0, row["name"])
            self.assertGreater(row["seconds"], 0.0, row["name"])
            self.assertGreater(row["us_per_instruction"], 0.0, row["name"])
        for name in ("corpus_us_per_instruction", "alu_us_per_instruction", "nop_us_per_instruction"):
            self.assertIn(name, report["measured"])
            self.assertIsNotNone(report["slowdown"][name])
        # A compute fixture needs the FSSM header the semantic tests pass, so it is reported, never silently counted.
        self.assertTrue(all("FSSM" in line for line in report["corpus"]["skipped"]), report["corpus"]["skipped"])

    def test_a_slowdown_check_passes_at_the_recorded_baseline(self):
        # 100x the recorded per-instruction time is never reached in a real run, so this only exercises the path.
        result = run("--runs", "1", "--size", "20", "--max-slowdown", "100")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("us/instr", result.stdout)

    def test_compare_runs_two_trees_and_prints_a_ratio(self):
        tools = os.path.join(ROOT, "tools")
        result = run("--compare", tools, tools, "--rounds", "1", "--runs", "1", "--size", "20")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ratio", result.stdout)
        self.assertIn(tools, result.stdout)
        self.assertIn("us_per_instruction", result.stdout)

    def test_bad_arguments_are_rejected(self):
        for args in (("--runs", "0"), ("--size", "0")):
            result = run(*args)
            self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
