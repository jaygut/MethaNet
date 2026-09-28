"""Keep Apolo operational defaults portable across user accounts and checkouts."""
from pathlib import Path
import ast
import importlib.util
import os
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
PORTABLE_DOCS = [
    ROOT / "docs/apollo3_functional_mag_runbook.md",
    ROOT / "docs/apollo3_mag_functional_analytics_ops.md",
    ROOT / "ai_docs/functional_metagenomics_expansion/snakemake_backbone/config.apollo3.yaml",
]
OPERATIONAL_DOC_ROOTS = [ROOT / "docs", ROOT / "ai_docs/functional_metagenomics_expansion"]
VALIDATOR_PATH = ROOT / "scripts/validate_functional_mag_production_gates.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location("metha_functional_gates", VALIDATOR_PATH)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = VALIDATOR
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


class ApolloPathPortabilityTests(unittest.TestCase):
    def test_scripts_and_runbooks_have_no_account_specific_home_paths(self):
        candidates = [
            path for path in (ROOT / "scripts").rglob("*")
            if path.is_file() and path.suffix == ".sh"
        ]
        candidates.append(ROOT / "scripts/validate_functional_mag_production_gates.py")
        candidates.extend(PORTABLE_DOCS)
        candidates.extend(path for root in OPERATIONAL_DOC_ROOTS for path in root.rglob("*.md"))
        account_home = re.compile(r"/home/[^/\s|]+/")
        for path in candidates:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIsNone(account_home.search(path.read_text(errors="replace")))

    def test_database_defaults_are_overridable(self):
        scripts = [
            ROOT / "scripts/check_functional_mag_db_readiness_apollo3.sh",
            ROOT / "scripts/setup_functional_metagenomics_dbs_apollo3.sh",
            ROOT / "scripts/slurm/run_functional_mag_array_apollo3.sh",
        ]
        for path in scripts:
            with self.subTest(path=path.relative_to(ROOT)):
                content = path.read_text()
                self.assertIn("${DB_ROOT:-", content)
                self.assertNotRegex(content, r"/home/[^/\s|]+/")
        validator = (ROOT / "scripts/validate_functional_mag_production_gates.py").read_text()
        self.assertIn("default_db_root()", validator)
        self.assertEqual(
            VALIDATOR.default_db_root({"DB_ROOT": ""}),
            Path.home() / "scratch" / "methanet_db",
        )
        self.assertEqual(
            VALIDATOR.default_db_root({}), Path.home() / "scratch" / "methanet_db"
        )
        self.assertEqual(
            VALIDATOR.default_db_root({"DB_ROOT": "/persistent/custom/db"}),
            Path("/persistent/custom/db"),
        )

    def test_snakemake_scaffold_expands_home_relative_database_paths(self):
        config = PORTABLE_DOCS[-1].read_text()
        snakefile_path = ROOT / "ai_docs/functional_metagenomics_expansion/snakemake_backbone/Snakefile"
        snakefile = snakefile_path.read_text()
        self.assertIn("project_root: .", config)
        self.assertIn("db_root: ~/scratch/methanet_db", config)
        self.assertIn("def expand_home_paths(value):", snakefile)

        start = snakefile.index("def expand_home_paths(value):")
        end = snakefile.index("\n\nconfig = expand_home_paths(config)", start)
        tree = ast.parse(snakefile[start:end])
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "expand_home_paths")
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(snakefile_path), "exec"), namespace)
        resolved = namespace["expand_home_paths"]({
            "database": ["~/scratch/methanet_db", "relative/path", "/durable/db"]
        })
        self.assertEqual(resolved["database"], [
            str(Path.home() / "scratch/methanet_db"), "relative/path", "/durable/db"
        ])

    def test_database_defaults_do_not_use_ephemeral_scratch_or_dot_fallback(self):
        scripts = [path for path in (ROOT / "scripts").rglob("*.sh")
                   if "DB_ROOT=\"${DB_ROOT:-" in path.read_text(errors="replace")]
        for path in scripts:
            with self.subTest(path=path.relative_to(ROOT)):
                content = path.read_text(errors="replace")
                self.assertIn("${HOME:?Set HOME or DB_ROOT}/scratch/methanet_db", content)
                self.assertNotIn("${SCRATCH:-", content)
                self.assertNotIn("${HOME:-.", content)

    def test_database_setup_fails_closed_when_home_and_db_root_are_missing(self):
        script = ROOT / "scripts/check_functional_mag_db_readiness_apollo3.sh"
        result = subprocess.run(
            ["bash", str(script)], cwd=ROOT,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "DB_ROOT": ""},
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HOME: Set HOME or DB_ROOT", result.stderr)


if __name__ == "__main__":
    unittest.main()
