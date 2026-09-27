"""Protect the landing-only release boundary and runtime packaging list."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'web/emergentbiome-methanet'
SPEC = importlib.util.spec_from_file_location('landing_assemble', WEB / 'tools/assemble_landing.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LandingPackagingTests(unittest.TestCase):
    def test_new_css_and_data_are_packaged_by_both_release_paths(self):
        for path in [WEB / 'tools/publish_site.sh', ROOT / '.github/workflows/deploy-landing.yml']:
            text = path.read_text()
            for name in ['molecular-applications.css', 'molecular-network.css']:
                self.assertIn(name, text)
                self.assertIn(name, MODULE.ITEMS)

    def test_every_static_dependency_exists(self):
        for path in MODULE.local_dependencies(WEB):
            self.assertTrue((WEB / path).is_file(), path)

    def test_report_and_internal_material_excluded(self):
        for name in ['report', 'tools', 'design-qa.md', 'README.md', '_site']:
            self.assertNotIn(name, MODULE.ITEMS)
        self.assertEqual(MODULE.DATA_FILES, {
            'atlas.json', 'molecular-application-cases-public-v1.json',
            'molecular-evidence-network-public-v1.json'})

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with self.assertRaisesRegex(ValueError, 'new staging directory'):
                MODULE.assemble(target, target / 'absent-review.json', target / 'manifest.json')


if __name__ == '__main__':
    unittest.main()
