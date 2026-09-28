"""Hash-bound approval is required for the public evidence projections."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'landing_assemble', ROOT / 'web/emergentbiome-methanet/tools/assemble_landing.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicationReviewTests(unittest.TestCase):
    def test_review_accepts_exact_approved_projection_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            payloads = {
                'data/molecular-application-cases-public-v1.json': b'{"cases":[]}',
                'data/molecular-evidence-network-public-v1.json': b'{"cases":[]}',
            }
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
            receipt = root / 'publication-review.json'
            receipt.write_text(json.dumps({
                'decision': 'approved_scoped_landing',
                'scope': 'three_case_presentation_only',
                'reviewers': {
                    'science_provenance': 'approved',
                    'public_boundary': 'approved',
                    'operations_portability': 'approved',
                },
                'files': {name: hashlib.sha256(payload).hexdigest()
                          for name, payload in payloads.items()},
            }))

            checked = MODULE.validate_publication_review(receipt, root)
            self.assertEqual(set(checked['reviewed_files']), set(payloads))

    def test_review_rejects_stale_hashes_and_missing_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            path = root / 'data/molecular-evidence-network-public-v1.json'
            path.write_text('{}')
            receipt = root / 'publication-review.json'
            receipt.write_text(json.dumps({
                'decision': 'approved_scoped_landing',
                'scope': 'three_case_presentation_only',
                'reviewers': {
                    'science_provenance': 'approved',
                    'public_boundary': 'approved',
                    'operations_portability': 'approved',
                },
                'files': {
                    'data/molecular-application-cases-public-v1.json': '0' * 64,
                    'data/molecular-evidence-network-public-v1.json': '0' * 64,
                },
            }))
            with self.assertRaisesRegex(ValueError, 'hashes'):
                MODULE.validate_publication_review(receipt, root)

    def test_review_rejects_missing_or_unapproved_review_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            payloads = {
                'data/molecular-application-cases-public-v1.json': b'{}',
                'data/molecular-evidence-network-public-v1.json': b'{}',
            }
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
            receipt = root / 'publication-review.json'
            receipt.write_text(json.dumps({
                'decision': 'approved_scoped_landing',
                'scope': 'three_case_presentation_only',
                'reviewers': {'science_provenance': 'approved', 'public_boundary': 'pending'},
                'files': {name: hashlib.sha256(payload).hexdigest()
                          for name, payload in payloads.items()},
            }))
            with self.assertRaisesRegex(ValueError, 'three required review lanes'):
                MODULE.validate_publication_review(receipt, root)

    def test_deployment_gate_requires_tracked_committed_approval_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            root = repository / 'web/emergentbiome-methanet'
            data = root / 'data'
            data.mkdir(parents=True)
            payloads = {
                'data/molecular-application-cases-public-v1.json': b'{"cases":[]}',
                'data/molecular-evidence-network-public-v1.json': b'{"cases":[]}',
            }
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
            receipt = root / 'publication-review.json'
            receipt.write_text(json.dumps({
                'decision': 'approved_scoped_landing',
                'scope': 'three_case_presentation_only',
                'reviewers': {
                    'science_provenance': 'approved',
                    'public_boundary': 'approved',
                    'operations_portability': 'approved',
                },
                'files': {name: hashlib.sha256(payload).hexdigest()
                          for name, payload in payloads.items()},
            }))

            def git(*args):
                return subprocess.run(
                    ['git', '-C', str(repository), *args],
                    capture_output=True, text=True, check=True,
                )

            git('init', '-q')
            git('config', 'user.name', 'Review Test')
            git('config', 'user.email', 'review-test@example.invalid')
            with self.assertRaisesRegex(ValueError, 'not tracked by Git'):
                MODULE.validate_publication_review(receipt, root, require_tracked=True)

            git('add', 'web/emergentbiome-methanet')
            git('commit', '-q', '-m', 'approved fixture')
            MODULE.validate_publication_review(receipt, root, require_tracked=True)

            changed = b'{"cases":[{"changed":true}]}'
            (root / 'data/molecular-evidence-network-public-v1.json').write_bytes(changed)
            approval = json.loads(receipt.read_text())
            approval['files']['data/molecular-evidence-network-public-v1.json'] = hashlib.sha256(changed).hexdigest()
            receipt.write_text(json.dumps(approval))
            with self.assertRaisesRegex(ValueError, 'must be committed before deployment'):
                MODULE.validate_publication_review(receipt, root, require_tracked=True)


if __name__ == '__main__':
    unittest.main()
