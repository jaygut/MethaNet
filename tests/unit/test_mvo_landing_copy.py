"""The public case copy is identical across the landing-page data views."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'web/emergentbiome-methanet/data'

class LandingCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = json.loads((DATA / 'molecular-application-cases-public-v1.json').read_text())
        cls.network = json.loads((DATA / 'molecular-evidence-network-public-v1.json').read_text())

    def test_three_real_cases_have_stable_order_and_identity(self):
        expected = ['interpret', 'locate', 'design']
        self.assertEqual([case['id'] for case in self.application['cases']], expected)
        self.assertEqual([case['id'] for case in self.network['cases']], expected)
        for case in self.application['cases']:
            self.assertTrue(case['record'])
            self.assertTrue(case['proteome_id'])
            self.assertTrue(case['source_id'])

    def test_both_public_views_share_the_same_decision_narrative(self):
        narrative = ['question', 'record', 'finding', 'decision', 'next_action', 'boundary', 'status']
        for app_case, network_case in zip(self.application['cases'], self.network['cases']):
            for field in narrative:
                self.assertEqual(app_case[field], network_case[field], field)
                self.assertTrue(app_case[field])

    def test_copy_keeps_review_and_validation_limits_explicit(self):
        cases = {case['id']: case for case in self.application['cases']}
        self.assertIn('substrate-specific review', cases['interpret']['decision'])
        self.assertIn('sample membership', cases['locate']['decision'])
        self.assertIn('additional validation', cases['design']['boundary'])

if __name__=='__main__':unittest.main()
