"""Contract tests for the narrow, attributed public evidence projection."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'web/emergentbiome-methanet/data'


class PublicPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = json.loads((DATA / 'molecular-evidence-network-public-v1.json').read_text())
        cls.application = json.loads((DATA / 'molecular-application-cases-public-v1.json').read_text())

    def test_public_sources_and_separate_scope(self):
        self.assertEqual(self.public['publication_status'], 'scoped_public_demonstration')
        self.assertEqual({s['id'] for s in self.public['public_sources']}, {'mucc', 'futian'})
        for source in self.public['public_sources']:
            self.assertEqual(source['license'], 'CC BY 4.0')
            self.assertTrue(source['url'].startswith('https://doi.org/'))
            self.assertTrue(source['creators'])
            self.assertTrue(source['modifications'])

    def test_no_raw_database_or_internal_payload(self):
        text = json.dumps(self.public)
        for pattern in [r'\bK\d{5}\b', 'sourcePayload', 'kegg_hit', 'pfam_hits',
                        '/home/', 'results/', '/policy/', 'internal_review', 'unknown_no_external_export']:
            self.assertIsNone(re.search(pattern, text), pattern)
        self.assertFalse(any(n['kind'] == 'component' for c in self.public['cases'] for n in c['nodes']))

    def test_public_facts_have_stable_subjects_and_scoped_predicates(self):
        facts = self.public['canonical_assertions']
        self.assertEqual(len(facts), 459)
        self.assertEqual(sum(map(len, facts.values())), 4725)
        for subject, rows in facts.items():
            self.assertTrue(subject.startswith('https://emergentbiome.earth/id/'))
            self.assertTrue(rows)
            for row in rows:
                self.assertTrue(row['predicate'].startswith('http'))
                self.assertNotIn('sourcePayload', json.dumps(row))

    def test_navigation_tree_and_source_references_are_closed(self):
        for case in self.public['cases']:
            ids = {n['id'] for n in case['nodes']}
            self.assertEqual(len(ids), len(case['nodes']))
            self.assertEqual(len(case['edges']), len(ids)-1)
            self.assertEqual({e['target'] for e in case['edges']}, ids-{case['root']})
            for edge in case['edges']:
                self.assertIn(edge['source'], ids)
            for node in case['nodes']:
                self.assertEqual(node['children'], [e['target'] for e in case['edges'] if e['source']==node['id']])
                for ref in node['source_refs']:
                    self.assertIn(ref, self.public['canonical_assertions'])

    def test_values_units_and_missingness_are_explicit_on_both_public_surfaces(self):
        self.assertEqual(self.public['depths'], self.application['depths'])
        self.assertEqual(len(self.public['depths']), 5)
        missing = sum(row[key] is None for row in self.public['depths']
                      for key in ['salinity_psu', 'tp_mg_g', 'ts_mg_g'])
        self.assertEqual(missing, 15)
        for case in self.public['cases']:
            for node in case['nodes']:
                if node['kind'] in {'measurement', 'rna_cell'}:
                    self.assertTrue(node['source_refs'])
                    self.assertTrue(node['facts'])
                    self.assertIn(node['state'], {'recorded', 'review', 'pending'})

    def test_qc_method_and_review_actor_are_correctly_typed(self):
        for case in self.public['cases']:
            root = next(n for n in case['nodes'] if n['id']==case['root'])
            keys = dict(root['facts'])
            prefix = 'CheckM2' if case['id']=='locate' else 'Source-reported'
            self.assertEqual(keys[prefix+' completeness'], str(case['completeness'])+'%')
            self.assertEqual(keys[prefix+' contamination'], str(case['contamination'])+'%')
            for row in case['profile']:
                self.assertTrue(row['basis'])
                self.assertTrue(row['source_ids'])
                self.assertIn('EmergentBiome authored', row['review_origin'])
            self.assertEqual(case['profile'][-1]['status'], 'Proposed')

    def test_both_application_surfaces_use_identical_case_copy(self):
        for case, public in zip(self.application['cases'], self.public['cases']):
            for key, value in case.items():
                self.assertEqual(value, public[key])
        self.assertEqual(self.application['public_sources'], self.public['public_sources'])


if __name__ == '__main__':
    unittest.main()
