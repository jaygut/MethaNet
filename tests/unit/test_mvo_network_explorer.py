"""Contracts for the reviewed three-case public evidence-network payload."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
WEB=ROOT/'web/emergentbiome-methanet'


class EvidenceNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((WEB/'data/molecular-evidence-network-public-v1.json').read_text())
        cls.application=json.loads((WEB/'data/molecular-application-cases-public-v1.json').read_text())

    def test_cases_and_canonical_keys_preserved(self):
        self.assertEqual([c['id'] for c in self.data['cases']],['interpret','locate','design'])
        self.assertEqual(self.data['snapshot'],self.application['snapshot'])
        fields=['id','question','record','setting','finding','decision','next_action',
                'boundary','numbers','status','proteome_id','source_id',
                'completeness','contamination']
        for c,source in zip(self.data['cases'],self.application['cases']):
            for key in fields:self.assertEqual(c[key],source[key],key)

    def test_unique_ids_and_closed_navigation_tree(self):
        for c in self.data['cases']:
            nodes={n['id']:n for n in c['nodes']}
            self.assertEqual(len(nodes),len(c['nodes']))
            self.assertEqual(len(c['edges']),len(nodes)-1)
            targets=[]
            for e in c['edges']:
                self.assertIn(e['source'],nodes);self.assertIn(e['target'],nodes)
                self.assertIn(e['target'],nodes[e['source']]['children'])
                self.assertIn('navigation',e['meaning'])
                targets.append(e['target'])
            self.assertEqual(len(targets),len(set(targets)))
            self.assertEqual(set(targets),set(nodes)-{c['root']})

    def test_every_source_reference_has_exact_assertions(self):
        refs={r for c in self.data['cases'] for n in c['nodes'] for r in n['source_refs']}
        self.assertEqual(refs,set(self.data['canonical_assertions']))
        self.assertEqual(len(refs),459)
        for ref,assertions in self.data['canonical_assertions'].items():
            self.assertTrue(ref.startswith('https://emergentbiome.earth/id/'))
            self.assertTrue(assertions)
            for a in assertions:self.assertTrue(a['predicate'].startswith('http'))

    def test_gene_and_review_grains(self):
        for cid,genes in [('interpret',4),('design',7)]:
            c=next(c for c in self.data['cases'] if c['id']==cid)
            self.assertEqual(sum(n['kind']=='gene' for n in c['nodes']),genes)
            self.assertNotIn('component',[n['kind'] for n in c['nodes']])
            self.assertTrue(any(n['kind']=='alternative' for n in c['nodes']))
            review=next(p for p in c['profile'] if p['key']=='review')
            self.assertIn(review['status'],['On hold','Review'])

    def test_depths_preserve_units_values_and_missingness(self):
        self.assertEqual(self.data['depths'],self.application['depths'])
        c=next(c for c in self.data['cases'] if c['id']=='locate')
        samples=[n for n in c['nodes'] if n['kind']=='sample']
        measures=[n for n in c['nodes'] if n['kind']=='measurement']
        self.assertEqual(len(samples),5);self.assertEqual(len(measures),40)
        self.assertEqual(sum(n['state']=='pending' for n in measures),15)
        self.assertTrue(all(len(n['children'])==8 for n in samples))
        self.assertTrue(all(n['state']=='pending' for n in samples))

    def test_rna_cells_stay_on_source_scale(self):
        c=next(c for c in self.data['cases'] if c['id']=='design')
        cells=[n for n in c['nodes'] if n['kind']=='rna_cell']
        self.assertEqual(len(cells),399)
        self.assertEqual(sum(float(dict(n['facts'])['Value on source scale'])>0 for n in cells),40)
        for n in cells:
            self.assertEqual(n['state'],'review')
            self.assertIn('/rna-observation/',n['source_refs'][0])
            self.assertEqual(dict(n['facts'])['Unit'],'source_processed_expression_scale')
            self.assertEqual(dict(n['facts'])['Field pairing'],'unresolved')

    def test_profile_is_qualitative_and_claim_safe(self):
        for c in self.data['cases']:
            self.assertEqual([p['key'] for p in c['profile']],['evidence','review','context','action'])
            for row in c['profile']:
                self.assertTrue({'key','label','status','state','basis','source_ids','review_origin'} <= set(row))
                self.assertTrue(row['source_ids'])
            required={'interpret':'experimental evidence','locate':'unresolved','design':'library assays'}
            self.assertIn(required[c['id']],c['boundary'].lower())
        self.assertEqual(self.data['publication_status'],'scoped_public_demonstration')
        self.assertIn('no abundance',self.data['layout_contract'])

    def test_no_private_paths_in_browser_export(self):
        self.assertNotIn('/home/',json.dumps(self.data))


if __name__=='__main__':unittest.main()
