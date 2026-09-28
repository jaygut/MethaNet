"""Scene names and the proposed-study design stay consistent across the landing."""
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'web/emergentbiome-methanet'


class LandingSceneNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = (WEB / 'config.js').read_text()
        cls.index = (WEB / 'index.html').read_text()
        cls.labels = {scene: (int(n), name) for scene, n, name in
                      re.findall(r'id: "(\w+)", n: (\d+), label: "\d+ · ([^"]+)"', cls.config)}

    def test_skip_links_match_rail_labels(self):
        skip = {scene: (int(n), name) for scene, n, name in
                re.findall(r'<a href="#scene-(\w+)">(\d+) · ([^<]+)</a>', self.index)}
        self.assertEqual(len(self.labels), 10)
        self.assertEqual(skip.keys() - {'ask'}, self.labels.keys())
        for scene, label in self.labels.items():
            self.assertEqual(skip[scene], label, scene)

    def test_native_scene_kickers_match_their_labels(self):
        # Scenes 07 and 08 render their own headings rather than injected copy.
        kickers = {
            'platform': re.search(r'application-kicker">07 · ([^<]+)<', self.index).group(1),
            'network': re.search(r'<p class="smallcaps">08 · ([^<]+)</p>', self.index).group(1),
        }
        for scene, kicker in kickers.items():
            self.assertEqual(kicker.lower(), self.labels[scene][1].lower(), scene)

    def test_proposed_study_design_is_internally_consistent(self):
        block = re.search(r'const study = \{(.*?)\n  \};', self.config, re.S).group(1)
        value = lambda key: int(re.search(rf'\b{key}: (\d+)', block).group(1))
        stages = re.findall(r'"([^"]+)"', re.search(r'stages: \[(.*?)\]', block).group(1))
        campaigns = re.findall(r'"([^"]+)"', re.search(r'campaigns: \[(.*?)\]', block).group(1))
        self.assertEqual(len(stages), 3)
        self.assertEqual(value('plots'), len(stages) * value('salinityPositions') * value('plotsPerCombination'))
        self.assertEqual(value('microsites'), value('plots') * value('micrositesPerPlot'))
        self.assertEqual(value('sampleEvents'), value('microsites') * len(campaigns))
        # The no-script default note states the same nested, repeated design.
        note = re.search(r'id="askStudyNote">([^<]+)<', self.index).group(1)
        self.assertIn(f"{value('sampleEvents')} sample-events = {value('plots')} plots", note)
        self.assertIn('not independent replicates', note)

    def test_unsourced_targets_are_not_reintroduced(self):
        for key in ('pairedFluxTargetLo', 'pairedFluxTargetHi', 'pipelineDays'):
            self.assertNotIn(key, self.config)


if __name__ == '__main__':
    unittest.main()
