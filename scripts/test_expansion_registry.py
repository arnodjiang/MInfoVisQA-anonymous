pass
import copy
import hashlib
import json
import unittest
from collections import Counter
from build_expansion_registry import enrich, propose_extension
from profile_and_sample import OUT, ROOT

class RegistryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pool = [json.loads(l) for l in (OUT / 'eligible_pool.jsonl').read_text().splitlines()]
        cls.registry = [json.loads(l) for l in (OUT / 'expansion_registry.jsonl').read_text().splitlines()]
        cls.manifest = {r['local_path']: r for r in json.loads((ROOT / 'data/download_manifest.json').read_text())['verified_files']}

    def test_identity_survives_row_file_relocation_and_answer_correction(self):
        for source in ['CharXiv', 'ChartQA', 'MMTU', 'TableVQA-Bench', 'ChartQAPro', 'Visual-TableQA']:
            r = next((r for r in self.pool if r['source'] == source))
            entry = next((e for e in self.registry if e['id'] == r['id']))
            im = {k: entry['features']['measured'][k] for k in ['image_sha256', 'width', 'height'] if k in entry['features']['measured']}
            moved = copy.deepcopy(r)
            moved.update(file='new-shard.parquet', row_index=999999, id='new-locator')
            manifest = dict(self.manifest, **{'new-shard.parquet': self.manifest[r['file']]})
            self.assertEqual(entry['base_id'], enrich(moved, im, manifest)['base_id'])
            moved['raw_record']['answer_correction_for_test'] = 'corrected'
            changed = enrich(moved, im, manifest)
            self.assertEqual(entry['base_id'], changed['base_id'])
            self.assertNotEqual(entry['record_version_sha256'], changed['record_version_sha256'])

    def test_append_proposal_deterministic_unique_and_within_budget(self):
        plan = propose_extension(self.registry, 32)
        self.assertEqual(plan, propose_extension(list(reversed(self.registry)), 32))
        self.assertEqual(len(plan), 32)
        self.assertEqual(len({r['base_id'] for r in plan}), 32)
        selected = [r for r in self.registry if r['membership'] == 'selected']
        selected_ids = {r['base_id'] for r in selected}
        self.assertTrue(selected_ids.isdisjoint((r['base_id'] for r in plan)))
        used_groups = {g for r in selected for g in r['dedup_group_keys']}
        for p in plan:
            r = next((r for r in self.registry if r['id'] == p['id']))
            self.assertTrue(used_groups.isdisjoint(r['dedup_group_keys']))
            used_groups.update(r['dedup_group_keys'])
        self.assertEqual((len(selected) + len(plan)) * 31, 4960)
        self.assertEqual(dict(Counter((p['source'] for p in plan))), {'CharXiv': 12, 'ChartQAPro': 6, 'TableVQA-Bench': 5, 'Visual-TableQA': 5, 'ChartQA': 2, 'MMTU': 2})

    def test_locked_candidates_and_reserves_unchanged(self):
        lock = json.loads((OUT / 'selection_lock.json').read_text())
        for (file, checksum) in lock['input_file_hashes'].items():
            self.assertEqual(hashlib.sha256((OUT / file).read_bytes()).hexdigest(), checksum)
        self.assertEqual(len(lock['selected_base_ids']), 128)
        self.assertEqual(len(lock['reserve_base_ids']), 128)
        self.assertEqual(len(set(lock['selected_base_ids'] + lock['reserve_base_ids'])), 256)
if __name__ == '__main__':
    unittest.main()
