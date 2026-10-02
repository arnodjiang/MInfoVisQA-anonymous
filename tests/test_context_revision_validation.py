import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.final_benchmark.api import digest, save
from scripts.final_benchmark.provenance import sha256
from scripts.final_benchmark.restore_context import source_paragraphs
from scripts.final_benchmark.validate_context_revision import validate
from scripts.evaluation.context_input import VERSION

class RevisionValidationTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (self.source, self.dataset, self.work) = (root / 'source', root / 'dataset', root / 'work')
        for p in (self.source, self.dataset):
            (p / 'validation_release').mkdir(parents=True)
            (p / 'image.png').write_bytes(b'same image')
        source = {'candidate': {'source': 'MMTU', 'raw_record': {'dataset': 'FinQA', 'metadata': json.dumps({'pre_text': ['Source value 137.']})}}}
        save(self.source / 'cases/case1/source.json', source)
        paragraphs = source_paragraphs(source)
        self.context = {'policy': VERSION, 'language': 'en', 'paragraphs': paragraphs, 'text_sha256': digest(paragraphs), 'source_text_sha256': digest(paragraphs)}
        save(self.work / 'contexts/case1/en.json', self.context)
        self.old = [{'id': 'one', 'case_id': 'case1', 'query': 'Q?', 'query_language': 'en', 'answer': '137', 'image_path': '../image.png', 'image_sha256': sha256(self.source / 'image.png')}, {'id': 'two', 'case_id': 'case2', 'query': 'Other?', 'query_language': 'en', 'answer': '5', 'image_path': '../image.png', 'image_sha256': sha256(self.source / 'image.png')}]
        self.new = [dict(self.old[0], source_context=self.context, input_condition='source_context_restored_text'), self.old[1]]
        (self.source / 'validation_release/val.candidates.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in self.old)))
        save(self.dataset / 'invalidated_evaluation_inputs.json', {'variants': [{'variant_id': 'one'}]})

    def run_validation(self):
        (self.dataset / 'validation_release/val.candidates.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in self.new)))
        with patch('scripts.final_benchmark.validate_context_revision.dependency_audit', return_value={'cases': [{'case_id': 'case1'}], 'affected_variants': 1}), patch('scripts.final_benchmark.validate_context_revision.LANGUAGES', ['en']):
            return validate(self.source, self.dataset, self.work)

    def test_accepts_only_context_change(self):
        report = self.run_validation()
        self.assertEqual(report['changed_variants'], 1)
        self.assertEqual(report['unchanged_variants'], 1)

    def test_rejects_reference_edit_or_wrong_context(self):
        self.new[0]['answer'] = '999'
        with self.assertRaisesRegex(ValueError, 'non_context_fields_changed'):
            self.run_validation()
        self.new[0]['answer'] = '137'
        self.new[0]['source_context'] = dict(self.context, language='zh')
        with self.assertRaisesRegex(ValueError, 'context_policy'):
            self.run_validation()

    def test_rejects_silent_image_or_unaffected_query_changes(self):
        (self.dataset / 'image.png').write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError, 'image_hash_mismatch'):
            self.run_validation()
        (self.dataset / 'image.png').write_bytes(b'same image')
        self.new[1] = dict(self.old[1], query='Changed other question')
        with self.assertRaisesRegex(ValueError, 'unaffected_row_changed'):
            self.run_validation()
if __name__ == '__main__':
    unittest.main()
