import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.evaluation.publish_results import update_row, publish
from scripts.evaluation.run import Runner
from scripts.evaluation.judge_config import resolve
from scripts.final_benchmark.api import save

class PublicationTests(unittest.TestCase):

    def test_update_only_sol_row(self):
        text = 'before\n    ' + '\\bottomrule' + '\nafter\n'
        astra = update_row(text, 'gpt-6-astra', [80.5] * 27)
        pending = update_row(astra, 'gpt-5.6-sol', [None] * 27)
        updated = update_row(pending, 'gpt-5.6-sol', [75.1] * 27)
        self.assertEqual(updated.count('GPT-5.6-Sol'), 1)
        self.assertEqual(updated.count('80.5'), 27)
        self.assertEqual(updated.count('75.1'), 27)
        self.assertNotIn('textemdash', updated)
        self.assertTrue(updated.startswith('before\n'))
        self.assertTrue(updated.endswith('after\n'))

    def test_incomplete_judge_cannot_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            table = out / 'table.tex'
            table.write_text('unchanged')
            save(out / 'progress.json', {'state': 'running', 'remaining': 1})
            save(out / 'manifest.json', {})
            with self.assertRaisesRegex(ValueError, 'not complete'):
                publish(out, table)
            self.assertEqual(table.read_text(), 'unchanged')

    def test_inference_override_does_not_change_judge_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'dataset'
            (dataset / 'validation_release').mkdir(parents=True)
            (dataset / 'validation_release/val.candidates.jsonl').write_text('')
            base = {'OPENAI_API_KEY': 'fixture', 'OPENAI_MODEL': 'gpt-6-astra'}
            args = SimpleNamespace(output=root / 'run', dataset=dataset, model='gpt-5.6-sol', max_output_tokens=8192, smoke=True, workers=1, judge_model='gpt-6-astra')
            with patch('scripts.evaluation.run.load', return_value=dict(base)):
                runner = Runner(args)
            self.assertEqual(runner.identity['model'], 'gpt-5.6-sol')
            self.assertEqual(resolve(base, model='gpt-6-astra')['OPENAI_MODEL'], 'gpt-6-astra')
            runner.lock_file.close()
