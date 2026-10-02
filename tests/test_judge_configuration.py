import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.evaluation.judge_config import resolve
from scripts.evaluation.freeze_inputs import freeze, image_path
from scripts.evaluation.release import ROOT, check_dataset
from scripts.final_benchmark.api import read, save

class JudgeConfigurationTests(unittest.TestCase):

    def test_independent_overrides_and_default_inheritance(self):
        base = {'OPENAI_MODEL': 'gpt-6-astra', 'OPENAI_BASE_URL': 'old', 'OPENAI_API_KEY': 'old-key'}
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(resolve(base), base)
            self.assertEqual(resolve(dict(base, JUDGE_MODEL='judge'))['OPENAI_MODEL'], 'judge')
            self.assertEqual(resolve(dict(base, JUDGE_MODEL='judge'), 'cli')['OPENAI_MODEL'], 'cli')
        with patch.dict('os.environ', {'JUDGE_MODEL': 'env', 'JUDGE_BASE_URL': 'new', 'JUDGE_API_KEY': 'new-key'}, clear=True):
            cfg = resolve(dict(base, JUDGE_MODEL='file'))
            self.assertEqual([cfg['OPENAI_' + k] for k in ['MODEL', 'BASE_URL', 'API_KEY']], ['env', 'new', 'new-key'])
            self.assertEqual(base['OPENAI_MODEL'], 'gpt-6-astra')

    def test_retired_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Dataset retired'):
            check_dataset(ROOT / 'data/visual_benchmark/final_128_24lang_v1')
        with self.assertRaisesRegex(ValueError, 'Dataset retired'):
            check_dataset(ROOT / 'data/visual_benchmark/final_128_24lang_v3')
        check_dataset(ROOT / 'data/visual_benchmark/final_128_24lang_v4_context')

    def test_frozen_pixels_survive_old_dataset_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old'
            new = root / 'new'
            run = root / 'run'
            run.mkdir()
            rows = []
            for folder in (old, new):
                (folder / 'validation_release').mkdir(parents=True)
            for (i, content) in enumerate([b'shared', b'old-changed']):
                (old / (str(i) + '.png')).write_bytes(content)
                rows.append(dict(id=str(i), image_path='../' + str(i) + '.png', image_sha256=hashlib.sha256(content).hexdigest()))
            revised = [dict(r) for r in rows]
            (new / '0.png').write_bytes(b'shared')
            (new / '1.png').write_bytes(b'new-changed')
            revised[1]['image_sha256'] = hashlib.sha256(b'new-changed').hexdigest()
            (run / 'references.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in rows)))
            (new / 'validation_release/val.candidates.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in revised)))
            manifest = dict(run_key='key', dataset=str(old), references_sha256=hashlib.sha256((run / 'references.jsonl').read_bytes()).hexdigest())
            save(run / 'manifest.json', manifest)
            self.assertEqual(freeze(run, new), {'unique_images': 2, 'archived_images': 1})
            shutil.rmtree(old)
            frozen = read(run / 'frozen_images.json')
            for row in rows:
                self.assertEqual(hashlib.sha256(image_path(run, manifest, row, frozen).read_bytes()).hexdigest(), row['image_sha256'])
            with self.assertRaisesRegex(ValueError, 'index_mismatch'):
                image_path(run, dict(manifest, run_key='wrong'), rows[0], frozen)
