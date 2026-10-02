import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.final_benchmark.api import save, read, digest
from scripts.final_benchmark.provenance import sha256
from scripts.evaluation.judge import Judge, parse_verdict, load_run, upgrade_binary_policy, PROMPT
from scripts.evaluation.reference_corrections import load, corrected_reference

class JudgeTests(unittest.TestCase):

    def test_correction_is_bound_to_source_not_prediction(self):
        c = next(iter(load().values()))
        row = dict(case_id=c['case_id'], source=c['source'], provenance={k: c[k] for k in ['file', 'row_index']}, original_answer=c['original_answer'], answer='translated original')
        self.assertEqual(corrected_reference(row, load()), ('0°、45°', c['id']))
        row['provenance']['row_index'] += 1
        with self.assertRaises(ValueError):
            corrected_reference(row, load())

    def test_judge_output_is_validated(self):
        good = {'verdict': 'equivalent', 'category': 'rounding', 'language_compliance': 'not_applicable', 'reason': 'Same rounded value.'}
        self.assertEqual(parse_verdict(good), good)
        normalized = parse_verdict(dict(good, verdict='uncertain'))
        self.assertEqual(normalized['verdict'], 'different')
        self.assertEqual(normalized['reason'], good['reason'])
        for changed in [dict(good, verdict=True), dict(good, verdict='yes'), dict(good, reason=''), dict(good, language_compliance='ok')]:
            with self.assertRaises(ValueError):
                parse_verdict(changed)

    def test_binary_migration_preserves_results_without_api_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            legacy_prompt = 'legacy text-only prompt'
            old_sha = digest(legacy_prompt)
            policy = {'protocol': 'strict_match_then_text_equivalence_v2', 'prompt_sha256': digest(PROMPT), 'verdict_policy': 'binary'}
            old = dict(policy, prompt_sha256=old_sha, uncertain_policy='not correct; reported separately; fixed denominator')
            del old['verdict_policy']
            save(out / 'manifest.json', old)
            (out / 'prompt.txt').write_text(legacy_prompt)
            refs = [{'id': 'a', 'answer': '1'}, {'id': 'b', 'answer': '2'}]
            preds = {'a': {'prediction': 'one'}, 'b': {'prediction': 'green'}}
            for (row, verdict) in zip(refs, ['equivalent', 'uncertain']):
                save(out / 'judgments' / (row['id'] + '.json'), {'id': row['id'], 'verdict': verdict, 'reason': 'Original reason.', 'input_fingerprint': digest([row, preds[row['id']], corrected_reference(row, {}), old_sha])})
            with patch('scripts.evaluation.judge.LEGACY_TEXT_PROMPT_SHA256', old_sha), patch('scripts.evaluation.judge.API.call') as api:
                upgrade_binary_policy(out, policy, refs, preds, {})
                upgrade_binary_policy(out, policy, refs, preds, {})
                api.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'policy_changed'):
                    upgrade_binary_policy(out, dict(policy, judge_model='changed'), refs, preds, {})
            self.assertEqual(read(out / 'judgments/a.json')['verdict'], 'equivalent')
            self.assertEqual(read(out / 'judgments/b.json')['verdict'], 'different')
            self.assertEqual(read(out / 'judgments/b.json')['reason'], 'Original reason.')
            self.assertEqual(read(out / 'binary_migration.json')['uncertain_converted_to_different'], 1)

    def test_retired_inference_cannot_start_judge(self):
        from scripts.evaluation.release import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            save(run / 'manifest.json', dict(dataset=str(ROOT / 'data/visual_benchmark/final_128_24lang_v1')))
            with self.assertRaisesRegex(ValueError, 'Dataset retired'):
                load_run(run)

    def test_frozen_run_smoke_resume_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'run'
            run.mkdir()
            image = root / 'image.png'
            image.write_bytes(b'fixture image')
            refs = [dict(id=str(i), case_id='fixture', answer='1', original_answer='1', query='Number?', source='fixture', provenance={}, visual_kind='chart', image_language='en', query_language='en', answer_language='en', image_path=str(image), image_sha256=sha256(image)) for i in range(3)]
            (run / 'references.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in refs)))
            save(run / 'manifest.json', dict(run_key='key', model='fixture', dataset=str(root), references_sha256=sha256(run / 'references.jsonl')))
            for (i, pred) in enumerate(['1', 'one', None]):
                save(run / 'predictions' / (str(i) + '.json'), dict(id=str(i), run_key='key', status='completed' if pred else 'failed', prediction=pred))
            corrections = root / 'corrections.json'
            save(corrections, {'corrections': []})
            image.unlink()
            args = SimpleNamespace(run=run, output=None, corrections=corrections, model=None, workers=1, ids=None, limit=1)
            decision = dict(verdict='equivalent', category='paraphrase', language_compliance='pass', reason='One means 1.')
            with patch('scripts.evaluation.judge.load_config', return_value={'OPENAI_MODEL': 'fixture', 'OPENAI_API_KEY': 'fake'}), patch('scripts.evaluation.judge.API.call', return_value=(decision, 'request')) as call:
                judge = Judge(args)
                judge.execute()
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.args[3]['candidate_answer'], 'one')
                self.assertNotIn('model', call.call_args.args[3])
                self.assertEqual(set(call.call_args.args[3]), {'question', 'reference_answer', 'candidate_answer', 'answer_language'})
                self.assertNotIn('image', call.call_args.kwargs)
                self.assertEqual(judge.out.name, 'llm_judge_text_v2')
                result = json.loads((judge.out / 'scores.json').read_text())
                self.assertAlmostEqual(result['cohorts']['all']['micro_ACC'], 200 / 3)
                self.assertEqual(result['cohorts']['all']['failed_or_missing'], 1)
                self.assertEqual(result['verdict_counts'], {'equivalent': 2, 'different': 1})
                judge.lockfile.close()
                judge = Judge(args)
                judge.execute()
                self.assertEqual(call.call_count, 1)
                judge.lockfile.close()
            (run / 'predictions/2.json').unlink()
            with self.assertRaisesRegex(ValueError, 'inference_not_complete'):
                load_run(run)
if __name__ == '__main__':
    unittest.main()
