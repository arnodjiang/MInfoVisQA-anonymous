import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from scripts.final_benchmark.api import digest, read, save
from scripts.final_benchmark.restore_context import source_paragraphs, protect, restore
from scripts.final_benchmark.provenance import sha256
from scripts.evaluation.context_input import VERSION, context_prompt, input_text, protocol
from scripts.evaluation.context_rerun import seed_unchanged_judgments
from scripts.evaluation.migrate_token_router import reusable
from scripts.evaluation.judge import PROMPT as JUDGE_PROMPT

class ContextTests(unittest.TestCase):

    def test_chartqapro_keeps_document_and_excludes_answer_and_year_flags(self):
        source = {'candidate': {'source': 'ChartQAPro', 'raw_record': {'Paragraph': 'Survey of 3,375 people.\n\nDetails.', 'Answer': ['secret'], 'Year': ['YES'], 'Question': ['Q?']}}}
        paragraphs = source_paragraphs(source)
        self.assertEqual('\n'.join((p['text'] for p in paragraphs)), source['candidate']['raw_record']['Paragraph'])
        self.assertEqual([p['source_field'] for p in paragraphs], ['Paragraph'] * 3)
        self.assertNotIn('secret', json.dumps(paragraphs))
        self.assertNotIn('YES', json.dumps(paragraphs))
        source['candidate']['raw_record']['Paragraph'] = ''
        self.assertEqual(source_paragraphs(source), [])

    def test_table_transcriptions_and_other_qa_labels_are_not_context(self):
        for dataset in ('TableVQA-Bench', 'CharXiv', 'ChartQA', 'Visual-TableQA'):
            self.assertEqual(source_paragraphs({'candidate': {'source': dataset, 'raw_record': {'text_html_table': '<table><tr><td>secret</td></tr></table>', 'text_markdown_table': '| secret |', 'descriptive_a1': 'secret', 'answer': 'secret'}}}), [])

    def context(self):
        paragraphs = [{'id': 'post_000', 'text': 'The expenses were 137 and 133.'}]
        return {'policy': VERSION, 'language': 'en', 'paragraphs': paragraphs, 'text_sha256': digest(paragraphs)}

    def test_source_allowlist_excludes_labels_and_gold_selectors(self):
        source = {'candidate': {'source': 'MMTU', 'raw_record': {'dataset': 'FinQA', 'metadata': json.dumps({'pre_text': ['Before.'], 'post_text': ['After.'], 'label': 'secret', 'qa': {'gold_inds': {'x': 'secret'}, 'answer': 'secret'}})}}}
        paragraphs = source_paragraphs(source)
        self.assertEqual([p['text'] for p in paragraphs], ['Before.', 'After.'])
        self.assertNotIn('secret', json.dumps(paragraphs))

    def test_protected_numbers_cannot_be_changed_or_added(self):
        originals = [{'id': 'pre_000', 'text': '$ 375 million at 5.0% in 2012.'}]
        (masked, tokens) = protect(originals)
        self.assertEqual(restore({'paragraphs': masked}, masked, originals, tokens), originals)
        wrong = [{'id': 'pre_000', 'text': masked[0]['text'].replace('[[N_000000]]', '376')}]
        with self.assertRaisesRegex(ValueError, 'placeholder_changed'):
            restore({'paragraphs': wrong}, masked, originals, tokens)
        wrong = [{'id': 'pre_000', 'text': masked[0]['text'] + ' Answer 3.'}]
        with self.assertRaisesRegex(ValueError, 'number_added'):
            restore({'paragraphs': wrong}, masked, originals, tokens)

    def test_context_input_preserves_question_and_uses_query_language(self):
        row = {'query': 'How much?', 'query_language': 'en', 'answer': 'secret'}
        base = 'Use only the supplied image and question. If the image does not support an answer, return UNANSWERABLE.'
        self.assertEqual(input_text(row), row['query'])
        self.assertEqual(context_prompt(base, row), base)
        self.assertEqual(protocol([row], base, base), {})
        changed = dict(row, source_context=self.context())
        payload = json.loads(input_text(changed))
        self.assertEqual(payload['question'], row['query'])
        self.assertNotIn('secret', input_text(changed))
        self.assertIn('image and source context', context_prompt(base, changed))
        self.assertFalse(reusable(row, changed))
        changed['query_language'] = 'zh'
        with self.assertRaisesRegex(ValueError, 'context_policy'):
            input_text(changed)

    def test_judge_reuse_requires_identical_reference_prediction_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'old'
            out = root / 'new/llm_judge_text_v2'
            policy = {'protocol': 'v2', 'judge_model': 'judge', 'prompt_sha256': digest(JUDGE_PROMPT)}
            save(source / 'llm_judge_text_v2/manifest.json', policy)
            save(out / 'manifest.json', policy)
            save(source / 'manifest.json', {'model': 'model'})
            row = {'id': 'one', 'answer': '1', 'query': 'Q?', 'original_query': 'Older wording?'}
            (source / 'references.jsonl').write_text(json.dumps(row) + '\n')
            prediction = {'id': 'one', 'prediction': 'one', 'status': 'completed'}
            path = source / 'predictions/one.json'
            save(path, prediction)
            new_prediction = dict(prediction, reused_from={'source_prediction_sha256': sha256(path)})
            fingerprint = digest([row, prediction, ('1', None), digest(JUDGE_PROMPT)])
            save(source / 'llm_judge_text_v2/judgments/one.json', {'id': 'one', 'input_fingerprint': fingerprint, 'verdict': 'equivalent'})
            from scripts.final_benchmark.clean_release import clean_reference
            judge = SimpleNamespace(out=out, manifest={'model': 'model'}, refs=[clean_reference(row)], results={}, predictions={'one': new_prediction}, corrections={}, fingerprint=lambda _: 'new-fp')
            self.assertEqual(seed_unchanged_judgments(judge, source, {'one'}), 0)
            self.assertEqual(seed_unchanged_judgments(judge, source, set()), 1)
            self.assertEqual(read(out / 'judgments/one.json')['input_fingerprint'], 'new-fp')
            judge.results = {}
            judge.predictions['one']['prediction'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'prediction_changed'):
                seed_unchanged_judgments(judge, source, set())
if __name__ == '__main__':
    unittest.main()
