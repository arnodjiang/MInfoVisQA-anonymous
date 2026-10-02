import unittest
from scripts.evaluation.run import extract, usage_summary
from scripts.evaluation.score import correct, summarize, LANGUAGES

class EvaluationTests(unittest.TestCase):

    def test_answer_parser(self):
        self.assertEqual(extract('{"answer":"750"}'), '750')
        self.assertEqual(extract('```json\n{"answer":"1"}\n```'), '1')
        for raw in ['{"answer":1}', '{"answer":""}', '{"answer":"1","reason":"x"}']:
            with self.assertRaises(ValueError):
                extract(raw)

    def test_exact_numeric_not_tolerance(self):
        self.assertTrue(correct('750.0', '750'))
        self.assertTrue(correct('−1,000.00', '-1000'))
        self.assertTrue(correct(' é ', 'é'))
        self.assertFalse(correct('749', '750'))
        self.assertFalse(correct('50', '50%'))
        self.assertFalse(correct('Yes', 'yes'))

    def test_macro_not_mean_of_visible_columns(self):
        rows = []
        predictions = {}
        for v in LANGUAGES:
            for q in {v, 'zh', 'en'}:
                rid = v + '/' + q
                rows.append(dict(id=rid, source='fixture', visual_kind='chart', image_language=v, query_language=q, answer='1'))
                predictions[rid] = {'status': 'completed', 'prediction': '1' if v == q else '0'}
        (result, _) = summarize(rows, predictions)
        self.assertEqual(result['XQA'], {'ZH': 0, 'EN': 0})
        self.assertAlmostEqual(result['AVG'], 24 / 70 * 100)
        self.assertEqual(result['successful_predictions'], 70)
        del predictions['en/en']
        (result, _) = summarize(rows, predictions)
        self.assertEqual(result['failed_or_missing'], 1)
        self.assertEqual(result['LQA']['EN'], 0)
        self.assertAlmostEqual(result['AVG'], 23 / 70 * 100)

    def test_usage_unknown_and_retry_accounting(self):
        result = usage_summary([{'attempt_number': 1, 'retryable': True, 'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120, 'input_tokens_details': {'cached_tokens': 50}, 'output_tokens_details': {'reasoning_tokens': 15}}}, {'attempt_number': 2, 'usage': None}])
        self.assertEqual(result['api_attempts'], 2)
        self.assertEqual(result['transport_retries'], 1)
        self.assertEqual(result['total_tokens'], 120)
        self.assertEqual(result['cached_input_tokens'], 50)
        self.assertEqual(result['reasoning_output_tokens'], 15)
        self.assertFalse(result['usage_is_complete'])
        self.assertIsNone(usage_summary([{}])['total_tokens'])
        manual = usage_summary([{'id': 'x', 'attempt_number': 1, 'http_status': 403, 'retryable': False}, {'id': 'x', 'attempt_number': 2, 'http_status': 200}])
        self.assertEqual(manual['successful_http_responses'], 1)
        self.assertEqual(manual['transport_retries'], 0)
        self.assertEqual(manual['non_transport_retry_attempts'], 1)
