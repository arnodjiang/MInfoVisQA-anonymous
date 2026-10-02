import unittest
from scripts.final_benchmark.query_policy import is_plain_number, with_reply

class QueryPolicyTests(unittest.TestCase):

    def test_numbers(self):
        for answer in ['750', '0', '-2', '−2.5', '+3', '0.25', '.5', '1,234.50', 750]:
            with self.subTest(answer=answer):
                self.assertTrue(is_plain_number(answer))
                self.assertEqual(with_reply('Question?', answer, 'Reply in English.'), 'Question?')

    def test_keep_language_for_non_plain_answers(self):
        for answer in ['50%', '10 kg', '约750', '1, 2', '1/2', '1e3', '一百', '١٢٣', '', None, True, '3–5', '12,34']:
            with self.subTest(answer=answer):
                self.assertFalse(is_plain_number(answer))
                self.assertEqual(with_reply('Q', answer, 'Instruction'), 'Q\nInstruction')
