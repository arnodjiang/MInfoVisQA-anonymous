pass
import ast
import tempfile
import unittest
from pathlib import Path
from .pipeline import LANGUAGES, bind, runtime_module, table_spec
from .export import REPLY, code_prelude, same_localized_numbers
from .repair_json import apply_edits

class Contracts(unittest.TestCase):

    def test_unique_language_configurations(self):
        configs = {(v, q, q) for v in LANGUAGES for q in {v, 'zh', 'en'}}
        self.assertEqual(len(configs), 31)
        self.assertEqual(sum((v == q for (v, q, a) in configs)), 11)
        self.assertEqual(sum((q == 'zh' and v != q for (v, q, a) in configs)), 10)
        self.assertEqual(sum((q == 'en' and v != q for (v, q, a) in configs)), 10)
        self.assertEqual(set(REPLY), set(LANGUAGES))

    def test_linked_reference_and_numeric_cell(self):
        source = {'rows': [[{'text': 'Product'}, {'text': '1e3'}], [{'text': 'Diapers'}, {'text': '03.00'}]]}
        spec = table_spec(source)
        self.assertIsNone(spec['data']['rows'][0][1]['label_key'])
        self.assertEqual(spec['data']['rows'][1][1]['text'], '03.00')
        key = spec['data']['rows'][1][0]['label_key']
        self.assertEqual(bind('Value for [[' + key + ']]?', {key: '纸尿裤'}), 'Value for 纸尿裤?')

    def test_adapter_contract(self):
        rt = runtime_module()
        rt.validate_code('def render(data, labels):\n    return finish(plt.figure(), labels, [])')
        for text in ['import os', 'def render(data, labels):\n    return open(".env").read()', 'def render(data, labels):\n    return np.load("private.npy")']:
            with self.assertRaises(ValueError):
                rt.validate_code(text)

    def test_exported_prelude_compiles(self):
        compile(code_prelude(), '<snapshot>', 'exec')

    def test_long_finish_label_wraps_inside_canvas(self):
        rt = runtime_module()
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(2, 1), dpi=100)
        (image, boxes) = rt.finish(fig, {'long': '日本語の長い副标题，说明接触减少和其他措施的重要性'}, [{'key': 'long', 'x': 0.5, 'y': 0.95, 'size': 10, 'max_width': 0.98}])
        self.assertTrue(boxes[0]['inside_canvas'])

    def test_multiline_shaping_does_not_treat_newline_as_glyph(self):
        rt = runtime_module()
        helper = rt.mask.__wrapped__.__globals__
        helper['MISSING_GLYPHS'].clear()
        rt.mask.cache_clear()
        single = rt.mask('Heading', 24)
        multiple = rt.mask('Heading\n中文 العربية', 24)
        self.assertGreater(multiple.height, single.height)
        self.assertEqual(helper['MISSING_GLYPHS'], [])
        self.assertEqual(rt.wrap('First\nSecond\nThird', 500, 24), ['First', 'Second', 'Third'])
        self.assertTrue(all((rt.mask(line, 27).width <= 235 for line in rt.wrap('ジョージア・ドーム • ジョージア州アトランタ（サウスイースタン・カンファレンス優勝決定戦）', 235, 27))))

    def test_arabic_mixed_numbers_keep_digit_order(self):
        rt = runtime_module()
        helper = rt.mask.__wrapped__.__globals__
        runs = helper['bidi_runs']('18 يوليو 2013')
        self.assertEqual(runs, [('2013', 'ltr'), (' يوليو ', 'rtl'), ('18', 'ltr')])

    def test_numeric_checks_respect_cjk_dates_and_ordinals(self):
        self.assertTrue(same_localized_numbers('6th', '第6名'))
        self.assertTrue(same_localized_numbers('July 11, 2013', '2013年7月11日'))
        self.assertFalse(same_localized_numbers('July 11, 2013', '2013年7月12日'))
        self.assertFalse(same_localized_numbers('6th', '第7名'))
        self.assertTrue(same_localized_numbers('4-14 (K-8)', '4-14 岁（幼儿园至 8 年级）'))
        self.assertFalse(same_localized_numbers('-8', '8'))

    def test_json_repair_preserves_values_and_code(self):
        raw = '{"x":[1,2]],"code":"a[0]"}'
        self.assertEqual(apply_edits(raw, [{'old': '[1,2]]', 'new': '[1,2]'}]), '{"x":[1,2],"code":"a[0]"}')
        with self.assertRaises(ValueError):
            apply_edits(raw, [{'old': '[1,2]]', 'new': '[1,3]'}])
        with self.assertRaises(ValueError):
            apply_edits(raw, [{'old': 'a[0]', 'new': 'a{0}'}])
if __name__ == '__main__':
    unittest.main()
