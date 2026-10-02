import unittest
from scripts.final_benchmark.classify_visuals import validate, TABLE_SPANS
from scripts.evaluation.run_token_router import smoke_rows

class VisualClassificationTests(unittest.TestCase):

    def result(self, kind):
        spans = TABLE_SPANS[kind]
        return dict(case_id='case', visual_kind=kind, secondary_types=[], layout='single_panel', panel_count=1, table_structure=dict(row_spans=spans[0], column_spans=spans[1]), confidence=0.9, evidence='Visible header cell spans two columns.', ambiguities=[])

    def test_table_structure_and_identity_are_bound(self):
        for kind in TABLE_SPANS:
            validate(self.result(kind), 'case')
        value = self.result('Column-Spanning Table')
        value['table_structure']['row_spans'] = True
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            validate(value, 'case')
        with self.assertRaisesRegex(ValueError, 'Invalid case'):
            validate(self.result('Simple Table'), 'other')

    def test_unknown_category_is_rejected(self):
        value = self.result('Simple Table')
        value['visual_kind'] = 'Some table'
        with self.assertRaises(ValueError):
            validate(value, 'case')

    def test_smoke_selection_supports_fine_types(self):
        rows = [dict(visual_kind='Grouped Bar Chart', visual_family='chart', image_language='en', query_language='en'), dict(visual_kind='Mixed-Spanning Table', visual_family='table', image_language='zh', query_language='zh')]
        self.assertEqual(smoke_rows(rows), rows)
