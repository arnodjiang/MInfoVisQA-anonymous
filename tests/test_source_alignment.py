import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.final_benchmark.api import API, read, save
from scripts.final_benchmark.pipeline import Builder, table_spec
from scripts.final_benchmark.provenance import source_binding, verify_polish, verify_locale, locale_parent, render_binding, render_is_current

class SourceAlignmentTests(unittest.TestCase):

    def test_telugu_and_bengali_use_fonts_that_shape_conjuncts(self):
        import os
        if not Path(os.environ.get('MVISQA_FONT', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf')).is_file():
            self.skipTest('Rendering fonts are not installed; API evaluation does not require them')
        from scripts.final_benchmark.pipeline import runtime_module
        import uharfbuzz as hb
        runtime_module()
        import multilingual_drawing as drawing
        for (script, text) in [('TELUGU', 'ప్ర'), ('BENGALI', 'ক্ষ')]:
            if script not in drawing.SCRIPT_FONTS:
                self.skipTest('Dedicated script font unavailable')
            runs = drawing.font_runs(text, 'ltr')
            self.assertEqual(len(runs), 1)
            (run, path) = runs[0]
            font = hb.Font(drawing.font_face(path))
            hb.ot_font_set_funcs(font)
            buf = hb.Buffer()
            buf.add_str(run)
            buf.guess_segment_properties()
            hb.shape(font, buf)
            self.assertTrue(all((g.codepoint for g in buf.glyph_infos)))
            self.assertEqual({g.cluster for g in buf.glyph_infos}, {0})

    def fixture(self, root):
        folder = root / 'cases' / 'case1'
        folder.mkdir(parents=True)
        (folder / 'original').mkdir()
        image = folder / 'original/original.png'
        image.write_bytes(b'first image')
        candidate = dict(id='case1', source='fixture', file='source.parquet', row_index=3, slot='qa', question='Original question?', answer='source answer', raw_record={'id': 'upstream'})
        identity = dict(id='case1', base_id='qa_first', source='fixture', question=candidate['question'], provenance={k: candidate[k] for k in ('file', 'row_index', 'slot')}, fingerprints={'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}, membership='reserve')
        save(folder / 'source.json', dict(candidate=candidate, identity=identity))
        binding = source_binding(folder)
        spec = dict(id='case1', base_id='qa_first', source='fixture', labels={'k': 'Source label'}, data={})
        qa = dict(question='What is [[k]]?', answer='source answer', answer_template='source answer', source_binding=binding)
        save(folder / 'spec.json', spec)
        save(folder / 'qa.json', qa)
        return (folder, candidate, identity, binding, spec, qa)

    def test_wrong_source_image_and_query_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (folder, c, i, b, s, q) = self.fixture(Path(tmp))
            self.assertEqual(source_binding(folder, c, dict(i, membership='selected')), b)
            with self.assertRaisesRegex(ValueError, 'candidate_changed'):
                source_binding(folder, dict(c, question='Other question?'), i)
            (folder / 'original/original.png').write_bytes(b'other entry image')
            with self.assertRaisesRegex(ValueError, 'original_image_mismatch'):
                source_binding(folder)

    def test_polish_requires_same_id_language_source_query_and_bindings(self):
        loc = dict(question='What is [[k]]?', labels={'k': 'label'})
        r = dict(id='one', language='en', original_source_query='Source?', current_query=loc['question'], protected_bindings=loc['labels'])
        verify_polish(r, 'one', 'en', 'Source?', loc)
        for (field, value) in [('id', 'two'), ('language', 'zh'), ('original_source_query', 'Other?'), ('protected_bindings', {'k': 'different'})]:
            with self.assertRaises(ValueError):
                verify_polish(dict(r, **{field: value}), 'one', 'en', 'Source?', loc)

    def test_locale_cannot_be_reused_after_source_or_qa_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            (folder, c, i, b, s, q) = self.fixture(Path(tmp))
            loc = dict(labels=s['labels'], question=q['question'], answer_template=q['answer_template'], input_binding=locale_parent(s, q, b))
            verify_locale(loc, s, q, b)
            with self.assertRaises(ValueError):
                verify_locale(loc, s, dict(q, question='Different [[k]]?'), b)
            with self.assertRaises(ValueError):
                verify_locale(loc, s, q, dict(b, case_id='other'))

    def test_changed_labels_or_image_invalidate_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            (folder, c, i, b, s, q) = self.fixture(Path(tmp))
            image = folder / 'en.png'
            image.write_bytes(b'rendered')
            save(image.with_suffix('.layout.json'), dict(render_binding=render_binding(s, s['labels']), image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), all_text_inside_canvas=True, all_text_inside_cells=True, missing_glyphs=[]))
            self.assertTrue(render_is_current(image, s, s['labels']))
            self.assertFalse(render_is_current(image, s, {'k': 'Different language'}))
            image.write_bytes(b'another case')
            self.assertFalse(render_is_current(image, s, s['labels']))

    def test_diagram_cannot_be_flattened_to_table_notes(self):
        for extraction in [dict(rows=[], non_tabular_visuals=[{'type': 'diagram'}]), dict(rows=[], recovery={'uncertainties': ['The texts are labels from the circular diagram below the table.']})]:
            with self.assertRaisesRegex(ValueError, 'mixed_visual_requires_full_reconstruction'):
                table_spec(extraction)
        with self.assertRaisesRegex(ValueError, 'missing_full_visual_inventory'):
            table_spec({'rows': []}, require_visual_inventory=True)

    def test_translation_receives_matching_source_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (folder, c, i, b, s, q) = self.fixture(root)
            runner = Builder.__new__(Builder)
            runner.out = root
            runner.by_id = {c['id']: c}
            runner.identities = {c['id']: i}
            result = {'locales': {'zh': dict(labels={'k': '源标签'}, question='[[k]]是什么？', answer_template='源答案')}}
            from unittest.mock import Mock
            runner.api = Mock()
            runner.api.call.return_value = (result, 'key')
            runner.log = Mock()
            runner.translate(('case1', ['zh']))
            (args, kwargs) = runner.api.call.call_args
            self.assertEqual(args[3]['original_source_query'], c['question'])
            self.assertEqual(args[3]['original_source_answer'], c['answer'])
            self.assertEqual(args[3]['source_binding'], b)
            self.assertEqual(kwargs['image'], folder / 'original/original.png')
            verify_locale(read(folder / 'locales/zh.json'), s, q, b)

    def test_api_sends_the_same_bytes_it_hashes_and_checks_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'image.png'
            image.write_bytes(b'before')
            api = API(root, {'OPENAI_MODEL': 'fixture', 'OPENAI_API_KEY': 'fake'})

            class MutatingLock:

                def __enter__(self):
                    image.write_bytes(b'after')

                def __exit__(self, *args):
                    pass
            api.lock = MutatingLock()
            response = SimpleNamespace(id='r', model='fixture', status='completed', usage=None)
            with patch('scripts.final_benchmark.api.create', return_value=response) as create, patch('scripts.final_benchmark.api.response_text', return_value='{"ok":true}'):
                (_, key) = api.call('translate', 'case1', 'prompt', {}, image=image)
                sent = create.call_args.kwargs['image_uris'][0].split(',')[1]
                self.assertEqual(base64.b64decode(sent), b'before')
            request = read(root / 'api/translate/case1' / key / 'request.json')
            self.assertEqual(request['image_sha256'], hashlib.sha256(b'before').hexdigest())
            image.write_bytes(b'before')
            save(root / 'api/translate/case1' / key / 'result.json', {'parsed': {}, 'request_sha256': 'different'})
            with self.assertRaisesRegex(ValueError, 'cached_request_binding_mismatch'):
                api.call('translate', 'case1', 'prompt', {}, image=image)

    def test_api_wrong_image_blocked_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (folder, c, i, b, s, q) = self.fixture(root)
            other = root / 'other.png'
            other.write_bytes(b'wrong')
            api = API(root, {'OPENAI_MODEL': 'fixture', 'OPENAI_API_KEY': 'fake'})
            with patch('scripts.final_benchmark.api.create') as create:
                with self.assertRaisesRegex(ValueError, 'api_source_image_mismatch'):
                    api.call('translate', 'case1', 'prompt', {'source_binding': b}, image=other)
                create.assert_not_called()

    def test_worker_records_the_rendered_spec_and_labels(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {'kind': 'table', 'data': {'rows': [[{'text': 'Name', 'label_key': 'heading', 'rowspan': 1, 'colspan': 1}], [{'text': '10', 'label_key': None, 'rowspan': 1, 'colspan': 1}]]}}
            labels = {'heading': 'Name'}
            save(root / 'spec.json', spec)
            save(root / 'en.json', {'labels': labels})
            worker = Path(__file__).resolve().parents[1] / 'scripts/final_benchmark/worker.py'
            env = dict(os.environ, MPLCONFIGDIR=str(root / 'mpl'), PYTHONPYCACHEPREFIX=str(root / 'pycache'))
            subprocess.run([sys.executable, str(worker), str(root / 'spec.json'), str(root / 'en.json'), str(root / 'en.png')], env=env, check=True, capture_output=True, timeout=60)
            self.assertTrue(render_is_current(root / 'en.png', spec, labels))
            self.assertFalse(render_is_current(root / 'en.png', spec, {'heading': 'Other label'}))

    def test_mixed_table_executes_adapter_instead_of_discarding_diagram(self):
        import os
        import subprocess
        import sys
        from PIL import Image
        from scripts.final_benchmark.val_audit import table_features
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {'kind': 'table', 'render_mode': 'custom', 'data': {'color': '#ffa500', 'components': ['table', 'diagram']}, 'python_code': "def render(data, labels):\n    fig=plt.figure(figsize=(2,2),dpi=100)\n    ax=fig.add_axes([0,0,1,1])\n    ax.set_axis_off()\n    ax.add_patch(Circle((0.5,0.5),0.2,facecolor=data['color']))\n    return finish(fig,labels,[])\n"}
            save(root / 'spec.json', spec)
            save(root / 'en.json', {'labels': {}})
            worker = Path(__file__).resolve().parents[1] / 'scripts/final_benchmark/worker.py'
            subprocess.run([sys.executable, str(worker), str(root / 'spec.json'), str(root / 'en.json'), str(root / 'en.png')], env=dict(os.environ, MPLCONFIGDIR=str(root / 'mpl')), check=True, capture_output=True, timeout=60)
            self.assertEqual(Image.open(root / 'en.png').getpixel((100, 100)), (255, 165, 0))
            self.assertEqual(table_features(spec)['type'], 'mixed_visual')
            self.assertTrue(render_is_current(root / 'en.png', spec, {}))
