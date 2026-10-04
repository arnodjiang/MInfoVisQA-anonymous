import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlunsplit
import httpx
from scripts.translation import GoogleTranslator, backend, ensure_backend, load_config
from scripts.final_benchmark.api import API
CONFIG = {'GOOGLE_TRANSLATE_API_KEY': 'test-secret-key', 'GOOGLE_TRANSLATE_ENDPOINT': urlunsplit(('https', 'translate.invalid', '/language/translate/v2', '', ''))}

def translated_response(*args, **kwargs):
    body = kwargs['json']
    return httpx.Response(200, json={'data': {'translations': [{'translatedText': text.replace('Value', 'Valeur').replace('What', 'Quel').replace('Hello', 'Bonjour')} for text in body['q']]}})

class TranslationBackendTests(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = directory.name

    def test_default_and_config_precedence(self):
        self.assertEqual(backend({}), 'google')
        self.assertEqual(backend({'TRANSLATION_BACKEND': 'llm'}), 'llm')
        with self.assertRaises(ValueError):
            backend({'TRANSLATION_BACKEND': 'unknown'})
        with tempfile.TemporaryDirectory() as d:
            Path(d, '.env').write_text('TRANSLATION_BACKEND=llm\nGOOGLE_TRANSLATE_API_KEY=file\n')
            with patch.dict(os.environ, {'TRANSLATION_BACKEND': 'google'}, clear=True):
                self.assertEqual(load_config(d)['TRANSLATION_BACKEND'], 'google')
                self.assertEqual(load_config(d)['GOOGLE_TRANSLATE_API_KEY'], 'file')

    def test_google_dispatch_without_openai_or_image(self):
        with tempfile.TemporaryDirectory() as d, patch('scripts.translation.httpx.post', side_effect=translated_response) as post, patch('scripts.final_benchmark.api.create') as llm:
            api = API(d, CONFIG)
            payload = {'languages': {'fr': 'French'}, 'labels': {'k': 'Value 3.0'}, 'question': 'What is [[k]] at -10.5%?', 'answer_template': '3.0'}
            (result, key) = api.call('translation_fr', 'case', 'optional LLM prompt', payload, image=Path(d) / 'absent.png')
            locale = result['locales']['fr']
            self.assertEqual(locale['labels']['k'], 'Valeur 3.0')
            self.assertEqual(locale['question'], 'Quel is [[k]] at -10.5%?')
            self.assertEqual(locale['answer_template'], '3.0')
            self.assertEqual(locale['translation_backend'], 'google')
            llm.assert_not_called()
            self.assertEqual(post.call_args.kwargs['json']['model'], 'nmt')
            self.assertEqual(post.call_args.kwargs['json']['format'], 'text')
            self.assertEqual(post.call_args.kwargs['params']['key'], CONFIG['GOOGLE_TRANSLATE_API_KEY'])
            calls = post.call_count
            self.assertEqual(api.call('translation_fr', 'case', '', payload), (result, key))
            self.assertEqual(post.call_count, calls)
            for p in Path(d).rglob('*.json'):
                self.assertNotIn(CONFIG['GOOGLE_TRANSLATE_API_KEY'], p.read_text())

    def test_optional_llm_uses_existing_adapter(self):
        with tempfile.TemporaryDirectory() as d, patch.object(API, '_call_once', return_value=({'locales': {}}, 'llm-key')) as llm, patch('scripts.translation.httpx.post') as google:
            self.assertEqual(API(d, {'TRANSLATION_BACKEND': 'llm'}).call('translation_fr', 'case', 'prompt', {}), ({'locales': {}}, 'llm-key'))
            llm.assert_called_once()
            google.assert_not_called()

    def test_context_preserves_ids_and_metadata(self):
        with tempfile.TemporaryDirectory() as d, patch('scripts.translation.httpx.post', side_effect=translated_response):
            paragraphs = [{'id': 'paragraph_001', 'text': 'Value [[N_000001]]', 'source_field': 'pre_text', 'source_index': 1}]
            (result, _) = API(d, CONFIG).call('source_context_translation_fr', 'case', '', {'target_language': 'fr', 'paragraphs': paragraphs})
            self.assertEqual(result['paragraphs'][0], dict(paragraphs[0], text='Valeur [[N_000001]]'))

    def test_identity_numeric_and_blank_text_do_not_call_api(self):
        with patch('scripts.translation.httpx.post') as post:
            t = GoogleTranslator(self.root, {})
            source = {'answer': '-1,200.50%', 'empty': '', 'ref': '[[label]]'}
            self.assertEqual(t.tree(source, 'zh'), source)
            self.assertEqual(t.tree({'label': 'Hello'}, 'en'), {'label': 'Hello'})
            post.assert_not_called()

    def test_changed_marker_or_added_number_rejected(self):
        for output in ['Valeur 900', 'Valeur ZXQKEEP000000QXZ 900', '']:
            with patch('scripts.translation.httpx.post', return_value=httpx.Response(200, json={'data': {'translations': [{'translatedText': output}]}})):
                with self.assertRaises(ValueError):
                    GoogleTranslator(self.root, CONFIG).tree('Value 10', 'fr')

    def test_batch_limits_and_long_text(self):
        with patch('scripts.translation.httpx.post', side_effect=translated_response) as post:
            source = {'k' + str(i): 'Hello ' + chr(65 + i % 26) * (i + 1) for i in range(140)}
            source['long'] = 'Hello ' * 1600
            result = GoogleTranslator(self.root, CONFIG).tree(source, 'fr')
            self.assertEqual(set(result), set(source))
            self.assertEqual(result['long'], 'Bonjour ' * 1600)
            for call in post.call_args_list:
                q = call.kwargs['json']['q']
                self.assertLessEqual(len(q), 128)
                self.assertLessEqual(sum(map(len, q)), 4500)

    def test_transport_retry_and_auth_error_redaction(self):
        with patch('scripts.translation.time.sleep'), patch('scripts.translation.httpx.post', side_effect=[httpx.ConnectError('secret'), httpx.Response(429), translated_response(json={'q': ['Hello']})]) as post:
            self.assertEqual(GoogleTranslator(self.root, CONFIG).tree('Hello', 'fr'), 'Bonjour')
            self.assertEqual(post.call_count, 3)
        with patch('scripts.translation.httpx.post', return_value=httpx.Response(403, text='test-secret-key')) as post:
            with self.assertRaisesRegex(RuntimeError, '^Google translation HTTP 403$'):
                GoogleTranslator(self.root, CONFIG).tree('Hello', 'fr')
            self.assertEqual(post.call_count, 1)

    def test_cache_provider_required(self):
        for record in [{}, {'translation_backend': 'llm'}]:
            with self.assertRaises(ValueError):
                ensure_backend(record, 'google')
        ensure_backend({'translation_backend': 'google'}, 'google')

    def test_empty_or_invalid_response_fails_closed(self):
        for data in [{}, {'data': {'translations': []}}, {'data': {'translations': [{'translatedText': 5}]}}]:
            with patch('scripts.translation.httpx.post', return_value=httpx.Response(200, json=data)):
                with self.assertRaises(ValueError):
                    GoogleTranslator(self.root, CONFIG).tree('Hello', 'zh')
if __name__ == '__main__':
    unittest.main()
