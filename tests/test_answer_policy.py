import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
from scripts.evaluation.answer_policy import apply_policy, extract_answer, next_mode
from scripts.evaluation.run import Runner
from scripts.evaluation.run_token_router import TokenRouterRunner

class AnswerPolicyTests(unittest.TestCase):

    def fixture(self, root, provider):
        (root / 'validation_release').mkdir()
        (root / 'image.png').write_bytes(b'fixture')
        cls = TokenRouterRunner if provider == 'gemini' else Runner
        r = cls.__new__(cls)
        r.out = root / 'run'
        r.out.mkdir()
        r.source = root / 'validation_release/val.candidates.jsonl'
        r.args = SimpleNamespace(dataset=root, model='fixture', temperature=0.7, max_output_tokens=8192, request_interval=0, timeout=1)
        r.prompt = 'json only'
        r.identity = {'prompt_sha256': 'p', 'model': 'fixture'}
        r.run_key = 'r'
        r.config = {'TOKEN_ROUTER_API_KEY': 'fixture', 'OPENAI_API_KEY': 'fixture', 'OPENAI_MODEL': 'fixture'}
        r.url = 'CONFIGURE_LOCALLY'
        r.results = {}
        r.attempts = []
        r.abort = threading.Event()
        r.abort_reason = None
        r.mutex = threading.Lock()
        r.rate = threading.Lock()
        r.last_start = 0
        (r.out / 'manifest.json').write_text(json.dumps({'sessions': [{}]}))
        row = {'id': 'one', 'query': 'Q', 'image_path': '../image.png', 'image_sha256': hashlib.sha256(b'fixture').hexdigest()}
        r.rows = [row]
        return (r, row)

    def response(self, text, provider):
        if provider == 'gemini':
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': text}]}}], 'usageMetadata': {'promptTokenCount': 10, 'totalTokenCount': 12}}, request=httpx.Request('POST', 'CONFIGURE_LOCALLY'))
        return SimpleNamespace(id='resp', model='fixture', status='completed', raw=text, model_dump=lambda **kw: {'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}})

    def test_both_providers_fallback_once_and_keep_usage(self):
        for provider in ('gemini', 'gpt'):
            for fallback in ('最高为750。', '', 'connection_error'):
                with self.subTest(provider=provider, fallback=fallback), tempfile.TemporaryDirectory() as tmp:
                    (r, row) = self.fixture(Path(tmp), provider)
                    module = 'scripts.evaluation.run_token_router' if provider == 'gemini' else 'scripts.evaluation.run'
                    target = module + ('.httpx.Client' if provider == 'gemini' else '.OpenAI')
                    second = httpx.ConnectError('offline') if fallback == 'connection_error' else self.response(fallback, provider)
                    with patch(target) as client, patch('scripts.evaluation.run.response_text', side_effect=lambda res: res.raw):
                        connection = client.return_value.__enter__.return_value
                        call = connection.post if provider == 'gemini' else connection.responses.create
                        call.side_effect = [self.response('not valid json', provider), second]
                        r.evaluate(row)
                        self.assertEqual(call.call_count, 2)
                        last = call.call_args.kwargs
                        if provider == 'gemini':
                            self.assertEqual(last['json']['generationConfig']['responseMimeType'], 'text/plain')
                            self.assertNotIn('responseJsonSchema', last['json']['generationConfig'])
                            self.assertEqual(last['json']['contents'][0]['parts'][0]['text'], 'Q')
                        else:
                            self.assertEqual(last['text']['format']['type'], 'text')
                            self.assertEqual(last['input'][0]['content'][0]['text'], 'Q')
                    self.assertEqual(len(r.attempts), 2)
                    self.assertEqual(r.attempts[1]['answer_mode'], 'plain')
                    if fallback == '最高为750。':
                        self.assertEqual(r.results['one']['prediction'], fallback)
                        self.assertNotIn('error', r.results['one'])
                        self.assertEqual(json.loads((r.out / 'progress.json').read_text())['total_tokens'], 24)
                    else:
                        self.assertEqual(r.results['one']['status'], 'failed')
                        self.assertFalse(r.attempts[1]['retryable'])
                    r.results = {}
                    with patch(target) as client:
                        r.evaluate(row)
                        client.assert_not_called()

    def test_saved_fences_recovered_and_malformed_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            (r, row) = self.fixture(Path(tmp), 'gemini')
            (r.out / 'predictions').mkdir()
            for (rid, raw) in [('one', '```json\n{"answer":"1"}\n```'), ('two', 'malformed')]:
                result = {'id': rid, 'status': 'failed', 'error_type': 'JSONDecodeError', 'run_key': 'r'}
                r.results[rid] = result
                (r.out / 'predictions' / (rid + '.json')).write_text(json.dumps(result))
                r.attempts.append({'id': rid, 'attempt_number': 1, 'error_type': 'JSONDecodeError', 'raw_output': raw})
            apply_policy(r)
            self.assertEqual(r.results['one']['prediction'], '1')
            self.assertNotIn('two', r.results)
            self.assertTrue((r.out / 'prediction_history/two/json_then_one_plain_answer_v1.json').exists())
            self.assertEqual(next_mode(r.attempts), 'plain')
            self.assertIsNone(next_mode([{'answer_mode': 'plain', 'status': 'started'}]))
            self.assertEqual(extract_answer('12', plain=True), '12')
