pass
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import re
import time
import uuid
from urllib.parse import urlsplit
import httpx
from dotenv import dotenv_values
VERSION = 'google-nmt-protected-text-v1'
PROTECTED = re.compile('\\[\\[[^\\]\\n]+\\]\\]|[+−-]?\\d+(?:[.,]\\d+)*(?:[eE][+−-]?\\d+)?%?|(?m:^[ \\t]*:?-{3,}:?[ \\t]*$)')
MARKER = re.compile('ZXQKEEP\\d{6}QXZ')

def load_config(root):
    config = dict(dotenv_values(Path(root) / '.env', interpolate=False))
    for key in ('TRANSLATION_BACKEND', 'GOOGLE_TRANSLATE_API_KEY', 'GOOGLE_TRANSLATE_ENDPOINT', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'):
        if key in os.environ:
            config[key] = os.environ[key]
    backend(config)
    return config

def backend(config):
    value = (config.get('TRANSLATION_BACKEND') or 'google').strip().lower()
    if value not in ('google', 'llm'):
        raise ValueError('TRANSLATION_BACKEND must be google or llm')
    return value

def translation_stage(stage):
    return stage.startswith(('translation_', 'source_context_translation_', 'mixed_visual_translation_repair_', 'remaining_chunk_'))

def ensure_backend(record, selected):
    actual = record.get('translation_backend')
    if actual is None:
        raise ValueError('Translation cache has no provider identity; use a new output revision')
    if actual != selected:
        raise ValueError('Translation backend changed; use a new output revision')

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

class GoogleTranslator:

    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config

    def _request(self, texts, target):
        endpoint = self.config.get('GOOGLE_TRANSLATE_ENDPOINT') or ''
        key = self.config.get('GOOGLE_TRANSLATE_API_KEY') or ''
        url = urlsplit(endpoint)
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment or (not url.path.endswith('/language/translate/v2')):
            raise ValueError('Configure GOOGLE_TRANSLATE_ENDPOINT with the Google Cloud Translation Basic v2 translate endpoint')
        if not key:
            raise ValueError('Configure GOOGLE_TRANSLATE_API_KEY')
        body = {'q': texts, 'source': 'en', 'target': 'zh-CN' if target == 'zh' else target, 'format': 'text', 'model': 'nmt'}
        from scripts.final_benchmark.api import save
        request_key = fingerprint({'body': body, 'endpoint_sha256': fingerprint(endpoint), 'policy': VERSION})
        log = self.root / 'google_requests' / request_key
        save(log / 'request.json', {'body': body, 'endpoint_sha256': fingerprint(endpoint), 'policy': VERSION})
        for attempt in range(4):
            attempt_path = log / ('attempt_' + uuid.uuid4().hex + '.json')
            meta = {'translation_backend': 'google', 'model': 'nmt', 'retry_index': attempt, 'source_characters': sum(map(len, texts)), 'status': 'started'}
            save(attempt_path, meta)
            try:
                response = httpx.post(endpoint, params={'key': key}, json=body, timeout=60)
            except httpx.TransportError:
                save(attempt_path, dict(meta, status='transport_error'))
                if attempt == 3:
                    raise RuntimeError('Google translation transport retries exhausted') from None
                time.sleep(2 ** attempt)
                continue
            save(attempt_path, dict(meta, status='http_response', http_status=response.status_code))
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 3:
                    raise RuntimeError('Google translation transport retries exhausted')
                time.sleep(2 ** attempt)
                continue
            if response.status_code != 200:
                raise RuntimeError('Google translation HTTP ' + str(response.status_code))
            try:
                result = response.json()['data']['translations']
                if len(result) != len(texts) or not all((isinstance(x.get('translatedText'), str) for x in result)):
                    raise ValueError()
                save(attempt_path, dict(meta, status='completed', http_status=200))
                return [html.unescape(x['translatedText']) for x in result]
            except (KeyError, TypeError, ValueError):
                save(attempt_path, dict(meta, status='invalid_response', http_status=200))
                raise ValueError('Malformed Google translation response') from None
        raise RuntimeError('Google translation failed')

    def tree(self, value, target):
        if target == 'en':
            return copy.deepcopy(value)
        leaves = []

        def collect(v):
            if isinstance(v, str):
                leaves.append(v)
            elif isinstance(v, dict):
                for item in v.values():
                    collect(item)
            elif isinstance(v, list):
                for item in v:
                    collect(item)
            else:
                raise ValueError('Translation text tree must contain only strings, lists and dictionaries')
        collect(value)
        unique = list(dict.fromkeys(leaves))
        translated = {}
        pending = []
        mappings = {}
        parts = {}
        for original in unique:
            if not original.strip() or not any((c.isalpha() for c in PROTECTED.sub('', original))):
                translated[original] = original
                continue
            if MARKER.search(original):
                raise ValueError('Source text collides with reserved translation markers')
            tokens = []

            def mask(match):
                marker = 'ZXQKEEP%06dQXZ' % len(tokens)
                tokens.append(match.group())
                return marker
            masked = PROTECTED.sub(mask, original)
            mappings[original] = tokens
            segments = []
            for piece in re.split('(\\r?\\n|\\|+)', masked):
                while len(piece) > 4000:
                    boundary = piece.rfind(' ', 0, 4000)
                    if boundary <= 0:
                        raise ValueError('Translation text has an unsplittable segment exceeding 4000 characters')
                    segments.extend([piece[:boundary], piece[boundary:boundary + 1]])
                    piece = piece[boundary + 1:]
                segments.append(piece)
            parts[original] = segments
            pending.extend((s for s in segments if s.strip() and (not re.fullmatch('\\|+', s))))
        translated_parts = {}
        batch = []
        size = 0

        def flush():
            if not batch:
                return
            outputs = self._request(batch, target)
            for (source, result) in zip(batch, outputs):
                if not result.strip() or sorted(MARKER.findall(source)) != sorted(MARKER.findall(result)):
                    raise ValueError('Google translation changed protected markers or returned empty text')
                if re.search('\\d', MARKER.sub('', result)):
                    raise ValueError('Google translation introduced an unprotected number')
                translated_parts[source] = result
        for segment in dict.fromkeys(pending):
            if batch and (len(batch) == 128 or size + len(segment) > 4500):
                flush()
                batch = []
                size = 0
            batch.append(segment)
            size += len(segment)
        flush()
        for (original, segments) in parts.items():
            result = ''.join((translated_parts.get(s, s) for s in segments))
            for (i, token) in enumerate(mappings[original]):
                result = result.replace('ZXQKEEP%06dQXZ' % i, token)
            if sorted(PROTECTED.findall(original)) != sorted(PROTECTED.findall(result)):
                raise ValueError('Google translation changed protected values')
            translated[original] = result

        def rebuild(v):
            if isinstance(v, str):
                return translated[v]
            if isinstance(v, dict):
                return {k: rebuild(x) for (k, x) in v.items()}
            return [rebuild(x) for x in v]
        return rebuild(value)

    def call(self, stage, case_id, prompt, payload, image=None, max_tokens=None):
        from scripts.final_benchmark.api import save, read
        request = {'stage': stage, 'case_id': case_id, 'system': VERSION, 'payload': payload, 'translation_backend': 'google', 'model': 'nmt', 'source_language': 'en', 'endpoint_sha256': fingerprint(self.config.get('GOOGLE_TRANSLATE_ENDPOINT')), 'image_path': None, 'image_sha256': None}
        key = fingerprint(request)
        folder = self.root / 'api' / stage / case_id / key
        if (folder / 'result.json').exists():
            saved = read(folder / 'result.json')
            if read(folder / 'request.json') != request or saved['request_sha256'] != key:
                raise ValueError('Translation cache binding changed')
            return (saved['parsed'], key)
        save(folder / 'request.json', request)
        if 'languages' in payload:
            source = {k: payload[k] for k in ('labels', 'question', 'answer_template')}
            parsed = {'locales': {lang: dict(self.tree(source, lang), notes=[], translation_backend='google', translation_model='nmt') for lang in payload['languages']}}
        elif 'paragraphs' in payload:
            texts = self.tree([p['text'] for p in payload['paragraphs']], payload['target_language'])
            parsed = {'paragraphs': [dict(p, text=t) for (p, t) in zip(payload['paragraphs'], texts)]}
        elif 'source_labels' in payload:
            parsed = self.tree(payload['source_labels'], payload['target_language'])
        elif 'text_tree' in payload:
            parsed = self.tree(payload['text_tree'], payload['target_language'])
        else:
            parsed = dict(self.tree({k: payload[k] for k in ('question', 'answer_template')}, payload['language']), notes=[])
        save(folder / 'result.json', {'parsed': parsed, 'request_sha256': key, 'translation_backend': 'google', 'translation_model': 'nmt'})
        return (parsed, key)

def google_tree(root, config, case_id, source, language):
    return GoogleTranslator(root, config).call('translation_' + language, case_id, '', {'target_language': language, 'text_tree': source})

def main():
    import argparse
    from scripts.final_benchmark.api import API, save
    parser = argparse.ArgumentParser(description='Translate a labels/QA JSON text tree with Google Translate API by default')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--translation-backend', choices=['google', 'llm'])
    args = parser.parse_args()
    config = load_config(Path(__file__).resolve().parents[1])
    if args.translation_backend:
        config['TRANSLATION_BACKEND'] = args.translation_backend
    source = json.loads(args.input.read_text())
    selected = backend(config)
    if args.output.exists():
        ensure_backend(json.loads(args.output.read_text()), selected)
    prompt = 'Translate every text value of text_tree from English to target_language. Preserve dictionary keys, list order, numerical tokens and [[label]] references exactly. Return the translated text_tree alone as JSON. Do not solve or recompute QA.'
    (parsed, key) = API(args.output.parent / 'translation_cache', config).call('translation_' + args.language, 'text_tree', prompt, {'text_tree': source, 'target_language': args.language})
    save(args.output, {'translation': parsed, 'translation_backend': selected, 'translation_model': 'nmt' if selected == 'google' else config['OPENAI_MODEL'], 'request_sha256': key})
if __name__ == '__main__':
    main()
