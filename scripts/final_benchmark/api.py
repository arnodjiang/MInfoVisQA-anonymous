pass
import base64
import hashlib
import json
import re
import threading
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from openai import APIStatusError, APIConnectionError
import httpx
from scripts.responses_client import create, response_text

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%s.%s.tmp' % (os.getpid(), threading.get_ident()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

class RetryableTransportError(RuntimeError):
    pass

class API:

    def __init__(self, root, config, retry_failed=False):
        self.root = Path(root)
        self.config = config
        self.retry_failed = retry_failed
        self.lock = threading.Lock()
        self.last_start = 0.0
        self.connection_failures = 0
        self.circuit_open = False

    def call(self, stage, case_id, prompt, payload, image=None, max_tokens=18000):
        from scripts.translation import backend, translation_stage, GoogleTranslator
        if translation_stage(stage) and backend(self.config) == 'google':
            return GoogleTranslator(self.root, self.config).call(stage, case_id, prompt, payload, image, max_tokens)
        for retry_index in range(11):
            try:
                return self._call_once(stage, case_id, prompt, payload, image, max_tokens, retry_index)
            except RetryableTransportError:
                if retry_index == 10:
                    raise RuntimeError('transport_retries_exhausted:' + stage + ':' + case_id) from None
                time.sleep(5)

    def _call_once(self, stage, case_id, prompt, payload, image=None, max_tokens=18000, retry_index=0):
        effective_max_tokens = int(max_tokens) * 3
        image_paths = [Path(p) for p in image] if isinstance(image, (list, tuple)) else [Path(image)] if image else []
        image_bytes = [p.read_bytes() for p in image_paths]
        binding = payload.get('source_binding') if isinstance(payload, dict) else None
        if binding:
            if binding['case_id'] != case_id:
                raise ValueError('api_source_case_mismatch')
            if image_bytes and (len(image_bytes) != 1 or hashlib.sha256(image_bytes[0]).hexdigest() != binding['original_image_sha256']):
                raise ValueError('api_source_image_mismatch')
        request = {'stage': stage, 'case_id': case_id, 'system': prompt, 'payload': payload, 'model': self.config['OPENAI_MODEL'], 'max_output_tokens': effective_max_tokens, 'endpoint_sha256': digest(self.config.get('OPENAI_BASE_URL')), 'image_path': str(image_paths[0]) if len(image_paths) == 1 else None, 'image_sha256': hashlib.sha256(image_bytes[0]).hexdigest() if len(image_paths) == 1 else None}
        if len(image_paths) > 1:
            request['images'] = [{'path': str(p), 'sha256': hashlib.sha256(blob).hexdigest()} for (p, blob) in zip(image_paths, image_bytes)]
        key = digest(request)
        folder = self.root / 'api' / stage / case_id / key
        complete = folder / 'result.json'
        if complete.exists():
            cached = read(complete)
            if cached.get('request_sha256') != key or read(folder / 'request.json') != request:
                raise ValueError('cached_request_binding_mismatch:' + key)
            return (cached['parsed'], key)
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / 'request.json', request)
        attempts = sorted(folder.glob('attempt_*.json'))
        if attempts and (not self.retry_failed) and (retry_index == 0):
            old = read(attempts[-1])
            if old.get('finish_reason') == 'stop' and old.get('raw_output'):
                try:
                    parsed = self.parse(old['raw_output'])
                    save(complete, {'parsed': parsed, 'request_sha256': key})
                    return (parsed, key)
                except (ValueError, TypeError):
                    pass
            raise RuntimeError('previous_attempt_requires_explicit_retry:' + key)
        path = folder / ('attempt_%03d.json' % (len(attempts) + 1))
        meta = {'status': 'started', 'started_at': now(), 'request_sha256': key, 'automatic_transport_retries': retry_index, 'max_transport_retries': 10, 'transport_retry_delay_seconds': 5}
        save(path, meta)
        with self.lock:
            gap = 3.0 - (time.monotonic() - self.last_start)
            if gap > 0:
                time.sleep(gap)
            self.last_start = time.monotonic()
        image_uris = []
        for (image_path, blob) in zip(image_paths, image_bytes):
            mime = 'image/png' if image_path.suffix.lower() == '.png' else 'image/jpeg'
            uri = 'data:' + mime + ';base64,' + base64.b64encode(blob).decode()
            image_uris.append(uri)
        begin = time.monotonic()
        try:
            response = create(self.config, prompt, payload, image_uris=image_uris, max_tokens=effective_max_tokens, timeout=900)
            meta.update(response_id=response.id, returned_model=response.model, transport='responses', status=response.status)
            if response.usage:
                meta['usage'] = response.usage.model_dump()
            raw = response_text(response).replace(self.config['OPENAI_API_KEY'], '[REDACTED]')
            meta['raw_output'] = raw
            if response.status != 'completed':
                raise ValueError('incomplete_api_response')
            parsed = self.parse(raw)
            meta['status'] = 'completed'
            with self.lock:
                self.connection_failures = 0
            save(complete, {'parsed': parsed, 'request_sha256': key})
            return (parsed, key)
        except Exception as exc:
            meta.update(status='failed', error_type=type(exc).__name__, error_message=str(exc).replace(self.config['OPENAI_API_KEY'], '[REDACTED]')[:2000])
            if isinstance(exc, APIStatusError):
                meta['http_status'] = exc.status_code
                body = getattr(exc, 'body', None)
                if isinstance(body, dict):
                    error = body.get('error', body)
                    if isinstance(error, dict):
                        meta['provider_error_code'] = error.get('code')
                        meta['provider_error_type'] = error.get('type')
            reason = ':provider_concurrency_limit' if 'concurrency limit' in meta['error_message'].lower() else ''
            if isinstance(exc, (APIConnectionError, httpx.TransportError)) or (isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500)):
                meta['retryable_transport_failure'] = True
                raise RetryableTransportError('api_failed:' + stage + ':' + case_id + ':' + type(exc).__name__ + reason) from None
            raise RuntimeError('api_failed:' + stage + ':' + case_id + ':' + type(exc).__name__ + reason) from None
        finally:
            meta.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - begin, 2))
            save(path, meta)

    @staticmethod
    def parse(raw):
        return json.loads(re.sub('^```(?:json)?\\s*|\\s*```$', '', raw.strip()))
