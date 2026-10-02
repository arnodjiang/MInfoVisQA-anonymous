pass
import argparse
import base64
import fcntl
import hashlib
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx
from openai import OpenAI, APIConnectionError, APIStatusError
from scripts.openai_config import load
from scripts.responses_client import response_text
from scripts.final_benchmark.api import save, read, digest, now
from scripts.evaluation.score import score_run, METRIC
from scripts.evaluation.answer_policy import AnswerFormatError, extract_answer, next_mode, apply_policy, PLAIN_PROMPT, VERSION
from scripts.evaluation.context_input import context_prompt, input_text, protocol, request_context
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer'], 'additionalProperties': False}

def extract(raw):
    return extract_answer(raw)

def usage_summary(attempts):
    fields = {'input_tokens': ('input_tokens',), 'output_tokens': ('output_tokens',), 'total_tokens': ('total_tokens',), 'cached_input_tokens': ('input_tokens_details', 'cached_tokens'), 'reasoning_output_tokens': ('output_tokens_details', 'reasoning_tokens')}
    index = {(a.get('id'), a.get('attempt_number')): a for a in attempts}
    retries = [a for a in attempts if a.get('attempt_number', 1) > 1 and (a.get('id'), a['attempt_number'] - 1) in index]
    automatic = sum((bool(index.get((a.get('id'), a['attempt_number'] - 1), {}).get('retryable')) for a in retries))
    result = {'api_attempts': len(attempts), 'attempts_with_usage': sum((isinstance(a.get('usage'), dict) for a in attempts)), 'successful_http_responses': sum((bool(a.get('response_id')) or (isinstance(a.get('http_status'), int) and 200 <= a['http_status'] < 300) for a in attempts)), 'retry_attempts': len(retries), 'transport_retries': automatic, 'non_transport_retry_attempts': len(retries) - automatic}
    result['plain_answer_fallback_calls'] = sum((a.get('answer_mode') == 'plain' for a in attempts))
    for (name, path) in fields.items():
        values = []
        for a in attempts:
            value = a.get('usage')
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, int):
                values.append(value)
        result[name] = sum(values) if values else None
        result[name + '_reported_attempts'] = len(values)
    result['logical_qa_attempted'] = len({a.get('id') for a in attempts if a.get('id')})
    result['sum_attempt_elapsed_seconds'] = round(sum((a.get('elapsed_seconds', 0) for a in attempts)), 3)
    result['attempts_without_usage'] = len(attempts) - result['attempts_with_usage']
    result['usage_is_complete'] = result['attempts_without_usage'] == 0
    result['cost_currency'] = None
    result['cost_note'] = 'Token totals are provider-reported; unknown usage is not zero. Monetary cost needs provider pricing/billing.'
    return result

class Runner:

    def __init__(self, args):
        from scripts.evaluation.release import check_dataset
        check_dataset(args.dataset)
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.out / 'run.lock').open('a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.config = load(ROOT)
        if getattr(args, 'model', None):
            self.config['OPENAI_MODEL'] = args.model
        self.prompt = Path(__file__).with_name('answer_prompt.txt').read_text()
        source = args.dataset.resolve() / 'validation_release/val.candidates.jsonl'
        data = source.read_bytes()
        self.rows = [json.loads(s) for s in data.decode().splitlines()]
        if len({r['id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Duplicate dataset IDs')
        self.identity = {'model': self.config['OPENAI_MODEL'], 'endpoint_sha256': digest(self.config.get('OPENAI_BASE_URL')), 'references_sha256': hashlib.sha256(data).hexdigest(), 'prompt_sha256': digest(self.prompt), 'schema': SCHEMA, 'max_output_tokens': args.max_output_tokens, 'image_detail': 'high', 'metric': METRIC, 'repetitions': 1, 'temperature': 'provider default', 'reasoning_effort': 'provider default'}
        self.identity.update(protocol(self.rows, self.prompt, PLAIN_PROMPT))
        self.run_key = digest(self.identity)
        mp = self.out / 'manifest.json'
        if mp.exists():
            manifest = read(mp)
            if manifest['run_key'] != self.run_key:
                raise ValueError('Run inputs changed; select a new output directory')
        else:
            manifest = dict(self.identity, run_key=self.run_key, created_at=now(), dataset=str(args.dataset.resolve()), n=len(self.rows), sessions=[], format='strict Responses JSON schema', tools=[], reference_answers_sent=False)
            (self.out / 'references.jsonl').write_bytes(data)
            (self.out / 'prompt.txt').write_text(self.prompt)
        manifest['sessions'].append({'started_at': now(), 'pid': os.getpid(), 'smoke': args.smoke, 'workers': args.workers, 'llm_judge': getattr(args, 'llm_judge', False), 'judge_model': getattr(args, 'judge_model', None), 'publish_table': str(args.publish_table) if getattr(args, 'publish_table', None) else None})
        save(mp, manifest)
        template = ROOT / 'paper/tables/main_results.tex'
        if template.exists() and (not (self.out / 'table_template.tex').exists()):
            (self.out / 'table_template.tex').write_bytes(template.read_bytes())
        self.assets = {}
        for row in self.rows:
            path = (source.parent / row['image_path']).resolve()
            if args.dataset.resolve() not in path.parents:
                raise ValueError('Unsafe image path')
            if path not in self.assets:
                self.assets[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.assets[path] != row['image_sha256']:
                raise ValueError('Image changed: ' + row['id'])
        self.results = {p.stem: read(p) for p in (self.out / 'predictions').glob('*.json')}
        self.attempts = [read(p) for p in (self.out / 'attempts').glob('*/*.json')]
        apply_policy(self)
        self.mutex = threading.Lock()
        self.rate = threading.Lock()
        self.last_start = 0.0
        self.abort = threading.Event()
        self.progress('ready')

    def progress(self, state):
        save(self.out / 'progress.json', {'state': state, 'updated_at': now(), 'expected': len(self.rows), 'terminal': len(self.results), 'completed': sum((r['status'] == 'completed' for r in self.results.values())), 'failed': sum((r['status'] == 'failed' for r in self.results.values())), 'repetitions': 1, 'run_sessions': len(read(self.out / 'manifest.json')['sessions']), **usage_summary(self.attempts)})

    def evaluate(self, row):
        rid = row['id']
        if rid in self.results or self.abort.is_set():
            return
        image = (self.args.dataset.resolve() / 'validation_release' / row['image_path']).resolve()
        question_text = input_text(row)
        request_meta = {'id': rid, 'run_key': self.run_key, 'model': self.identity['model'], 'question': row['query'], 'image_sha256': row['image_sha256'], 'image_path': row['image_path'], 'schema': SCHEMA, 'max_output_tokens': self.args.max_output_tokens, 'prompt_sha256': digest(context_prompt(self.prompt, row)), **request_context(row)}
        save(self.out / 'requests' / (rid + '.json'), request_meta)
        image_bytes = image.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row['image_sha256']:
            raise ValueError('Image changed during inference')
        uri = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode()
        existing = sorted((self.out / 'attempts' / rid).glob('*.json'))
        result = {'id': rid, 'status': 'failed', 'prediction': None, 'run_key': self.run_key}
        prior = [read(p) for p in existing]
        mode = next_mode(prior)
        start_number = max((a['attempt_number'] for a in prior), default=0) + 1
        for number in range(start_number, max(start_number, 12) + 1):
            if mode is None or (mode == 'json' and number > 11):
                break
            if self.abort.is_set():
                return
            with self.rate:
                delay = self.args.request_interval - (time.monotonic() - self.last_start)
                if delay > 0 and self.abort.wait(delay):
                    return
                if self.abort.is_set():
                    return
                self.last_start = time.monotonic()
            path = self.out / 'attempts' / rid / ('%03d.json' % number)
            attempt = {'id': rid, 'attempt_number': number, 'status': 'started', 'started_at': now(), 'answer_mode': mode, 'answer_policy': VERSION}
            effective_prompt = context_prompt(PLAIN_PROMPT if mode == 'plain' else self.prompt, row)
            attempt['prompt_sha256'] = digest(effective_prompt)
            save(path, attempt)
            start = time.monotonic()
            retry = False
            format_retry = False
            try:
                with OpenAI(api_key=self.config['OPENAI_API_KEY'], base_url=self.config.get('OPENAI_BASE_URL') or None, timeout=self.args.timeout, max_retries=0) as client:
                    response = client.responses.create(model=self.config['OPENAI_MODEL'], instructions=effective_prompt, input=[{'role': 'user', 'content': [{'type': 'input_text', 'text': question_text}, {'type': 'input_image', 'image_url': uri, 'detail': 'high'}]}], text={'format': {'type': 'text'}} if mode == 'plain' else {'format': {'type': 'json_schema', 'name': 'benchmark_answer', 'strict': True, 'schema': SCHEMA}}, max_output_tokens=self.args.max_output_tokens)
                payload = response.model_dump(mode='json')
                payload = json.loads(json.dumps(payload, ensure_ascii=False).replace(self.config['OPENAI_API_KEY'], '[REDACTED]'))
                save(self.out / 'responses' / rid / ('%03d.json' % number), payload)
                raw = response_text(response)
                attempt.update(response_id=response.id, returned_model=response.model, response_status=response.status, usage=payload.get('usage'), raw_output=raw)
                if response.status != 'completed':
                    raise ValueError('Incomplete model response')
                prediction = extract_answer(raw, plain=mode == 'plain')
                attempt['status'] = 'completed'
                result.update(status='completed', prediction=prediction, response_id=response.id, returned_model=response.model, usage=payload.get('usage'), selected_attempt=number, answer_mode=mode, answer_policy=VERSION)
                result.pop('error', None)
                result.pop('error_type', None)
            except Exception as exc:
                attempt.update(status='failed', error_type=type(exc).__name__, error=str(exc).replace(self.config['OPENAI_API_KEY'], '[REDACTED]')[:1800])
                if isinstance(exc, APIStatusError):
                    attempt['http_status'] = exc.status_code
                retry = isinstance(exc, (APIConnectionError, httpx.TransportError)) or (isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500))
                if mode == 'plain':
                    retry = False
                format_retry = isinstance(exc, AnswerFormatError) and mode == 'json'
                attempt['retryable'] = retry
                if isinstance(exc, APIStatusError) and exc.status_code in (401, 403):
                    self.abort.set()
                result.update(error_type=type(exc).__name__, error=attempt['error'])
            finally:
                attempt.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - start, 3))
                save(path, attempt)
                with self.mutex:
                    self.attempts.append(attempt)
                    self.progress('running')
            if format_retry:
                mode = 'plain'
                continue
            if result['status'] == 'completed' or not retry:
                break
            if number < 11 and self.abort.wait(5):
                return
        result.update(finished_at=now(), attempts=len(list((self.out / 'attempts' / rid).glob('*.json'))))
        save(self.out / 'predictions' / (rid + '.json'), result)
        with self.mutex:
            self.results[rid] = result
            self.progress('running')
            if len(self.results) % 10 == 0:
                print('Terminal QA:', len(self.results), '/', len(self.rows), flush=True)

    def run(self):
        pending = [r for r in self.rows if r['id'] not in self.results]
        if self.args.smoke:
            pending = pending[:1]
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = [pool.submit(self.evaluate, r) for r in pending]
            for future in as_completed(futures):
                future.result()
        from scripts.evaluation.lineage import snapshot_run
        snapshot_run(self.out)
        if len(self.results) == len(self.rows):
            self.progress('scoring')
            score_run(self.out)
            if getattr(self.args, 'llm_judge', False):
                from scripts.evaluation.judge import run_judge
                self.progress('llm_judging')
                judge_model = getattr(self.args, 'judge_model', None)
                judged = run_judge(self.out, model=judge_model) if judge_model else run_judge(self.out)
                if getattr(self.args, 'publish_table', None):
                    from scripts.evaluation.publish_results import publish
                    publish(judged, self.args.publish_table, expected_judge=judge_model)
            self.progress('finished')
            save(self.out / 'usage_summary.json', usage_summary(self.attempts))
            print('Finished inference and scoring:', self.out, flush=True)
        else:
            self.progress('blocked' if self.abort.is_set() else 'preflight_complete' if self.args.smoke else 'incomplete')
        manifest = read(self.out / 'manifest.json')
        manifest['sessions'][-1]['finished_at'] = now()
        manifest['sessions'][-1]['state'] = read(self.out / 'progress.json')['state']
        save(self.out / 'manifest.json', manifest)
        if self.args.smoke:
            print(json.dumps(read(self.out / 'progress.json'), ensure_ascii=False), flush=True)
            if not pending or self.results.get(pending[0]['id'], {}).get('status') == 'completed':
                return
            raise SystemExit('Preflight failed; inspect saved attempt before starting full batch')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--llm-judge', action='store_true', help='Run separate semantic judging after all inference completes')
    p.add_argument('--model', help='Override only the inference model; OPENAI_* credentials are unchanged')
    p.add_argument('--judge-model', help='Pin the independent Judge model (e.g. gpt-6-astra)')
    p.add_argument('--publish-table', type=Path, help='After complete judging, update only this model row in the LaTeX table')
    p.add_argument('--request-interval', type=float, default=3)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--max-output-tokens', type=int, default=8192)
    args = p.parse_args()
    if args.workers < 1 or args.request_interval < 0:
        p.error('Invalid concurrency/rate')
    if args.publish_table and (not args.llm_judge):
        p.error('--publish-table requires --llm-judge')
    runner = Runner(args)
    signal.signal(signal.SIGTERM, lambda *_: runner.abort.set())
    signal.signal(signal.SIGINT, lambda *_: runner.abort.set())
    try:
        runner.run()
    except BaseException:
        runner.progress('interrupted')
        raise
if __name__ == '__main__':
    main()
