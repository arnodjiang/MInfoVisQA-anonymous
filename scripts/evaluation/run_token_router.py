pass
import argparse
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
from scripts.final_benchmark.api import save, read, digest, now
from scripts.evaluation.run import Runner, usage_summary, ROOT
from scripts.evaluation.score import score_run, METRIC, correct
from scripts.evaluation.answer_policy import AnswerFormatError, extract_answer, next_mode, apply_policy, PLAIN_PROMPT, VERSION
from scripts.evaluation.token_router import MODEL, SCHEMA, load_config, endpoint, request_body, answer_text, normalize_usage, quota_exhausted

class QuotaExhaustedError(RuntimeError):
    pass

def resume_answer_mode(attempts, retry_failed=False):
    mode = next_mode(attempts)
    if mode is None and retry_failed:
        last = max(attempts, key=lambda a: a['attempt_number'])
        status = last.get('http_status')
        if last.get('answer_mode') == 'plain' and last.get('status') == 'failed' and (not last.get('raw_output')) and (status == 429 or (isinstance(status, int) and status >= 500) or last.get('quota_exhausted') or (last.get('error_type') in ('ConnectError', 'ConnectTimeout', 'ReadTimeout', 'ReadError'))):
            return 'plain'
    return mode

def smoke_rows(rows):
    selected = []
    for (kind, lang) in [('chart', 'en'), ('table', 'zh')]:
        choices = [r for r in rows if r.get('visual_family', r['visual_kind']) == kind and r['image_language'] == lang and (r['query_language'] == lang)]
        if not choices:
            raise ValueError('Smoke test requires English chart and Chinese table configurations')
        selected.append(choices[0])
    return selected

class TokenRouterRunner(Runner):

    def __init__(self, args):
        from scripts.evaluation.release import check_dataset
        check_dataset(args.dataset)
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.out / 'run.lock').open('a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.config = load_config(ROOT)
        self.url = endpoint(self.config['TOKEN_ROUTER_BASE_URL'], args.model)
        self.prompt = Path(__file__).with_name('answer_prompt.txt').read_text()
        self.source = args.dataset.resolve() / 'validation_release/val.candidates.jsonl'
        data = self.source.read_bytes()
        self.rows = [json.loads(s) for s in data.decode().splitlines()]
        if len({r['id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Duplicate reference IDs')
        self.identity = {'provider': 'token_router', 'transport': 'gemini_generateContent', 'model': args.model, 'endpoint_sha256': digest(self.url), 'references_sha256': hashlib.sha256(data).hexdigest(), 'prompt_sha256': digest(self.prompt), 'schema': SCHEMA, 'max_output_tokens': args.max_output_tokens, 'temperature': args.temperature, 'reasoning_effort': 'provider default', 'image_detail': 'provider default', 'metric': METRIC, 'repetitions': 1}
        from scripts.evaluation.context_input import protocol
        self.identity.update(protocol(self.rows, self.prompt, PLAIN_PROMPT))
        self.run_key = digest(self.identity)
        mp = self.out / 'manifest.json'
        if mp.exists():
            manifest = read(mp)
            if manifest['run_key'] != self.run_key:
                raise ValueError('Run inputs changed; use a new output directory')
        else:
            manifest = dict(self.identity, run_key=self.run_key, created_at=now(), dataset=str(args.dataset.resolve()), n=len(self.rows), sessions=[], format='generateContent JSON schema', tools=[], reference_answers_sent=False)
            (self.out / 'references.jsonl').write_bytes(data)
            (self.out / 'prompt.txt').write_text(self.prompt)
        manifest['sessions'].append({'started_at': now(), 'pid': os.getpid(), 'smoke': args.smoke, 'workers': args.workers, 'request_interval': args.request_interval, 'defer_scoring': args.defer_scoring, 'llm_judge': getattr(args, 'llm_judge', False)})
        save(mp, manifest)
        template = ROOT / 'paper/tables/main_results.tex'
        if template.exists() and (not (self.out / 'table_template.tex').exists()):
            (self.out / 'table_template.tex').write_bytes(template.read_bytes())
        self.results = {p.stem: read(p) for p in (self.out / 'predictions').glob('*.json')}
        self.attempts = [read(p) for p in (self.out / 'attempts').glob('*/*.json')]
        apply_policy(self)
        self.mutex = threading.Lock()
        self.rate = threading.Lock()
        self.last_start = 0.0
        self.abort = threading.Event()
        self.abort_reason = None
        self.progress('ready')

    def progress(self, state):
        reason = getattr(self, 'abort_reason', None)
        if getattr(self, 'abort', None) is not None and self.abort.is_set():
            state = 'quota_exhausted' if reason == 'quota_exhausted' else 'stopping'
        super().progress(state)
        if reason:
            record = read(self.out / 'progress.json')
            record['stop_reason'] = reason
            save(self.out / 'progress.json', record)

    def stop(self, reason):
        with self.mutex:
            if self.abort_reason != 'quota_exhausted':
                self.abort_reason = reason
            self.abort.set()
            save(self.out / 'stop_reason.json', {'reason': self.abort_reason, 'updated_at': now(), 'policy': 'No further submissions or retries; drain already in-flight requests before exit.'})

    def evaluate(self, row):
        from scripts.evaluation.context_input import context_prompt, input_text, request_context
        rid = row['id']
        if rid in self.results or self.abort.is_set():
            return
        image = (self.source.parent / row['image_path']).resolve()
        if self.args.dataset.resolve() not in image.parents:
            raise ValueError('Unsafe image path')
        content = image.read_bytes()
        if hashlib.sha256(content).hexdigest() != row['image_sha256']:
            raise ValueError('Image hash mismatch')
        question_text = input_text(row)
        body = request_body(context_prompt(self.prompt, row), question_text, content, temperature=self.args.temperature, max_tokens=self.args.max_output_tokens)
        save(self.out / 'requests' / (rid + '.json'), {'id': rid, 'run_key': self.run_key, 'model': self.args.model, 'transport': 'gemini_generateContent', 'question': row['query'], 'image_path': row['image_path'], 'image_sha256': row['image_sha256'], 'generationConfig': body['generationConfig'], 'prompt_sha256': digest(context_prompt(self.prompt, row)), **request_context(row)})
        result = {'id': rid, 'status': 'failed', 'prediction': None, 'run_key': self.run_key}
        prior = list((self.out / 'attempts' / rid).glob('*.json'))
        mode = resume_answer_mode([read(p) for p in prior], getattr(self.args, 'retry_failed', False))
        start_number = max((int(p.stem) for p in prior), default=0) + 1
        for number in range(start_number, max(start_number, 12) + 1):
            if mode is None or (mode == 'json' and number > 11):
                break
            if self.abort.is_set():
                return
            if mode == 'plain':
                body = request_body(context_prompt(PLAIN_PROMPT, row), question_text, content, temperature=self.args.temperature, max_tokens=self.args.max_output_tokens)
                body['generationConfig'].pop('responseJsonSchema', None)
                body['generationConfig']['responseMimeType'] = 'text/plain'
            with self.rate:
                wait = self.args.request_interval - (time.monotonic() - self.last_start)
                if wait > 0 and self.abort.wait(wait):
                    return
                if self.abort.is_set():
                    return
                self.last_start = time.monotonic()
            path = self.out / 'attempts' / rid / ('%03d.json' % number)
            attempt = {'id': rid, 'attempt_number': number, 'status': 'started', 'started_at': now(), 'transport': 'gemini_generateContent', 'answer_mode': mode, 'answer_policy': VERSION}
            attempt['prompt_sha256'] = digest(context_prompt(PLAIN_PROMPT if mode == 'plain' else self.prompt, row))
            save(path, attempt)
            start = time.monotonic()
            retry = False
            format_retry = False
            try:
                with httpx.Client(timeout=self.args.timeout, follow_redirects=False) as client:
                    response = client.post(self.url, headers={'Content-Type': 'application/json', 'x-goog-api-key': self.config['TOKEN_ROUTER_API_KEY']}, json=body)
                attempt['http_status'] = response.status_code
                raw = response.text.replace(self.config['TOKEN_ROUTER_API_KEY'], '[REDACTED]')
                try:
                    payload = json.loads(raw)
                except ValueError:
                    payload = {'raw_http_body': raw}
                save(self.out / 'responses' / rid / ('%03d.json' % number), payload)
                error = payload.get('error')
                if isinstance(error, dict):
                    attempt.update(provider_error_code=error.get('code'), provider_error_message=error.get('message'))
                if quota_exhausted(payload):
                    attempt['quota_exhausted'] = True
                    self.stop('quota_exhausted')
                    raise QuotaExhaustedError('Provider reports insufficient account credit/quota')
                response.raise_for_status()
                attempt.update(response_id=payload.get('responseId'), returned_model=payload.get('modelVersion'), usage_metadata=payload.get('usageMetadata'), usage=normalize_usage(payload.get('usageMetadata')))
                attempt['finish_reasons'] = [c.get('finishReason') for c in payload.get('candidates', [])]
                text = answer_text(payload)
                attempt['raw_output'] = text
                prediction = extract_answer(text, plain=mode == 'plain')
                attempt['status'] = 'completed'
                result.update(status='completed', prediction=prediction, response_id=attempt['response_id'], returned_model=attempt['returned_model'], usage=attempt['usage'], usage_metadata=attempt['usage_metadata'], selected_attempt=number, answer_mode=mode, answer_policy=VERSION)
                result.pop('error', None)
                result.pop('error_type', None)
            except Exception as exc:
                attempt.update(status='failed', error_type=type(exc).__name__, error=str(exc).replace(self.config['TOKEN_ROUTER_API_KEY'], '[REDACTED]')[:1800])
                retry = isinstance(exc, httpx.TransportError) or (isinstance(exc, httpx.HTTPStatusError) and (exc.response.status_code == 429 or exc.response.status_code >= 500))
                if mode == 'plain':
                    retry = False
                format_retry = isinstance(exc, AnswerFormatError) and mode == 'json'
                attempt['retryable'] = retry
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404):
                    self.stop('http_' + str(exc.response.status_code))
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
            print('QA', rid, result['status'], flush=True)

    def run(self):
        selected = smoke_rows(self.rows) if self.args.smoke else self.rows
        if self.args.retry_failed:
            for row in selected:
                rid = row['id']
                if self.results.get(rid, {}).get('status') == 'failed':
                    old = self.results.pop(rid)
                    target = self.out / 'prediction_history' / rid / (now().replace(':', '-') + '.json')
                    save(target, old)
                    (self.out / 'predictions' / (rid + '.json')).unlink()
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = [pool.submit(self.evaluate, r) for r in selected if r['id'] not in self.results]
            for future in as_completed(futures):
                future.result()
        from scripts.evaluation.lineage import snapshot_run
        snapshot_run(self.out)
        if getattr(self, 'abort', None) is not None and self.abort.is_set():
            self.progress('blocked')
        elif len(self.results) == len(self.rows):
            if self.args.defer_scoring:
                (self.out / 'predictions.jsonl').write_text(''.join((json.dumps(self.results[r['id']], ensure_ascii=False) + '\n' for r in self.rows)))
                self.progress('inference_finished_scoring_deferred')
            else:
                self.progress('scoring')
                score_run(self.out)
                if getattr(self.args, 'llm_judge', False):
                    from scripts.evaluation.judge import run_judge
                    self.progress('llm_judging')
                    run_judge(self.out)
                self.progress('finished')
        else:
            self.progress('blocked' if self.abort.is_set() else 'preflight_complete' if self.args.smoke else 'incomplete')
        save(self.out / 'usage_summary.json', usage_summary(self.attempts))
        manifest = read(self.out / 'manifest.json')
        manifest['sessions'][-1].update(finished_at=now(), state=read(self.out / 'progress.json')['state'])
        save(self.out / 'manifest.json', manifest)
        if getattr(self, 'abort_reason', None) == 'quota_exhausted':
            print('Stopped automatically: insufficient Token Router quota.', flush=True)
            raise SystemExit(3)
        if self.args.smoke:
            details = []
            for r in selected:
                pred = self.results.get(r['id'], {})
                details.append({'id': r['id'], 'case_id': r['case_id'], 'visual_kind': r['visual_kind'], 'image_language': r['image_language'], 'query_language': r['query_language'], 'question': r['query'], 'reference': r['answer'], 'prediction': pred.get('prediction'), 'status': pred.get('status', 'not_attempted'), 'correct': correct(pred['prediction'], r['answer']) if pred.get('status') == 'completed' else None, 'returned_model': pred.get('returned_model'), 'usage_metadata': pred.get('usage_metadata')})
            report = {'model': self.args.model, 'examples': details, 'usage': usage_summary(self.attempts), 'note': 'Two-case interface check, not benchmark ACC.'}
            save(self.out / 'smoke_report.json', report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if any((r['status'] != 'completed' for r in details)):
                raise SystemExit('Smoke test did not complete both examples')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--retry-failed', action='store_true', help='Archive terminal failures and retry them within the existing 11-attempt budget')
    p.add_argument('--defer-scoring', action='store_true', help='Save final predictions and usage; run score.py separately later')
    p.add_argument('--llm-judge', action='store_true', help='Use configured OPENAI judge after scoring; defer-scoring postpones this too')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--request-interval', type=float, default=6.5, help='Minimum seconds between all request starts, including retries (default: 6.5 for a 10 RPM limit)')
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--max-output-tokens', type=int, default=8192)
    args = p.parse_args()
    if args.workers < 1 or args.request_interval < 0 or (not 0 <= args.temperature <= 2):
        p.error('Invalid settings')
    if args.defer_scoring and args.llm_judge:
        p.error('--llm-judge requires scoring; remove --defer-scoring')
    runner = TokenRouterRunner(args)
    signal.signal(signal.SIGTERM, lambda *_: runner.stop('requested_stop'))
    signal.signal(signal.SIGINT, lambda *_: runner.stop('requested_stop'))
    try:
        runner.run()
    except BaseException:
        if read(runner.out / 'progress.json')['state'] not in ('blocked', 'stopping', 'quota_exhausted'):
            runner.progress('interrupted')
        raise
if __name__ == '__main__':
    main()
