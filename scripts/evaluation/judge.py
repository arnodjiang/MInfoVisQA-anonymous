pass
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
from pathlib import Path
import signal
import threading
from scripts.final_benchmark.api import API, read, save, digest, now
from scripts.final_benchmark.provenance import sha256
from scripts.openai_config import load as load_config
from scripts.evaluation.judge_config import resolve as resolve_judge_config
from scripts.evaluation.reference_corrections import load as load_corrections, corrected_reference, DEFAULT
from scripts.evaluation.release import check_dataset
from scripts.evaluation.score import correct, summarize, LANGUAGES, latex_escape
ROOT = Path(__file__).resolve().parents[2]
PROMPT = Path(__file__).with_name('judge_prompt.txt').read_text()
LEGACY_TEXT_PROMPT_SHA256 = 'ea85bbfce7e7b542e6ea5f9bb92fe613d2d7a13616a2d6aa321b32ef04fd878e'

def normalize_verdict(value):
    value = dict(value)
    if value.get('verdict') == 'uncertain':
        value.update(verdict='different', verdict_normalization='uncertain_to_different')
    return value

def parse_verdict(value):
    if not isinstance(value, dict) or value.get('verdict') not in {'equivalent', 'different', 'uncertain'}:
        raise ValueError('invalid_judge_verdict')
    if value.get('language_compliance') not in {'pass', 'fail', 'not_applicable', 'uncertain'}:
        raise ValueError('invalid_language_compliance')
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('missing_judge_reason')
    if not isinstance(value.get('category'), str):
        raise ValueError('missing_judge_category')
    return normalize_verdict(value)

def upgrade_binary_policy(out, policy, refs, predictions, corrections):
    pass
    old = read(out / 'manifest.json')
    expected = dict(policy, prompt_sha256=LEGACY_TEXT_PROMPT_SHA256, uncertain_policy='not correct; reported separately; fixed denominator')
    expected.pop('verdict_policy')
    if old != expected:
        raise ValueError('judge_policy_changed_use_new_output')
    old_prompt = (out / 'prompt.txt').read_text()
    if digest(old_prompt) != LEGACY_TEXT_PROMPT_SHA256:
        raise ValueError('legacy_prompt_changed')
    changed = []
    converted = 0
    for row in refs:
        path = out / 'judgments' / (row['id'] + '.json')
        if not path.exists():
            continue
        result = read(path)
        old_fp = digest([row, predictions[row['id']], corrected_reference(row, corrections), LEGACY_TEXT_PROMPT_SHA256])
        new_fp = digest([row, predictions[row['id']], corrected_reference(row, corrections), digest(PROMPT)])
        if result.get('input_fingerprint') not in {old_fp, new_fp}:
            raise ValueError('stale_judgment_during_binary_upgrade')
        if result.get('verdict') not in {'equivalent', 'different', 'uncertain'}:
            raise ValueError('invalid_saved_verdict')
        converted += result['verdict'] == 'uncertain' or result.get('verdict_normalization') == 'uncertain_to_different'
        result = normalize_verdict(result)
        result.update(input_fingerprint=new_fp, policy_migration='binary_verdict_v1', original_prompt_sha256=LEGACY_TEXT_PROMPT_SHA256)
        changed.append((path, result))
    history = out / 'policy_history/before_binary'
    save(history / 'manifest.json', old)
    (history / 'prompt.txt').write_text(old_prompt)
    for (path, result) in changed:
        save(path, result)
    save(out / 'binary_migration.json', {'updated_at': now(), 'preserved_decisions': len(changed), 'uncertain_converted_to_different': converted, 'api_calls': 0, 'accuracy_unchanged': True, 'rule': 'Only equivalent is correct; all uncertain outcomes become different.'})

def load_run(run):
    manifest = read(run / 'manifest.json')
    check_dataset(manifest['dataset'])
    if sha256(run / 'references.jsonl') != manifest['references_sha256']:
        raise ValueError('reference_snapshot_changed')
    refs = [json.loads(s) for s in (run / 'references.jsonl').read_text().splitlines()]
    records = [read(p) for p in sorted((run / 'predictions').glob('*.json'))]
    predictions = {r['id']: r for r in records}
    if len(predictions) != len(records) or len({r['id'] for r in refs}) != len(refs):
        raise ValueError('duplicate_ids')
    if set(predictions) != {r['id'] for r in refs}:
        raise ValueError('inference_not_complete')
    if any((p.get('run_key') != manifest['run_key'] or p.get('status') not in {'completed', 'failed'} for p in records)):
        raise ValueError('invalid_prediction_snapshot')
    return (manifest, refs, predictions)

class Judge:

    def __init__(self, args):
        self.args = args
        self.run = Path(args.run).resolve()
        self.out = Path(args.output).resolve() if args.output else self.run / 'llm_judge_text_v2'
        self.out.mkdir(parents=True, exist_ok=True)
        self.lockfile = (self.out / 'judge.lock').open('a')
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (self.manifest, self.refs, self.predictions) = load_run(self.run)
        self.corrections = load_corrections(args.corrections)
        config = resolve_judge_config(load_config(ROOT), args.model)
        self.api = API(self.out, config, retry_failed=True)
        self.stop = threading.Event()
        self.mutex = threading.Lock()
        policy = {'protocol': 'strict_match_then_text_equivalence_v2', 'source_run': str(self.run), 'source_run_key': self.manifest['run_key'], 'references_sha256': self.manifest['references_sha256'], 'predictions_sha256': digest(self.predictions), 'judge_model': config['OPENAI_MODEL'], 'endpoint_sha256': digest(config.get('OPENAI_BASE_URL')), 'prompt_sha256': digest(PROMPT), 'corrections_sha256': digest(self.corrections), 'same_model_judge': config['OPENAI_MODEL'] == self.manifest['model'], 'verdict_policy': 'binary: equivalent is correct; different or unconfirmed equivalence is incorrect', 'language_policy': 'semantic ACC; language compliance recorded separately', 'strict_matches': 'accepted without API; other completed answers sent to judge', 'image_policy': 'text only: question, reference answer, candidate answer and answer language; no images, tables or reconstruction data'}
        if (self.out / 'manifest.json').exists() and read(self.out / 'manifest.json') != policy:
            upgrade_binary_policy(self.out, policy, self.refs, self.predictions, self.corrections)
        save(self.out / 'manifest.json', policy)
        save(self.out / 'reference_corrections.json', self.corrections)
        (self.out / 'prompt.txt').write_text(PROMPT)
        self.results = {}
        for r in self.refs:
            path = self.out / 'judgments' / (r['id'] + '.json')
            if path.exists():
                result = read(path)
                if result.get('input_fingerprint') != self.fingerprint(r):
                    raise ValueError('stale_judgment')
                if result.get('verdict') not in {'equivalent', 'different'}:
                    raise ValueError('non_binary_saved_verdict')
                self.results[r['id']] = result
        for r in self.refs:
            p = self.predictions[r['id']]
            (answer, cid) = corrected_reference(r, self.corrections)
            if r['id'] in self.results:
                continue
            if p['status'] == 'failed':
                self.record(r, {'verdict': 'different', 'category': 'inference_failure', 'reason': 'No completed model answer.', 'language_compliance': 'not_applicable'}, 'inference_failure')
            elif correct(p['prediction'], answer):
                self.record(r, {'verdict': 'equivalent', 'category': 'strict_match', 'reason': 'Exact numeric/text match to effective reference.', 'language_compliance': 'not_applicable'}, 'deterministic')
        self.progress('ready')

    def fingerprint(self, r):
        return digest([r, self.predictions[r['id']], corrected_reference(r, self.corrections), digest(PROMPT)])

    def record(self, r, result, method, key=None):
        result = normalize_verdict(result)
        if result.get('verdict') not in {'equivalent', 'different'}:
            raise ValueError('non_binary_verdict')
        (answer, cid) = corrected_reference(r, self.corrections)
        record = dict(result, id=r['id'], case_id=r['case_id'], method=method, request_sha256=key, reference_original=r['answer'], reference_effective=answer, reference_correction_id=cid, prediction=self.predictions[r['id']].get('prediction'), input_fingerprint=self.fingerprint(r), finished_at=now())
        save(self.out / 'judgments' / (r['id'] + '.json'), record)
        with self.mutex:
            self.results[r['id']] = record

    def progress(self, state):
        with self.mutex:
            rows = list(self.results.values())
        save(self.out / 'progress.json', {'updated_at': now(), 'state': state, 'expected': len(self.refs), 'completed': len(rows), 'remaining': len(self.refs) - len(rows), 'methods': dict(Counter((r['method'] for r in rows))), 'verdicts': dict(Counter((r['verdict'] for r in rows)))})

    def evaluate(self, r):
        if self.stop.is_set():
            return
        (answer, cid) = corrected_reference(r, self.corrections)
        payload = {'question': r['query'], 'reference_answer': answer, 'candidate_answer': self.predictions[r['id']]['prediction'], 'answer_language': r['answer_language']}
        try:
            (value, key) = self.api.call('answer_equivalence', r['id'], PROMPT, payload, max_tokens=900)
            self.record(r, parse_verdict(value), 'llm', key)
        except Exception as exc:
            self.record(r, {'verdict': 'different', 'category': 'judge_error', 'language_compliance': 'uncertain', 'reason': type(exc).__name__ + ': ' + str(exc)[:200]}, 'judge_error')
            for p in (self.out / 'api/answer_equivalence' / r['id']).glob('*/attempt_*.json'):
                a = read(p)
                reason = str(a.get('error_message', '')).lower()
                if a.get('http_status') in (401, 403) or 'insufficient' in reason or 'quota_exhausted' in reason:
                    self.stop.set()
        self.progress('running')
        if len(self.results) % 25 == 0:
            self.accounting()

    def accounting(self):
        attempts = [read(p) for p in (self.out / 'api').glob('*/*/*/attempt_*.json')]
        totals = {k: sum(((a.get('usage') or {}).get(k, 0) for a in attempts)) for k in ['input_tokens', 'output_tokens', 'total_tokens']}
        save(self.out / 'usage_summary.json', {'api_attempts': len(attempts), 'transport_retries': sum((a.get('automatic_transport_retries', 0) > 0 for a in attempts)), 'known_usage': totals, 'attempts_without_usage': sum((not a.get('usage') for a in attempts)), 'note': 'Separate from inference token usage; unknown usage is not zero.'})

    def finish(self):
        self.accounting()
        if len(self.results) != len(self.refs):
            self.progress('paused' if self.stop.is_set() else 'partial')
            return
        verdicts = {r['id']: r['verdict'] == 'equivalent' for r in self.results.values()}
        corrected = [dict(r, answer=corrected_reference(r, self.corrections)[0]) for r in self.refs]
        cohorts = {}
        for (name, rows) in [('all', corrected), ('screened', [r for r in corrected if r.get('audit', {}).get('status') == 'accepted']), ('chart', [r for r in corrected if r.get('visual_family', r['visual_kind']) == 'chart']), ('table', [r for r in corrected if r.get('visual_family', r['visual_kind']) == 'table'])]:
            (cohorts[name], _) = summarize(rows, self.predictions, verdicts)
        report = {'metric': 'Semantic ACC: strict matching plus text-only LLM equivalence adjudication', 'model': self.manifest['model'], 'judge_model': self.api.config['OPENAI_MODEL'], 'same_model_judge': read(self.out / 'manifest.json')['same_model_judge'], 'cohorts': cohorts, 'verdict_counts': dict(Counter((r['verdict'] for r in self.results.values()))), 'reference_corrections': list(self.corrections.values()), 'source_inference_dataset': self.manifest['dataset'], 'note': 'Post-hoc scoring protocol, not independent human adjudication. Existing inference/images and original strict scores are unchanged.'}
        save(self.out / 'scores.json', report)
        (self.out / 'judgments.jsonl').write_text(''.join((json.dumps(self.results[r['id']], ensure_ascii=False) + '\n' for r in self.refs)))
        with (self.out / 'leaderboard.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['cohort', 'model', 'XQA_ZH', 'XQA_EN'] + ['LQA_' + l.upper() for l in LANGUAGES] + ['AVG'])
            for (name, s) in cohorts.items():
                writer.writerow([name, report['model'], s['XQA']['ZH'], s['XQA']['EN']] + list(s['LQA'].values()) + [s['AVG']])
        s = cohorts['all']
        values = [s['XQA']['ZH'], s['XQA']['EN']] + list(s['LQA'].values()) + [s['AVG']]
        row = latex_escape(report['model']) + ' & ' + ' & '.join(('\\textemdash' if v is None else f'{v:.1f}' for v in values)) + ' \\\\' + '\n'
        (self.out / 'main_table_row.tex').write_text('% Semantic ACC with declared reference corrections; see scores.json.\n' + row)
        self.progress('finished')

    def execute(self):
        pending = [r for r in self.refs if r['id'] not in self.results]
        if self.args.ids:
            pending = [r for r in pending if r['id'] in self.args.ids.split(',')]
        if self.args.limit is not None:
            pending = pending[:self.args.limit]
        self.progress('running')
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            for f in as_completed([pool.submit(self.evaluate, r) for r in pending]):
                f.result()
        self.finish()
        print(json.dumps(read(self.out / 'progress.json')), flush=True)

def run_judge(run, model=None):
    args = argparse.Namespace(run=run, output=None, corrections=DEFAULT, model=model, workers=2, ids=None, limit=None)
    judge = Judge(args)
    try:
        judge.execute()
        if read(judge.out / 'progress.json')['state'] != 'finished':
            raise RuntimeError('Text Judge incomplete; resume the same run before reporting final scores')
        if (judge.run / 'lineage/latest.json').exists():
            from scripts.evaluation.lineage import snapshot_run
            snapshot_run(judge.run, frozen_inputs_only=True)
        return judge.out
    finally:
        judge.lockfile.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--corrections', type=Path, default=DEFAULT)
    p.add_argument('--model', help='Judge model: CLI > JUDGE_MODEL > OPENAI_MODEL')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    p.add_argument('--ids', help='Comma-separated variant IDs for a bounded smoke test')
    a = p.parse_args()
    if a.workers < 1 or (a.limit is not None and a.limit < 0):
        p.error('invalid workers/limit')
    judge = Judge(a)
    signal.signal(signal.SIGTERM, lambda *_: judge.stop.set())
    signal.signal(signal.SIGINT, lambda *_: judge.stop.set())
    judge.execute()
if __name__ == '__main__':
    main()
