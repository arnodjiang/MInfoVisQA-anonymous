pass
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace
from scripts.final_benchmark.api import read, save, now
from scripts.evaluation.run import usage_summary
from scripts.evaluation.run_token_router import TokenRouterRunner
from scripts.evaluation.context_rerun import seed_unchanged_judgments
from scripts.evaluation.judge import Judge
from scripts.evaluation.reference_corrections import DEFAULT
from scripts.evaluation.lineage import snapshot_run
from scripts.evaluation.publish_results import publish
ROOT = Path(__file__).resolve().parents[2]

def targets(predictions):
    return {rid for (rid, p) in predictions.items() if p.get('status') == 'failed' and 'MAX_TOKENS' in p.get('error', '')}

def execute(source, output, max_tokens=32768):
    (source, output) = (Path(source).resolve(), Path(output).resolve())
    job = output.parent / (output.name + '.job.json')
    with job.with_suffix('.lock').open('a') as job_lock:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(job, {'state': 'waiting_for_source_completion', 'source': str(source), 'output': str(output), 'max_output_tokens': max_tokens, 'updated_at': now()})
        with (source / 'run.lock').open('a') as source_lock:
            while True:
                try:
                    fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(30)
            progress = read(source / 'progress.json')
            if progress['state'] != 'finished' or progress['terminal'] != progress['expected']:
                raise RuntimeError('Source stopped before completion; repair was not started')
            manifest = read(source / 'manifest.json')
            if manifest['provider'] != 'token_router' or max_tokens <= manifest['max_output_tokens']:
                raise ValueError('Expected native Gemini and a larger output budget')
            predictions = {p.stem: read(p) for p in (source / 'predictions').glob('*.json')}
            ids = targets(predictions)
            if not ids:
                save(job, {'state': 'finished_no_repairs_needed', 'updated_at': now()})
                return
            from scripts.evaluation.audit_resume import audit
            audit(source)
            args = SimpleNamespace(dataset=Path(manifest['dataset']), output=output, model=manifest['model'], workers=2, request_interval=6.5, timeout=300, max_output_tokens=max_tokens, temperature=manifest['temperature'], smoke=False, retry_failed=False, defer_scoring=False, llm_judge=False)
            fresh = not output.exists()
            runner = TokenRouterRunner(args)
            signal.signal(signal.SIGTERM, lambda *_: runner.stop('requested_stop'))
            signal.signal(signal.SIGINT, lambda *_: runner.stop('requested_stop'))
            try:
                if fresh:
                    for (rid, prediction) in predictions.items():
                        if rid in ids:
                            continue
                        original = source / 'predictions' / (rid + '.json')
                        reuse = {'source_run': str(source), 'source_run_key': manifest['run_key'], 'source_prediction_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'reason': 'Unchanged original outcome; only MAX_TOKENS failures receive the increased budget.', 'original_max_output_tokens': manifest['max_output_tokens']}
                        request = read(source / 'requests' / (rid + '.json'))
                        save(output / 'predictions' / (rid + '.json'), dict(prediction, run_key=runner.run_key, reused_from=reuse))
                        save(output / 'requests' / (rid + '.json'), dict(request, run_key=runner.run_key, reused_from=reuse))
                        for folder in ('attempts', 'responses'):
                            path = source / folder / rid
                            if path.exists():
                                shutil.copytree(path, output / folder / rid)
                    save(output / 'migration.json', {'source_run': str(source), 'changed_input_ids': sorted(ids), 'reused_count': len(predictions) - len(ids), 'kind': 'output_budget_repair', 'note': 'Images/questions unchanged. Scope denotes increased generation budget, not changed data.'})
                    current = read(output / 'manifest.json')
                    current['output_budget_policy'] = {'kind': 'mixed_budget_targeted_recovery', 'original_max_output_tokens': manifest['max_output_tokens'], 'retry_max_output_tokens': max_tokens, 'retry_ids': sorted(ids), 'selection': 'Only terminal MAX_TOKENS failures; no correctness-based resampling.', 'untouched_predictions': len(predictions) - len(ids)}
                    save(output / 'manifest.json', current)
                elif set(read(output / 'migration.json')['changed_input_ids']) != ids:
                    raise ValueError('Repair scope changed')
                runner.results = {p.stem: read(p) for p in (output / 'predictions').glob('*.json')}
                runner.attempts = [read(p) for p in (output / 'attempts').glob('*/*.json')]
                pending = {r['id'] for r in runner.rows if r['id'] not in runner.results}
                if not pending <= ids:
                    raise ValueError('Refusing to rerun an unaffected sample')
                save(job, {'state': 'repairing', 'ids': sorted(ids), 'max_output_tokens': max_tokens, 'updated_at': now()})
                runner.run()
                if runner.abort.is_set() or len(runner.results) != len(runner.rows):
                    raise RuntimeError('Repair interrupted; resume this repair output')
                runner.progress('llm_judging')
                judge = Judge(SimpleNamespace(run=output, output=None, corrections=DEFAULT, model='gpt-6-astra', workers=2, ids=None, limit=None))
                signal.signal(signal.SIGTERM, lambda *_: judge.stop.set())
                signal.signal(signal.SIGINT, lambda *_: judge.stop.set())
                try:
                    seed_unchanged_judgments(judge, source, ids)
                    judge.execute()
                    if len(judge.results) != len(judge.refs):
                        raise RuntimeError('Judge incomplete')
                    snapshot_run(output, frozen_inputs_only=True)
                    publish(judge.out, ROOT / 'paper/tables/main_results.tex', expected_judge='gpt-6-astra')
                finally:
                    judge.lockfile.close()
                report = {'state': 'finished', 'updated_at': now(), 'source_run': str(source), 'output': str(output), 'retry_ids': sorted(ids), 'max_output_tokens': max_tokens, 'recovered': sum((runner.results[i]['status'] == 'completed' for i in ids)), 'still_failed': sum((runner.results[i]['status'] == 'failed' for i in ids)), 'unchanged_predictions_preserved': len(predictions) - len(ids), 'new_inference_usage': usage_summary([a for a in runner.attempts if a['id'] in ids]), 'original_failed_usage': usage_summary([read(p) for rid in ids for p in (source / 'attempts' / rid).glob('*.json')]), 'judge_usage': read(judge.out / 'usage_summary.json')}
                save(output / 'max_tokens_repair_summary.json', report)
                save(job, report)
                runner.progress('finished')
            finally:
                runner.lock_file.close()
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-output-tokens', type=int, default=32768)
    a = p.parse_args()
    try:
        execute(a.source, a.output, a.max_output_tokens)
    except BaseException as exc:
        save(a.output.parent / (a.output.name + '.job.json'), {'state': 'failed', 'updated_at': now(), 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
