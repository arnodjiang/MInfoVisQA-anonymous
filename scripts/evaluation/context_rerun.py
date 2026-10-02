pass
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from scripts.final_benchmark.api import digest, now, read, save
from scripts.final_benchmark.provenance import sha256
from scripts.final_benchmark.restore_context import prepare, publish as publish_dataset
from scripts.evaluation.migrate_token_router import migrate
from scripts.evaluation.judge import Judge, PROMPT as JUDGE_PROMPT
from scripts.evaluation.reference_corrections import DEFAULT, corrected_reference
from scripts.evaluation.lineage import snapshot_run
from scripts.evaluation.run import Runner, usage_summary
from scripts.evaluation.run_token_router import TokenRouterRunner
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/visual_benchmark/final_128_24lang_v3'
DATASET = ROOT / 'data/benchmark'
WORK = ROOT / 'data/context_restoration/all_sources_context_v1'
RUNS = [('gpt6_astra_v3_repair_20260916', 'gpt6_astra_v4_context_20260916', 'openai'), ('gemini38_flash_tokenrouter_v3_20260916', 'gemini38_flash_tokenrouter_v4_context_20260916', 'token_router'), ('gpt56_sol_v3_20260916', 'gpt56_sol_v4_context_20260916', 'openai')]
JUDGE_MODEL = 'gpt-6-astra'

def check_pause():
    if (WORK / 'pause_requested.json').exists():
        raise RuntimeError('Paused by user. Resume only after explicit authorization and removal of pause_requested.json.')

def seed_unchanged_judgments(judge, source, changed_ids):
    pass
    previous = Path(source) / 'llm_judge_text_v2'
    if not (previous / 'manifest.json').exists():
        return 0
    old_policy = read(previous / 'manifest.json')
    new_policy = read(judge.out / 'manifest.json')
    keys = ['protocol', 'judge_model', 'endpoint_sha256', 'prompt_sha256', 'corrections_sha256', 'verdict_policy', 'language_policy', 'strict_matches', 'image_policy']
    if any((old_policy.get(k) != new_policy.get(k) for k in keys)):
        raise ValueError('cannot_reuse_changed_judge_policy')
    old_manifest = read(Path(source) / 'manifest.json')
    if old_manifest['model'] != judge.manifest['model']:
        raise ValueError('cannot_reuse_other_model_judgments')
    old_rows = {r['id']: r for r in map(json.loads, (Path(source) / 'references.jsonl').read_text().splitlines())}
    reused = 0
    for row in judge.refs:
        rid = row['id']
        path = previous / 'judgments' / (rid + '.json')
        if rid in changed_ids or rid in judge.results or (not path.exists()):
            continue
        from scripts.final_benchmark.clean_release import clean_reference
        if old_rows[rid] != row and clean_reference(old_rows[rid]) != row:
            raise ValueError('unchanged_judge_reference_changed:' + rid)
        old_prediction_path = Path(source) / 'predictions' / (rid + '.json')
        old_prediction = read(old_prediction_path)
        prediction = judge.predictions[rid]
        if prediction.get('reused_from', {}).get('source_prediction_sha256') != sha256(old_prediction_path) or prediction.get('prediction') != old_prediction.get('prediction') or prediction['status'] != old_prediction['status']:
            raise ValueError('unchanged_judge_prediction_changed:' + rid)
        decision = read(path)
        expected = digest([old_rows[rid], old_prediction, corrected_reference(old_rows[rid], judge.corrections), digest(JUDGE_PROMPT)])
        if decision['input_fingerprint'] != expected or decision['verdict'] not in ('equivalent', 'different'):
            raise ValueError('old_judgment_not_bound_to_input:' + rid)
        decision.update(input_fingerprint=judge.fingerprint(row), reused_from_judgment={'path': str(path), 'sha256': sha256(path), 'reason': 'Identical question, reference, prediction and text-Judge policy.'})
        save(judge.out / 'judgments' / (rid + '.json'), decision)
        judge.results[rid] = decision
        reused += 1
    save(judge.out / 'reuse_summary.json', {'reused_decisions': reused, 'source': str(previous), 'new_api_calls_for_reused_decisions': 0})
    return reused

def evaluate(run, provider, retry_failed=False):
    check_pause()
    run = Path(run).resolve()
    manifest = read(run / 'manifest.json')
    from scripts.evaluation.audit_resume import audit
    audit(run)
    migration = read(run / 'migration.json')
    source = Path(migration['source_run'])
    changed = set(migration['changed_input_ids'])
    args = SimpleNamespace(dataset=Path(manifest['dataset']), output=run, model=manifest['model'], workers=2 if provider == 'token_router' else 3, request_interval=6.5 if provider == 'token_router' else 3, timeout=300, max_output_tokens=manifest['max_output_tokens'], temperature=manifest['temperature'], smoke=False, retry_failed=retry_failed, defer_scoring=False, llm_judge=False, judge_model=JUDGE_MODEL, publish_table=None)
    runner = TokenRouterRunner(args) if provider == 'token_router' else Runner(args)
    signal.signal(signal.SIGTERM, lambda *_: runner.abort.set())
    signal.signal(signal.SIGINT, lambda *_: runner.abort.set())
    try:
        runner.rows.sort(key=lambda row: row['id'] not in changed)
        runner.run()
        if runner.abort.is_set() or len(runner.results) != len(runner.rows):
            raise RuntimeError('inference_incomplete; resume same context run')
        runner.progress('llm_judging')
        judge = Judge(SimpleNamespace(run=run, output=None, corrections=DEFAULT, model=JUDGE_MODEL, workers=2, ids=None, limit=None))
        signal.signal(signal.SIGTERM, lambda *_: judge.stop.set())
        signal.signal(signal.SIGINT, lambda *_: judge.stop.set())
        try:
            seed_unchanged_judgments(judge, source, changed)
            judge.execute()
            if len(judge.results) != len(judge.refs):
                raise RuntimeError('context_judge_incomplete')
        finally:
            judge.lockfile.close()
        snapshot_run(run, frozen_inputs_only=True)
        pairs = []
        for rid in sorted(changed):
            old_path = source / 'predictions' / (rid + '.json')
            old = read(old_path) if old_path.exists() else {}
            old_judge_path = source / 'llm_judge_text_v2/judgments' / (rid + '.json')
            old_judge = read(old_judge_path) if old_judge_path.exists() else {}
            pairs.append({'variant_id': rid, 'old_prediction': old.get('prediction'), 'old_status': old.get('status', 'not_attempted'), 'new_prediction': runner.results[rid].get('prediction'), 'old_correct': old_judge.get('verdict') == 'equivalent' if old_judge else None, 'new_correct': judge.results[rid]['verdict'] == 'equivalent'})
        comparable = [p for p in pairs if p['old_correct'] is not None]
        save(run / 'context_comparison.json', {'condition': 'source_context_restored_text', 'changed_variants': len(changed), 'paired_scored_count': len(comparable), 'pairs': pairs, 'accuracy_change_percentage_points': 100 * sum((int(p['new_correct']) - int(p['old_correct']) for p in comparable)) / len(comparable) if comparable else None, 'correct_to_wrong': sum((p['old_correct'] and (not p['new_correct']) for p in comparable)), 'wrong_to_correct': sum((not p['old_correct'] and p['new_correct'] for p in comparable)), 'note': 'Restored evidence changes available information; this is not a pure rendering ablation.'})
        save(run / 'context_rerun_summary.json', {'state': 'finished', 'updated_at': now(), 'source_run': str(source), 'changed_variants': len(changed), 'unchanged_predictions_reused': migration['reused_count'], 'changed_inference_usage': usage_summary([a for a in runner.attempts if a['id'] in changed]), 'judge_usage': read(judge.out / 'usage_summary.json')})
        from scripts.evaluation.publish_results import publish
        publish(judge.out, ROOT / 'paper/tables/main_results.tex', expected_judge=JUDGE_MODEL)
        runner.progress('finished')
    finally:
        runner.lock_file.close()

def submit(after_data_pid=None):
    check_pause()
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if after_data_pid:
            save(WORK / 'job.json', {'state': 'waiting_for_current_data_job', 'pid': os.getpid(), 'data_pid': after_data_pid, 'updated_at': now(), 'evaluation_after_validation': True})
            while True:
                check_pause()
                status = read(WORK / 'data_job.json')
                if status.get('state') == 'data_ready':
                    break
                try:
                    os.kill(after_data_pid, 0)
                except ProcessLookupError:
                    if status.get('state') == 'failed':
                        raise RuntimeError('data_job_failed; inspect data_job.json before resuming')
                    break
                time.sleep(10)
        save(WORK / 'job.json', {'state': 'translating_or_verifying_cache', 'pid': os.getpid(), 'updated_at': now()})
        save(WORK / 'data_job.json', {'state': 'translating', 'pid': os.getpid(), 'updated_at': now(), 'clean_export': True, 'evaluation_after_validation': True})
        prepare(SOURCE, WORK, workers=2)
        check_pause()
        dataset = publish_dataset(SOURCE, WORK, DATASET)
        from scripts.final_benchmark.validate_context_revision import validate
        print('Validated context revision', validate(SOURCE, dataset, WORK), flush=True)
        save(WORK / 'data_job.json', {'state': 'data_ready', 'pid': os.getpid(), 'updated_at': now(), 'dataset': str(dataset), 'clean_export': True, 'evaluation_after_validation': True})
        changed = {r['variant_id'] for r in read(dataset / 'invalidated_evaluation_inputs.json')['variants']}
        scope = read(dataset / 'prompt_context_manifest.json')
        if len(changed) != scope['affected_variants'] or len(changed) != scope['source_cases'] * 70:
            raise ValueError('context_scope_does_not_match_full_configuration_coverage')
        plan = []
        for (old_name, new_name, provider) in RUNS:
            (old, new) = (ROOT / 'data/evaluation' / old_name, ROOT / 'data/evaluation' / new_name)
            if not new.exists():
                snapshot_run(old)
                result = migrate(old, dataset, new, provider=provider, preserve_failed=True, expected_changed=changed, allow_partial=True)
                print('Migrated', new_name, result, flush=True)
            else:
                m = read(new / 'migration.json')
                if set(m['changed_input_ids']) != changed or m['source_run'] != str(old):
                    raise ValueError('existing_context_migration_mismatch')
            plan.append((new, provider))
        config_path = ROOT / 'configs/benchmark_release.json'
        config = read(config_path)
        config['active_dataset'] = str(dataset.relative_to(ROOT))
        config['previous_context_condition'] = str(SOURCE.relative_to(ROOT))
        save(config_path, config)
        table_snapshot = WORK / 'main_results_before_context.tex'
        if not table_snapshot.exists():
            table_snapshot.write_bytes((ROOT / 'paper/tables/main_results.tex').read_bytes())
        children = []
        for (run, provider) in plan:
            check_pause()
            log = WORK / (run.name + '.log')
            with log.open('a') as stream:
                proc = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.evaluation.context_rerun', '--run', str(run), '--provider', provider], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append({'run': str(run), 'pid': proc.pid, 'log': str(log), 'provider': provider})
        save(WORK / 'job.json', {'state': 'evaluation_submitted', 'updated_at': now(), 'dataset': str(dataset), 'changed_variants_per_model': len(changed), 'children': children})
        print(json.dumps(children), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path)
    p.add_argument('--provider', choices=['openai', 'token_router'])
    p.add_argument('--retry-failed', action='store_true', help='Retry saved failures after restoring provider access')
    p.add_argument('--after-data-pid', type=int, help='Wait for the existing data-only producer before validating and resuming evaluations')
    args = p.parse_args()
    try:
        if args.run:
            if not args.provider:
                p.error('--provider is required with --run')
            evaluate(args.run, args.provider, retry_failed=args.retry_failed)
        else:
            submit(args.after_data_pid)
    except BaseException as exc:
        dest = args.run / 'context_job_error.json' if args.run else WORK / 'job_error.json'
        save(dest, {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'updated_at': now()})
        raise
