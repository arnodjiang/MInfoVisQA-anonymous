pass
import argparse
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from scripts.evaluation.migrate_token_router import migrate
from scripts.evaluation.run import Runner, usage_summary
from scripts.evaluation.judge import Judge
from scripts.evaluation.reference_corrections import DEFAULT
from scripts.final_benchmark.api import read, save, now

def repair(source, dataset, output):
    (source, dataset, output) = (Path(p).resolve() for p in (source, dataset, output))
    invalidated = read(dataset / 'invalidated_evaluation_inputs.json')
    ids = {r['variant_id'] for r in invalidated['variants'] if r['requires_new_prediction']}
    if not output.exists():
        print(json.dumps(migrate(source, dataset, output, provider='openai', preserve_failed=True, expected_changed=ids)), flush=True)
    migration = read(output / 'migration.json')
    if set(migration['changed_input_ids']) != ids or migration['source_run'] != str(source) or migration['target_dataset'] != str(dataset):
        raise ValueError('repair_scope_changed')
    manifest = read(output / 'manifest.json')
    args = SimpleNamespace(dataset=dataset, output=output, workers=3, request_interval=3, timeout=300, max_output_tokens=manifest['max_output_tokens'], smoke=False, llm_judge=False)
    runner = Runner(args)
    signal.signal(signal.SIGTERM, lambda *_: runner.abort.set())
    signal.signal(signal.SIGINT, lambda *_: runner.abort.set())
    try:
        pending = {r['id'] for r in runner.rows if r['id'] not in runner.results}
        if not pending <= ids:
            raise ValueError('refusing_to_rerun_unchanged_inputs')
        runner.run()
        if runner.abort.is_set() or len(runner.results) != len(runner.rows):
            raise RuntimeError('repair_inference_incomplete')
    finally:
        runner.lock_file.close()
    judge_args = SimpleNamespace(run=output, output=None, corrections=DEFAULT, model=None, workers=2, ids=None, limit=None)
    judge = Judge(judge_args)
    signal.signal(signal.SIGTERM, lambda *_: judge.stop.set())
    signal.signal(signal.SIGINT, lambda *_: judge.stop.set())
    try:
        judge.execute()
        complete = len(judge.results) == len(judge.refs)
        save(output / 'repair_summary.json', {'updated_at': now(), 'state': 'finished' if complete else 'partial', 'source_run': str(source), 'dataset': str(dataset), 'repaired_ids': sorted(ids), 'repaired_predictions': sum((runner.results[i]['status'] == 'completed' for i in ids)), 'repaired_judgments': sum((i in judge.results for i in ids)), 'unchanged_predictions_preserved': len(runner.rows) - len(ids), 'new_inference_usage': usage_summary([a for a in runner.attempts if a['id'] in ids]), 'judge_usage': read(judge.out / 'usage_summary.json'), 'note': 'Only changed inputs rerun. All combined predictions use one resumable text-only Judge stage: strict match first, LLM for mismatches. No legacy judgment merging.'})
    finally:
        judge.lockfile.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    repair(a.source, a.dataset, a.output)
