pass
import argparse
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from scripts.final_benchmark.api import read, save, now
from scripts.evaluation.run import Runner, usage_summary
from scripts.evaluation.run_token_router import TokenRouterRunner
from scripts.evaluation.freeze_inputs import image_path

def reusable(old, new):
    fields = ('id', 'case_id', 'source', 'query', 'image_sha256', 'source_context', 'query_language', 'image_language', 'answer_language')
    return all((old.get(k) == new.get(k) for k in fields)) and old.get('provenance') == new.get('provenance')

def verify_images(dataset, rows):
    dataset = Path(dataset).resolve()
    seen = {}
    for row in rows:
        image = (dataset / 'validation_release' / row['image_path']).resolve()
        if dataset not in image.parents:
            raise ValueError('unsafe_image_path')
        if image not in seen:
            seen[image] = hashlib.sha256(image.read_bytes()).hexdigest()
        if seen[image] != row['image_sha256']:
            raise ValueError('image_hash_mismatch:' + row['id'])

def migrate(source, dataset, output, provider='token_router', preserve_failed=False, expected_changed=None, allow_partial=False):
    (source, dataset, output) = (Path(p).resolve() for p in (source, dataset, output))
    if output.exists():
        raise ValueError('Migration requires a fresh output directory')
    with (source / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(source / 'manifest.json')
        raw = (source / 'references.jsonl').read_bytes()
        if hashlib.sha256(raw).hexdigest() != previous['references_sha256']:
            raise ValueError('source_reference_snapshot_changed')
        old = {r['id']: r for r in map(json.loads, raw.decode().splitlines())}
        if len(old) != previous['n']:
            raise ValueError('invalid_source_ids')
        new = [json.loads(s) for s in (dataset / 'validation_release/val.candidates.jsonl').read_text().splitlines()]
        changed_ids = {r['id'] for r in new if r['id'] not in old or not reusable(old[r['id']], r)}
        if expected_changed is not None and changed_ids != set(expected_changed):
            raise ValueError('changed_inputs_do_not_match_authorized_repair')
        if preserve_failed and (not allow_partial) and any((not (source / 'predictions' / (r['id'] + '.json')).exists() for r in new if r['id'] not in changed_ids)):
            raise ValueError('unchanged_prediction_missing')
        frozen = read(source / 'frozen_images.json') if (source / 'frozen_images.json').exists() else None
        if frozen is None:
            verify_images(previous['dataset'], old.values())
        else:
            seen = {}
            for row in old.values():
                path = image_path(source, previous, row, frozen)
                if path not in seen:
                    seen[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                if seen[path] != row['image_sha256']:
                    raise ValueError('frozen_source_image_mismatch:' + row['id'])
        verify_images(dataset, new)
        args = SimpleNamespace(dataset=dataset, output=output, model=previous['model'], workers=2, request_interval=6.5, smoke=False, defer_scoring=False, llm_judge=True, retry_failed=False, timeout=300, max_output_tokens=previous['max_output_tokens'], temperature=previous['temperature'])
        if provider not in {'openai', 'token_router'}:
            raise ValueError('Unsupported provider')
        runner = Runner(args) if provider == 'openai' else TokenRouterRunner(args)
        try:
            for (key, value) in runner.identity.items():
                if key == 'context_protocol' and key not in previous and (expected_changed is not None):
                    continue
                if key != 'references_sha256' and previous.get(key) != value:
                    raise ValueError('inference_setting_changed:' + key)
            if (source / 'prompt.txt').read_text() != runner.prompt:
                raise ValueError('source_prompt_changed')
            policy = read(output / 'manifest.json')['policy_amendments'][-1]
            if not any((all((p.get(k) == v for (k, v) in policy.items() if k != 'applied_at')) for p in previous.get('policy_amendments', []))):
                raise ValueError('answer_format_policy_changed')
            (reused, changed, failed) = ([], [], [])
            for row in new:
                rid = row['id']
                if rid not in old or not reusable(old[rid], row):
                    changed.append(rid)
                    continue
                path = source / 'predictions' / (rid + '.json')
                if not path.exists():
                    continue
                result = read(path)
                if result.get('id') != rid or result.get('run_key') != previous['run_key']:
                    raise ValueError('invalid_source_prediction:' + rid)
                if result['status'] not in {'completed', 'failed'}:
                    raise ValueError('invalid_source_status:' + rid)
                if result['status'] != 'completed' and (not preserve_failed):
                    failed.append(rid)
                    continue
                request = read(source / 'requests' / (rid + '.json'))
                from scripts.evaluation.context_input import context_prompt, input_text
                expected_prompt = context_prompt(runner.prompt, row)
                from scripts.final_benchmark.api import digest
                if request['run_key'] != previous['run_key'] or request['question'] != row['query'] or request['image_sha256'] != row['image_sha256'] or (request['model'] != previous['model']) or (request['prompt_sha256'] != digest(expected_prompt)) or (request.get('input_text', request['question']) != input_text(row)):
                    raise ValueError('request_provenance_mismatch:' + rid)
                if result['status'] == 'completed':
                    selected = read(source / 'attempts' / rid / ('%03d.json' % result['selected_attempt']))
                    if selected.get('status') != 'completed' and (not result.get('recovery')):
                        raise ValueError('selected_attempt_not_completed:' + rid)
                lineage = {'source_run': str(source), 'source_run_key': previous['run_key'], 'source_prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'reason': 'Identical question, image bytes, source entry and inference settings.'}
                result.update(run_key=runner.run_key, reused_from=lineage)
                request.update(run_key=runner.run_key, image_path=row['image_path'], reused_from=lineage)
                save(output / 'predictions' / (rid + '.json'), result)
                save(output / 'requests' / (rid + '.json'), request)
                for folder in ('attempts', 'responses'):
                    src = source / folder / rid
                    if src.exists():
                        shutil.copytree(src, output / folder / rid)
                reused.append(rid)
            runner.results = {p.stem: read(p) for p in (output / 'predictions').glob('*.json')}
            runner.attempts = [read(p) for p in (output / 'attempts').glob('*/*.json')]
            report = {'created_at': now(), 'source_run': str(source), 'target_dataset': str(dataset), 'reused_count': len(reused), 'reused_ids': reused, 'changed_input_count': len(changed), 'changed_input_ids': changed, 'failed_to_retry_ids': failed, 'remaining': len(new) - len(reused), 'preserved_failures': sum((r['status'] == 'failed' for r in runner.results.values())), 'source_usage': usage_summary([read(p) for p in (source / 'attempts').glob('*/*.json')]), 'inherited_usage': usage_summary(runner.attempts), 'accounting_note': 'Target usage includes copied attempts. Do not sum source and target totals: inherited usage overlaps. Old excluded calls remain recorded in source_usage.'}
            save(output / 'migration.json', report)
            manifest = read(output / 'manifest.json')
            manifest['migration'] = {'source_run': str(source), 'report': 'migration.json'}
            manifest['sessions'][-1].update(finished_at=now(), state='migration_prepared', stage='migration')
            save(output / 'manifest.json', manifest)
            runner.progress('migration_prepared')
            save(output / 'usage_summary.json', usage_summary(runner.attempts))
            return {k: report[k] for k in ('reused_count', 'changed_input_count', 'remaining')}
        finally:
            runner.lock_file.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.dataset, args.output)))
