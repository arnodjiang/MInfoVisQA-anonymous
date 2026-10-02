pass
import argparse
import hashlib
import json
from pathlib import Path
from scripts.final_benchmark.api import digest, now, read, save
from scripts.final_benchmark.clean_release import verify_clean
from scripts.evaluation.context_input import context_prompt, input_text
from scripts.evaluation.answer_policy import PLAIN_PROMPT
from scripts.evaluation.migrate_token_router import reusable, verify_images

def audit(run):
    run = Path(run).resolve()
    manifest = read(run / 'manifest.json')
    dataset = Path(manifest['dataset'])
    verify_clean(dataset, write_report=False)
    raw = (dataset / 'validation_release/val.candidates.jsonl').read_bytes()
    if raw != (run / 'references.jsonl').read_bytes():
        raise ValueError('Saved references differ from current dataset; migrate before resume')
    if hashlib.sha256(raw).hexdigest() != manifest['references_sha256']:
        raise ValueError('Reference hash mismatch')
    rows = {r['id']: r for r in map(json.loads, raw.decode().splitlines())}
    verify_images(dataset, rows.values())
    prompt = (run / 'prompt.txt').read_text()
    if digest(prompt) != manifest['prompt_sha256']:
        raise ValueError('Saved prompt changed')
    changed = set(read(run / 'migration.json')['changed_input_ids'])
    old_refs = {}
    completed = failed = reused = changed_done = 0
    for path in (run / 'predictions').glob('*.json'):
        result = read(path)
        rid = result['id']
        row = rows[rid]
        request = read(run / 'requests' / (rid + '.json'))
        if path.stem != rid or result['run_key'] != manifest['run_key'] or request['run_key'] != manifest['run_key'] or (request['model'] != manifest['model']) or (request['question'] != row['query']) or (request['image_sha256'] != row['image_sha256']) or (request.get('source_context') != row.get('source_context')) or (request.get('input_text', request['question']) != input_text(row)) or (request['prompt_sha256'] != digest(context_prompt(prompt, row))):
            raise ValueError('Stale request: ' + rid)
        lineage = result.get('reused_from')
        if lineage:
            source = Path(lineage['source_run'])
            if source not in old_refs:
                old_refs[source] = {r['id']: r for r in map(json.loads, (source / 'references.jsonl').read_text().splitlines())}
            original = source / 'predictions' / (rid + '.json')
            old_result = read(original)
            if rid in changed or not reusable(old_refs[source][rid], row) or hashlib.sha256(original.read_bytes()).hexdigest() != lineage['source_prediction_sha256'] or (result.get('prediction') != old_result.get('prediction')):
                raise ValueError('Invalid old prediction reuse: ' + rid)
            reused += 1
        if result['status'] == 'completed':
            attempt = read(run / 'attempts' / rid / ('%03d.json' % result['selected_attempt']))
            expected = digest(context_prompt(PLAIN_PROMPT if attempt.get('answer_mode') == 'plain' else prompt, row))
            if attempt.get('prompt_sha256', request['prompt_sha256']) != expected:
                raise ValueError('Selected attempt prompt mismatch: ' + rid)
            completed += 1
        elif result['status'] == 'failed':
            failed += 1
        else:
            raise ValueError('Unexpected prediction state: ' + rid)
        changed_done += rid in changed
    report = dict(state='passed', updated_at=now(), dataset=str(dataset), references_sha256=manifest['references_sha256'], expected=len(rows), successful=completed, failed=failed, verified_reused=reused, changed_scope=len(changed), changed_processed=changed_done, stale_predictions=0, remaining=len(rows) - completed - failed)
    save(run / 'resume_input_audit.json', report)
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().run)))
