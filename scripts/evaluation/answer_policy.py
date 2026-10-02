pass
import json
import re
from pathlib import Path
from scripts.final_benchmark.api import save, read, now, digest
VERSION = 'json_then_one_plain_answer_v1'
PLAIN_PROMPT = Path(__file__).with_name('plain_answer_prompt.txt').read_text()

class AnswerFormatError(ValueError):
    pass

def extract_answer(raw, plain=False):
    text = raw.strip()
    fenced = re.fullmatch('```(?:json|text)?\\s*\\n(.*?)\\n```', text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict) and set(value) == {'answer'} and isinstance(value['answer'], str) and value['answer'].strip():
        return value['answer']
    if plain and text:
        return text
    raise AnswerFormatError('Invalid answer JSON; one plain-answer fallback is allowed')

def format_failure(attempt):
    if attempt.get('answer_mode') == 'plain':
        return False
    return attempt.get('error_type') in ('JSONDecodeError', 'AnswerFormatError') or (attempt.get('error_type') == 'ValueError' and attempt.get('error') == 'Invalid answer schema')

def next_mode(attempts):
    if any((a.get('answer_mode') == 'plain' for a in attempts)):
        return None
    return 'plain' if any((format_failure(a) for a in attempts)) else 'json'

def apply_policy(runner):
    pass
    manifest_path = runner.out / 'manifest.json'
    manifest = read(manifest_path)
    policy = {'version': VERSION, 'plain_prompt_sha256': digest(PLAIN_PROMPT), 'max_plain_calls_per_qa': 1, 'fenced_json_accepted': True, 'selection': 'format-only; no reference answers used'}
    manifest['sessions'][-1]['answer_policy'] = policy
    if not any((p.get('version') == VERSION for p in manifest.get('policy_amendments', []))):
        manifest.setdefault('policy_amendments', []).append(dict(policy, applied_at=now()))
    save(manifest_path, manifest)
    (runner.out / 'plain_answer_prompt.txt').write_text(PLAIN_PROMPT)
    indexed = {}
    for attempt in runner.attempts:
        indexed.setdefault(attempt['id'], []).append(attempt)
    recovered = queued = 0
    for (rid, result) in list(runner.results.items()):
        attempts = indexed.get(rid, [])
        if result.get('status') != 'failed' or next_mode(attempts) != 'plain':
            continue
        failed = max((a for a in attempts if format_failure(a)), key=lambda a: a['attempt_number'])
        save(runner.out / 'prediction_history' / rid / (VERSION + '.json'), result)
        try:
            prediction = extract_answer(failed.get('raw_output', ''))
        except AnswerFormatError:
            (runner.out / 'predictions' / (rid + '.json')).unlink()
            del runner.results[rid]
            queued += 1
            continue
        updated = {k: v for (k, v) in result.items() if k not in ('error', 'error_type')}
        updated.update(status='completed', prediction=prediction, selected_attempt=failed['attempt_number'], response_id=failed.get('response_id'), returned_model=failed.get('returned_model'), usage=failed.get('usage'), usage_metadata=failed.get('usage_metadata'), answer_policy=VERSION, recovery='existing_fenced_json', recovered_at=now())
        save(runner.out / 'predictions' / (rid + '.json'), updated)
        runner.results[rid] = updated
        recovered += 1
    save(runner.out / 'format_recovery.json', dict(updated_at=now(), recovered_without_api=recovered, queued_for_one_plain_call=queued, policy=policy))
