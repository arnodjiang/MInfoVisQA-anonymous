pass
import argparse
from collections import Counter
import hashlib
from importlib import metadata
import platform
from pathlib import Path
from .api import now, read, save
from .pipeline import ROOT, DEFAULT_OUT

def audit(out):
    attempts = Counter()
    stages = Counter()
    errors = Counter()
    usage = Counter()
    models = Counter()
    with_usage = 0
    for path in sorted((out / 'api').glob('*/*/*/attempt_*.json')):
        item = read(path)
        attempts[item.get('status', 'unknown')] += 1
        stages[path.relative_to(out / 'api').parts[0]] += 1
        if item.get('error_type'):
            errors[item['error_type']] += 1
        if item.get('returned_model'):
            models[item['returned_model']] += 1
        if isinstance(item.get('usage'), dict):
            with_usage += 1
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                value = item['usage'].get(key)
                if isinstance(value, int):
                    usage[key] += value
    versions = {}
    for name in ('openai', 'python-dotenv', 'numpy', 'matplotlib', 'Pillow', 'freetype-py', 'uharfbuzz', 'python-bidi', 'pyarrow'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    code = {}
    for subdir in ('scripts/final_benchmark', 'skills/multilingual-visual-benchmark'):
        for path in sorted((ROOT / subdir).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.md', '.txt', '.json') and ('__pycache__' not in path.parts):
                code[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    folders = list((out / 'cases').glob('*'))
    counts = {key: sum(((p / filename).exists() for p in folders)) for (key, filename) in [('reconstructed', 'recovery_complete.json'), ('qa_normalized', 'qa.json'), ('cases_rendered', 'render_complete.json')]}
    counts['images_rendered'] = len(list((out / 'cases').glob('*/images/*.png')))
    counts['locales_saved'] = len(list((out / 'cases').glob('*/locales/*.json')))
    reviews = Counter((read(p)['status'] for p in (out / 'cases').glob('*/review.json')))
    result = {'created_at': now(), 'python': platform.python_version(), 'platform': platform.platform(), 'dependencies': versions, 'code_sha256': code, 'actual_counts': counts, 'review_status': dict(reviews), 'api_attempt_status': dict(attempts), 'api_attempts_by_stage': dict(stages), 'api_errors': dict(errors), 'returned_models': dict(models), 'observed_usage': dict(usage), 'attempts_with_usage': with_usage, 'attempts_without_usage': sum(attempts.values()) - with_usage, 'usage_caveat': 'Observed provider usage only; missing usage is unknown, not zero. Started attempts may be active or interrupted. Not a billing statement.'}
    save(out / 'reproducibility_audit.json', result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    result = audit(Path(args.output))
    print(result['actual_counts'])
    print('API attempt status:', result['api_attempt_status'])
if __name__ == '__main__':
    main()
