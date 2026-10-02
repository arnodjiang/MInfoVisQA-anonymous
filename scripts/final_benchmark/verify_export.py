pass
import argparse
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageChops
from .api import now, read, save
from .pipeline import DEFAULT_OUT, FONT, ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(DEFAULT_OUT))
    parser.add_argument('--ids', help='Optional comma-separated base IDs')
    parser.add_argument('--languages', default='en,zh,ar,hi', help='Comma-separated languages to execute')
    args = parser.parse_args()
    out = Path(args.output)
    cases = read(out / 'case_manifest.json')
    if args.ids:
        cases = [c for c in cases if c['id'] in args.ids.split(',')]
    else:
        chosen = {}
        for case in cases:
            chosen.setdefault(case['kind'], case)
        cases = list(chosen.values())
    checks = []
    for case in cases:
        for lang in args.languages.split(','):
            folder = out / 'cases' / case['id']
            code = folder / 'code' / ('original' if lang == 'en' else 'translated') / lang / 'render.py'
            target = out / 'reproduction_check' / case['id'] / (lang + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
            if os.environ.get('MVISQA_FALLBACK_FONTS'):
                env['MVISQA_FALLBACK_FONTS'] = os.environ['MVISQA_FALLBACK_FONTS']
            command = [str(ROOT / '.venv/bin/python'), str(code), '--output', str(target), '--font', FONT]
            run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
            equal = False
            if run.returncode == 0:
                with Image.open(target) as actual, Image.open(folder / 'images' / (lang + '.png')) as expected:
                    equal = actual.size == expected.size and ImageChops.difference(actual.convert('RGB'), expected.convert('RGB')).getbbox() is None
            checks.append({'id': case['id'], 'kind': case['kind'], 'language': lang, 'exit_code': run.returncode, 'pixels_identical': equal, 'error': run.stderr[-1200:] if run.returncode else None})
            print(case['id'], lang, 'identical' if equal else 'FAILED', flush=True)
    result = {'checked_at': now(), 'checks': checks, 'all_passed': bool(checks) and all((c['pixels_identical'] for c in checks)), 'scope': 'Standalone reproducibility only, not source or translation accuracy.'}
    save(out / 'standalone_verification.json', result)
    if not result['all_passed']:
        sys.exit(1)
if __name__ == '__main__':
    main()
