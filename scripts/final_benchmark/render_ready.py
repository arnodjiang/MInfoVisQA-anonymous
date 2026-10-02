pass
import argparse
from pathlib import Path
import shutil
import hashlib
from .pipeline import Builder, DEFAULT_OUT, LANGUAGES
from .export import assemble
from .api import read, save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--refresh-language', choices=list(LANGUAGES), help='Archive and rerender an existing language after a renderer fix')
    args = parser.parse_args()
    args.output = str(DEFAULT_OUT)
    args.retry_failed = False
    builder = Builder(args)
    if args.refresh_language:
        lang = args.refresh_language
        for case in builder.cases:
            folder = builder.folder(case['id'])
            image = folder / 'images' / (lang + '.png')
            if not image.exists():
                continue
            archive = folder / 'render_history' / hashlib.sha256(image.read_bytes()).hexdigest()[:16]
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, archive / image.name)
            shutil.copy2(image.with_suffix('.layout.json'), archive / (lang + '.layout.json'))
            builder.render_one(case['id'], folder / 'render_spec.json', folder / 'locales' / (lang + '.json'), image)
            print('Refreshed ' + case['id'] + ' ' + lang, flush=True)
        assemble(builder.out, partial=True)
        return
    attempts = {}
    while True:
        jobs = []
        for case in builder.cases:
            folder = builder.folder(case['id'])
            if builder.render_current(case):
                continue
            if attempts.get(case['id'], 0) >= 2:
                continue
            if all(((folder / 'locales' / (lang + '.json')).exists() for lang in LANGUAGES)):
                jobs.append(case)
        if jobs:
            builder.stage('render_ready', jobs, builder.render, args.workers)
            for case in jobs:
                attempts[case['id']] = attempts.get(case['id'], 0) + 1
            summary = assemble(builder.out, partial=True)
            shutil.copy2(Path(__file__).with_name('README.md'), builder.out / 'README.md')
            print('Exported %d samples from %d complete cases.' % (summary['samples'], summary['base_cases']), flush=True)
        if not args.watch or all((builder.render_current(c) for c in builder.cases)):
            break
        import time
        time.sleep(20)
if __name__ == '__main__':
    main()
