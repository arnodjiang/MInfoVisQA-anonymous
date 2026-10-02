pass
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from collections import Counter
from .api import now, read, save

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/visual_benchmark/baseline')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    src = Path(a.input).resolve()
    dest = Path(a.output).resolve()
    release = src / 'validation_release'
    rows = [json.loads(l) for l in (release / 'val.jsonl').read_text().splitlines()]
    assert rows and len({r['id'] for r in rows}) == len(rows)
    for r in rows:
        assert r['audit']['status'] == 'accepted'
        assert r['answer_type'] not in ('list', 'long_text', 'structured')
        assert r['query_language'] == r['answer_language']
        image = (release / r['image_path']).resolve()
        assert hashlib.sha256(image.read_bytes()).hexdigest() == r['image_sha256']
        assert (release / r['code_path']).resolve().is_file()
    dest.mkdir(parents=True, exist_ok=False)
    copied = set()

    def copy(source, target):
        if target in copied:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.add(target)
    for r in rows:
        cid = r['case_id']
        lang = r['image_language']
        image = f'images/{cid}/{lang}.png'
        code = f'code/{cid}/{lang}.py'
        copy((release / r['image_path']).resolve(), dest / image)
        copy((release / r['code_path']).resolve(), dest / code)
        r['image_path'] = image
        r['code_path'] = code
        review = f'metadata/reviews/{cid}.json'
        copy(release / 'reviews' / f'{cid}.json', dest / review)
        r['audit']['review_path'] = review
        for name in ('source.json', 'spec.json', 'render_spec.json', 'qa.json', 'table_extraction.json'):
            f = src / 'cases' / cid / name
            if f.exists():
                copy(f, dest / 'metadata/cases' / cid / name)
        for f in (src / 'cases' / cid / 'locales').glob('*.json'):
            copy(f, dest / 'metadata/cases' / cid / 'locales' / f.name)
        for f in (src / 'cases' / cid / 'original').glob('*'):
            if f.is_file():
                copy(f, dest / 'metadata/cases' / cid / 'original' / f.name)
    (dest / 'val.jsonl').write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in rows)))
    for name in ('audit_summary.json', 'quality_metrics.json', 'quality_issues_by_case.jsonl', 'EVALUATION.md'):
        if (release / name).exists():
            copy(release / name, dest / 'metadata' / name)
    base = {r['base_id']: r for r in rows}
    for cid in {r['case_id'] for r in rows}:
        for stage in (src / 'api').iterdir():
            if not stage.is_dir():
                continue
            for f in (stage / cid).rglob('*.json'):
                copy(f, dest / 'metadata/api' / stage.name / cid / f.relative_to(stage / cid))
    project = Path(__file__).resolve().parents[2]
    for directory in ('scripts/final_benchmark', 'skills/multilingual-visual-benchmark'):
        for f in (project / directory).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and (f.suffix in ('.py', '.md', '.json', '.txt')):
                copy(f, dest / 'reproducibility' / f.relative_to(project))
    summary = {'created_at': now(), 'split': 'validation', 'samples': len(rows), 'base_questions': len(base), 'images': len({r['image_path'] for r in rows}), 'source_counts_base': dict(Counter((r['source'] for r in base.values()))), 'source_build': str(src), 'human_verified': False, 'minimum_target': 2000, 'meets_size_target': 2000 <= len(rows) <= 5000, 'construction_complete': len(list((src / 'cases').glob('*/render_complete.json'))) == 128, 'audit_coverage': read(release / 'audit_summary.json')['reviewed_base_cases'], 'status': 'screened_release_candidate; unresolved full-benchmark requirements remain', 'scope': 'Only automatically admitted variants. Does not certify failed or pending candidates; no human verification claimed.'}
    save(dest / 'manifest.json', summary)
    (dest / 'README.md').write_text(f'# MVisQA screened validation candidate\n\nContains {len(rows)} admitted configurations from {len(base)} source questions. This is an automated-screened subset, not certification of the full planned benchmark.\n\nLoad `val.jsonl`; resolve image_path and code_path relative to this directory. Language variants of one base question are correlated; retain split_group_keys when splitting or bootstrapping. Use query, answer, query_language, image_language, answer_language, difficulty, chart_types, table_structure, and task_tags.\n\nReference evaluation: Unicode NFC and outer-whitespace-normalized exact match. Do not silently add numeric tolerances or unordered-list matching. Difficulty and visual classes are automated annotations. Upstream provenance is retained under metadata. Rendering code requires the project dependencies and a compatible multilingual font; no font is redistributed here.\n\nSource build: `{src}`. Per-case API records, original inputs and pipeline source are included under metadata/ and reproducibility/. Complete run history and excluded cases remain in the source build. Consult manifest.json for audit coverage and whether the requested 2K–5K size target has been reached.\n')
    hashes = {str(f.relative_to(dest)): hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.rglob('*') if f.is_file()}
    save(dest / 'SHA256SUMS.json', hashes)
    print(json.dumps(summary, ensure_ascii=False))
if __name__ == '__main__':
    main()
