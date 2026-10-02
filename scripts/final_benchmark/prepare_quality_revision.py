pass
import argparse
import json
from collections import Counter
from pathlib import Path
import shutil
from .api import read, save, now

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/visual_benchmark/baseline')
    p.add_argument('--output', default='data/visual_benchmark/final_128_v3')
    a = p.parse_args()
    src = Path(a.input).resolve()
    out = Path(a.output).resolve()
    if out.exists():
        raise ValueError('revision_output_already_exists')
    rows = [json.loads(l) for l in (src / 'validation_release/val.candidates.jsonl').read_text().splitlines()]
    grouped = {}
    for r in rows:
        grouped.setdefault(r['case_id'], []).append(r)
    repairs = {}
    held = {}
    retained = []
    for (cid, variants) in grouped.items():
        review_path = src / 'validation_release/reviews' / f'{cid}.json'
        if not review_path.exists():
            held[cid] = 'pending_audit'
            continue
        review = read(review_path)
        if all((r['audit']['status'] == 'accepted' for r in variants)):
            retained.append(cid)
            continue
        if variants[0]['answer_type'] in ('list', 'structured', 'long_text'):
            held[cid] = 'answer_format_requires_replacement'
            continue
        if review['answer_preservation'] != 'pass' or review['normal_qa'] != 'pass':
            held[cid] = 'source_answer_or_question_requires_adjudication_or_replacement'
            continue
        spec = read(src / 'cases' / cid / 'spec.json')
        if review['source_fidelity'] != 'pass':
            if variants[0]['source'] == 'Visual-TableQA':
                held[cid] = 'mixed_visual_or_source_problem_requires_replacement'
                continue
            repairs[cid] = 'source'
        else:
            repairs[cid] = 'layout' if spec['kind'] == 'chart' else 'translation'
    shutil.copytree(src, out, ignore=shutil.ignore_patterns('serial_runs', '.matplotlib_cache', '__pycache__'))
    save(out / 'quality_repair_plan.json', {'created_at': now(), 'parent_build': str(src), 'repairs': repairs, 'held': held, 'fully_admitted_cases_unchanged': retained, 'mode_counts': dict(Counter(repairs.values())), 'policy': 'Never alter source question or supplied answer to obtain a pass; immutable prior build retained. All changed cases require renewed multilingual audit.'})
    (out / 'BUILD_NOTES.md').write_text('# v3 quality repair revision\n\nThis quality-repair revision is cloned from v2. quality_repair_plan.json records eligible repairs and cases requiring replacement or adjudication. Each changed case archives its prior files in before_quality_repair/. Copied API records are historical reuse, not newly billed calls. Final admission requires renewed checks; the screened release manifest reports whether 2K–5K samples were achieved.\n')
    print({'repairs': len(repairs), 'modes': dict(Counter(repairs.values())), 'held': len(held), 'unchanged': len(retained)})
if __name__ == '__main__':
    main()
