pass
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
OUTLIERS = {'0bccf72675f62463de4b', '311aff4cd44119b31867', '3577ae17737fb64d741f', '48b010662f74389a9d51', '6503ad662b7f50d04ef5', '695d46740bbada87a08a', '6ba9e5985977cdcd23cd', 'a5f0cc1f70c843131bc2', 'd7ed2940a356a9921746', 'f03b6707dd90a5fd9d47'}
REPLACEMENTS = {'b5ed01c0223fd6aee634', '667f444290a8733126aa', '3003d2db8a0f8562b7ca', '31b80870eb436346754d', '88679fa8758854690e7c', '0efd9b1928fbf1e23895', '2873b6610879880f28c9', 'c10f553cd51894a5a726', 'c8bed98b3b7e85891063', '7a281c6b9338cffbaab6'}

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/processed/normal_qa_v3')
    parser.add_argument('--output', default='data/processed/normal_qa_v4')
    args = parser.parse_args()
    (source, output) = (Path(args.source), Path(args.output))
    selected = read_jsonl(source / 'candidates.jsonl')
    pool = {r['id']: r for r in read_jsonl(source / 'eligible_pool.jsonl')}
    missing = OUTLIERS - {r['id'] for r in selected}
    if missing:
        raise ValueError('outlier_ids_missing:' + repr(sorted(missing)))
    if not REPLACEMENTS <= set(pool):
        raise ValueError('replacement_ids_missing')
    expected_sources = {'ChartQA', 'ChartQAPro', 'TableVQA-Bench', 'MMTU'}
    if any((pool[i]['source'] not in expected_sources for i in REPLACEMENTS)):
        raise ValueError('replacement_source_failure')
    if OUTLIERS & REPLACEMENTS:
        raise ValueError('replacement_overlaps_outlier')
    rows = [r for r in selected if r['id'] not in OUTLIERS] + [pool[i] for i in sorted(REPLACEMENTS)]
    if len(rows) != 128 or len({r['id'] for r in rows}) != 128:
        raise ValueError('selection_size_failure')
    if any((r['id'] in OUTLIERS for r in rows)):
        raise ValueError('outlier_survived')
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name in ('candidates.jsonl', 'selection_lock.json', 'source_manifest.json'):
            continue
        target = output / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path, target)
    (output / 'candidates.jsonl').write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in rows)))
    identity = read_jsonl(source / 'selected_reserve_identity.jsonl')
    for entry in identity:
        if entry['id'] in OUTLIERS:
            entry['membership'] = 'excluded_answer_format'
        elif entry['id'] in REPLACEMENTS:
            entry['membership'] = 'selected'
    (output / 'selected_reserve_identity.jsonl').write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in identity)))
    summary = json.loads((source / 'sampling_summary.json').read_text())
    summary['source_counts'] = dict(Counter((r['source'] for r in rows)))
    summary['quotas'] = dict(Counter((r['stratum'] for r in rows)))
    summary['selection_version'] = 'normal_qa_v4'
    (output / 'sampling_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    coverage = {'selection_version': 'normal_qa_v4', 'selected_cases': len(rows), 'coverage': {'selected': dict(Counter((t for r in identity if r['membership'] == 'selected' for t in r['diversity_tokens'])))}, 'note': 'Source-feature coverage; actual rendered chart classes require visual review.'}
    (output / 'diversity_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2))
    for r in rows:
        (output / 'assets' / r['id']).mkdir(parents=True, exist_ok=True)
    (output / 'replacement_manifest.json').write_text(json.dumps({'version': 'normal_qa_v4', 'removed_outliers': sorted(OUTLIERS), 'added_replacements': sorted(REPLACEMENTS), 'policy': 'replace all v1 long_text/structured normalized answer types with diverse unused chart/table candidates; API re-normalization is required before admission.', 'source_quota_preserved': False, 'source_counts_v1': {s: sum((r['source'] == s for r in selected)) for s in sorted({r['source'] for r in selected})}, 'source_counts_v2': {s: sum((r['source'] == s for r in rows)) for s in sorted({r['source'] for r in rows})}}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'removed': len(OUTLIERS), 'added': len(REPLACEMENTS), 'rows': len(rows)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
