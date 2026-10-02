pass
import argparse
from collections import defaultdict
import json
from pathlib import Path
import unicodedata
import random

def normalize(s):
    return unicodedata.normalize('NFC', s).strip()

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def score(refs, preds, bootstrap=2000, seed=42):
    if len({r['id'] for r in refs}) != len(refs):
        raise ValueError('duplicate reference IDs')
    if len({r['id'] for r in preds}) != len(preds):
        raise ValueError('duplicate prediction IDs')
    index = {r['id']: r['prediction'] for r in preds}
    if set(index) - {r['id'] for r in refs}:
        raise ValueError('unknown prediction IDs')
    if any((not isinstance(s, str) for s in index.values())):
        raise ValueError('predictions must be strings')
    scores = {r['id']: int(r['id'] in index and normalize(index[r['id']]) == normalize(r['answer'])) for r in refs}
    groups = {}
    for field in ('source', 'base_id', 'query_language', 'image_language', 'answer_language', 'configuration', 'visual_kind'):
        buckets = defaultdict(list)
        for r in refs:
            buckets[r[field]].append(scores[r['id']])
        groups[field] = {k: {'accuracy': mean(v), 'n': len(v)} for (k, v) in buckets.items()}
    for field in ('difficulty', 'table_type', 'chart_types', 'task_tags'):
        buckets = defaultdict(list)
        for r in refs:
            if field == 'difficulty':
                keys = [r.get('difficulty', {}).get('label', 'unknown')]
            elif field == 'table_type':
                keys = [r['table_structure']['type']] if r.get('table_structure') else []
            else:
                keys = r.get(field, [])
            for key in keys:
                buckets[key].append(scores[r['id']])
        groups[field] = {k: {'accuracy': mean(v), 'n': len(v)} for (k, v) in buckets.items()}
    matched = {(r['base_id'], r['image_language'], r['query_language']): scores[r['id']] for r in refs}
    gaps = {}
    for q in ('en', 'zh'):
        differences = []
        for ((b, v, l), s) in matched.items():
            if l == v and v != q and ((b, v, q) in matched):
                differences.append(s - matched[b, v, q])
        gaps[q] = {'monolingual_minus_cross_accuracy': mean(differences), 'matched_pairs': len(differences)}
    parent = {r['base_id']: r['base_id'] for r in refs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen = {}
    for r in refs:
        for key in r.get('split_group_keys', []):
            if key in seen:
                parent[find(r['base_id'])] = find(seen[key])
            else:
                seen[key] = r['base_id']
    clusters = defaultdict(list)
    for r in refs:
        clusters[find(r['base_id'])].append(scores[r['id']])
    arrays = list(clusters.values())
    rng = random.Random(seed)
    samples = []
    if arrays:
        for _ in range(bootstrap):
            draw = rng.choices(arrays, k=len(arrays))
            samples.append(sum(map(sum, draw)) / sum(map(len, draw)))
    samples.sort()
    ci = [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]] if samples else None
    return {'metric': 'NFC + outer-whitespace normalized strict EM', 'n': len(refs), 'missing_predictions': len(refs) - len(index), 'micro_EM': mean(list(scores.values())), 'macro_base_EM': mean([x['accuracy'] for x in groups['base_id'].values()]), 'macro_source_EM': mean([x['accuracy'] for x in groups['source'].values()]), 'groups': groups, 'matched_cross_language_gaps': gaps, 'cluster_bootstrap': {'clusters': len(arrays), 'resamples': bootstrap, 'seed': seed, 'micro_EM_95pct_CI': ci}, 'language_compliance': 'not measured; requires language-aware annotation; numeric responses excluded'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--references', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    refs = [json.loads(s) for s in Path(a.references).read_text().splitlines()]
    preds = [json.loads(s) for s in Path(a.predictions).read_text().splitlines()]
    result = score(refs, preds)
    Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
if __name__ == '__main__':
    main()
