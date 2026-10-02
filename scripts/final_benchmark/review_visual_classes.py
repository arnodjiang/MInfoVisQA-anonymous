pass
import argparse
import json
from collections import Counter
from pathlib import Path
from scripts.final_benchmark.classify_visuals import TAXONOMY, PROMPT, validate
from scripts.final_benchmark.api import API, read, save, digest, now
from scripts.openai_config import load
ROOT = Path(__file__).resolve().parents[2]

def review(dataset, work):
    (dataset, work) = (Path(dataset), Path(work))
    progress = read(work / 'progress.json')
    if progress['state'] not in ('finished', 'failed'):
        raise ValueError('Finish initial classification first')
    config = load(ROOT)
    config['OPENAI_MODEL'] = 'gpt-6-astra'
    api = API(work, config, retry_failed=True)
    revised = []
    rows = {r['case_id']: r for r in map(json.loads, (dataset / 'validation_release/val.candidates.jsonl').read_text().splitlines()) if r['image_language'] == 'en' and r['query_language'] == 'en'}
    for (cid, error) in progress.get('errors', {}).items():
        row = rows[cid]
        (value, key) = api.call('visual_classification_validation_retry', cid, PROMPT, {'case_id': cid, 'image_sha256': row['image_sha256'], 'taxonomy': TAXONOMY, 'validation_feedback': error + '. Copy case_id exactly from this request.'}, image=dataset / 'cases' / cid / 'images/en.png', max_tokens=2000)
        validate(value, cid)
        save(work / 'classifications' / (cid + '.json'), {**value, 'image_language': 'en', 'image_sha256': row['image_sha256'], 'model': 'gpt-6-astra', 'request_sha256': key, 'prompt_sha256': digest(PROMPT), 'taxonomy_version': TAXONOMY['version']})
    for path in sorted((work / 'classifications').glob('*.json')):
        old = read(path)
        if old.get('reviewed_with_expanded_taxonomy'):
            continue
        if old['visual_kind'] != 'Other Scientific Visualization' and old['confidence'] >= 0.9:
            continue
        cid = old['case_id']
        image = dataset / 'cases' / cid / 'images/en.png'
        (value, key) = api.call('visual_classification_review', cid, PROMPT, {'case_id': cid, 'image_sha256': old['image_sha256'], 'taxonomy': TAXONOMY, 'review_instruction': 'Re-examine the image using this expanded category list. Prefer a supported specific type; do not force specificity if the geometry is ambiguous.'}, image=image, max_tokens=2000)
        validate(value, cid)
        save(work / 'classification_history' / (cid + '.json'), old)
        updated = {**old, **value, 'request_sha256': key, 'reviewed_with_expanded_taxonomy': True, 'taxonomy_sha256': digest(TAXONOMY), 'reviewed_at': now()}
        save(path, updated)
        revised.append({'case_id': cid, 'before': old['visual_kind'], 'after': value['visual_kind']})
    values = [read(p) for p in (work / 'classifications').glob('*.json')]
    summary = read(work / 'summary.json')
    summary['categories'] = dict(Counter((v['visual_kind'] for v in values)))
    summary['secondary_review'] = revised
    usage = Counter()
    attempts = list((work / 'api').glob('*/*/*/attempt_*.json'))
    for p in attempts:
        for (k, v) in read(p).get('usage', {}).items():
            if k in ('input_tokens', 'output_tokens', 'total_tokens') and isinstance(v, int):
                usage[k] += v
    summary.update(api_attempts=len(attempts), usage=dict(usage))
    if len(values) != len(rows):
        raise ValueError('Missing final classifications')
    summary.update(state='finished', cases=len(values), errors={})
    summary['low_confidence_cases'] = [v['case_id'] for v in values if v['confidence'] < 0.8]
    summary['review_caution_cases'] = [v['case_id'] for v in values if v['confidence'] < 0.9]
    save(work / 'progress.json', {'state': 'finished', 'updated_at': now(), 'expected': len(rows), 'completed': len(values), 'failed': 0, 'errors': {}})
    save(work / 'summary.json', summary)
    print(summary)
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    review(a.dataset, a.work)
