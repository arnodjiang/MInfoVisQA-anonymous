pass
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
from scripts.final_benchmark.api import API, read, save, digest, now
from scripts.final_benchmark.provenance import sha256
from scripts.openai_config import load
ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = read(ROOT / 'configs/visual_taxonomy.json')
PROMPT = Path(__file__).with_name('visual_classification_prompt.txt').read_text()
CATEGORIES = set(TAXONOMY['chart_categories']) | set(TAXONOMY['table_categories'])
TABLE_SPANS = {'Simple Table': (False, False), 'Row-Spanning Table': (True, False), 'Column-Spanning Table': (False, True), 'Mixed-Spanning Table': (True, True)}

def validate(result, case_id):
    if result.get('case_id') != case_id or result.get('visual_kind') not in CATEGORIES:
        raise ValueError('Invalid case/category')
    if not isinstance(result.get('secondary_types'), list) or any((v not in CATEGORIES or v == result['visual_kind'] for v in result['secondary_types'])):
        raise ValueError('Invalid secondary categories')
    if result.get('layout') not in {'single_panel', 'same_type_small_multiples', 'mixed_panels', 'integrated_composite'}:
        raise ValueError('Invalid layout')
    if type(result.get('panel_count')) is not int or result['panel_count'] < 1:
        raise ValueError('Invalid panel count')
    if type(result.get('confidence')) not in (float, int) or not 0 <= result['confidence'] <= 1:
        raise ValueError('Invalid confidence')
    if not isinstance(result.get('evidence'), str) or not result['evidence'].strip():
        raise ValueError('Missing evidence')
    if not isinstance(result.get('ambiguities'), list):
        raise ValueError('Missing ambiguities')
    structure = result.get('table_structure')
    if result['visual_kind'] in TABLE_SPANS:
        if not isinstance(structure, dict) or any((type(structure.get(k)) is not bool for k in ('row_spans', 'column_spans'))):
            raise ValueError('Invalid table structure')
        if (structure['row_spans'], structure['column_spans']) != TABLE_SPANS[result['visual_kind']]:
            raise ValueError('Inconsistent table category')
    return result

def classify(dataset, work, workers=2):
    (dataset, work) = (Path(dataset).resolve(), Path(work).resolve())
    work.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (dataset / 'validation_release/val.candidates.jsonl').read_text().splitlines()]
    representatives = {r['case_id']: r for r in rows if r['image_language'] == 'en' and r['query_language'] == 'en'}
    if len(representatives) != len({r['case_id'] for r in rows}):
        raise ValueError('Missing English representative')
    config = load(ROOT)
    config['OPENAI_MODEL'] = 'gpt-6-astra'
    api = API(work, config, retry_failed=True)
    mutex = threading.Lock()
    done = {}
    errors = {}

    def status():
        save(work / 'progress.json', {'state': 'running', 'updated_at': now(), 'expected': len(representatives), 'completed': len(done), 'failed': len(errors), 'errors': errors})
    status()

    def one(cid, row):
        image = (dataset / 'validation_release' / row['image_path']).resolve()
        if sha256(image) != row['image_sha256']:
            raise ValueError('Image changed')
        payload = {'case_id': cid, 'image_sha256': row['image_sha256'], 'taxonomy': TAXONOMY}
        (value, key) = api.call('visual_classification', cid, PROMPT, payload, image=image, max_tokens=2000)
        validate(value, cid)
        result = {**value, 'image_language': 'en', 'image_sha256': row['image_sha256'], 'model': config['OPENAI_MODEL'], 'request_sha256': key, 'prompt_sha256': digest(PROMPT), 'taxonomy_version': TAXONOMY['version']}
        save(work / 'classifications' / (cid + '.json'), result)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(one, cid, row): cid for (cid, row) in representatives.items()}
        for future in as_completed(jobs):
            cid = jobs[future]
            try:
                done[cid] = future.result()
            except Exception as exc:
                errors[cid] = str(exc)
            with mutex:
                status()
            print('Classified', len(done), '/', len(representatives), 'failed', len(errors), flush=True)
    usage = Counter()
    attempts = list((work / 'api').glob('*/*/*/attempt_*.json'))
    for p in attempts:
        for (k, v) in read(p).get('usage', {}).items():
            if k in ('input_tokens', 'output_tokens', 'total_tokens') and isinstance(v, int):
                usage[k] += v
    save(work / 'summary.json', {'state': 'failed' if errors else 'finished', 'cases': len(done), 'errors': errors, 'categories': dict(Counter((v['visual_kind'] for v in done.values()))), 'low_confidence_cases': [cid for (cid, v) in done.items() if v['confidence'] < 0.8], 'api_attempts': len(attempts), 'usage': dict(usage), 'model': 'gpt-6-astra'})
    save(work / 'progress.json', {'state': 'failed' if errors else 'finished', 'updated_at': now(), 'expected': len(representatives), 'completed': len(done), 'failed': len(errors), 'errors': errors})
    if errors:
        raise RuntimeError('Classification incomplete; inspect cached failures before publishing')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    classify(a.dataset, a.work, a.workers)
