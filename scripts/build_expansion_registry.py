pass
import argparse
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
import pyarrow.parquet as pq
from PIL import Image
from profile_and_sample import OUT, ROOT, main as build_catalog

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def normalized(text):
    return re.sub('\\s+', ' ', unicodedata.normalize('NFKC', str(text))).strip().casefold()
TASK_PATTERNS = {'counting': 'how\\s*many|number of', 'sum_aggregation': '\\bsum\\b|\\btotal\\b|combined', 'difference': 'difference|how much more|how many more|net change', 'ratio_percentage': 'percent|ratio|proportion|fraction|\\btimes\\b', 'mean_median': 'average|\\bmean\\b|median', 'extrema_ranking': 'highest|lowest|largest|smallest|greatest|least|maximum|minimum|second|third|rank|most ', 'comparison': 'compar|greater|smaller|higher|lower|more than|less than|exceed|above|below', 'trend_change': 'trend|increase|decrease|decline|growth|change|monotonic|stabili', 'intersection_threshold': 'intersect|cross|threshold|surpass|first exceed', 'correlation_distribution': 'correlat|distribution|density|spread|variance|contour', 'spatial_subplot': 'subplot|left|right|top|bottom|region|area|panel', 'legend_series_grounding': 'legend|curve|line|label|color|colour|blue|red|green|black', 'conditional_filter': 'where|whose|which.*(have|has)|greater than|less than|at least|but.*not|when ', 'temporal_lookup': '\\byear\\b|\\bmonth\\b|\\bday\\b|\\b19\\d{2}\\b|\\b20\\d{2}\\b|before|after', 'approximate_reading': 'approximat|estimat|roughly'}

def answer_hint(answer):
    values = answer if isinstance(answer, list) else [answer]
    if len(values) > 1:
        return 'multiple_values_or_aliases'
    text = str(values[0]).strip()
    if re.fullmatch('[\\s$€£¥+−\\-\\d.,%eE]+', text):
        return 'numeric'
    if len(text.split()) > 30:
        return 'long_answer_needs_canonicalization'
    return 'short_text_or_expression'

def native_ids(r):
    raw = r['raw_record']
    if r['source'] == 'CharXiv':
        return {'paper_id': raw['original_id'], 'figure_path': raw['original_figure_path'], 'slot': r['slot']}
    if r['source'] == 'TableVQA-Bench':
        return {'subset': r['split'], 'qa_id': raw['qa_id']}
    if r['source'] == 'Visual-TableQA':
        return {'table_id': raw['table_id']}
    if r['source'] == 'MMTU':
        meta = json.loads(raw['metadata'])
        return {'dataset': raw['dataset'], 'task': raw['task'], 'test_case': meta['test_case']}
    return {}

def enrich(r, image_info, manifest):
    raw = r['raw_record']
    tables = [raw[k] for k in ('text_markdown_table', 'text_html_table') if raw.get(k)]
    if r['source'] == 'MMTU':
        tables = re.findall('(?m)(?:^[ \\t]*\\|[^\\n]*(?:\\n|$))+', raw['prompt'])
    exact_table_hash = digest(tables) if tables else None
    normalized_table_hash = digest([normalized(t) for t in tables]) if tables else None
    natives = native_ids(r)
    image_hash = image_info.get('image_sha256')
    content_visual = image_hash or exact_table_hash
    if content_visual is None:
        raise ValueError(f"Missing identity anchor: {r['id']}")
    identity_anchor = natives or {'visual_content_hash': content_visual}
    base_id = 'qa_' + digest([r['source'], identity_anchor, r['question'], r['slot']])
    record_version = digest([r['raw_record'], image_hash])
    tasks = [name for (name, pat) in TASK_PATTERNS.items() if re.search(pat, r['question'], re.I)]
    native_features = {k: raw[k] for k in ('category', 'difficulty', 'num_subplots', 'year', 'human_or_machine') if k in raw}
    if r['source'] == 'MMTU':
        native_features['underlying_dataset'] = raw['dataset']
        native_features['qtype'] = json.loads(raw['metadata']).get('qtype')
    if r['source'] == 'TableVQA-Bench':
        native_features['subset'] = r['split']
    measured = dict(image_info)
    measured['table_text_chars'] = sum(map(len, tables))
    measured['table_block_count'] = len(tables)
    measured['question_chars'] = len(r['question'])
    measured['markdown_line_count'] = sum((len(t.splitlines()) for t in tables if t.lstrip().startswith('|')))
    tokens = ['source=' + r['source'], 'answer=' + answer_hint(r['answer'])]
    tokens += ['task=' + task for task in tasks or ['lookup_or_other']]
    for k in ('category', 'difficulty', 'underlying_dataset', 'subset'):
        if native_features.get(k) is not None:
            tokens.append(f'{k}={native_features[k]}')
    panels = raw.get('num_subplots')
    if panels is not None:
        tokens.append('subplot_bin=' + ('1' if panels == 1 else '2-4' if panels <= 4 else '5-9' if panels <= 9 else '10+'))
    if image_info:
        ratio = image_info['width'] / image_info['height']
        tokens.append('aspect=' + ('wide' if ratio > 1.5 else 'tall' if ratio < 0.67 else 'balanced'))
    if tables:
        size = sum(map(len, tables))
        tokens.append('table_chars=' + ('<1000' if size < 1000 else '1000-4999' if size < 5000 else '5000+'))
    evidence = manifest[r['file']]
    dedup_keys = ['visual=' + content_visual]
    if normalized_table_hash:
        dedup_keys.append('table=' + normalized_table_hash)
    if r['source'] == 'CharXiv':
        dedup_keys.append('paper=' + raw['original_id'])
    return {'id': r['id'], 'base_id': base_id, 'identity_schema': 'mvisqa-identity-v1', 'record_version_sha256': record_version, 'native_ids': natives, 'provenance': {'repo': evidence['repo'], 'revision': evidence['revision'], 'file_sha256_at_download': evidence['sha256'], 'file': r['file'], 'row_index': r['row_index'], 'slot': r['slot']}, 'source': r['source'], 'stratum': r['stratum'], 'question': r['question'], 'fingerprints': {'image_sha256': image_hash, 'table_exact_sha256': exact_table_hash, 'table_normalized_sha256': normalized_table_hash, 'question_normalized_sha256': digest(normalized(r['question'])), 'visual_question_sha256': digest([content_visual, normalized(r['question'])])}, 'dedup_group_keys': dedup_keys, 'features': {'native': native_features, 'measured': measured, 'inferred': {'task_tags': tasks, 'answer_type': answer_hint(r['answer']), 'method': 'regex-v1; unverified; overlapping labels'}, 'verified': {'primary_task': None, 'visual_type': None, 'difficulty': None, 'domain': None}}, 'diversity_tokens': tokens, 'review_status': 'pending'}

def propose_extension(registry, add):
    selected = [r for r in registry if r['membership'] == 'selected']
    counts = Counter((t for r in selected for t in r['diversity_tokens']))
    used_ids = {r['base_id'] for r in selected}
    used_groups = {g for r in selected for g in r['dedup_group_keys']}
    weights = Counter((r['source'] for r in selected))
    desired = {s: add * n // len(selected) for (s, n) in weights.items()}
    for s in sorted(weights, key=lambda s: (-(add * weights[s] % len(selected)), s))[:add - sum(desired.values())]:
        desired[s] += 1
    proposals = []
    for source in sorted(desired, key=lambda s: (-weights[s], s)):
        for _ in range(desired[source]):
            eligible = [r for r in registry if r['source'] == source and r['membership'] != 'selected' and (r['base_id'] not in used_ids) and (not used_groups.intersection(r['dedup_group_keys']))]
            if not eligible:
                raise ValueError(f'No eligible unused records for {source}')

            def score(r):
                return sum(((2 if t.startswith('task=') else 1) / (1 + counts[t]) for t in r['diversity_tokens']))
            best = min(eligible, key=lambda r: (-score(r), r['base_id']))
            proposals.append({'base_id': best['base_id'], 'id': best['id'], 'source': source, 'score': score(best), 'diversity_tokens': best['diversity_tokens'], 'previously_uncovered_tokens': [t for t in best['diversity_tokens'] if counts[t] == 0], 'status': 'proposed_not_added'})
            used_ids.add(best['base_id'])
            used_groups.update(best['dedup_group_keys'])
            counts.update(best['diversity_tokens'])
    return proposals

def write_jsonl(path, records):
    with path.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

def main(add=32):
    if add < 0 or (128 + add) * 31 > 5000:
        raise ValueError('Current 31-config budget supports 0–33 additional base QA.')
    build_catalog(catalog_only=True)
    paths = [OUT / 'candidates.jsonl', OUT / 'reserves.jsonl']
    original_bytes = {p: p.read_bytes() for p in paths}
    membership = {json.loads(line)['id']: label for (p, label) in zip(paths, ['selected', 'reserve']) for line in p.read_text().splitlines()}
    pool = [json.loads(line) for line in (OUT / 'eligible_pool.jsonl').read_text().splitlines()]
    manifest = {r['local_path']: r for r in json.loads((ROOT / 'data/download_manifest.json').read_text())['verified_files']}
    image_info = {}
    byfile = defaultdict(set)
    for r in pool:
        if r['source'] != 'MMTU':
            byfile[r['file']].add(r['row_index'])
    for (filename, wanted) in sorted(byfile.items()):
        row_index = 0
        for batch in pq.ParquetFile(ROOT / filename).iter_batches(batch_size=16, columns=['image']):
            for raw in batch.column(0).to_pylist():
                if row_index in wanted:
                    blob = raw.get('bytes') if isinstance(raw, dict) else raw
                    with Image.open(io.BytesIO(blob)) as im:
                        image_info[filename, row_index] = {'image_sha256': hashlib.sha256(blob).hexdigest(), 'width': im.width, 'height': im.height}
                row_index += 1
        print(f'Image fingerprints: {filename}', flush=True)
    registry = []
    for r in pool:
        entry = enrich(r, image_info.get((r['file'], r['row_index']), {}), manifest)
        entry['membership'] = membership.get(r['id'], 'available')
        registry.append(entry)
    registry.sort(key=lambda r: (r['base_id'], r['id']))
    assert set(membership) <= {r['id'] for r in registry}
    locked = [r for r in registry if r['membership'] != 'available']
    snapshot_hash = digest([(r['base_id'], r['record_version_sha256'], r['membership']) for r in locked])
    lock_path = OUT / 'selection_lock.json'
    lock = {'schema': 'selection-lock-v1', 'snapshot_hash': snapshot_hash, 'input_file_hashes': {p.name: hashlib.sha256(v).hexdigest() for (p, v) in original_bytes.items()}, 'selected_base_ids': [r['base_id'] for r in locked if r['membership'] == 'selected'], 'reserve_base_ids': [r['base_id'] for r in locked if r['membership'] == 'reserve']}
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError('Selection changed: create a new version instead of replacing the existing lock.')
    write_jsonl(OUT / 'expansion_registry.jsonl', registry)
    write_jsonl(OUT / 'selected_reserve_identity.jsonl', locked)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2))
    proposals = propose_extension(registry, add)
    write_jsonl(OUT / f'extension_proposal_{add}.jsonl', proposals)
    groups = defaultdict(list)
    for r in registry:
        keys = ['base_id=' + r['base_id']] + r['dedup_group_keys']
        keys.append('question_only=' + r['fingerprints']['question_normalized_sha256'])
        for key in keys:
            groups[key].append({'id': r['id'], 'base_id': r['base_id'], 'source': r['source'], 'membership': r['membership'], 'record_version_sha256': r['record_version_sha256']})
    (OUT / 'duplicate_groups.json').write_text(json.dumps({k: v for (k, v) in groups.items() if len(v) > 1}, ensure_ascii=False, indent=2))
    coverage = {label: dict(Counter((t for r in registry if r['membership'] == label for t in r['diversity_tokens']))) for label in ['selected', 'reserve', 'available']}
    duplicate_ids = Counter((r['base_id'] for r in registry))
    report = {'registry_records': len(registry), 'membership': dict(Counter((r['membership'] for r in registry))), 'duplicate_base_ids_in_pool': {k: n for (k, n) in duplicate_ids.items() if n > 1}, 'coverage': coverage, 'proposed_additions': len(proposals), 'planned_configs_if_accepted': (128 + len(proposals)) * 31, 'selection_unchanged': all((p.read_bytes() == v for (p, v) in original_bytes.items())), 'limitations': ['Task tags are overlapping keyword hints, not verified cognition labels.', 'Image SHA detects identical bytes only; re-encoded and semantic duplicates need review.', 'No source revision or original content was modified.', 'Proposal is not committed to the selection; reserves remain eligible for promotion.']}
    (OUT / 'diversity_coverage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ['registry_records', 'membership', 'proposed_additions', 'selection_unchanged']}))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--add', type=int, default=32, help='Propose additions without modifying the current selection')
    main(parser.parse_args().add)
