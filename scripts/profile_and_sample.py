pass
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('MVISQA_CATALOG_OUT', str(ROOT / 'data/processed/normal_qa_v3'))).resolve()
SEED = 20260909

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def infer_task(question):
    pass
    q = question.lower()
    rules = [('arithmetic', 'average|sum |total|difference|ratio|percent|how much more|how many more'), ('comparison', 'highest|lowest|largest|smallest|maximum|minimum|greater|less than|most |least '), ('trend', 'trend|increase|decrease|decline|growth|change|correlat'), ('counting', 'how many|number of'), ('structure', 'axis|axes|legend|subplot|title|label')]
    return next((name for (name, pattern) in rules if re.search(pattern, q)), 'lookup_or_other')

def normal_qa(question, answer):
    if not question or answer is None:
        return False
    values = answer if isinstance(answer, list) else [answer]
    banned = {'yes', 'no', 'true', 'false', 'unanswerable', 'not applicable', 'n/a', 'unknown', 'cannot be determined'}
    if not values or any((str(a).strip().lower().rstrip('.') in banned for a in values)):
        return False
    if re.match('\\s*(is|are|was|were|do|does|did|has|have|had|can|could|would|will|should)\\b', question, re.I):
        return False
    if re.search('\\b(if|hypothetically|suppose|assuming)\\b|\\n\\s*[A-Da-d][).]', question, re.I):
        return False
    if re.search('\\b(explain|describe|why|imply|significance|demonstrate|categorize)\\b|general observation|what characteristics|what leaf characteristics|what specific feature|are these mappings', question, re.I):
        return False
    return True

def main(catalog_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    pools = defaultdict(list)
    inventory = {}
    for directory in sorted((ROOT / 'data/raw').iterdir()):
        if not directory.is_dir():
            continue
        source = directory.name
        stats = {'rows': 0, 'files': [], 'native_counts': defaultdict(Counter)}
        for path in sorted(directory.rglob('*.parquet')):
            parquet = pq.ParquetFile(path)
            cols = [c for c in parquet.schema_arrow.names if c != 'image']
            rows = parquet.read(columns=cols).to_pylist()
            rel = str(path.relative_to(ROOT))
            split = path.stem.split('-')[0]
            stats['files'].append({'path': rel, 'rows': len(rows), 'columns': parquet.schema_arrow.names})
            stats['rows'] += len(rows)
            for (row_index, r) in enumerate(rows):
                for key in ('task', 'dataset', 'Question Type', 'difficulty', 'category', 'human_or_machine', 'num_subplots'):
                    if key in r:
                        stats['native_counts'][key][str(r[key])] += 1
                base = {'source': source, 'file': rel, 'row_index': row_index, 'split': split, 'raw_record': r, 'status': 'needs_review', 'visual_type': None, 'verified_task': None, 'verified_difficulty': None}
                variants = []
                if source == 'CharXiv':
                    if r['figure_path'] == 'images/0.jpg':
                        continue
                    variants = [('reasoning', r['reasoning_q'], r['reasoning_a'])]
                elif source == 'ChartQAPro':
                    if r['Question Type'] != 'Factoid' or len(r['Question']) != 1:
                        continue
                    if len(r['Question']) != len(r['Answer']):
                        raise ValueError(f'Unaligned turns: {rel}:{row_index}')
                    variants = [(f"turn_{len(r['Question']) - 1}", r['Question'][-1], r['Answer'][-1])]
                elif source == 'ChartQA':
                    variants = [('qa', r['query'], r['label'])]
                elif source == 'TableVQA-Bench':
                    if split == 'vtabfact':
                        continue
                    variants = [('qa', r['question'], r['gt'])]
                elif source == 'Visual-TableQA':
                    if split != 'test':
                        continue
                    variants = [('qa', r['question'], r['answer'])]
                elif source == 'MMTU':
                    if r['task'] != 'Table-QA':
                        continue
                    metadata = json.loads(r['metadata'])
                    answer = metadata.get('label', metadata.get('output'))
                    variants = [('qa', metadata.get('question'), answer)]
                for (slot, question, answer) in variants:
                    if not normal_qa(question, answer):
                        continue
                    item = dict(base, slot=slot, question=question, answer=answer)
                    item['id'] = hashlib.sha256(f'{rel}:{row_index}:{slot}'.encode()).hexdigest()[:20]
                    item['task_hint'] = infer_task(question or '')
                    item['task_hint_method'] = 'keyword_heuristic_not_verified'
                    item['review_flags'] = ['verify_answer', 'verify_task', 'verify_visual', 'check_semantic_duplicates']
                    if source == 'CharXiv':
                        kind = 'reasoning' if slot == 'reasoning' else 'descriptive'
                        item['stratum'] = f"{source}/{r['category']}/{kind}"
                        item['group_hint'] = r['original_id']
                        if kind == 'descriptive':
                            item['question_template_id'] = r['descriptive_q' + slot.split('_')[-1]]
                            item['review_flags'].append('resolve_official_template_and_subplot_scope')
                            item['task_hint'] = 'unanswerable' if answer == 'Not Applicable' else 'structure'
                    elif source == 'TableVQA-Bench':
                        item['stratum'] = f'{source}/{split}'
                        table = r['text_markdown_table'] or r['text_html_table']
                        item['group_hint'] = hashlib.sha256(table.encode()).hexdigest() if table else r['qa_id']
                        if not table:
                            item['review_flags'].append('missing_structured_table')
                        if split == 'vtabfact':
                            item['task_hint'] = 'fact_verification'
                    elif source == 'ChartQAPro':
                        item['stratum'] = f"{source}/{r['Question Type']}"
                        item['context_turns'] = [{'question': q, 'answer': a} for (q, a) in zip(r['Question'][:-1], r['Answer'][:-1])]
                        item['scored_turn_count'] = 1
                        item['review_flags'].append('decide_paragraph_input_policy')
                    elif source == 'Visual-TableQA':
                        item['stratum'] = f"{source}/{r['difficulty']}"
                        item['group_hint'] = r['table_id']
                        item['review_flags'].append('extract_canonical_answer_from_rationale')
                    elif source == 'ChartQA':
                        item['stratum'] = f"{source}/{r['human_or_machine']}"
                    else:
                        item['stratum'] = f"{source}/Table-QA/{r['dataset']}"
                        item['review_flags'] += ['render_tables', 'remove_text_table_bypass', 'check_external_knowledge_requirement']
                    pools[item['stratum']].append(item)
        stats['native_counts'] = {k: dict(v) for (k, v) in stats['native_counts'].items()}
        inventory[source] = stats
    if catalog_only:
        with (OUT / 'eligible_pool.jsonl').open('w') as stream:
            for stratum in sorted(pools):
                for item in pools[stratum]:
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'Catalogued {sum(map(len, pools.values()))} eligible records; selection unchanged.')
        return
    quotas = {f'TableVQA-Bench/{s}': n for (s, n) in [('fintabnetqa', 8), ('vwtq', 6), ('vwtq_syn', 6)]}
    quotas.update({f'CharXiv/{d}/{k}': n for d in inventory['CharXiv']['native_counts']['category'] for (k, n) in [('reasoning', 6)]})
    quotas.update({'ChartQAPro/Factoid': 24})
    quotas.update({'Visual-TableQA/standard': 14, 'Visual-TableQA/challenging': 6, 'ChartQA/0': 4, 'ChartQA/1': 4})
    quotas.update({f'MMTU/Table-QA/{t}': n for (t, n) in [('FinQA', 3), ('WikiQA', 3), ('TableBench', 2)]})
    (selected, reserves) = ([], [])
    seen_groups = set()
    rng = random.Random(SEED)
    for (stratum, quota) in sorted(quotas.items()):
        rows = sorted(pools[stratum], key=lambda x: x['id'])
        rng.shuffle(rows)
        hints = Counter()
        chosen = []
        while len(chosen) < quota * 2:
            eligible = [r for r in rows if (r['source'], r.get('group_hint', (r['file'], r['row_index']))) not in seen_groups]
            if not eligible:
                raise ValueError(f'Insufficient distinct groups: {stratum}')
            r = min(eligible, key=lambda x: hints[x['task_hint']])
            rows.remove(r)
            hints[r['task_hint']] += 1
            seen_groups.add((r['source'], r.get('group_hint', (r['file'], r['row_index']))))
            chosen.append(r)
        selected.extend(chosen[:quota])
        reserves.extend(chosen[quota:])
    assert len(selected) == 128 and len(reserves) == 128
    assert len({x['id'] for x in selected + reserves}) == 256
    outputs = {name: ''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in records)) for (name, records) in [('candidates.jsonl', selected), ('reserves.jsonl', reserves)]}
    if (OUT / 'selection_lock.json').exists():
        for (name, text) in outputs.items():
            if (OUT / name).read_bytes() != text.encode():
                raise ValueError('Selection is locked. Use a new output version for changes; do not overwrite existing IDs.')
    for (name, text) in outputs.items():
        (OUT / name).write_text(text)
    dump('inventory.json', inventory)
    dump('sampling_summary.json', {'seed': SEED, 'quotas': quotas, 'base_candidates': len(selected), 'reserve_candidates': len(reserves), 'assumed_languages': 11, 'configs_per_base': 31, 'planned_configs': len(selected) * 31, 'generated_multilingual_items': 0, 'source_counts': dict(Counter((r['source'] for r in selected))), 'task_hints_unverified': dict(Counter((r['task_hint'] for r in selected))), 'unresolved_question_templates': sum((r['question'] is None for r in selected)), 'eligible_pool_by_stratum': {k: len(v) for (k, v) in sorted(pools.items())}, 'policy': 'single-turn answerable open-ended QA only; no CharXiv descriptive, boolean, MCQ, conversation, hypothetical, or non-QA tasks', 'notes': ['Native source strata are enforced; cognitive task quotas are not yet enforced.', 'Group hints do not establish cross-source image or semantic deduplication.', 'All selected targets are single-turn QA; empty context only.', 'No candidate is certified as evaluation-ready.']})
    print((OUT / 'sampling_summary.json').read_text())
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog-only', action='store_true', help='Index eligible pool without changing selections')
    main(catalog_only=parser.parse_args().catalog_only)
