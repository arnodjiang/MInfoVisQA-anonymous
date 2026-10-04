pass
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
from scripts.evaluation.context_input import VERSION
from scripts.evaluation.score import LANGUAGES
from scripts.final_benchmark.api import API, digest, now, read, save
from scripts.final_benchmark.provenance import source_binding, sha256
from scripts.openai_config import load
from scripts.final_benchmark.clean_release import clean_reference, copy_current, write_current_views, verify_clean, VERSION as RELEASE_VERSION
ROOT = Path(__file__).resolve().parents[2]
PROMPT = Path(__file__).with_name('context_translation_prompt.txt').read_text()
NUMBERS = re.compile('[+−-]?\\d+(?:[.,]\\d+)*%?')
TOKENS = re.compile('\\[\\[N_\\d{6}\\]\\]')

def source_paragraphs(source):
    candidate = source['candidate']
    if candidate['source'] == 'ChartQAPro':
        paragraph = candidate['raw_record'].get('Paragraph') or ''
        if not paragraph.strip():
            return []
        return [{'id': 'document_%03d' % i, 'text': text, 'source_field': 'Paragraph', 'source_index': i} for (i, text) in enumerate(paragraph.split('\n'))]
    if candidate['source'] != 'MMTU' or candidate['raw_record']['dataset'] != 'FinQA':
        return []
    metadata = json.loads(candidate['raw_record']['metadata'])
    return [{'id': '%s_%03d' % (key.split('_')[0], i), 'text': text, 'source_field': 'metadata.' + key, 'source_index': i} for key in ('pre_text', 'post_text') for (i, text) in enumerate(metadata.get(key) or [])]

def protect(paragraphs):
    tokens = {}

    def replace(match):
        token = '[[N_%06d]]' % len(tokens)
        tokens[token] = match.group()
        return token
    protected = [{'id': p['id'], 'text': NUMBERS.sub(replace, p['text'])} for p in paragraphs]
    return (protected, tokens)

def restore(value, protected, originals, tokens):
    if not isinstance(value, dict) or set(value) != {'paragraphs'}:
        raise ValueError('invalid_context_translation')
    rows = value['paragraphs']
    if not isinstance(rows, list) or [p.get('id') for p in rows] != [p['id'] for p in protected]:
        raise ValueError('paragraph_ids_or_order_changed')
    result = []
    for (output, masked, original) in zip(rows, protected, originals):
        text = output.get('text')
        if not isinstance(text, str) or bool(text.strip()) != bool(original['text'].strip()):
            raise ValueError('missing_paragraph')
        if Counter(TOKENS.findall(text)) != Counter(TOKENS.findall(masked['text'])):
            raise ValueError('numeric_placeholder_changed')
        if re.search('\\d', TOKENS.sub('', text)):
            raise ValueError('unprotected_number_added')
        text = TOKENS.sub(lambda m: tokens[m.group()], text)
        result.append(dict(original, text=text))
    return result

def prepare(source, work, workers=2, limit=None, case_ids=None):
    (source, work) = (Path(source).resolve(), Path(work).resolve())
    work.mkdir(parents=True, exist_ok=True)
    config = load(ROOT)
    api = API(work, config, retry_failed=True)
    from scripts.translation import backend, ensure_backend
    selected_backend = backend(config)
    cases = {}
    for path in sorted((source / 'cases').glob('*/source.json')):
        original = read(path)
        paragraphs = source_paragraphs(original)
        if not paragraphs:
            continue
        cid = path.parent.name
        binding = source_binding(path.parent)
        record = {'case_id': cid, 'base_id': binding['base_id'], 'source_binding': binding, 'paragraphs': paragraphs, 'source_text_sha256': digest(paragraphs)}
        save(work / 'sources' / (cid + '.json'), record)
        cases[cid] = record
        save(work / 'contexts' / cid / 'en.json', {'policy': VERSION, 'language': 'en', 'paragraphs': paragraphs, 'text_sha256': digest(paragraphs), 'source_text_sha256': digest(paragraphs), 'translation_request_sha256': None})
    tasks = [(cid, language) for cid in cases for language in LANGUAGES if language != 'en']
    if case_ids is not None:
        if not set(case_ids) <= set(cases):
            raise ValueError('unknown_context_case_id')
        tasks = [(cid, lang) for (cid, lang) in tasks if cid in case_ids]
    if limit is not None:
        tasks = tasks[:limit]
    counts = Counter()
    failures = []

    def translate(cid, language):
        original = cases[cid]
        target = work / 'contexts' / cid / (language + '.json')
        if target.exists():
            saved = read(target)
            ensure_backend(saved, selected_backend)
            if saved['source_text_sha256'] != original['source_text_sha256'] or saved['text_sha256'] != digest(saved['paragraphs']):
                raise ValueError('cached_context_changed')
            return 'cached'
        (protected, tokens) = protect(original['paragraphs'])
        (result, key) = api.call('source_context_translation_' + language, cid, PROMPT, {'target_language': language, 'source_binding': original['source_binding'], 'paragraphs': protected}, max_tokens=6000)
        paragraphs = restore(result, protected, original['paragraphs'], tokens)
        save(target, {'policy': VERSION, 'language': language, 'paragraphs': paragraphs, 'text_sha256': digest(paragraphs), 'source_text_sha256': original['source_text_sha256'], 'translation_request_sha256': key, 'translation_backend': selected_backend, 'translation_model': 'nmt' if selected_backend == 'google' else config['OPENAI_MODEL']})
        return 'translated'
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(translate, cid, lang): (cid, lang) for (cid, lang) in tasks}
        for future in as_completed(futures):
            (cid, lang) = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as exc:
                failures.append({'case_id': cid, 'language': lang, 'error': str(exc)})
            save(work / 'progress.json', {'updated_at': now(), 'source_cases': len(cases), 'requested_translations': len(tasks), 'completed': sum(counts.values()), 'counts': dict(counts), 'failures': failures})
            print('Context', cid, lang, 'completed', sum(counts.values()), 'failures', len(failures), flush=True)
    attempts = [read(p) for p in (work / 'api').glob('*/*/*/attempt_*.json')]
    google_attempts = [read(p) for p in (work / 'google_requests').glob('*/attempt_*.json')]
    save(work / 'usage_summary.json', {'translation_backend': selected_backend, 'google_api_attempts': len(google_attempts), 'google_submitted_characters': sum((a['source_characters'] for a in google_attempts)), 'api_attempts': len(attempts), 'known_tokens': {k: sum(((a.get('usage') or {}).get(k, 0) for a in attempts)) for k in ('input_tokens', 'output_tokens', 'total_tokens')}, 'attempts_without_usage': sum((not a.get('usage') for a in attempts))})
    if failures:
        raise RuntimeError('context_translation_incomplete; see ' + str(work / 'progress.json'))
    return cases

def publish(source, work, output):
    (source, work, output) = (Path(p).resolve() for p in (source, work, output))
    cases = {p.stem: read(p) for p in (work / 'sources').glob('*.json')}
    contexts = {(cid, lang): read(work / 'contexts' / cid / (lang + '.json')) for cid in cases for lang in LANGUAGES}
    for ((cid, lang), context) in contexts.items():
        if context['language'] != lang or context['policy'] != VERSION or context['source_text_sha256'] != cases[cid]['source_text_sha256'] or (context['text_sha256'] != digest(context['paragraphs'])):
            raise ValueError('context_publish_binding_changed:' + cid + ':' + lang)
    if not cases:
        raise ValueError('no_context_cases')
    source_hash = sha256(source / 'validation_release/val.candidates.jsonl')
    version = {'policy': VERSION, 'release_policy': RELEASE_VERSION, 'parent_references_sha256': source_hash, 'context_hash': digest(contexts_as_list(contexts)), 'source_cases': sorted(cases), 'context_language': 'query_language', 'images_unchanged': True, 'questions_unchanged': True, 'answers_unchanged': True, 'table_input': 'image_only', 'condition': 'source_context_restored_text'}
    if output.exists():
        if read(output / 'context_revision.json') != version:
            raise ValueError('context_revision_changed')
        verify_clean(output)
        return output
    staging = output.with_name(output.name + '.building')
    if staging.exists():
        shutil.rmtree(staging)
    original_rows = [json.loads(s) for s in (source / 'validation_release/val.candidates.jsonl').read_text().splitlines()]
    copy_current(source, staging, original_rows)
    changed = []
    current_rows = [clean_reference(row) for row in original_rows]
    for row in current_rows:
        cid = row['case_id']
        if cid in cases:
            row['source_context'] = contexts[cid, row['query_language']]
            row['input_condition'] = 'source_context_restored_text'
            changed.append({'variant_id': row['id'], 'case_id': cid, 'requires_new_prediction': True, 'reason': 'Original document prose restored as translated prompt context.'})
    write_current_views(staging, current_rows)
    for ((cid, lang), context) in contexts.items():
        save(staging / 'prompt_contexts' / cid / (lang + '.json'), context)
    save(staging / 'context_revision.json', version)
    save(staging / 'invalidated_evaluation_inputs.json', {'variants': changed})
    save(staging / 'prompt_context_manifest.json', {'version': VERSION, 'source_cases': len(cases), 'context_locales': len(contexts), 'affected_variants': len(changed), 'allowed_source_fields': ['metadata.pre_text', 'metadata.post_text', 'Paragraph'], 'excluded_fields': ['label', 'qa', 'gold_inds', 'model_input', 'steps', 'program', 'Year', 'Answer', 'text_html_table', 'text_markdown_table']})
    save(staging / 'release_manifest.json', {'policy': RELEASE_VERSION, 'qa_count': len(current_rows), 'case_count': len({r['case_id'] for r in current_rows}), 'old_qa_in_release': False, 'source_archive': 'External content-addressed provenance assets; see provenance_index.json hashes.'})
    verify_clean(staging)
    staging.rename(output)
    return output

def contexts_as_list(contexts):
    return [{'case_id': cid, **value} for ((cid, lang), value) in sorted(contexts.items())]
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    p.add_argument('--case-ids', help='Restrict translation calls to comma-separated case IDs; publication still requires all contexts')
    args = p.parse_args()
    save(args.work / 'data_job.json', {'state': 'translating', 'pid': os.getpid(), 'updated_at': now(), 'evaluation_autostart': False})
    try:
        prepare(args.source, args.work, args.workers, args.limit, args.case_ids.split(',') if args.case_ids else None)
        if args.output:
            publish(args.source, args.work, args.output)
        save(args.work / 'data_job.json', {'state': 'data_ready' if args.output else 'translation_batch_complete', 'pid': os.getpid(), 'updated_at': now(), 'dataset': str(args.output) if args.output else None, 'evaluation_autostart': False})
    except BaseException as exc:
        save(args.work / 'data_job.json', {'state': 'failed', 'updated_at': now(), 'error': str(exc), 'evaluation_autostart': False})
        raise
