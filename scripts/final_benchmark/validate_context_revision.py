pass
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.final_benchmark.api import digest, read, save
from scripts.final_benchmark.provenance import source_binding, sha256, verify_locale
from scripts.final_benchmark.restore_context import source_paragraphs, protect, restore
from scripts.evaluation.context_input import input_text, VERSION
from scripts.evaluation.score import LANGUAGES
from scripts.final_benchmark.clean_release import clean_reference, verify_clean

def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]

def dependency_audit(source, work):
    (source, work) = (Path(source).resolve(), Path(work).resolve())
    references = rows(source / 'validation_release/val.candidates.jsonl')
    report = []
    for folder in sorted((source / 'cases').iterdir()):
        if not (folder / 'source.json').exists():
            continue
        original = read(folder / 'source.json')
        paragraphs = source_paragraphs(original)
        if not paragraphs:
            continue
        binding = source_binding(folder)
        (spec, qa) = (read(folder / 'spec.json'), read(folder / 'qa.json'))
        subset = [r for r in references if r['case_id'] == folder.name]
        if len(subset) != 70 or {r['query_language'] for r in subset} != set(LANGUAGES):
            raise ValueError('incomplete_affected_configuration_coverage')
        for lang in LANGUAGES:
            verify_locale(read(folder / 'locales' / (lang + '.json')), spec, qa, binding)
        verified_images = {}
        for row in subset:
            image = (source / 'validation_release' / row['image_path']).resolve()
            if image not in verified_images:
                verified_images[image] = sha256(image)
            if verified_images[image] != row['image_sha256']:
                raise ValueError('affected_image_changed')
        answers = {str(r['answer']) for r in subset}
        if len(answers) != 1:
            raise ValueError('affected_answers_need_language_review')
        from decimal import Decimal, InvalidOperation
        try:
            value = Decimal(next(iter(answers)))
            if not value.is_finite():
                raise InvalidOperation()
        except InvalidOperation:
            raise ValueError('affected_non_numeric_answer_requires_translation_review')
        report.append({'case_id': folder.name, 'source': binding['source'], 'variants': len(subset), 'image_languages_verified': len(verified_images), 'numeric_answer': next(iter(answers)), 'translate_context': True, 'retranslate_question': False, 'retranslate_answer': False, 'rerender_image': False, 'source_text_sha256': digest(paragraphs)})
    result = {'state': 'passed', 'cases': report, 'affected_variants': sum((r['variants'] for r in report)), 'reason': 'Source-aligned questions, numeric answers and existing localized images are unchanged; only omitted document prose is added.'}
    save(work / 'dependency_audit.json', result)
    return result

def validate(source, dataset, work):
    (source, dataset, work) = (Path(p).resolve() for p in (source, dataset, work))
    dependencies = dependency_audit(source, work)
    old = {r['id']: r for r in rows(source / 'validation_release/val.candidates.jsonl')}
    new_rows = rows(dataset / 'validation_release/val.candidates.jsonl')
    new = {r['id']: r for r in new_rows}
    if set(old) != set(new) or len(new) != len(new_rows):
        raise ValueError('variant_ids_changed_or_duplicated')
    affected = {r['case_id'] for r in dependencies['cases']}
    images = {}
    contexts = set()
    changed = set()
    for (rid, row) in new.items():
        image = (dataset / 'validation_release' / row['image_path']).resolve()
        if dataset not in image.parents:
            raise ValueError('unsafe_image_path')
        if image not in images:
            images[image] = sha256(image)
        if images[image] != row['image_sha256']:
            raise ValueError('new_image_hash_mismatch')
        if row['case_id'] not in affected:
            if row != clean_reference(old[rid]):
                raise ValueError('unaffected_row_changed')
            continue
        unchanged = {k: v for (k, v) in row.items() if k not in ('source_context', 'input_condition')}
        if unchanged != clean_reference(old[rid]):
            raise ValueError('non_context_fields_changed')
        if row.get('input_condition') != 'source_context_restored_text':
            raise ValueError('missing_condition_marker')
        context = row['source_context']
        input_text(row)
        key = (row['case_id'], row['query_language'])
        expected = read(work / 'contexts' / key[0] / (key[1] + '.json'))
        if context != expected:
            raise ValueError('row_context_language_binding_changed')
        if key not in contexts:
            original = read(source / 'cases' / key[0] / 'source.json')
            paragraphs = source_paragraphs(original)
            if context['source_text_sha256'] != digest(paragraphs):
                raise ValueError('context_parent_source_changed')
            if key[1] == 'en':
                if context['paragraphs'] != paragraphs:
                    raise ValueError('english_context_modified')
            else:
                request_key = context['translation_request_sha256']
                api_folder = work / 'api' / ('source_context_translation_' + key[1]) / key[0] / request_key
                request = read(api_folder / 'request.json')
                result = read(api_folder / 'result.json')
                (protected, tokens) = protect(paragraphs)
                if digest(request) != request_key or request['case_id'] != key[0] or result['request_sha256'] != request_key or (request['payload']['target_language'] != key[1]) or (request['payload']['paragraphs'] != protected) or (request['payload']['source_binding'] != source_binding(source / 'cases' / key[0])):
                    raise ValueError('translation_request_source_mismatch')
                if restore(result['parsed'], protected, paragraphs, tokens) != context['paragraphs']:
                    raise ValueError('translated_context_differs_from_api_output')
            contexts.add(key)
        changed.add(rid)
    expected_changed = {r['variant_id'] for r in read(dataset / 'invalidated_evaluation_inputs.json')['variants']}
    if changed != expected_changed or len(changed) != dependencies['affected_variants']:
        raise ValueError('invalidated_scope_mismatch')
    if len(contexts) != len(affected) * len(LANGUAGES):
        raise ValueError('context_locale_coverage_incomplete')
    if (dataset / 'release_manifest.json').exists():
        verify_clean(dataset)
    report = {'state': 'passed', 'references_sha256': sha256(dataset / 'validation_release/val.candidates.jsonl'), 'variants': len(new), 'changed_variants': len(changed), 'unchanged_variants': len(new) - len(changed), 'verified_image_files': len(images), 'verified_context_locales': len(contexts), 'questions_images_answers_unchanged': True, 'validation_scope': 'Source/locale binding, protected numbers, exact API outputs and byte hashes; not a human translation-quality certification.'}
    save(dataset / 'context_validation.json', report)
    save(work / 'context_validation.json', report)
    return report
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--dataset', type=Path)
    args = p.parse_args()
    result = validate(args.source, args.dataset, args.work) if args.dataset else dependency_audit(args.source, args.work)
    print(json.dumps(result, ensure_ascii=False, indent=2))
