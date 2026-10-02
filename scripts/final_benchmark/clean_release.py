pass
import json
from pathlib import Path
import shutil
from scripts.final_benchmark.api import digest, read, save
from scripts.final_benchmark.provenance import sha256
VERSION = 'current_artifacts_only_v1'
REFERENCE_FIELDS = {'id', 'base_id', 'case_id', 'split', 'query', 'answer', 'image_path', 'image_sha256', 'code_path', 'query_language', 'image_language', 'answer_language', 'visual_kind', 'source', 'configuration', 'task_tags', 'answer_type', 'provenance', 'visual_family', 'reference_correction_id', 'upstream_answer_sha256', 'source_context', 'input_condition'}
AUDIT_FIELDS = {'status', 'human_verified', 'source_fidelity'}
FORBIDDEN_KEYS = {'original_query', 'original_answer', 'upstream_original_answer', 'upstream_source_answer', 'source_question', 'source_answer', 'question_without_instruction', 'raw_record', 'reused_from', 'prediction', 'candidate_answer'}
ROOT_FILES = {'README.md', 'benchmark.jsonl', 'case_manifest.json', 'image_manifest.json', 'provenance_index.json', 'release_manifest.json', 'context_revision.json', 'invalidated_evaluation_inputs.json', 'prompt_context_manifest.json', 'context_validation.json', 'clean_release_validation.json', 'visual_classification.json', 'visual_taxonomy.json'}

def clean_reference(row):
    result = {k: v for (k, v) in row.items() if k in REFERENCE_FIELDS}
    if 'audit' in row:
        result['audit'] = {k: v for (k, v) in row['audit'].items() if k in AUDIT_FIELDS}
    upstream = row.get('upstream_original_answer', row.get('original_answer', row.get('upstream_source_answer', row.get('source_answer'))))
    if upstream is not None:
        result['upstream_answer_sha256'] = digest(upstream)
    return result

def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in records)))

def copy_current(source, destination, references):
    from scripts.evaluation.lineage import Archive, DEFAULT_STORE, source_record
    (source, destination) = (Path(source).resolve(), Path(destination).resolve())
    destination.mkdir(parents=True, exist_ok=True)
    archive = Archive(DEFAULT_STORE)
    (index, cases, images) = ({}, {}, {})
    copied = set()
    for row in references:
        cid = row['case_id']
        if cid not in index:
            record = source_record(source, row, archive)
            archived = archive.put((json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n').encode(), '.json')
            index[cid] = {'case_id': cid, 'original_sample_id': row['base_id'], 'source': row['source'], 'source_record_sha256': archived['sha256']}
            cases[cid] = {'id': cid, 'base_id': row['base_id'], 'source': row['source'], 'kind': row['visual_kind']}
        for field in ('image_path', 'code_path'):
            path = (source / 'validation_release' / row[field]).resolve()
            if source not in path.parents:
                raise ValueError('unsafe_current_artifact_path')
            if path in copied:
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.add(path)
        key = (cid, row['image_language'])
        image_path = (source / 'validation_release' / row['image_path']).resolve()
        code_path = (source / 'validation_release' / row['code_path']).resolve()
        images[key] = {'case_id': cid, 'base_id': row['base_id'], 'visual_language': row['image_language'], 'image': str(image_path.relative_to(source)), 'image_sha256': row['image_sha256'], 'code': str(code_path.relative_to(source)), 'code_sha256': sha256(code_path)}
    save(destination / 'provenance_index.json', {'schema': VERSION, 'cases': index, 'archive_store': 'External content-addressed provenance_assets; original QA is not part of this release.'})
    save(destination / 'case_manifest.json', list(cases.values()))
    save(destination / 'image_manifest.json', list(images.values()))
    (destination / 'README.md').write_text('# MInfoVisQA current data\n\nThis release contains only the current QA, localized images, rendering code, and source-context text.\nThe evaluation entry point is validation_release/val.candidates.jsonl.\nbenchmark.jsonl is a dataset-root view of the same current variants.\nOnly query + optional source_context + the image enter model inference; answer is for scoring.\nOriginal sample IDs and provenance hashes support paired analysis. Original source QA, previous\nqueries, previous labels, API logs and historical results are stored outside this release.\ncode/original/en means the current English reconstruction, not a previous QA version.\n')

def write_current_views(destination, references):
    destination = Path(destination)
    jsonl(destination / 'validation_release/val.candidates.jsonl', references)
    jsonl(destination / 'validation_release/val.jsonl', [r for r in references if r.get('audit', {}).get('status') == 'accepted'])
    jsonl(destination / 'validation_release/val.needs_review.jsonl', [r for r in references if r.get('audit', {}).get('status') != 'accepted'])
    root_view = []
    for row in references:
        view = dict(row)
        for key in ('image_path', 'code_path'):
            view[key] = str((destination / 'validation_release' / row[key]).resolve().relative_to(destination.resolve()))
        root_view.append(view)
    jsonl(destination / 'benchmark.jsonl', root_view)

def verify_clean(dataset, write_report=True):
    dataset = Path(dataset).resolve()
    refs = [json.loads(s) for s in (dataset / 'validation_release/val.candidates.jsonl').read_text().splitlines()]
    if len({r['id'] for r in refs}) != len(refs):
        raise ValueError('duplicate_current_ids')
    allowed = {Path(p) for p in ROOT_FILES}
    allowed.update((Path('validation_release') / n for n in ('val.candidates.jsonl', 'val.jsonl', 'val.needs_review.jsonl')))
    for row in refs:
        if row != clean_reference(row):
            raise ValueError('noncurrent_reference_fields')
        for key in ('image_path', 'code_path'):
            asset = (dataset / 'validation_release' / row[key]).resolve()
            if not asset.is_file():
                raise ValueError('missing_current_artifact')
            allowed.add(asset.relative_to(dataset))
        if row.get('source_context'):
            context_path = Path('prompt_contexts') / row['case_id'] / (row['query_language'] + '.json')
            if not (dataset / context_path).is_file():
                raise ValueError('missing_current_context')
            allowed.add(context_path)

    def reject_old_fields(value):
        if isinstance(value, dict):
            if FORBIDDEN_KEYS & value.keys():
                raise ValueError('historical_content_in_latest_release')
            for child in value.values():
                reject_old_fields(child)
        elif isinstance(value, list):
            for child in value:
                reject_old_fields(child)
    for path in dataset.rglob('*'):
        if path.is_symlink():
            raise ValueError('release_symlink_not_allowed')
        if not path.is_file():
            continue
        if path.relative_to(dataset) not in allowed:
            raise ValueError('unexpected_release_file:' + str(path.relative_to(dataset)))
        if path.suffix == '.json':
            reject_old_fields(read(path))
        if path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                reject_old_fields(json.loads(line))
    expected_root = []
    for row in refs:
        view = dict(row)
        for key in ('image_path', 'code_path'):
            view[key] = str((dataset / 'validation_release' / row[key]).resolve().relative_to(dataset))
        expected_root.append(view)
    actual_root = [json.loads(s) for s in (dataset / 'benchmark.jsonl').read_text().splitlines()]
    if actual_root != expected_root:
        raise ValueError('root_and_evaluation_view_mismatch')
    for (name, accepted) in [('val.jsonl', True), ('val.needs_review.jsonl', False)]:
        expected = [r for r in refs if (r.get('audit', {}).get('status') == 'accepted') == accepted]
        actual = [json.loads(s) for s in (dataset / 'validation_release' / name).read_text().splitlines()]
        if actual != expected:
            raise ValueError('stale_current_subset_view:' + name)
    report = {'state': 'passed', 'policy': VERSION, 'current_variants': len(refs), 'historical_qa_files': 0, 'historical_question_fields': 0, 'references_sha256': sha256(dataset / 'validation_release/val.candidates.jsonl')}
    if write_report:
        save(dataset / 'clean_release_validation.json', report)
    return report
