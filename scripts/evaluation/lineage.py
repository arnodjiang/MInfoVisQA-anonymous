pass
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import threading
from scripts.final_benchmark.api import digest, now, read, save
from scripts.final_benchmark.provenance import source_binding
from scripts.evaluation.context_input import input_text, context_prompt, protocol
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = ROOT / 'data/evaluation/provenance_assets'
SCHEMA = 'mstructqa-source-input-output-v1'

class Archive:

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}

    def put(self, data, suffix):
        sha = hashlib.sha256(data).hexdigest()
        path = self.root / sha[:2] / (sha + suffix)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError('archive_content_changed:' + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + '.%s.%s.tmp' % (os.getpid(), threading.get_ident()))
            tmp.write_bytes(data)
            tmp.replace(path)
        return {'sha256': sha, 'archive_path': str(path)}

    def file(self, path):
        path = Path(path).resolve()
        if path not in self.cache:
            self.cache[path] = dict(self.put(path.read_bytes(), path.suffix), source_path=str(path))
        return self.cache[path]

def source_record(dataset, row, archive, allow_public=False):
    folder = dataset / 'cases' / row['case_id']
    if not (folder / 'source.json').exists() and (dataset / 'provenance_index.json').exists():
        entry = read(dataset / 'provenance_index.json')['cases'][row['case_id']]
        if entry['case_id'] != row['case_id'] or entry['original_sample_id'] != row['base_id'] or entry['source'] != row['source']:
            raise ValueError('public_source_identity_mismatch')
        key = entry['source_record_sha256']
        path = archive.root / key[:2] / (key + '.json')
        if not path.exists() and allow_public:
            return {'original_sample_id': row['base_id'], 'case_id': row['case_id'], 'source_binding': {'base_id': row['base_id'], 'source': row['source']}, 'provenance': row.get('provenance', {}), 'source_record_sha256': key, 'upstream_question': None, 'upstream_answer': None, 'upstream_image': None, 'upstream_assets': [], 'original_visual_status': 'not_in_public_release', 'baseline_evaluation_status': 'not_established_by_this_run'}
        if hashlib.sha256(path.read_bytes()).hexdigest() != key:
            raise ValueError('external_source_archive_changed')
        source = read(path)
        if source['case_id'] != row['case_id'] or source['original_sample_id'] != row['base_id'] or source['source_binding']['source'] != row['source']:
            raise ValueError('external_source_archive_identity_mismatch')
        return source
    source = read(folder / 'source.json')
    binding = source_binding(folder)
    if binding['base_id'] != row['base_id'] or binding['source'] != row['source']:
        raise ValueError('source_row_mismatch:' + row['id'])
    candidate = source['candidate']
    assets = [archive.file(p) for p in sorted((folder / 'original').glob('*')) if p.is_file()]
    image = next((a for a in assets if a['sha256'] == binding['original_image_sha256']), None)
    return {'original_sample_id': row['base_id'], 'case_id': row['case_id'], 'source_binding': binding, 'provenance': source['identity']['provenance'], 'source_snapshot': archive.file(folder / 'source.json'), 'upstream_question': candidate['question'], 'upstream_answer': candidate['answer'], 'upstream_image': image, 'upstream_assets': assets, 'original_visual_status': 'available' if image else 'requires_faithful_table_rendering', 'baseline_evaluation_status': 'not_established_by_this_run'}

def pairing_key(row):
    pass
    return 'pair_' + digest([row['base_id'], row['query_language'], row['image_language'], row['answer_language']])

def snapshot_run(run, store=DEFAULT_STORE, frozen_inputs_only=False):
    run = Path(run).resolve()
    manifest = read(run / 'manifest.json')
    dataset = Path(manifest['dataset']).resolve()
    archive = Archive(store)
    (frozen_sources, frozen_samples) = ({}, {})
    if frozen_inputs_only:
        latest = read(run / 'lineage/latest.json')
        frozen = read(latest['snapshot']['archive_path'])
        frozen_sources = {s['case_id']: s for s in frozen['source_records']}
        frozen_samples = {s['variant_id']: s for s in frozen['samples']}
    reference_file = archive.file(run / 'references.jsonl')
    if reference_file['sha256'] != manifest['references_sha256']:
        raise ValueError('reference_snapshot_changed')
    rows = [json.loads(s) for s in (run / 'references.jsonl').read_text().splitlines()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate_variant_ids')
    (sources, records) = ({}, [])
    for row in rows:
        (cid, rid) = (row['case_id'], row['id'])
        if cid not in sources:
            sources[cid] = frozen_sources[cid] if frozen_inputs_only else source_record(dataset, row, archive, allow_public=True)
        source = sources[cid]
        if source['original_sample_id'] != row['base_id']:
            raise ValueError('case_base_id_changed')
        image_path = (dataset / 'validation_release' / row['image_path']).resolve()
        if dataset not in image_path.parents:
            raise ValueError('unsafe_image_path')
        image = frozen_samples[rid]['input']['image'] if frozen_inputs_only else archive.file(image_path)
        if image['sha256'] != row['image_sha256']:
            raise ValueError('evaluation_image_changed:' + rid)
        input_data = {'question': row['query'], 'image': image, 'prompt': archive.file(run / 'prompt.txt'), 'reference_answer': row['answer'], 'reference_correction_id': row.get('reference_correction_id')}
        if row.get('source_context'):
            input_data.update(source_context=row['source_context'], input_text=input_text(row))
            input_data['effective_prompt'] = archive.put(context_prompt((run / 'prompt.txt').read_text(), row).encode(), '.txt')
        item = {'schema': SCHEMA, 'original_sample_id': row['base_id'], 'case_id': cid, 'variant_id': rid, 'pair_id': pairing_key(row), 'condition': row.get('input_condition', 'reconstructed_localized'), 'is_upstream_original_input': False, 'dataset_version': dataset.name, 'references_sha256': manifest['references_sha256'], 'languages': {k: row[k] for k in ('query_language', 'image_language', 'answer_language')}, 'source_record_id': digest(source), 'input': input_data, 'model': manifest['model'], 'run_key': manifest['run_key'], 'visual_question_id': 'vq_' + digest([row['query'], image['sha256']]), 'prediction_status': 'pending', 'attempts': []}
        item['inference_input_id'] = 'input_' + digest([input_text(row), image['sha256'], context_prompt((run / 'prompt.txt').read_text(), row)])
        request_path = run / 'requests' / (rid + '.json')
        prediction_path = run / 'predictions' / (rid + '.json')
        if prediction_path.exists():
            prediction = read(prediction_path)
            if prediction['id'] != rid or prediction['run_key'] != manifest['run_key']:
                raise ValueError('prediction_identity_changed:' + rid)
            request = read(request_path)
            if request['id'] != rid or request['question'] != row['query'] or request['image_sha256'] != image['sha256']:
                raise ValueError('request_input_mismatch:' + rid)
            if request.get('input_text', request['question']) != input_text(row):
                raise ValueError('request_context_mismatch:' + rid)
            item.update(prediction_status=prediction['status'], request=archive.file(request_path), prediction_record=archive.file(prediction_path), extracted_answer=prediction.get('prediction'), selected_attempt=prediction.get('selected_attempt'), reused_from=prediction.get('reused_from'))
            for path in sorted((run / 'attempts' / rid).glob('*.json')):
                attempt = read(path)
                if attempt.get('status') == 'started':
                    continue
                response_path = run / 'responses' / rid / path.name
                record = {'attempt_number': attempt['attempt_number'], 'record': archive.file(path), 'response': archive.file(response_path) if response_path.exists() else None, 'is_selected': attempt['attempt_number'] == prediction.get('selected_attempt')}
                record['model_io_id'] = 'io_' + digest([item['request']['sha256'], record['record']['sha256'], record['response'] and record['response']['sha256']])
                item['attempts'].append(record)
                if record['is_selected']:
                    item['model_io_id'] = record['model_io_id']
                    item['raw_model_output'] = attempt.get('raw_output')
            if prediction['status'] == 'completed' and 'model_io_id' not in item:
                raise ValueError('selected_attempt_missing:' + rid)
            judge_path = run / 'llm_judge_text_v2/judgments' / (rid + '.json')
            if judge_path.exists():
                judgment = read(judge_path)
                if judgment['id'] != rid or judgment['prediction'] != prediction.get('prediction'):
                    raise ValueError('judgment_prediction_mismatch:' + rid)
                item.update(judgment=archive.file(judge_path), correct=judgment['verdict'] == 'equivalent', reference_effective=judgment['reference_effective'])
        records.append(item)
    run_assets = {'manifest': archive.file(run / 'manifest.json')}
    plain_prompt = Path(__file__).with_name('plain_answer_prompt.txt')
    expected_plain = {a['plain_prompt_sha256'] for a in manifest.get('policy_amendments', []) if a.get('plain_prompt_sha256')}
    if expected_plain:
        if expected_plain != {digest(plain_prompt.read_text())}:
            raise ValueError('plain_prompt_changed_since_inference')
        run_assets['plain_answer_prompt'] = archive.file(plain_prompt)
    if manifest.get('context_protocol'):
        actual = protocol(rows, (run / 'prompt.txt').read_text(), plain_prompt.read_text())
        if actual['context_protocol'] != manifest['context_protocol']:
            raise ValueError('context_prompt_protocol_changed')
        run_assets['context_plain_answer_prompt'] = archive.put(context_prompt(plain_prompt.read_text(), {'source_context': True}).encode(), '.txt')
    payload = {'schema': SCHEMA, 'run_assets': run_assets, 'source_records': list(sources.values()), 'samples': records}
    blob = archive.put((json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n').encode(), '.json')
    report = {'schema': SCHEMA, 'updated_at': now(), 'snapshot': blob, 'run_key': manifest['run_key'], 'model': manifest['model'], 'source_samples': len(sources), 'variants': len(records), 'prediction_status': dict(Counter((r['prediction_status'] for r in records))), 'original_visual_status': dict(Counter((s['original_visual_status'] for s in sources.values()))), 'judged': sum(('correct' in r for r in records)), 'baseline_evaluation_status': 'not_established_by_this_run', 'references': reference_file}
    judge_manifest = run / 'llm_judge_text_v2/manifest.json'
    if judge_manifest.exists():
        report['judge_policy'] = archive.file(judge_manifest)
    save(run / 'lineage/snapshots' / (blob['sha256'] + '.json'), report)
    save(run / 'lineage/latest.json', report)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()
    print(json.dumps(snapshot_run(args.run, args.store), ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
