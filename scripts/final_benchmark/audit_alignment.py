pass
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import unicodedata
from pathlib import Path
from .api import read, save, digest, now
from .provenance import source_binding, verify_spec, verify_locale, verify_polish, sha256

def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def upstream_qa(source, raw, slot):
    if source == 'CharXiv':
        return (raw['reasoning_q'], raw['reasoning_a'])
    if source == 'ChartQA':
        return (raw['query'], raw['label'])
    if source == 'ChartQAPro':
        index = int(slot.split('_')[-1])
        return (raw['Question'][index], raw['Answer'][index])
    if source == 'TableVQA-Bench':
        return (raw['question'], raw['gt'])
    if source == 'Visual-TableQA':
        return (raw['question'], raw['answer'])
    if source == 'MMTU':
        meta = json.loads(raw['metadata'])
        return (meta['question'], meta.get('label', meta.get('output')))
    raise ValueError('Unsupported source: ' + source)

def audit(dataset, project, run_dirs=()):
    import pyarrow.parquet as pq
    from .pipeline import bind
    from .query_policy import with_reply
    from .languages24 import REPLY as NEW_REPLY
    from .export import REPLY as BASE_REPLY
    REPLY = dict(BASE_REPLY, **NEW_REPLY)
    (dataset, project) = (Path(dataset).resolve(), Path(project).resolve())
    provenance_paths = read(dataset / 'provenance_paths.json') if (dataset / 'provenance_paths.json').exists() else {}
    corrections = {c['case_id']: c for c in read(dataset / 'reference_corrections.json')['corrections']} if (dataset / 'reference_corrections.json').exists() else {}
    (issues, warnings) = ([], [])
    counts = Counter()

    def check(ok, kind, cid, detail=None):
        if not ok:
            issues.append(dict(check=kind, case_id=cid, detail=detail))
    cases = lines(dataset / 'source/candidates.jsonl')
    identities = {r['id']: r for r in lines(dataset / 'source/selected_reserve_identity.jsonl')}
    check(len({c['id'] for c in cases}) == len(cases), 'duplicate_case_ids', None)
    groups = defaultdict(list)
    for c in cases:
        groups[c['file']].append(c)
    for (filename, wanted) in groups.items():
        byrow = defaultdict(list)
        for c in wanted:
            byrow[c['row_index']].append(c)
        pf = pq.ParquetFile(project / filename)
        cols = [k for k in pf.schema_arrow.names if k != 'image' or wanted[0]['source'] != 'MMTU']
        offset = 0
        found = set()
        for batch in pf.iter_batches(batch_size=16, columns=cols):
            for index in sorted((n for n in byrow if offset <= n < offset + len(batch))):
                raw = batch.slice(index - offset, 1).to_pylist()[0]
                im = raw.pop('image', None)
                blob = im.get('bytes') if isinstance(im, dict) else im
                for c in byrow[index]:
                    cid = c['id']
                    counts['raw_entries_checked'] += 1
                    found.add(index)
                    check(raw == c['raw_record'], 'raw_record_mismatch', cid)
                    (q, a) = upstream_qa(c['source'], raw, c['slot'])
                    check(q == c['question'] and a == c['answer'], 'raw_qa_mismatch', cid)
                    expected_id = hashlib.sha256(f"{filename}:{index}:{c['slot']}".encode()).hexdigest()[:20]
                    check(cid == expected_id, 'source_locator_id_mismatch', cid)
                    expected = identities[cid]['fingerprints'].get('image_sha256')
                    if expected:
                        check(bool(blob) and hashlib.sha256(blob).hexdigest() == expected, 'raw_image_mismatch', cid)
                        counts['raw_images_checked'] += 1
            offset += len(batch)
            if offset > max(byrow):
                break
        check(found == set(byrow), 'source_rows_missing', filename)
        print('Source checked:', filename, len(wanted), flush=True)
    roots = [dataset]
    for root in roots:
        for (filename, field) in [('expansion_provenance.json', 'source'), ('quality_repair_plan.json', 'parent_build'), ('repair_revision.json', 'parent'), ('reference_revision.json', 'parent')]:
            path = root / filename
            if path.exists():
                parent = Path(read(path)[field]).resolve()
                if parent not in roots:
                    roots.append(parent)
    request_index = {}
    for root in roots:
        for p in (root / 'api').glob('*/*/*/request.json'):
            request_index.setdefault(p.parent.name, p)
    request_cache = {}

    def request(key, cid):
        path = request_index.get(key)
        if not path:
            warnings.append(dict(check='historical_api_request_unavailable', case_id=cid, key=key))
            return None
        if key not in request_cache:
            request_cache[key] = read(path)
        r = request_cache[key]
        check(r['case_id'] == cid, 'api_case_id_mismatch', cid, str(path))
        check(digest(r) == key, 'api_fingerprint_mismatch', cid, str(path))
        counts['api_artifact_links_checked'] += 1
        return r

    def parsed_result(key, cid):
        path = request_index[key].with_name('result.json')
        if path.exists():
            return read(path)['parsed']
        for attempt in sorted(path.parent.glob('attempt_*.json'), reverse=True):
            raw = read(attempt).get('raw_output', '')
            for suffix in ('', '}', '}}', '}}}'):
                try:
                    parsed = json.loads(raw + suffix)
                except ValueError:
                    continue
                warnings.append(dict(check='legacy_result_missing_verified_from_raw_response', case_id=cid, request=key, attempt=str(attempt), appended_closing_braces=len(suffix)))
                return parsed
        raise ValueError('historical_parsed_output_unavailable:' + key)
    case_data = {}
    image_hashes = {}
    for c in cases:
        cid = c['id']
        folder = dataset / 'cases' / cid
        try:
            binding = source_binding(folder, c, identities[cid])
            (spec, qa) = (read(folder / 'spec.json'), read(folder / 'qa.json'))
            correction = corrections.get(cid)
            if correction:
                check(correction['original_answer'] == c['answer'] and correction['file'] == c['file'] and (correction['row_index'] == c['row_index']), 'adjudication_source_mismatch', cid)
                check(qa.get('reference_correction') == correction and qa['answer'] == correction['corrected_answer'] and (qa['answer_template'] == correction['corrected_answer']), 'adjudication_output_mismatch', cid)
            render = read(folder / 'render_spec.json')
            verify_spec(spec, binding)
            verify_spec(render, binding)
            data_without_layout = {k: v for (k, v) in render['data'].items() if k != 'layout'}
            check(data_without_layout == {k: v for (k, v) in spec['data'].items() if k != 'layout'}, 'render_data_changed', cid)
            qreq = request(qa.get('request_sha256'), cid) if qa.get('request_sha256') else None
            if qreq:
                check(qreq['payload'].get('source_question') == c['question'] and qreq['payload'].get('source_answer') == c['answer'], 'qa_normalization_source_mismatch', cid)
                parsed = parsed_result(qa['request_sha256'], cid)
                if correction:
                    parsed = dict(parsed, answer=correction['corrected_answer'], answer_template=correction['corrected_answer'])
                check(all((qa[k] == parsed[k] for k in ('question', 'answer', 'answer_template'))), 'qa_normalization_output_mismatch', cid)
            locales = {}
            for p in (folder / 'locales').glob('*.json'):
                lang = p.stem
                loc = read(p)
                locales[lang] = loc
                verify_locale(loc, spec, qa, binding)
                counts['locales_checked'] += 1
                key = loc.get('request_sha256') or loc.get('qa_translation_request')
                if key:
                    req = request(key, cid)
                    if req:
                        payload = req['payload']
                        original_qa = read(folder / 'before_reference_correction/qa.json') if correction else qa
                        check(payload.get('question') == original_qa['question'] and payload.get('answer_template') == original_qa['answer_template'], 'translation_qa_input_mismatch', cid, lang)
                        if 'labels' in payload:
                            check(payload['labels'] == spec['labels'], 'translation_labels_input_mismatch', cid, lang)
                        parsed = parsed_result(key, cid)
                        expected = dict(parsed['locales'][lang] if 'locales' in parsed else parsed)
                        if correction:
                            expected['answer_template'] = correction['corrected_answer']
                        if loc.get('digit_normalization'):

                            def digits(s):
                                return ''.join((str(unicodedata.decimal(c)) if unicodedata.category(c) == 'Nd' else c for c in s))
                            expected['question'] = digits(expected['question'])
                            expected['answer_template'] = digits(expected['answer_template'])
                            if 'labels' in expected:
                                expected['labels'] = {k: digits(v) for (k, v) in expected['labels'].items()}
                        fit = loc.get('label_fit_repair')
                        if fit and 'labels' in expected:
                            fitreq = request(fit['request'], cid)
                            if fitreq:
                                expected['labels'][fit['key']] = parsed_result(fit['request'], cid)['label']
                        check(expected['question'] == loc.get('question_original', loc['question']) and expected['answer_template'] == loc['answer_template'], 'translation_qa_output_mismatch', cid, lang)
                        if 'labels' in expected:
                            check(expected['labels'] == loc['labels'], 'translation_labels_output_mismatch', cid, lang)
                elif lang == 'en':
                    check(loc['labels'] == spec['labels'] and loc['answer_template'] == qa['answer_template'] and (loc.get('question_original', loc['question']) == qa['question']), 'english_locale_source_mismatch', cid)
                chunk_labels = {}
                for key in loc.get('chunk_translation_requests', []):
                    req = request(key, cid)
                    if req:
                        check(all((spec['labels'].get(k) == v for (k, v) in req['payload']['source_labels'].items())), 'translation_chunk_source_mismatch', cid, lang)
                        chunk_labels.update(parsed_result(key, cid))
                if chunk_labels:
                    check(chunk_labels == loc['labels'], 'translation_chunk_output_mismatch', cid, lang)
                polish = loc.get('query_polish_provenance')
                if polish:
                    path = dataset / provenance_paths[polish['path']] if polish['path'] in provenance_paths else project / polish['path']
                    record = read(path)
                    check(sha256(path) == polish['sha256'], 'query_polish_fingerprint_mismatch', cid, lang)
                    verify_polish(record, cid, lang, c['question'], dict(loc, question=loc['question_original']))
                    check(loc['question'] == record['query'], 'query_polish_output_mismatch', cid, lang)
                code = folder / 'code' / ('original' if lang == 'en' else 'translated') / lang / 'render.py'
                constants = {}
                for node in ast.parse(code.read_text()).body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in ('DATA', 'LABELS', 'LANGUAGE', 'BASE_ID'):
                                constants[target.id] = ast.literal_eval(node.value)
                check(constants == dict(DATA=render['data'], LABELS=loc['labels'], LANGUAGE=lang, BASE_ID=binding['base_id']), 'exported_code_binding_mismatch', cid, lang)
                image = folder / 'images' / (lang + '.png')
                image_hashes[str(image)] = sha256(image)
                counts['images_and_code_checked'] += 1
            case_data[cid] = (c, binding, render, locales)
        except Exception as exc:
            issues.append(dict(check='case_validation_error', case_id=cid, detail=str(exc)))
    benchmark = lines(dataset / 'benchmark.jsonl')
    byvariant = {r['variant_id']: r for r in benchmark}
    check(len(byvariant) == len(benchmark), 'duplicate_variant_ids', None)
    for row in benchmark:
        cid = row['id']
        if cid not in case_data:
            continue
        (c, binding, spec, locales) = case_data[cid]
        (vl, ql) = (row['visual_language'], row['query_language'])
        loc = locales[ql]
        expected = 'variant_' + digest([binding['base_id'], vl, ql, ql])
        image = dataset / 'cases' / cid / 'images' / (vl + '.png')
        answer = bind(loc['answer_template'], loc['labels'])
        question = bind(loc['question'], loc['labels'])
        check(row['variant_id'] == expected and row['base_id'] == binding['base_id'] and (row['answer_language'] == ql) and (ql in {vl, 'en', 'zh'}), 'variant_identity_mismatch', cid)
        check((dataset / row['image']).resolve() == image and row['image_sha256'] == image_hashes[str(image)], 'variant_image_mismatch', cid, row['variant_id'])
        expected_answer = corrections[cid]['corrected_answer'] if cid in corrections else c['answer']
        check(row['source_question'] == c['question'] and row['source_answer'] == expected_answer, 'variant_original_qa_mismatch', cid)
        check(row['answer'] == answer and row['question'] == with_reply(question, answer, REPLY[ql]), 'variant_localized_qa_mismatch', cid, row['variant_id'])
        counts['variants_checked'] += 1
    val = lines(dataset / 'validation_release/val.candidates.jsonl')
    valmap = {r['id']: r for r in val}
    check(set(valmap) == set(byvariant), 'validation_variant_set_mismatch', None)
    for r in val:
        b = byvariant.get(r['id'])
        if not b:
            continue
        check(r['case_id'] == b['id'] and r['query'] == b['question'] and (r['answer'] == b['answer']) and (r['image_language'] == b['visual_language']) and (r['query_language'] == b['query_language']), 'validation_qa_mismatch', b['id'], r['id'])
        check((dataset / 'validation_release' / r['image_path']).resolve() == (dataset / b['image']).resolve() and r['image_sha256'] == b['image_sha256'], 'validation_image_mismatch', b['id'], r['id'])
    for run in map(Path, run_dirs):
        for ref in lines(run / 'references.jsonl'):
            check(ref == valmap.get(ref['id']), 'evaluation_snapshot_mismatch', ref.get('case_id'), str(run))
        for p in (run / 'requests').glob('*.json'):
            req = read(p)
            ref = valmap.get(req['id'])
            check(ref is not None and req['question'] == ref['query'] and (req['image_path'] == ref['image_path']) and (req['image_sha256'] == ref['image_sha256']), 'evaluation_request_mismatch', ref.get('case_id') if ref else None, str(p))
            counts['evaluation_requests_checked'] += 1
    semantic = defaultdict(list)
    for r in val:
        a = r.get('audit', {})
        if any((a.get(k) in ('fail', 'uncertain') for k in ('source_fidelity', 'answer_preservation', 'normal_qa'))):
            semantic[r['case_id']].append(r['id'])
    return dict(checked_at=now(), dataset=str(dataset), counts=dict(counts), alignment_pass=not issues, issues=issues, warnings=warnings, semantic_review_flagged_cases=len(semantic), semantic_review_flagged_variants=sum(map(len, semantic.values())), semantic_review_case_ids=sorted(semantic), scope='Source-row identity, byte hashes, saved request inputs and deterministic binding. Does not certify semantic translations, source answers, or visual fidelity.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--run', type=Path, action='append', default=[])
    a = p.parse_args()
    result = audit(a.dataset, a.project, a.run)
    save(a.report, result)
    print(json.dumps({k: v for (k, v) in result.items() if k not in ('issues', 'warnings', 'semantic_review_case_ids')}, ensure_ascii=False))
    print('issues', len(result['issues']), 'warnings', len(result['warnings']))
    if result['issues']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
