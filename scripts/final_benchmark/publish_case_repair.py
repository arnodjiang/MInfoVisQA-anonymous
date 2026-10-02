pass
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import zipfile
from . import pipeline, export, val_verify
from .api import read, save, digest, now
from .provenance import sha256, source_binding, verify_locale, render_is_current
from .query_policy import with_reply
from .repair_mixed_visual import CASE, builder
from .val_audit import table_features

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def jsonl(path, rows):
    path.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in rows)))

def verify_text_bounds(spec, layout):
    data = spec['data']
    segments = list(data['segments'])
    for row in data['row_geometry']:
        for (x1, i, x2, j) in data['row_segment_templates']:
            segments.append([x1, row[i], x2, row[j]])
    positions = {r[0]: r[1:3] for r in data['texts']}
    for box in layout['boxes']:
        key = box['label_key']
        (x, y) = positions[key]
        if key == 'title' or key.startswith('diagram_'):
            continue
        ys = [a[1] for a in segments if a[1] == a[3] and a[0] <= x <= a[2]]
        top = max((t for t in ys if t < y))
        bottom = min((t for t in ys if t > y))
        scale = layout['height'] / data['coordinate_extent'][1]
        assert box['box'][1] >= top * scale + 1 and box['box'][3] <= bottom * scale - 1, 'text_crosses_cell_border:' + key
    for (i, a) in enumerate(layout['boxes']):
        for b in layout['boxes'][i + 1:]:
            (aa, bb) = (a['box'], b['box'])
            assert not (min(aa[2], bb[2]) > max(aa[0], bb[0]) and min(aa[3], bb[3]) > max(aa[1], bb[1])), 'text_boxes_overlap'

def publish(out, parent):
    b = builder(out, parent)
    f = b.folder(CASE)
    spec = read(f / 'render_spec.json')
    qa = read(f / 'qa.json')
    audit = read(f / 'repair_audit.json')
    review = read(f / 'review.json')
    binding = b.source_binding(CASE)
    assert review['spec_sha256'] == digest(read(f / 'spec.json')) and review['status'] == 'pass'
    assert b.render_current(b.by_id[CASE])
    locales = {l: read(f / 'locales' / (l + '.json')) for l in pipeline.LANGUAGES}
    assert set(audit['languages']) == set(locales)
    assert not any((v.get(k) == 'fail' for v in audit['languages'].values() for k in ['translation', 'answer_equivalence', 'render_readability', 'visual_fidelity'])), 'unresolved_known_language_defect'
    assert audit['render_spec_sha256'] == digest(spec), 'stale_source_audit'
    for (l, loc) in locales.items():
        assert audit['locale_sha256s'][l] == digest(loc), 'stale_locale_audit:' + l
        assert audit['image_sha256s'][l] == sha256(f / 'images' / (l + '.png')), 'stale_image_audit:' + l
    for loc in locales.values():
        verify_locale(loc, spec, qa, binding)
    assert source_binding(parent / 'cases' / CASE) == binding
    prelude = export.code_prelude()
    images = {}
    checks = {}
    manifest = read(parent / 'image_manifest.json')
    for entry in manifest:
        if entry['id'] != CASE:
            continue
        l = entry['visual_language']
        image = f / 'images' / (l + '.png')
        layout = read(image.with_suffix('.layout.json'))
        assert render_is_current(image, spec, locales[l]['labels'])
        assert not layout['missing_glyphs'] and layout['all_text_inside_canvas'] and layout['all_text_inside_cells']
        verify_text_bounds(spec, layout)
        from PIL import Image, ImageColor
        with Image.open(image) as im:
            scale_x = im.width / spec['data']['coordinate_extent'][0]
            scale_y = im.height / spec['data']['coordinate_extent'][1]
            for circle in spec['data']['circles']:
                if circle['face'] == 'none':
                    continue
                (x, y) = circle['center']
                radius = circle['radius']
                crop = im.convert('RGB').crop((int((x - radius) * scale_x), int((y - radius) * scale_y), int((x + radius) * scale_x), int((y + radius) * scale_y)))
                color = ImageColor.getrgb(circle['face'])
                assert sum((pixel == color for pixel in crop.getdata())) > 20, 'missing_colored_marker:' + l
        code = export.emit_code(f, spec, l, locales[l], prelude)
        entry.update(image_sha256=sha256(image), code_sha256=sha256(code), data_sha256=digest(spec['data']), labels_sha256=digest(locales[l]['labels']), width=layout['width'], height=layout['height'], text_inside_canvas=True, text_inside_cells=True, missing_glyphs=[])
        images[l] = entry
        checks[l] = export.localization_issues(locales['en'], locales[l])
    execution = []
    for (l, entry) in images.items():
        target = f / 'standalone_check' / (l + '.png')
        command = [str(pipeline.ROOT / '.venv/bin/python'), str(out / entry['code']), '--output', str(target)]
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'MVISQA_FALLBACK_FONTS') if k in os.environ}
        proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        if proc.returncode:
            raise ValueError('standalone_execution_failed:' + l + ':' + proc.stderr[-1000:])
        if sha256(target) != entry['image_sha256']:
            raise ValueError('standalone_pixel_mismatch:' + l)
        execution.append({'language': l, 'image_sha256': sha256(target), 'matches': True})
    save(f / 'standalone_verification.json', {'checked_at': now(), 'executed': execution})
    rows = lines(parent / 'benchmark.jsonl')
    records = lines(parent / 'validation_release/val.candidates.jsonl')
    byvariant = {r['id']: r for r in records}
    changed = []
    for row in rows:
        if row['id'] != CASE:
            continue
        (vl, ql) = (row['visual_language'], row['query_language'])
        loc = locales[ql]
        question = pipeline.bind(loc['question'], loc['labels'])
        answer = pipeline.bind(loc['answer_template'], loc['labels'])
        relevant = {l: audit['languages'][l] for l in {vl, ql}}
        accepted = not qa.get('review_flags') and (not any((checks[l] for l in relevant))) and all((not v.get('issues') and all((v.get(k) == 'pass' for k in ['translation', 'answer_equivalence', 'render_readability', 'visual_fidelity'])) for v in relevant.values()))
        old = copy.deepcopy(row)
        assert sha256(parent / old['image']) == old['image_sha256'], 'parent_evaluation_image_changed'
        row.update(image_sha256=images[vl]['image_sha256'], data_sha256=digest(spec['data']), question=with_reply(question, answer, export.REPLY[ql]), question_without_instruction=question, answer=answer, canonical_answer_en=qa['answer'], answer_type=qa['answer_type'], reference_label_keys=pipeline.placeholder_keys(loc['question']), status='api_reviewed_candidate' if accepted else 'needs_review', review_flags=[] if accepted else ['repair_localization_needs_review'])
        row['query_revision'] = digest([row['variant_id'], row['question']])
        row['input_revision'] = digest([row['variant_id'], row['image_sha256'], row['question'], row['answer']])
        changed.append({'variant_id': row['variant_id'], 'old_image_sha256': old['image_sha256'], 'new_image_sha256': row['image_sha256'], 'question_changed': old['question'] != row['question'], 'answer_wording_changed': old['answer'] != row['answer'], 'requires_new_prediction': True})
        r = byvariant[row['variant_id']]
        r.update(query=row['question'], answer=answer, image_sha256=row['image_sha256'], answer_type=qa['answer_type'], table_structure=table_features(spec), input_revision=row['input_revision'])
        r['visual_metadata'].update(width=images[vl]['width'], height=images[vl]['height'], aspect_ratio=images[vl]['width'] / images[vl]['height'], data_sha256=row['data_sha256'], label_count=len(spec['labels']))
        r['features'].update(visual_features=['merged_table', 'radial_diagram', 'colored_markers'], panel_count=spec['recovery'].get('panel_count'), visual_component_count=2)
        r['audit'] = {'status': 'accepted' if accepted else 'needs_review', 'mechanical_issues': [], 'curation_issues': qa.get('review_flags', []), 'localization_flags': {l: checks[l] for l in relevant}, 'source_fidelity': 'pass', 'answer_preservation': 'pass', 'normal_qa': 'pass', 'source_qa_flags': qa.get('review_flags', []), 'language_checks': relevant, 'critical_issues': [], 'review_path': 'reviews/' + CASE + '.json', 'human_verified': False, 'scope': 'Original image and current English reconstruction reviewed; all 24 current locales and PNGs reviewed by API. Not human certification.'}
    assert len(changed) == 70 and len(rows) == 8960 and (len(records) == 8960) and (len(images) == 24)
    assert [r for r in rows if r['id'] != CASE] == [r for r in lines(parent / 'benchmark.jsonl') if r['id'] != CASE]
    assert [r for r in records if r['case_id'] != CASE] == [r for r in lines(parent / 'validation_release/val.candidates.jsonl') if r['case_id'] != CASE]
    for entry in manifest:
        if entry['id'] != CASE:
            assert sha256(out / entry['image']) == entry['image_sha256']
    save(out / 'image_manifest.json', manifest)
    save(out / 'validation_release/reviews' / (CASE + '.json'), dict(audit, source_fidelity='pass', answer_preservation='pass', normal_qa='pass', critical_issues=[]))
    for (name, subset) in [('benchmark.jsonl', rows), ('benchmark.api_reviewed.jsonl', [r for r in rows if r['status'] == 'api_reviewed_candidate'])]:
        jsonl(out / name, subset)
    for (name, subset) in [('val.candidates.jsonl', records), ('val.jsonl', [r for r in records if r['audit']['status'] == 'accepted']), ('val.needs_review.jsonl', [r for r in records if r['audit']['status'] != 'accepted'])]:
        jsonl(out / 'validation_release' / name, subset)
    cases = read(parent / 'case_manifest.json')
    for c in cases:
        if c['id'] == CASE:
            c.update(review=review, recovery=spec['recovery'], render_mode='custom')
    save(out / 'case_manifest.json', cases)
    summary = read(parent / 'validation.json')
    summary.update(generated_at=now(), status_counts=dict(Counter((r['status'] for r in rows))))
    save(out / 'validation.json', summary)
    save(out / 'localization_checks.json', [r for r in read(parent / 'localization_checks.json') if r['id'] != CASE] + [{'id': CASE, 'language': l, 'issues': v} for (l, v) in checks.items() if v])
    summary_path = parent / 'validation_release/audit_summary.json'
    audit_summary = read(summary_path) if summary_path.exists() else {}
    audit_summary.update(created_at=now(), candidate_samples=len(records), accepted_samples=sum((r['audit']['status'] == 'accepted' for r in records)), needs_review_samples=sum((r['audit']['status'] != 'accepted' for r in records)), source_fidelity_counts=dict(Counter((next((r['audit']['source_fidelity'] for r in records if r['case_id'] == cid)) for cid in {r['case_id'] for r in records}))), accepted_source_counts=dict(Counter((r['source'] for r in records if r['audit']['status'] == 'accepted'))))
    audit_summary['cases'] = [{'id': cid, 'review_available': True, 'source_fidelity': next((r['audit']['source_fidelity'] for r in records if r['case_id'] == cid))} for cid in sorted({r['case_id'] for r in records})]
    audit_summary.update(base_cases=len(audit_summary['cases']), reviewed_base_cases=len(audit_summary['cases']))
    save(out / 'validation_release/audit_summary.json', audit_summary)
    val_verify.DEFAULT_OUT = out
    val_verify.main()
    code = read(out / 'validation_release/code_invariance.json')
    assert code['checked'] == 3072 and (not code['failures'])
    save(out / 'invalidated_evaluation_inputs.json', {'reason': 'Images rebuilt from original; old predictions remain valid only for v1. Rerun these 70 configurations for v2.', 'case_id': CASE, 'variants': changed})
    revision = read(out / 'repair_revision.json')
    revision.update(status='repaired', finished_at=now(), changed_cases=1, changed_images=24, changed_configurations=70, source_binding=binding, other_127_cases_unchanged=True, human_verified=False)
    save(out / 'repair_revision.json', revision)
    request_keys = {spec['recovery_request_sha256'], qa['request_sha256'], review['request_sha256']}
    request_keys.update((loc['request_sha256'] for loc in locales.values() if loc.get('request_sha256')))
    request_keys.update(audit['requests'])
    attempts = []
    for p in (out / 'api').glob('*/' + CASE + '/*/attempt_*.json'):
        if not (parent / p.relative_to(out)).exists():
            entry = read(p)
            request_keys.add(p.parent.name)
            attempts.append({'path': str(p.relative_to(out)), 'status': entry.get('status'), 'usage': entry.get('usage'), 'elapsed_seconds': entry.get('elapsed_seconds')})
    totals = {k: sum(((r['usage'] or {}).get(k, 0) for r in attempts)) for k in ['input_tokens', 'output_tokens', 'total_tokens']}
    save(out / 'repair_api_usage.json', {'distinct_requests': len(request_keys), 'attempts': len(attempts), 'usage_totals': totals, 'attempts_without_usage': sum((r['usage'] is None for r in attempts)), 'records': attempts})
    font_paths = {r['path'] for r in read(parent / 'font_manifest.json')['fonts']}
    for p in (f / 'images').glob('*.layout.json'):
        font_paths.update(read(p)['fonts_used'])
    save(out / 'font_manifest.json', {'redistributed': False, 'fonts': [{'path': p, 'sha256': sha256(p)} for p in sorted(font_paths)]})
    expansion = read(out / 'expansion_provenance.json')
    expansion.update(source_reconstruction_unchanged=False, subsequent_repair_manifest='repair_revision.json', note='Original expansion ancestry retained; this revision subsequently replaces one source reconstruction and its 24 locales.')
    save(out / 'expansion_provenance.json', expansion)
    old_readme = (parent / 'README.md').read_text()
    (out / 'README.md').write_text('# MInfoVisQA — repaired data revision v2\n\nThis revision repairs case `' + CASE + '` from the original Visual-TableQA image. The accompanying radial diagram is restored, all 24 locale images and QA templates are regenerated using the configured API, and 70 QA configurations are updated. The other 127 cases are unchanged.\n\nAutomated review is not human certification. See `cases/' + CASE + '/repair_audit.json` for per-language verdicts. Old model predictions and paper scores refer to v1; rerun the 70 inputs in `invalidated_evaluation_inputs.json` before reporting v2 results.\n\nThe following construction documentation is inherited from the parent; revision-specific facts above and `repair_revision.json` take precedence.\n\n' + old_readme)
    save(out / 'completion.json', {'complete': True, 'finished_at': now(), 'languages': 24, 'base_cases': 128, 'images': 3072, 'samples': 8960, 'accepted_samples': sum((r['audit']['status'] == 'accepted' for r in records)), 'needs_review_samples': sum((r['audit']['status'] != 'accepted' for r in records)), 'all_standalone_constants_verified': True, 'repaired_case_all_24_standalone_executions_match': True, 'human_verified': False, 'revision_manifest': 'repair_revision.json'})
    status = read(out / 'status.json')
    status.update(build_complete=True, updated_at=now(), revision_manifest='repair_revision.json')
    save(out / 'status.json', status)
    with zipfile.ZipFile(out / 'reproducible_code.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for pattern in ['*/code/*/*/render.py', '*/spec.json', '*/render_spec.json', '*/qa.json', '*/locales/*.json']:
            for p in sorted((out / 'cases').glob(pattern)):
                z.write(p, p.relative_to(out))
        for p in sorted((out / 'prompts').glob('*.txt')):
            z.write(p, p.relative_to(out))
        for name in ['README.md', 'font_manifest.json', 'selection_lock.json', 'repair_revision.json', 'validation.json', 'invalidated_evaluation_inputs.json']:
            z.write(out / name, name)
    print(json.dumps(revision, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--parent', required=True)
    a = p.parse_args()
    publish(Path(a.output).resolve(), Path(a.parent).resolve())
