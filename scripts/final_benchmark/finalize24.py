pass
from .query_policy import with_reply
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
from . import pipeline, export
from .api import read, save, digest, now
from .languages24 import activate, NEW_LANGUAGES

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]

def finalize(out):
    activate()
    pipeline.DEFAULT_OUT = out
    from .polish_queries import COMMON, RULES as old_rules
    from .languages24 import RULES as new_rules
    for (l, rule) in dict(old_rules, **new_rules).items():
        (out / 'prompts' / (l + '_query_copyedit.txt')).write_text(COMMON + '\nTarget language: ' + pipeline.LANGUAGES[l] + '\n' + rule)
    from . import val_verify, verify_export
    val_verify.DEFAULT_OUT = out
    val_verify.main()
    code = read(out / 'validation_release/code_invariance.json')
    if code['checked'] != 3072 or code['failures']:
        raise ValueError('code_invariance_failed')
    sys.argv = ['verify_export', '--output', str(out), '--languages', ','.join(pipeline.LANGUAGES)]
    verify_export.main()
    rows = lines(out / 'benchmark.jsonl')
    assert len(rows) == 8960
    assert len({r['variant_id'] for r in rows}) == 8960
    source = Path(read(out / 'expansion_provenance.json')['source'])
    prior = lines(source / 'validation_release/val.candidates.jsonl')
    prior_ids = {r['id'] for r in lines(source / 'validation_release/val.jsonl')}
    templates = {r['case_id']: r for r in prior}
    dest = out / 'validation_release'
    records = []
    bycase = {}
    for row in rows:
        bycase.setdefault(row['id'], []).append(row)
    for (cid, variants) in bycase.items():
        assert len(variants) == 70
        folder = out / 'cases' / cid
        spec = read(folder / 'spec.json')
        oldspec = read(source / 'cases' / cid / 'spec.json')
        assert spec == oldspec, 'source_spec_changed'
        assert read(folder / 'qa.json') == read(source / 'cases' / cid / 'qa.json'), 'source_qa_changed'
        assert len({r['data_sha256'] for r in variants}) == 1
        loc = {l: read(folder / 'locales' / (l + '.json')) for l in pipeline.LANGUAGES}
        audit = read(out / 'expansion_audit/reviews' / (cid + '.json'))
        for l in NEW_LANGUAGES:
            assert audit['locale_sha256s'][l] == digest(loc[l]), 'stale_language_audit'
        inherited = read(source / 'validation_release/reviews' / (cid + '.json'))
        langchecks = dict(inherited['languages'], **audit['languages'])
        for row in variants:
            (ql, vl) = (row['query_language'], row['visual_language'])
            checks = []
            assert row['answer_language'] == ql and ql in {vl, 'en', 'zh'}
            assert row['question'] == with_reply(pipeline.bind(loc[ql]['question'], loc[ql]['labels']), row['answer'], export.REPLY[ql])
            assert row['answer'] == pipeline.bind(loc[ql]['answer_template'], loc[ql]['labels'])
            image = out / row['image']
            layout = read(image.with_suffix('.layout.json'))
            assert hashlib.sha256(image.read_bytes()).hexdigest() == row['image_sha256']
            if not layout['all_text_inside_canvas'] or not layout['all_text_inside_cells'] or layout['missing_glyphs']:
                checks.append('render_geometry_or_glyphs')
            flags = {l: export.localization_issues(loc['en'], loc[l]) for l in {ql, vl}}
            source_ok = all((inherited.get(k) == 'pass' for k in ['source_fidelity', 'answer_preservation', 'normal_qa'])) and (not inherited.get('critical_issues'))
            language_ok = all((all((langchecks[l].get(k) == 'pass' for k in ['translation', 'answer_equivalence', 'render_readability'])) and (l not in NEW_LANGUAGES or langchecks[l].get('query_fluency') == 'pass') and (not langchecks[l].get('issues')) for l in {ql, vl}))
            qa = read(folder / 'qa.json')
            curation = bool(qa.get('review_flags')) or row['answer_type'] in ['long_text', 'structured', 'list']
            accepted = source_ok and language_ok and (not checks) and (not any(flags.values())) and (not curation)
            if vl not in NEW_LANGUAGES and ql not in NEW_LANGUAGES:
                accepted = accepted and row['variant_id'] in prior_ids
            record = copy.deepcopy(templates[cid])
            record.update(id=row['variant_id'], query=row['question'], answer=row['answer'], image_path='../' + row['image'], code_path='../' + row['code'], image_sha256=row['image_sha256'], query_language=ql, image_language=vl, answer_language=ql, configuration=row['configuration'])
            record['visual_metadata'].update(width=layout['width'], height=layout['height'], aspect_ratio=layout['width'] / layout['height'], data_sha256=row['data_sha256'])
            record['audit'].update(status='accepted' if accepted else 'needs_review', mechanical_issues=checks, localization_flags=flags, language_checks={l: langchecks[l] for l in {ql, vl}}, review_path=os.path.relpath(source / 'validation_release/reviews' / (cid + '.json'), dest), expansion_review_path='../expansion_audit/reviews/' + cid + '.json', human_verified=False, scope='Source audit and old-language semantics inherited; new-language text and images reviewed separately; all code and geometry rechecked.')
            records.append(record)
            row['status'] = 'api_reviewed_candidate' if accepted else 'needs_review'
            row['review_flags'] = ['expanded_release_needs_review'] if not accepted else []
            row['query_revision'] = digest([row['variant_id'], row['question']])
    for (name, subset) in [('val.candidates.jsonl', records), ('val.jsonl', [r for r in records if r['audit']['status'] == 'accepted']), ('val.needs_review.jsonl', [r for r in records if r['audit']['status'] != 'accepted'])]:
        (dest / name).write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in subset)))
    for (name, subset) in [('benchmark.jsonl', rows), ('benchmark.api_reviewed.jsonl', [r for r in rows if r['status'] == 'api_reviewed_candidate'])]:
        (out / name).write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in subset)))
    summary = read(out / 'validation.json')
    summary['status_counts'] = dict(Counter((r['status'] for r in rows)))
    save(out / 'validation.json', summary)
    fonts = {pipeline.FONT}
    for p in (out / 'cases').glob('*/images/*.layout.json'):
        fonts.update(read(p).get('fonts_used', []))
    save(out / 'font_manifest.json', {'redistributed': False, 'fonts': [{'path': p, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in sorted(fonts)]})
    with zipfile.ZipFile(out / 'reproducible_code.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted((out / 'cases').glob('*/code/*/*/render.py')):
            z.write(p, p.relative_to(out))
        for pattern in ['*/spec.json', '*/render_spec.json', '*/qa.json', '*/locales/*.json']:
            for p in sorted((out / 'cases').glob(pattern)):
                z.write(p, p.relative_to(out))
        for p in sorted((out / 'prompts').glob('*.txt')):
            z.write(p, p.relative_to(out))
        for name in ['README.md', 'font_manifest.json', 'selection_lock.json', 'expansion_provenance.json', 'validation.json', 'standalone_verification.json']:
            z.write(out / name, name)
    save(out / 'completion.json', {'complete': True, 'finished_at': now(), 'languages': 24, 'base_cases': 128, 'images': 3072, 'samples': 8960, 'accepted_samples': sum((r['audit']['status'] == 'accepted' for r in records)), 'needs_review_samples': sum((r['audit']['status'] != 'accepted' for r in records)), 'source_spec_and_qa_unchanged': True, 'all_standalone_constants_verified': True, 'representative_execution': 'all24 languages on one chart and one table', 'human_verified': False})
    save(out / 'expansion_audit/progress.json', {'completed': len(bycase), 'expected': len(bycase), 'running': 0, 'failures': [], 'updated_at': now(), 'note': 'All case audits completed; semantic fail/uncertain verdicts remain in their review records. Historical request failures remain in API attempt logs.'})
    status = read(out / 'status.json')
    status.update(build_complete=True, updated_at=now())
    save(out / 'status.json', status)
    print(json.dumps(read(out / 'completion.json')), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(pipeline.ROOT / 'data/visual_benchmark/mstructqa_24'))
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    out = Path(a.output)

    def ready():
        if not (out / 'validation.json').exists() or not (out / 'benchmark.jsonl').exists():
            return False
        report = read(out / 'validation.json')
        return not report.get('missing_cases') and report.get('samples') == 8960 and (len(list((out / 'expansion_audit/reviews').glob('*.json'))) == 128)
    while not ready():
        if not a.watch:
            raise SystemExit('Generation or expanded-language review incomplete.')
        time.sleep(20)
    finalize(out)
if __name__ == '__main__':
    main()
