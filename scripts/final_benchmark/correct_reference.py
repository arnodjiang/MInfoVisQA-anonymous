pass
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from .api import read, save, digest, now
from .provenance import source_binding, locale_parent, sha256
from .languages24 import activate
from . import export
from scripts.evaluation.reference_corrections import load

def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def write_rows(p, rows):
    p.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in rows)))

def apply(parent, out):
    if parent == out:
        raise ValueError('new_revision_required')
    if not out.exists():
        if sys.platform == 'darwin':
            subprocess.run(['cp', '-cR', str(parent), str(out)], check=True)
        else:
            shutil.copytree(parent, out)
    corrections = load()
    activate()
    changed = []
    for (cid, c) in corrections.items():
        f = out / 'cases' / cid
        source = read(f / 'source.json')
        assert source['candidate']['answer'] == c['original_answer']
        assert source['candidate']['file'] == c['file'] and source['candidate']['row_index'] == c['row_index']
        archive = f / 'before_reference_correction'
        if not archive.exists():
            archive.mkdir()
            shutil.copy2(f / 'qa.json', archive / 'qa.json')
            shutil.copytree(f / 'locales', archive / 'locales')
        qa = read(f / 'qa.json')
        qa.update(answer=c['corrected_answer'], answer_template=c['corrected_answer'], reference_correction=c)
        save(f / 'qa.json', qa)
        source.update(reference_answer_label=c['corrected_answer'], reference_correction=c)
        save(f / 'source.json', source)
        binding = source_binding(f)
        spec = read(f / 'spec.json')
        for p in (f / 'locales').glob('*.json'):
            loc = read(p)
            loc.update(answer_template=c['corrected_answer'], reference_correction=c, input_binding=locale_parent(spec, qa, binding))
            save(p, loc)
    rows = lines(out / 'benchmark.jsonl')
    for r in rows:
        c = corrections.get(r['id'])
        if not c:
            continue
        original = copy.deepcopy(r)
        assert sha256(out / r['image']) == r['image_sha256']
        r.update(answer=c['corrected_answer'], canonical_answer_en=c['corrected_answer'], source_answer=c['corrected_answer'], upstream_source_answer=c['original_answer'], reference_correction_id=c['id'], input_revision=digest([r['variant_id'], r['image_sha256'], r['question'], c['corrected_answer']]))
        changed.append({'id': r['variant_id'], 'case_id': r['id'], 'old_answer': original['answer'], 'corrected_answer': c['corrected_answer'], 'prediction_reusable': True, 'reason': 'Image and question unchanged; update scoring reference only.'})
    for (name, subset) in [('benchmark.jsonl', rows), ('benchmark.api_reviewed.jsonl', [r for r in rows if r['status'] == 'api_reviewed_candidate'])]:
        write_rows(out / name, subset)
    refs = lines(out / 'validation_release/val.candidates.jsonl')
    for r in refs:
        c = corrections.get(r['case_id'])
        if c:
            r.update(answer=c['corrected_answer'], original_answer=c['corrected_answer'], upstream_original_answer=c['original_answer'], reference_correction_id=c['id'])
            r['audit']['reference_correction'] = c
            r['audit']['historical_review_note'] = 'Visual audits predate this user-confirmed answer correction; no new visual certification is implied.'
    for (name, subset) in [('val.candidates.jsonl', refs), ('val.jsonl', [r for r in refs if r['audit']['status'] == 'accepted']), ('val.needs_review.jsonl', [r for r in refs if r['audit']['status'] != 'accepted'])]:
        write_rows(out / 'validation_release' / name, subset)
    save(out / 'reference_corrections.json', {'version': 1, 'corrections': list(corrections.values())})
    save(out / 'source/reference_corrections.json', read(out / 'reference_corrections.json'))
    save(out / 'reference_revision.json', {'parent': str(parent), 'created_at': now(), 'changes': changed, 'raw_downloads_unchanged': True, 'inference_images_and_questions_unchanged': True, 'supersedes_reference_labels_only': True})
    assert len(changed) == 70 * len(corrections)
    before = {r['variant_id']: r for r in lines(parent / 'benchmark.jsonl')}
    for r in rows:
        old = before[r['variant_id']]
        assert r['question'] == old['question'] and r['image_sha256'] == old['image_sha256']
        if r['id'] not in corrections:
            assert r == old
    (out / 'README.md').write_text('# Reference-label revision v3\n\nThe user-adjudicated CharXiv answer is now `0°、45°` in all 24 locales and 70 configurations. Images, questions and raw upstream downloads remain unchanged. Existing predictions can be rescored using the correction registry. See `reference_revision.json`.\n\n' + (parent / 'README.md').read_text())
    with zipfile.ZipFile(parent / 'reproducible_code.zip') as src, zipfile.ZipFile(out / 'reproducible_code.zip', 'w', zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            p = out / info.filename
            if p.is_file():
                dst.write(p, info.filename)
        for name in ['reference_corrections.json', 'reference_revision.json']:
            dst.write(out / name, name)
    completion = read(out / 'completion.json')
    completion.update(finished_at=now(), reference_revision='reference_revision.json')
    save(out / 'completion.json', completion)
    print('Corrected', len(corrections), 'source label;', len(changed), 'QA configurations; images/questions unchanged', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    apply(a.parent.resolve(), a.output.resolve())
