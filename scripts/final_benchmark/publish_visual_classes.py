pass
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from scripts.final_benchmark.api import read, save, digest, now
from scripts.final_benchmark.provenance import sha256
from scripts.final_benchmark.clean_release import copy_current, write_current_views, verify_clean
from scripts.final_benchmark.classify_visuals import validate, TAXONOMY, PROMPT
from scripts.evaluation.migrate_token_router import verify_images

def publish(source, work, output):
    (source, work, output) = [Path(p).resolve() for p in (source, work, output)]
    verify_clean(source, write_report=False)
    if read(work / 'progress.json')['state'] != 'finished':
        raise ValueError('Classification incomplete')
    if output.exists():
        raise ValueError('Output already exists')
    rows = [json.loads(s) for s in (source / 'validation_release/val.candidates.jsonl').read_text().splitlines()]
    cases = {r['case_id'] for r in rows}
    classes = {cid: read(work / 'classifications' / (cid + '.json')) for cid in cases}
    images = {(r['case_id'], r['image_language']): r['image_sha256'] for r in rows}
    for (cid, result) in classes.items():
        validate(result, cid)
        if result['image_sha256'] != images[cid, 'en'] or result['prompt_sha256'] != digest(PROMPT) or result['model'] != 'gpt-6-astra' or (result['taxonomy_version'] != TAXONOMY['version']):
            raise ValueError('Classification binding changed')
    updated = []
    for row in rows:
        new = dict(row, visual_kind=classes[row['case_id']]['visual_kind'], visual_family=row.get('visual_family', row['visual_kind']))
        assert {k: v for (k, v) in new.items() if k not in ('visual_kind', 'visual_family')} == {k: v for (k, v) in row.items() if k not in ('visual_kind', 'visual_family')}
        updated.append(new)
    staged = output.with_name(output.name + '.building')
    if staged.exists():
        raise ValueError('Unfinished staging directory exists; inspect before retry')
    copy_current(source, staged, updated)
    for name in ['prompt_contexts']:
        if (source / name).exists():
            shutil.copytree(source / name, staged / name)
    for name in ['prompt_context_manifest.json', 'context_revision.json', 'invalidated_evaluation_inputs.json']:
        if (source / name).exists():
            shutil.copy2(source / name, staged / name)
    write_current_views(staged, updated)
    report = {'schema': TAXONOMY['version'], 'model': 'gpt-6-astra', 'created_at': now(), 'classification_unit': 'one current English image per case; same geometry across 24 language variants', 'prompt_sha256': digest(PROMPT), 'source_references_sha256': sha256(source / 'validation_release/val.candidates.jsonl'), 'references_sha256': sha256(staged / 'validation_release/val.candidates.jsonl'), 'input_and_answers_unchanged': True, 'cases': classes, 'base_case_counts': dict(Counter((v['visual_kind'] for v in classes.values()))), 'qa_counts': dict(Counter((r['visual_kind'] for r in updated))), 'visual_family_policy': 'Retains the previous coarse chart/table cohort; visual_kind is the new fine-grained label.', 'review_policy': 'Model-generated labels; confidence is not calibrated or human verification.'}
    save(staged / 'visual_classification.json', report)
    save(staged / 'visual_taxonomy.json', TAXONOMY)
    release = read(source / 'release_manifest.json')
    release.update(visual_taxonomy=TAXONOMY['version'], metadata_only_revision=True, references_sha256=report['references_sha256'])
    save(staged / 'release_manifest.json', release)
    context = read(source / 'context_validation.json')
    context.update(references_sha256=report['references_sha256'], inherited_validation_references_sha256=report['source_references_sha256'], metadata_revision_note='All source_context, question, answer and image fields are byte-identical to the validated parent.')
    save(staged / 'context_validation.json', context)
    with (staged / 'README.md').open('a') as f:
        f.write('\nvisual_kind contains GPT-6 visual categories. visual_family retains the coarse chart/table cohort.\nSee visual_classification.json for per-case evidence and visual_taxonomy.json for categories.\n')
    verify_images(staged, updated)
    verify_clean(staged)
    staged.rename(output)
    print(json.dumps({k: report[k] for k in ('base_case_counts', 'qa_counts', 'references_sha256')}, ensure_ascii=False))
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'work', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    publish(a.source, a.work, a.output)
