pass
import argparse
import json
import os
from pathlib import Path
import shutil
from scripts.final_benchmark.api import read, save, now
from scripts.final_benchmark.provenance import sha256

def image_path(run, manifest, row, frozen=None):
    if frozen is not None:
        if frozen['run_key'] != manifest['run_key'] or frozen['references_sha256'] != manifest['references_sha256']:
            raise ValueError('frozen_input_index_mismatch')
        return (Path(run) / frozen['images'][row['image_sha256']]).resolve()
    return (Path(manifest['dataset']) / 'validation_release' / row['image_path']).resolve()

def freeze(run, replacement):
    (run, replacement) = (Path(run).resolve(), Path(replacement).resolve())
    manifest = read(run / 'manifest.json')
    if sha256(run / 'references.jsonl') != manifest['references_sha256']:
        raise ValueError('reference_snapshot_changed')
    refs = [json.loads(s) for s in (run / 'references.jsonl').read_text().splitlines()]
    current = [json.loads(s) for s in (replacement / 'validation_release/val.candidates.jsonl').read_text().splitlines()]
    existing = read(run / 'frozen_images.json') if (run / 'frozen_images.json').exists() else None
    choices = {r['image_sha256']: (replacement / 'validation_release' / r['image_path']).resolve() for r in current}
    images = {}
    archived = 0
    for row in refs:
        key = row['image_sha256']
        if key in images:
            continue
        original = image_path(run, manifest, row, existing)
        if sha256(original) != key:
            raise ValueError('historical_image_changed:' + row['id'])
        target = choices.get(key)
        if target is None:
            target = run / 'retired_input_assets' / (key + '.png')
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copy2(original, target)
            archived += 1
        if sha256(target) != key:
            raise ValueError('replacement_image_changed:' + row['id'])
        images[key] = os.path.relpath(target, run)
    report = {'created_at': now(), 'run_key': manifest['run_key'], 'references_sha256': manifest['references_sha256'], 'images': images, 'unique_images': len(images), 'archived_images': archived, 'note': 'Historical pixels only; not a runnable benchmark. Shared pixels resolve to the current release and remain SHA256-verified.'}
    save(run / 'frozen_images.json', report)
    return {k: report[k] for k in ('unique_images', 'archived_images')}
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--replacement-dataset', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(freeze(a.run, a.replacement_dataset)))
