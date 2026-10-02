pass
import argparse
import copy
import json
from pathlib import Path
from .api import digest, read, save, now
from .pipeline import Builder, DEFAULT_OUT, ROOT
from .prompts import CHART

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True, help='Explicit preserved pilot input directory')
    parser.add_argument('--output', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    args.retry_failed = True
    b = Builder(args)
    identifier = 'e525de67b11a063bafc6'
    folder = b.folder(identifier)
    if (folder / 'recovery_complete.json').exists():
        return
    pilot = args.pilot
    old = read(pilot / 'specs.json')[identifier]
    labels_all = read(pilot / 'labels_en.json')
    keys = {'learner_walltime', 'episode_return', 'unit_minutes', 'unit_hours', 'actors_title', 'series_single', 'series_2', 'series_8', 'series_16'}
    for panel in old['panels']:
        keys.update([panel['title_key'], panel['difficulty_key']])
    labels = {k: labels_all[k] for k in sorted(keys)}
    data = copy.deepcopy(old)
    for panel in data['panels']:
        for series in panel['series']:
            series.pop('observed_pixel_y', None)
    schema = copy.deepcopy(data)
    for panel in schema['panels']:
        panel.pop('calibration', None)
        for series in panel['series']:
            series['point_count'] = len(series['x'])
            series['x'] = series['x'][:3]
            series['y'] = series['y'][:3]
    prompt = CHART + '\nAn existing pixel-digitized dataset is retained locally for this exact image.\nUse it as-is, with the supplied schema, to write compact loop-based rendering code.\nThe schema shows only first three x/y values per series; full arrays are in the actual\ndata passed to render. Do not reproduce, regenerate or modify numerical arrays.\nRETURN ONLY {"python_code":"def render(data, labels): ...", "recovery":{...}}.\nNo labels or data in the response. Loop over data[\'panels\']; use stored xlim, ylim,\nxticks, yticks, row/col positions, title_key and difficulty_key. Plot every stored\nseries x/y using its color. Retain all21panels. Use a 2400x3000 canvas and translated\ntitle/axis/legend placements. The inherited pixel-digitization uncertainty remains;\ndo not claim exact recovery. This is a code-generation call, not new data extraction.\n'
    (result, key) = b.api.call('chart_code_from_preserved_digitization', identifier, prompt, {'labels': labels, 'data_schema': schema}, image=b.image(identifier), max_tokens=7000)
    b.rt.validate_code(result['python_code'])
    spec = {'kind': 'chart', 'id': identifier, 'base_id': b.identities[identifier]['base_id'], 'source': 'CharXiv', 'data': data, 'labels': labels, 'python_code': result['python_code'], 'recovery': dict(result['recovery'], method='API_renderer_over_preserved_pixel_digitization', prior_data_sha256=digest(old), prior_data_path=str(pilot / 'specs.json'), inherited_uncertainty=old.get('uncertainty')), 'recovery_request_sha256': key}
    save(folder / 'reconstruction_from_pilot.json', spec)
    save(folder / 'spec.json', spec)
    save(folder / 'english_labels.json', {'labels': labels})
    b.render_one(identifier, folder / 'spec.json', folder / 'english_labels.json', folder / 'baseline.png')
    save(folder / 'recovery_complete.json', {'finished_at': now(), 'spec_sha256': digest(spec), 'mechanical_render_pass': True, 'semantic_fidelity': 'pending_api_review'})
    b.normalize_qa(b.by_id[identifier])
    print('Dense chart API code generated from retained digitization.', flush=True)
if __name__ == '__main__':
    main()
