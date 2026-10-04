pass
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from dotenv import dotenv_values
from scripts.openai_config import load as load_openai_config
from .api import API, digest, now, read, save
from . import prompts
from .provenance import source_binding, verify_spec, verify_locale, locale_parent, render_is_current, render_inputs, require_table_only
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get('MVISQA_SOURCE', str(ROOT / 'data/processed/normal_qa_v3')))
DEFAULT_OUT = Path(os.environ.get('MVISQA_OUTPUT', str(ROOT / 'data/visual_benchmark/baseline'))).resolve()
FONT = os.environ.get('MVISQA_FONT', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
LANGUAGES = {'en': 'English', 'zh': 'Simplified Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'fr': 'French', 'de': 'German', 'es': 'Spanish', 'pt': 'Portuguese', 'ru': 'Russian', 'ar': 'Arabic', 'hi': 'Hindi'}

def placeholder_keys(text):
    return re.findall('\\[\\[([^\\]]+)\\]\\]', text)

def bind(text, labels):
    return re.sub('\\[\\[([^\\]]+)\\]\\]', lambda m: labels[m[1]], text)

def runtime_module():
    os.environ.setdefault('MPLCONFIGDIR', str(DEFAULT_OUT / '.matplotlib_cache'))
    sys.path.insert(0, str(ROOT / 'skills/multilingual-visual-benchmark/scripts'))
    import multilingual_drawing as drawing
    from . import render_runtime
    (render_runtime.put, render_runtime.wrap, render_runtime.mask) = (drawing.put, drawing.wrap, drawing.mask)
    return render_runtime

def parse_table(case):
    raw = case['raw_record'].get('text_html_table') or ''
    if raw:
        from visual_harness import TableParser
        parser = TableParser()
        parser.feed(raw)
        return (parser.rows, raw, 'source_html')
    path = SOURCE / 'assets' / case['id'] / 'table_00.md'
    if path.exists():
        raw = path.read_text()
        original = case['raw_record'].get('prompt') or case['raw_record'].get('text_markdown_table') or ''
        if raw not in original:
            raise ValueError('source_table_asset_mismatch:' + case['id'])
        rows = []
        for line in raw.splitlines():
            if not line.strip().startswith('|'):
                continue
            values = re.split('(?<!\\\\)\\|', line.strip().strip('|'))
            if all((re.fullmatch('[\\s:\\-]+', v) for v in values)):
                continue
            rows.append([{'text': v.strip().replace('\\|', '|'), 'rowspan': 1, 'colspan': 1} for v in values])
        return (rows, raw, 'source_markdown')
    return (None, None, None)

def table_spec(extracted, require_visual_inventory=False):
    require_table_only(extracted)
    if require_visual_inventory and (not isinstance(extracted.get('non_tabular_visuals'), list)):
        raise ValueError('missing_full_visual_inventory')
    rows = copy.deepcopy(extracted['rows'])
    corrections = []
    body_width = max((sum((int(c.get('colspan', 1)) for c in row)) for row in rows if len(row) > 1), default=0)
    for (ri, row) in enumerate(rows):
        if len(row) == 1 and body_width and (int(row[0].get('colspan', 1)) > body_width):
            corrections.append({'row': ri, 'original_colspan': row[0]['colspan'], 'rendered_colspan': body_width, 'reason': 'Source HTML footnote span exceeds the actual table width; raw input retained.'})
            row[0]['colspan'] = body_width
    labels = {}
    for (r, row) in enumerate(rows):
        for (c, cell) in enumerate(row):
            cell.setdefault('rowspan', 1)
            cell.setdefault('colspan', 1)
            if not isinstance(cell.get('text'), str):
                raise ValueError('cell_text_not_string')
            text = cell['text']
            numeric = re.fullmatch('\\s*[+−-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+−-]?\\d+)?\\s*', text)
            if not numeric and any((ch.isalpha() for ch in text)):
                key = 'cell_%03d_%03d' % (r, c)
                cell['label_key'] = key
                labels[key] = text
            else:
                cell['label_key'] = None
    rt = runtime_module()
    (_, ncols) = rt.table_geometry({'rows': rows})
    if extracted.get('title'):
        labels['table_title'] = extracted['title']
        rows.insert(0, [{'text': extracted['title'], 'label_key': 'table_title', 'colspan': ncols, 'rowspan': 1}])
    for (i, note) in enumerate(extracted.get('notes') or []):
        if not note:
            continue
        key = 'footnote_%03d' % i
        labels[key] = note
        rows.append([{'text': note, 'label_key': key, 'colspan': ncols, 'rowspan': 1}])
    return {'kind': 'table', 'labels': labels, 'data': {'rows': rows}, 'recovery': dict(extracted.get('recovery', {}), span_normalization=corrections), 'python_code': 'def render(data, labels):\n    return render_table(data, labels)\n'}

class Builder:

    def __init__(self, args):
        self.args = args
        self.out = Path(args.output).resolve()
        config = load_openai_config(ROOT)
        if getattr(args, 'translation_backend', None):
            config['TRANSLATION_BACKEND'] = args.translation_backend
        self.api = API(self.out, config, args.retry_failed)
        from scripts.translation import backend
        self.translation_backend = backend(config)
        self.rt = runtime_module()
        self.lock = threading.Lock()
        self.cases = [json.loads(x) for x in (SOURCE / 'candidates.jsonl').read_text().splitlines()]
        identity_rows = list(map(json.loads, (SOURCE / 'selected_reserve_identity.jsonl').read_text().splitlines()))
        self.identities = {r['id']: r for r in identity_rows}
        if len(self.identities) != len(identity_rows):
            raise ValueError('duplicate_source_identities')
        assert len(self.cases) == 128 and len({c['id'] for c in self.cases}) == 128
        self.by_id = {c['id']: c for c in self.cases}

    def prepare(self):
        self.out.mkdir(parents=True, exist_ok=True)
        lock = {'candidate_sha256': hashlib.sha256((SOURCE / 'candidates.jsonl').read_bytes()).hexdigest(), 'ids': [c['id'] for c in self.cases], 'base_ids': [self.identities[c['id']]['base_id'] for c in self.cases], 'languages': LANGUAGES, 'configurations_per_case': 3 * len(LANGUAGES) - 2, 'expected_samples': len(self.cases) * (3 * len(LANGUAGES) - 2), 'expected_images': len(self.cases) * len(LANGUAGES), 'answer_language_policy': f'answer follows query language; zh/en cross-lingual plus {len(LANGUAGES)} monolingual', 'requested_model': self.api.config['OPENAI_MODEL']}
        if (self.out / 'selection_lock.json').exists():
            if read(self.out / 'selection_lock.json') != lock:
                raise ValueError('frozen_selection_changed')
        else:
            save(self.out / 'selection_lock.json', lock)
        for name in ('candidates.jsonl', 'selected_reserve_identity.jsonl', 'diversity_coverage.json', 'sampling_summary.json'):
            dest = self.out / 'source' / name
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                shutil.copy2(SOURCE / name, dest)
        for case in self.cases:
            folder = self.out / 'cases' / case['id']
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / 'source.json').exists():
                save(folder / 'source.json', {'candidate': case, 'identity': self.identities[case['id']]})
            src = SOURCE / 'assets' / case['id']
            for path in src.iterdir():
                target = folder / 'original' / path.name
                if not target.exists():
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(path, target)
            self.source_binding(case['id'])

    def folder(self, identifier):
        return self.out / 'cases' / identifier

    def image(self, identifier):
        self.source_binding(identifier)
        return next(iter(sorted((self.folder(identifier) / 'original').glob('original.*'))), None)

    def source_binding(self, identifier):
        return source_binding(self.folder(identifier), self.by_id[identifier], self.identities[identifier])

    def log(self, stage, identifier, status, detail=None):
        with self.lock:
            event = {'time': now(), 'stage': stage, 'id': identifier, 'status': status, 'detail': detail}
            with (self.out / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            print(stage + ' ' + status + ' ' + identifier + (' ' + detail[:220] if detail else ''), flush=True)

    def render_one(self, identifier, spec_path, locale_path, image_path):
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
        env.update(MVISQA_FONT=FONT, MPLCONFIGDIR=str(self.folder(identifier) / '.matplotlib_cache'))
        if os.environ.get('MVISQA_FALLBACK_FONTS'):
            env['MVISQA_FALLBACK_FONTS'] = os.environ['MVISQA_FALLBACK_FONTS']
        command = [str(ROOT / '.venv/bin/python'), str(Path(__file__).with_name('worker.py')), str(spec_path), str(locale_path), str(image_path)]
        if sys.platform == 'darwin' and Path('/usr/bin/sandbox-exec').exists():
            profile = '(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (subpath ' + json.dumps(str(self.folder(identifier))) + ') (subpath "/private/tmp") (subpath "/private/var/folders"))(deny file-read* (literal ' + json.dumps(str(ROOT / '.env')) + ') (subpath ' + json.dumps(str(Path.home() / '.codex')) + '))'
            command = ['/usr/bin/sandbox-exec', '-p', profile] + command
        proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        if proc.returncode:
            raise ValueError('render_error:' + proc.stderr[-2200:])
        result = read(Path(image_path).with_suffix('.layout.json'))
        if result['missing_glyphs']:
            raise ValueError('missing_glyphs:' + repr(result['missing_glyphs'][:8]))
        if not result['all_text_inside_canvas'] or not result['all_text_inside_cells']:
            raise ValueError('text_overflow')
        return result

    def recover(self, case):
        identifier = case['id']
        folder = self.folder(identifier)
        done = folder / 'recovery_complete.json'
        binding = self.source_binding(identifier)
        if done.exists():
            verify_spec(read(folder / 'spec.json'), binding)
            return
        image = self.image(identifier)
        if image and image.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            image = None
        if case['source'] in ('TableVQA-Bench', 'MMTU', 'Visual-TableQA'):
            (rows, raw, method) = parse_table(case)
            if rows is None:
                (extracted, key) = self.api.call('table_extraction', identifier, prompts.TABLE, {'input': 'Extract all cells and identify any non-tabular visual elements.', 'source_binding': binding}, image=image, max_tokens=24000)
            else:
                extracted = {'rows': rows, 'recovery': {'method': method, 'fidelity_confidence': 'source_cells', 'uncertainties': []}}
                caption = re.search('<caption\\b[^>]*>(.*?)</caption>', raw or '', flags=re.I | re.S)
                if caption:
                    extracted['title'] = html.unescape(re.sub('<[^>]+>', '', caption[1])).strip()
                key = None
            save(folder / 'table_extraction.json', extracted)
            spec = table_spec(extracted, require_visual_inventory=rows is None)
            spec['recovery_request_sha256'] = key
        else:
            (recovered, key) = self.api.call('chart_recovery', identifier, prompts.CHART, {'input': 'Recover every visible panel and data mark in this chart.', 'source_binding': binding}, image=image, max_tokens=28000)
            spec = dict(recovered, kind='chart', recovery_request_sha256=key)
        spec.update(id=identifier, base_id=self.identities[identifier]['base_id'], source=case['source'], source_binding=binding)
        for attempt in range(3):
            save(folder / ('reconstruction_%02d.json' % attempt), spec)
            save(folder / 'spec.json', spec)
            save(folder / 'english_labels.json', {'labels': spec['labels']})
            try:
                if not isinstance(spec['data'], dict) or not all((isinstance(v, str) for v in spec['labels'].values())):
                    raise ValueError('invalid_reconstruction_schema')
                if spec['kind'] == 'chart':
                    self.rt.validate_code(spec['python_code'])
                self.render_one(identifier, folder / 'spec.json', folder / 'english_labels.json', folder / 'baseline.png')
                save(done, {'finished_at': now(), 'spec_sha256': digest(spec), 'mechanical_render_pass': True, 'semantic_fidelity': 'pending_api_review'})
                self.log('recovery', identifier, 'complete')
                return
            except Exception as exc:
                save(folder / ('reconstruction_%02d.error.json' % attempt), {'error': str(exc)})
                if spec['kind'] != 'chart' or attempt >= 2:
                    raise
                (repaired, key) = self.api.call('chart_repair_%d' % attempt, identifier, prompts.REPAIR, {'previous': spec, 'failure': str(exc)[-2600:]}, image=image, max_tokens=28000)
                spec = dict(repaired, kind='chart', id=identifier, base_id=self.identities[identifier]['base_id'], source=case['source'], recovery_request_sha256=key)

    def normalize_qa(self, case):
        identifier = case['id']
        folder = self.folder(identifier)
        path = folder / 'qa.json'
        binding = self.source_binding(identifier)
        if path.exists():
            if read(path).get('source_binding', binding) != binding:
                raise ValueError('qa_source_binding_mismatch')
            labels = read(folder / 'spec.json')['labels']
            if read(path).get('label_dictionary_sha256', digest(labels)) != digest(labels):
                raise ValueError('qa_label_dictionary_changed')
            return
        if not (folder / 'recovery_complete.json').exists():
            return
        spec = read(folder / 'spec.json')
        verify_spec(spec, binding)
        (qa, key) = self.api.call('qa_normalization', identifier, prompts.QA, {'source_question': case['question'], 'source_answer': case['answer'], 'visible_labels': spec['labels'], 'source_binding': binding}, image=self.image(identifier), max_tokens=5500)
        for name in ('question', 'answer', 'answer_template'):
            if not isinstance(qa.get(name), str) or not qa[name].strip():
                raise ValueError('invalid_normalized_qa:' + name)
        unknown = set(placeholder_keys(qa['question']) + placeholder_keys(qa['answer_template'])) - set(spec['labels'])
        if unknown:
            raise ValueError('unknown_qa_label:' + repr(unknown))
        qa['request_sha256'] = key
        qa['source_binding'] = binding
        qa['label_dictionary_sha256'] = digest(spec['labels'])
        save(path, qa)
        save(folder / 'locales/en.json', {'labels': spec['labels'], 'question': qa['question'], 'answer_template': qa['answer_template'], 'notes': [], 'policy': 'source_English_baseline', 'input_binding': locale_parent(spec, qa, binding)})
        self.log('qa', identifier, 'complete')

    def translation_jobs(self, cases):
        jobs = []
        for case in cases:
            folder = self.folder(case['id'])
            if not (folder / 'qa.json').exists():
                continue
            spec = read(folder / 'spec.json')
            binding = self.source_binding(case['id'])
            verify_spec(spec, binding)
            qa = read(folder / 'qa.json')
            for path in (folder / 'locales').glob('*.json'):
                verify_locale(read(path), spec, qa, binding)
                if path.stem != 'en':
                    from scripts.translation import ensure_backend
                    ensure_backend(read(path), self.translation_backend)
            chars = sum((len(v) for v in spec['labels'].values()))
            batch_size = 5 if chars < 900 else 3 if chars < 2200 else 2 if chars < 4000 else 1
            if getattr(self.args, 'translation_batch_size', None):
                batch_size = self.args.translation_batch_size
            languages = [x for x in LANGUAGES if x != 'en' and (not (folder / 'locales' / (x + '.json')).exists())]
            for start in range(0, len(languages), batch_size):
                batch = languages[start:start + batch_size]
                if all(((folder / 'locales' / (lang + '.json')).exists() for lang in batch)):
                    continue
                jobs.append((case['id'], batch))
        return jobs

    def translate(self, job):
        (identifier, languages) = job
        folder = self.folder(identifier)
        (spec, qa) = (read(folder / 'spec.json'), read(folder / 'qa.json'))
        binding = self.source_binding(identifier)
        verify_spec(spec, binding)
        if qa.get('source_binding', binding) != binding:
            raise ValueError('qa_source_binding_mismatch')
        source = self.by_id[identifier]
        payload = {'languages': {k: LANGUAGES[k] for k in languages}, 'labels': spec['labels'], 'question': qa['question'], 'answer_template': qa['answer_template'], 'source_binding': binding, 'original_source_query': source['question'], 'original_source_answer': source['answer']}
        archived_audit = folder / 'before_quality_repair/audit.json'
        if archived_audit.exists():
            previous = read(archived_audit)
            payload['quality_repair'] = {'instruction': 'Correct substantiated translation defects while preserving every numerical value, reference, and source answer. Do not solve the QA or alter the source meaning.', 'issues': previous.get('critical_issues', []), 'language_reviews': {l: previous.get('languages', {}).get(l, {}) for l in languages}}
        (result, key) = self.api.call('translation_' + '_'.join(languages), identifier, prompts.TRANSLATE, payload, image=self.image(identifier), max_tokens=28000)
        if set(result.get('locales', {})) != set(languages):
            raise ValueError('translation_language_set_mismatch')
        for lang in languages:
            loc = result['locales'][lang]
            if set(loc.get('labels', {})) != set(spec['labels']):
                raise ValueError('translation_label_key_mismatch:' + lang)
            empty_source = {k for (k, v) in spec['labels'].items() if not v.strip()}
            if any((not isinstance(v, str) or (not v.strip() and k not in empty_source) for (k, v) in loc['labels'].items())):
                raise ValueError('empty_translated_label:' + lang)
            for name in ('question', 'answer_template'):
                if not isinstance(loc.get(name), str) or sorted(placeholder_keys(loc[name])) != sorted(placeholder_keys(qa[name])):
                    raise ValueError('translation_reference_mismatch:' + lang + ':' + name)
            loc['translation_backend'] = self.translation_backend
            loc['translation_model'] = 'nmt' if self.translation_backend == 'google' else self.api.config['OPENAI_MODEL']
            loc['request_sha256'] = key
            loc['input_binding'] = locale_parent(spec, qa, binding)
        for lang in languages:
            save(folder / 'locales' / (lang + '.json'), result['locales'][lang])
        self.log('translation', identifier, 'complete', ','.join(languages))

    def render(self, case):
        identifier = case['id']
        folder = self.folder(identifier)
        if not all(((folder / 'locales' / (lang + '.json')).exists() for lang in LANGUAGES)):
            return
        spec = read(folder / 'spec.json')
        binding = self.source_binding(identifier)
        verify_spec(spec, binding)
        qa = read(folder / 'qa.json')
        for lang in LANGUAGES:
            verify_locale(read(folder / 'locales' / (lang + '.json')), spec, qa, binding)
        if spec['kind'] == 'table' and spec.get('render_mode') != 'custom':
            spec['data']['layout'] = self.rt.table_layout(spec['data'], [read(folder / 'locales' / (lang + '.json'))['labels'] for lang in LANGUAGES])
        save(folder / 'render_spec.json', spec)
        for lang in LANGUAGES:
            image = folder / 'images' / (lang + '.png')
            if render_is_current(image, spec, read(folder / 'locales' / (lang + '.json'))['labels']):
                continue
            if image.exists():
                archive = folder / 'render_history' / ('before_input_refresh_' + hashlib.sha256(image.read_bytes()).hexdigest()[:16])
                archive.mkdir(parents=True, exist_ok=True)
                for old in (image, image.with_suffix('.layout.json')):
                    if old.exists():
                        shutil.copy2(old, archive / old.name)
            self.render_one(identifier, folder / 'render_spec.json', folder / 'locales' / (lang + '.json'), image)
        save(folder / 'render_complete.json', {'finished_at': now(), 'images': len(LANGUAGES), 'data_sha256': digest(spec['data']), 'input_binding': render_inputs(folder, LANGUAGES)})
        self.log('render', identifier, 'complete')

    def render_current(self, case):
        folder = self.folder(case['id'])
        done = folder / 'render_complete.json'
        if not done.exists() or not all(((folder / 'locales' / (l + '.json')).exists() for l in LANGUAGES)):
            return False
        binding = self.source_binding(case['id'])
        verify_spec(read(folder / 'spec.json'), binding)
        if read(done).get('input_binding') != render_inputs(folder, LANGUAGES):
            return False
        spec = read(folder / 'render_spec.json')
        qa = read(folder / 'qa.json')
        for lang in LANGUAGES:
            verify_locale(read(folder / 'locales' / (lang + '.json')), spec, qa, binding)
        return all((render_is_current(folder / 'images' / (l + '.png'), spec, read(folder / 'locales' / (l + '.json'))['labels']) for l in LANGUAGES))

    def stage(self, name, jobs, worker, workers):
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_jobs = {pool.submit(worker, job): job for job in jobs}
            for future in as_completed(future_jobs):
                job = future_jobs[future]
                try:
                    future.result()
                except Exception as exc:
                    identifier = job['id'] if isinstance(job, dict) else job[0]
                    detail = str(exc)[-2800:]
                    self.log(name, identifier, 'failed', detail)
                    failures.append({'id': identifier, 'error': detail, 'job': job if not isinstance(job, dict) else identifier})
                self.status()
        save(self.out / (name + '_failures.json'), failures)
        return failures

    def status(self):
        with self.lock:
            status = {'updated_at': now(), 'expected_cases': len(self.cases), 'expected_images': len(self.cases) * len(LANGUAGES), 'expected_samples': len(self.cases) * (3 * len(LANGUAGES) - 2), 'reconstructed': len(list((self.out / 'cases').glob('*/recovery_complete.json'))), 'qa_normalized': len(list((self.out / 'cases').glob('*/qa.json'))), 'locales_saved': len(list((self.out / 'cases').glob('*/locales/*.json'))), 'images_rendered': len(list((self.out / 'cases').glob('*/images/*.png'))), 'cases_rendered': len(list((self.out / 'cases').glob('*/render_complete.json')))}
            save(self.out / 'status.json', status)
        return status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(DEFAULT_OUT))
    parser.add_argument('--stage', choices=['prepare', 'recovery', 'qa', 'translate', 'render', 'all'], default='all')
    parser.add_argument('--translation-backend', choices=['google', 'llm'], help='Default: Google Translate API; llm is opt-in')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids', help='comma-separated IDs for targeted repairs')
    parser.add_argument('--skip-ids', default='', help='IDs already being processed separately')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    builder = Builder(args)
    builder.prepare()
    cases = [c for c in builder.cases if not args.ids or c['id'] in args.ids.split(',')]
    cases = [c for c in cases if c['id'] not in args.skip_ids.split(',')]
    if args.limit:
        cases = cases[:args.limit]
    stages = ['recovery', 'qa', 'translate', 'render'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        if stage == 'recovery':
            builder.stage(stage, cases, builder.recover, args.workers)
        elif stage == 'qa':
            builder.stage(stage, cases, builder.normalize_qa, args.workers)
        elif stage == 'translate':
            builder.stage(stage, builder.translation_jobs(cases), builder.translate, args.workers)
        elif stage == 'render':
            builder.stage(stage, cases, builder.render, min(4, args.workers))
    print(json.dumps(builder.status()), flush=True)
if __name__ == '__main__':
    main()
