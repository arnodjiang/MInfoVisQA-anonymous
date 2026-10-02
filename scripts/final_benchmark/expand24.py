pass
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .languages24 import activate, NEW_LANGUAGES, RULES
from . import pipeline, prompts, export
from .api import read, save, digest, now
from .polish_queries import COMMON, RULES as ORIGINAL_QUERY_RULES
from .provenance import source_binding, verify_spec, verify_polish

def normalize_digits(folder):
    pass
    import unicodedata
    changed = False
    for lang in NEW_LANGUAGES:
        path = folder / 'locales' / (lang + '.json')
        if not path.exists():
            continue
        loc = read(path)

        def convert(text):
            return ''.join((str(unicodedata.decimal(c)) if unicodedata.category(c) == 'Nd' else c for c in text))
        fields = {k: convert(v) for (k, v) in loc['labels'].items()}
        (question, answer) = (convert(loc['question']), convert(loc['answer_template']))
        if fields == loc['labels'] and question == loc['question'] and (answer == loc['answer_template']):
            continue
        archive = folder / 'locale_history' / lang / (digest(loc) + '.json')
        save(archive, loc)
        loc.update(labels=fields, question=question, answer_template=answer, digit_normalization={'policy': 'Unicode decimal digits to equivalent ASCII; no numerical recomputation', 'original_sha256': digest(read(archive))})
        save(path, loc)
        changed = True
    if changed:
        stamp = now().replace(':', '').replace('.', '')
        for name in ['images', 'render_complete.json', 'render_spec.json']:
            path = folder / name
            if path.exists():
                dest = folder / 'render_history' / ('before_digit_normalization_' + stamp) / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))
    return changed

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(pipeline.ROOT / 'data/visual_benchmark/mstructqa_24'))
    p.add_argument('--source', default=str(pipeline.ROOT / 'data/visual_benchmark/final_128_v3'))
    p.add_argument('--query-polish', type=Path, help='Optional reviewed query overlay directory containing queries/<lang>/<case>.json')
    p.add_argument('--stage', choices=['prepare', 'translate', 'render', 'export', 'all'], default='all')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--translation-batch-size', type=int, choices=range(1, 14), help='Override languages per translation request; use 1 for repeated gateway timeouts.')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--ids')
    p.add_argument('--partial', action='store_true', help='Export an explicitly incomplete preview of rendered cases.')
    p.add_argument('--watch-render', action='store_true', help='Render each fully translated case as it becomes available, then export.')
    a = p.parse_args()
    activate()
    src = Path(a.source)
    out = Path(a.output)
    pipeline.SOURCE = src / 'source'
    b = pipeline.Builder(a)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'source').exists():
        shutil.copytree(src / 'source', out / 'source')
    lock = {'ids': [c['id'] for c in b.cases], 'languages': pipeline.LANGUAGES, 'configurations_per_case': 70, 'expected_samples': 8960, 'expected_images': 3072, 'source_selection_sha256': hashlib.sha256((src / 'source/candidates.jsonl').read_bytes()).hexdigest()}
    if (out / 'selection_lock.json').exists() and read(out / 'selection_lock.json') != lock:
        raise ValueError('expansion_selection_changed')
    save(out / 'selection_lock.json', lock)
    provenance = {}
    for c in b.cases:
        cid = c['id']
        origin = src / 'cases' / cid
        target = out / 'cases' / cid
        original_binding = source_binding(origin, c, b.identities[cid])
        target.mkdir(parents=True, exist_ok=True)
        if not (target / 'original').exists():
            shutil.copytree(origin / 'original', target / 'original')
        for name in ['source.json', 'spec.json', 'qa.json', 'review.json', 'recovery_complete.json', 'table_extraction.json']:
            if (origin / name).exists() and (not (target / name).exists()):
                shutil.copy2(origin / name, target / name)
        if source_binding(target, c, b.identities[cid]) != original_binding:
            raise ValueError('expansion_source_mismatch:' + cid)
        verify_spec(read(target / 'spec.json'), original_binding)
        if read(target / 'spec.json') != read(origin / 'spec.json') or read(target / 'qa.json') != read(origin / 'qa.json'):
            raise ValueError('expansion_parent_artifact_changed:' + cid)
        (target / 'locales').mkdir(exist_ok=True)
        for f in (origin / 'locales').glob('*.json'):
            if (target / 'locales' / f.name).exists():
                continue
            loc = read(f)
            polished = Path(a.query_polish) / 'queries' / f.stem / (cid + '.json') if a.query_polish else None
            if polished is not None and polished.exists():
                r = read(polished)
                verify_polish(r, cid, f.stem, c['question'], loc)
                loc['question_original'] = loc['question']
                loc['question'] = r['query']
                loc['query_polish_provenance'] = {'path': str(polished.relative_to(pipeline.ROOT)), 'sha256': hashlib.sha256(polished.read_bytes()).hexdigest(), 'status': r['status']}
            save(target / 'locales' / f.name, loc)
        provenance[cid] = {'source_spec_sha256': digest(read(origin / 'spec.json')), 'source_qa_sha256': digest(read(origin / 'qa.json'))}
    save(out / 'expansion_provenance.json', {'source': str(src), 'query_revision': str(a.query_polish) if a.query_polish else None, 'languages': pipeline.LANGUAGES, 'new_languages': NEW_LANGUAGES, 'cases': provenance, 'source_reconstruction_unchanged': True})
    pd = out / 'prompts'
    pd.mkdir(exist_ok=True)
    extra = '\nNative-language fluency requirements (apply the relevant language):\n' + '\n'.join((k + ': ' + v for (k, v) in RULES.items()))
    prompts.TRANSLATE += extra + '\nKeep the QA concise and fluent after binding labels; avoid duplicated temporal particles. Preserve time-point versus interval semantics. Do not append reply-language instructions.'
    (pd / 'translation.txt').write_text(prompts.TRANSLATE)
    for (l, rule) in dict(ORIGINAL_QUERY_RULES, **RULES).items():
        (pd / (l + '_query_copyedit.txt')).write_text(COMMON + '\nTarget language: ' + pipeline.LANGUAGES[l] + '\n' + rule)
    cases = [c for c in b.cases if not a.ids or c['id'] in a.ids.split(',')]
    stages = ['translate', 'render', 'export'] if a.stage == 'all' else ['render', 'export'] if a.watch_render else [a.stage]
    for stage in stages:
        if stage == 'translate':
            failures = b.stage(stage, b.translation_jobs(cases), b.translate, a.workers)
            if failures:
                raise SystemExit('Translation failures retained; inspect before intentional retry.')
        elif stage == 'render':
            attempted = set()
            while True:
                for c in cases:
                    folder = b.folder(c['id'])
                    if all(((folder / 'locales' / (l + '.json')).exists() for l in pipeline.LANGUAGES)):
                        normalize_digits(folder)
                ready = [c for c in cases if c['id'] not in attempted and (not b.render_current(c)) and all(((b.folder(c['id']) / 'locales' / (l + '.json')).exists() for l in pipeline.LANGUAGES))]
                if ready:
                    attempted.update((c['id'] for c in ready))
                    failures = b.stage(stage, ready, b.render, min(a.workers, 4))
                    if failures:
                        print('Rendering failures recorded; continuing other cases.', flush=True)
                if not a.watch_render or all((b.render_current(c) for c in cases)):
                    break
                if (out / 'translate_failures.json').exists() and (not ready):
                    raise SystemExit('Available cases processed; unresolved cases retained for the next run.')
                import time
                time.sleep(20)
        elif stage == 'export':
            result = export.assemble(out, partial=a.partial)
            benchmark_path = out / ('benchmark.partial.jsonl' if result['missing_cases'] else 'benchmark.jsonl')
            rows = [json.loads(x) for x in benchmark_path.read_text().splitlines()]
            for r in rows:
                if r['visual_language'] in NEW_LANGUAGES or r['query_language'] in NEW_LANGUAGES:
                    r['status'] = 'needs_review'
                    r['review_flags'].append('expanded_language_audit_pending')
            benchmark_path.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in rows)))
            (out / 'benchmark.api_reviewed.jsonl').write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in rows if r['status'] == 'api_reviewed_candidate')))
            from collections import Counter
            result['status_counts'] = dict(Counter((r['status'] for r in rows)))
            save(out / 'validation.json', result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps(b.status()), flush=True)
if __name__ == '__main__':
    main()
