pass
import argparse
import hashlib
import csv
import json
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from scripts.final_benchmark.api import save
from scripts.final_benchmark.query_policy import is_plain_number
LANGUAGES = 'en zh ja ko fr de es pt ru ar hi it nl pl tr vi id th sw fa ur bn ta te'.split()
METRIC = 'ACC: exact numeric value for plain numbers; otherwise NFC + strip strict match; no tolerance'

def latex_escape(text):
    escapes = {'\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#', '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}', '^': '\\textasciicircum{}'}
    return ''.join((escapes.get(c, c) for c in text))

def correct(prediction, answer):
    p = unicodedata.normalize('NFC', str(prediction)).strip()
    a = unicodedata.normalize('NFC', str(answer)).strip()
    if is_plain_number(p) and is_plain_number(a):
        return Decimal(p.replace(',', '').replace('−', '-')) == Decimal(a.replace(',', '').replace('−', '-'))
    return p == a

def summarize(refs, predictions, verdicts=None):
    buckets = defaultdict(list)
    details = []
    for r in refs:
        pred = predictions.get(r['id'], {})
        ok = pred.get('status') == 'completed'
        hit = bool(ok and (verdicts.get(r['id'], False) if verdicts is not None else correct(pred['prediction'], r['answer'])))
        key = (r['image_language'], r['query_language'])
        buckets[key].append(hit)
        details.append({'id': r['id'], 'source': r['source'], 'visual_kind': r['visual_kind'], 'image_language': key[0], 'query_language': key[1], 'correct': hit, 'prediction_status': pred.get('status', 'missing'), 'prediction': pred.get('prediction'), 'reference': r['answer']})
    configs = {'/'.join(k): {'ACC': 100 * sum(v) / len(v), 'n': len(v)} for (k, v) in buckets.items()}

    def macro(keys):
        return sum((configs[k]['ACC'] for k in keys)) / len(keys) if keys and all((k in configs for k in keys)) else None
    xqa = {q.upper(): macro([v + '/' + q for v in LANGUAGES if v != q]) for q in ('zh', 'en')}
    lqa = {l.upper(): configs.get(l + '/' + l, {}).get('ACC') for l in LANGUAGES}
    expected = [v + '/' + q for v in LANGUAGES for q in sorted({v, 'en', 'zh'})]
    n = len(refs)
    success = sum((predictions.get(r['id'], {}).get('status') == 'completed' for r in refs))
    return ({'n': n, 'successful_predictions': success, 'failed_or_missing': n - success, 'prediction_coverage': success / n if n else None, 'XQA': xqa, 'LQA': lqa, 'AVG': macro(expected), 'micro_ACC': 100 * sum((x['correct'] for x in details)) / n if n else None, 'configurations': configs, 'covered_configurations': len(configs), 'failure_policy': 'Failed/missing predictions count as incorrect in the fixed denominator.'}, details)

def score_run(run):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    if hashlib.sha256((run / 'references.jsonl').read_bytes()).hexdigest() != manifest['references_sha256']:
        raise ValueError('Reference snapshot changed')
    refs = [json.loads(s) for s in (run / 'references.jsonl').read_text().splitlines()]
    records = [json.loads(p.read_text()) for p in sorted((run / 'predictions').glob('*.json'))]
    if any((r.get('run_key') != manifest['run_key'] for r in records)):
        raise ValueError('Prediction belongs to a different run')
    preds = {r['id']: r for r in records}
    if len(preds) != len(records) or len({r['id'] for r in refs}) != len(refs):
        raise ValueError('Duplicate IDs')
    if set(preds) != {r['id'] for r in refs}:
        raise ValueError('Inference has not reached a terminal outcome for every reference; scoring deferred')
    if any((r['status'] not in ('completed', 'failed') for r in records)):
        raise ValueError('Nonterminal predictions')
    results = {}
    for (name, subset) in [('all', refs), ('screened', [r for r in refs if r.get('audit', {}).get('status') == 'accepted']), ('chart', [r for r in refs if r.get('visual_family', r['visual_kind']) == 'chart']), ('table', [r for r in refs if r.get('visual_family', r['visual_kind']) == 'table'])]:
        (results[name], details) = summarize(subset, preds)
        if name == 'all':
            (run / 'scored_predictions.jsonl').write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in details)))
    report = {'metric': METRIC, 'model': json.loads((run / 'manifest.json').read_text())['model'], 'answer_policy_amendments': manifest.get('policy_amendments', []), 'primary_cohort': 'all candidates; not a human-certified benchmark release', 'cohorts': results}
    save(run / 'scores.json', report)
    (run / 'predictions.jsonl').write_text(''.join((json.dumps(preds[r['id']], ensure_ascii=False) + '\n' for r in refs)))
    headers = ['cohort', 'model', 'XQA_ZH', 'XQA_EN'] + ['LQA_' + l.upper() for l in LANGUAGES] + ['AVG', 'n', 'successful_predictions']
    with (run / 'leaderboard.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for (cohort, s) in results.items():
            writer.writerow([cohort, report['model'], s['XQA']['ZH'], s['XQA']['EN']] + list(s['LQA'].values()) + [s['AVG'], s['n'], s['successful_predictions']])
    s = results['all']
    values = [s['XQA']['ZH'], s['XQA']['EN']] + list(s['LQA'].values()) + [s['AVG']]
    fmt = lambda v: '\\textemdash' if v is None else '100' if v == 100 else f'{v:.1f}'
    tex = '% ' + METRIC + '\n% Failed requests count as incorrect; see scores.json for coverage.\n'
    display_model = {'gpt-6-astra': 'GPT-6-Astra', 'google/gemini-3.8-flash': 'Gemini-3.8-Flash'}.get(report['model'], report['model'])
    display_model = latex_escape(display_model)
    tex += display_model + ' & ' + ' & '.join(map(fmt, values)) + ' \\\\' + '\n'
    (run / 'main_table_row.tex').write_text(tex)
    if (run / 'table_template.tex').exists():
        template = (run / 'table_template.tex').read_text()
        marker = template.index('\\midrule') + len('\\midrule')
        start = marker
        while template[start].isspace():
            start += 1
        end = template.index('\\bottomrule', start)
        row = tex.splitlines()[-1]
        table = template[:start] + row + '\n    ' + template[end:]
        table = table.replace('Accuracy (', 'Exact-value/strict-text accuracy (')
        table = table.replace('\\textemdash{} indicates an unevaluated entry.', 'Failed requests count as incorrect; prediction coverage is reported with the run.')
        (run / 'main_results.tex').write_text(table)
    return report
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--llm-judge', action='store_true', help='After strict scoring, run the API answer-equivalence judge in a separate output directory')
    a = p.parse_args()
    result = score_run(a.run)
    print(json.dumps(result['cohorts']['all'], ensure_ascii=False))
    if a.llm_judge:
        from scripts.evaluation.judge import run_judge
        run_judge(a.run)
