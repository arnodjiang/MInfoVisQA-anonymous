pass
import argparse, csv, hashlib, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scripts.final_benchmark.api import digest
from scripts.evaluation.context_input import input_text, context_prompt
from scripts.evaluation.reference_corrections import corrected_reference
from scripts.evaluation.score import LANGUAGES
ROOT = Path(__file__).resolve().parents[2]
RUNS = [('GPT-6 Astra', 'gpt6_astra_v4_context_20260916'), ('Gemini-3.8-Flash', 'gemini38_flash_tokenrouter_v4_context_max_tokens_repair_20260918'), ('GPT-5.6-Sol', 'gpt56_sol_v4_context_20260916'), ('GPT-5.6-Luna', 'gpt56_luna_v5_20260918')]
DEFAULT_DATA = ROOT / 'data/benchmark/validation_release/val.candidates.jsonl'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def indexed(p):
    rows = [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    out = {r['id']: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f'Duplicate IDs: {p}')
    return out

def seed_metrics(y, baseline_index):
    pass
    y = np.asarray(y, dtype=float)
    base = y[:, baseline_index]
    other = np.delete(y, baseline_index, axis=1)
    return np.column_stack([base, other.mean(1), other.mean(1) - base, ((base[:, None] == 1) & (other == 0)).mean(1), ((base[:, None] == 0) & (other == 1)).mean(1)]) * 100

def estimate(v, reps=10000, seed=20261002):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return {'n_seeds': 0, 'point': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (reps, len(v)))
    boot = v[idx].mean(axis=1)
    return {'n_seeds': len(v), 'point': v.mean(axis=0).tolist(), 'ci95': np.quantile(boot, [0.025, 0.975], axis=0).T.tolist()}

def classify_failure(pred, judgment):
    if judgment.get('method') == 'judge_error':
        return 'judge_error'
    if pred['status'] == 'completed':
        return None
    return 'inference_failure'

def analyze(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = indexed(args.dataset)
    ids = set(data)
    if len(data) != 8960:
        raise ValueError('Expected 8960 frozen configurations')
    seeds = sorted({r['case_id'] for r in data.values()})
    assert len(seeds) == 128
    expected = {(v, q) for v in LANGUAGES for q in {v, 'en', 'zh'}}
    for cid in seeds:
        group = [r for r in data.values() if r['case_id'] == cid]
        assert len(group) == 70 and {(r['image_language'], r['query_language']) for r in group} == expected
    assets = {}
    for r in data.values():
        path = (Path(args.dataset).parent / r['image_path']).resolve()
        if path not in assets:
            assets[path] = sha(path)
        assert assets[path] == r['image_sha256'], ('Changed visual', r['id'])
    audit = {r['case_id']: r for r in csv.DictReader(open(ROOT / 'outputs/figures/answer_translation_audit.csv'))}
    report = {'schema': 'paired-visual-language-v1', 'dataset_sha256': sha(args.dataset), 'bootstrap': {'repetitions': args.bootstrap, 'seed': args.seed, 'unit': 'seed cluster', 'interval': 'percentile 95%; sampling uncertainty, not API repetition variance'}, 'metric_order': ['baseline_acc', 'cross_visual_acc', 'delta_pp', 'correct_to_wrong_pct', 'wrong_to_correct_pct'], 'models': {}, 'classification_review': 'Existing answer-translation audit reused; borderline labels are not newly human-certified.'}
    details = []
    curves = []
    table = []
    subrows = []
    validations = {}
    for (display, name) in RUNS:
        run = ROOT / 'data/evaluation' / name
        manifest = read(run / 'manifest.json')
        refs = indexed(run / 'references.jsonl')
        jd = run / 'llm_judge_text_v2'
        jm = read(jd / 'manifest.json')
        judges = indexed(jd / 'judgments.jsonl')
        corrections = read(jd / 'reference_corrections.json')
        assert ids == set(refs) == set(judges)
        assert sha(run / 'references.jsonl') == manifest['references_sha256'] == jm['references_sha256']
        prompt = (run / 'prompt.txt').read_text()
        judge_prompt = (jd / 'prompt.txt').read_text()
        assert digest(prompt) == manifest['prompt_sha256']
        assert digest(judge_prompt) == jm['prompt_sha256']
        buckets = defaultdict(list)
        failures = []
        amendments = []
        preds = {}
        attempt_checks = Counter()
        for (rid, r) in refs.items():
            latest = data[rid]
            for k in ['case_id', 'base_id', 'query', 'answer', 'image_sha256', 'query_language', 'image_language', 'answer_language', 'source_context']:
                assert latest.get(k) == r.get(k), (display, rid, k)
            pred = read(run / 'predictions' / f'{rid}.json')
            request = read(run / 'requests' / f'{rid}.json')
            j = judges[rid]
            preds[rid] = pred
            assert pred['id'] == rid and pred['run_key'] == manifest['run_key']
            assert request['question'] == r['query'] and request['image_sha256'] == r['image_sha256']
            assert request.get('source_context') == r.get('source_context')
            assert request.get('input_text', request['question']) == input_text(r)
            assert request['prompt_sha256'] == digest(context_prompt(prompt, r))
            assert j['prediction'] == pred.get('prediction') and j['reference_effective'] == corrected_reference(r, corrections)[0]
            assert j['input_fingerprint'] == digest([r, pred, corrected_reference(r, corrections), digest(judge_prompt)]), (display, rid, 'stale judgment')
            assert j['verdict'] in ['equivalent', 'different']
            selected = pred.get('selected_attempt')
            attempt_path = run / 'attempts' / rid / f'{selected:03d}.json' if isinstance(selected, int) else None
            a = read(attempt_path) if attempt_path and attempt_path.exists() else {}
            mode = a.get('answer_mode', pred.get('answer_mode', 'legacy_unspecified'))
            attempt_checks[mode] += 1
            if a.get('prompt_sha256'):
                effective = (run / 'plain_answer_prompt.txt').read_text() if mode == 'plain' else prompt
                assert a['prompt_sha256'] == digest(context_prompt(effective, r)), (display, rid, 'attempt prompt')
            else:
                attempt_checks['no_attempt_prompt_hash'] += 1
            request = dict(request, selected_answer_mode=mode)
            assert j['verdict'] in ['equivalent', 'different']
            if r['query_language'] in ['en', 'zh']:
                failure = classify_failure(pred, j)
                if failure:
                    failures.append({'id': rid, 'case_id': r['case_id'], 'query_language': r['query_language'], 'visual_language': r['image_language'], 'kind': failure, 'error': pred.get('error', j.get('reason'))})
                buckets[r['case_id'], r['query_language']].append((r, pred, request, j, failure))
        strict_policy_counts = Counter()
        model = {}
        curvemodel = {}
        metricmodel = {}
        for q in ['en', 'zh']:
            Y = []
            clean = []
            failure_free = []
            groups = defaultdict(list)
            for (i, cid) in enumerate(seeds):
                entries = sorted(buckets[cid, q], key=lambda e: LANGUAGES.index(e[0]['image_language']))
                assert len(entries) == 24
                for field in ['query', 'answer', 'source_context']:
                    assert len({digest(e[0].get(field)) for e in entries}) == 1, (display, cid, q, field)
                assert len({digest(e[2]['prompt_sha256']) for e in entries}) == 1
                configs = {digest({k: e[2].get(k) for k in ['generationConfig', 'max_output_tokens', 'schema', 'model']}) for e in entries}
                modes = {e[2]['selected_answer_mode'] for e in entries}
                has_plain = 'plain' in modes
                if has_plain:
                    amendments.append({'case_id': cid, 'question_language': q, 'issue': 'selected plain-answer fallback changes effective output-format prompt'})
                assert len({digest(e[3]['reference_effective']) for e in entries}) == 1
                if len(configs) > 1:
                    amendments.append({'case_id': cid, 'question_language': q, 'issue': 'generation settings differ across visual variants'})
                vals = [int(e[1]['status'] == 'completed' and e[3]['verdict'] == 'equivalent') for e in entries]
                Y.append(vals)
                failure_free.append(not any((e[4] for e in entries)))
                clean.append(failure_free[-1] and len(configs) == 1 and (not has_plain))
                row = entries[0][0]
                meta = data[row['id']]
                family = meta['visual_family']
                kind = meta['visual_kind']
                groups['all'].append(i)
                groups[family].append(i)
                if kind == 'Simple Table':
                    groups['simple_table'].append(i)
                elif kind in ['Row-Spanning Table', 'Column-Spanning Table', 'Mixed-Spanning Table']:
                    groups['spanning_table'].append(i)
                elif family == 'table':
                    groups['composite_table'].append(i)
                a = audit[cid]
                en = next((r for r in data.values() if r['case_id'] == cid and r['query_language'] == r['image_language'] == 'en'))
                assert a['answer'] == str(en['answer'])
                groups[a['category']].append(i)
                b = vals[LANGUAGES.index(q)]
                for (val, e) in zip(vals, entries):
                    (r, p, req, j, f) = e
                    details.append({'model': display, 'case_id': cid, 'id': r['id'], 'question_language': q, 'visual_language': r['image_language'], 'visual_family': family, 'visual_kind': kind, 'answer_translation_category': a['category'], 'baseline_correct': b, 'correct': val, 'failure': f, 'effective_reference': j['reference_effective'], 'prediction': p.get('prediction'), 'request_sha256': sha(run / 'requests' / f"{r['id']}.json"), 'judge_fingerprint': j['input_fingerprint']})
            Y = np.array(Y)
            M = seed_metrics(Y, LANGUAGES.index(q))
            assert np.allclose(M[:, 2], M[:, 4] - M[:, 3])
            model[q] = {}
            metricmodel[q] = M
            curvemodel[q] = Y
            for (group, indices) in groups.items():
                stats = estimate(M[indices], args.bootstrap, args.seed)
                model[q][group] = stats
                subrows.append({'model': display, 'query_language': q, 'group': group, 'n_seeds': len(indices), **dict(zip(report['metric_order'], stats['point']))})
            weighted = sum((model[q][f]['n_seeds'] * model[q][f]['point'][2] for f in ['chart', 'table'])) / 128
            assert abs(weighted - model[q]['all']['point'][2]) < 1e-10
            model[q]['failure_free_sensitivity'] = estimate(M[np.array(failure_free)], args.bootstrap, args.seed)
            model[q]['complete_seed_sensitivity'] = estimate(M[np.array(clean)], args.bootstrap, args.seed)
            model[q]['complete_seed_sensitivity']['note'] = 'Exclude entire seed-language clusters with any inference/Judge failure or varying request settings or selected plain-answer fallback; all 24 variants remain paired. Inference failures include content failures, not only transport failures.'
            K = Y.sum(1)
            coverage = np.array([K >= k for k in range(1, 25)]).T * 100
            stats = estimate(coverage, args.bootstrap, args.seed)
            model[q]['coverage'] = stats
            for (k, (point, ci)) in enumerate(zip(stats['point'], stats['ci95']), 1):
                curves.append({'model': display, 'query_language': q, 'k': k, 'coverage_pct': point, 'ci_low': ci[0], 'ci_high': ci[1]})
        model['selected_attempt_checks'] = dict(attempt_checks)
        model['failures'] = failures
        model['settings_amendments'] = amendments
        report['models'][display] = model
        vals = [model[q][g]['point'][2] for g in ['chart', 'table'] for q in ['en', 'zh']] + [model[q]['all']['point'][j] for j in [3, 4] for q in ['en', 'zh']]
        table.append((display, vals))
        scores = read(jd / 'scores.json')['cohorts']['all']
        for q in ['en', 'zh']:
            assert abs(model[q]['all']['point'][0] - scores['LQA'][q.upper()]) < 1e-09
            assert abs(model[q]['all']['point'][1] - scores['XQA'][q.upper()]) < 1e-09
        validations[display] = {'run': name, 'references_sha256': sha(run / 'references.jsonl'), 'judgments_sha256': sha(jd / 'judgments.jsonl'), 'judge_manifest': jm, 'all_8960_input_and_judgment_fingerprints_verified': True, 'fixed_EN_ZH_text_context_nominal_prompt_verified': True, 'selected_attempt_checks': dict(attempt_checks), 'current_visual_files_verified': len(assets), 'legacy_attempt_prompt_hash_missing_is_not_verified': True, 'generation_settings_amendments': amendments}

    def save(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    save('paired_metrics.json', report)
    save('input_validation.json', validations)
    (out / 'paired_records.jsonl').write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in details)))
    for (name, records) in [('subgroup_metrics.csv', subrows), ('coverage_curves.csv', curves)]:
        with (out / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    tex = '% Requires booktabs and multirow. Semantic ACC; each seed is a bootstrap cluster.\n\\begin{table*}[t]\n\\centering\n\\small\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrr@{}}\n\\toprule\n\\multirow{2}{*}{Model} & \\multicolumn{2}{c}{Charts $\\Delta$ACC} & \\multicolumn{2}{c}{Tables $\\Delta$ACC} & \\multicolumn{2}{c}{Correct$\\to$Wrong} & \\multicolumn{2}{c}{Wrong$\\to$Correct} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n& EN & ZH & EN & ZH & EN & ZH & EN & ZH \\\\\n\\midrule\n'
    for (name, vals) in table:
        tex += name + ' & ' + ' & '.join((f'{v:+.1f}' if i < 4 else f'{v:.1f}' for (i, v) in enumerate(vals))) + ' \\\\' + '\n'
    tex += '\\bottomrule\n\\end{tabular*}\n\\caption{Paired sensitivity to visual-language changes. Questions and answers remain in EN or ZH. $\\Delta$ACC (percentage points) compares the other 23 visual languages with the same-language baseline, separately for charts ($n=87$) and tables ($n=41$). Both flip rates use all $128\\times23$ pairs as their denominator (\\%). Failures retain the main-table scoring policy.}\n\\label{tab:paired-language-sensitivity}\n\\end{table*}\n'
    (out / 'paired_results.tex').write_text(tex)
    plot(out, curves)
    print(json.dumps({'output': str(out), 'table': table}, ensure_ascii=False))

def plot(out, rows):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/minfovisqa-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'pdf.fonttype': 42, 'font.size': 8})
    (fig, axs) = plt.subplots(1, 2, figsize=(6.9, 2.55), sharey=True)
    for (ax, q) in zip(axs, ['en', 'zh']):
        for ((model, _), color, style) in zip(RUNS, ['#154677', '#B08235', '#548979', '#A35D67'], ['-', '--', '-.', ':']):
            r = [x for x in rows if x['model'] == model and x['query_language'] == q]
            ax.plot([x['k'] for x in r], [x['coverage_pct'] for x in r], label=model, color=color, linestyle=style, linewidth=1.5)
        ax.set_title(f'{q.upper()} questions and answers', fontsize=9)
        ax.set_xlim(1, 24)
        ax.set_ylim(0, 100)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel('Minimum visual languages answered correctly')
        ax.grid(axis='y', alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
    axs[0].set_ylabel('Seeds meeting threshold (%)')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.87, bottom=0.3, wspace=0.13)
    fig.savefig(out / 'cross_language_coverage.pdf')
    fig.savefig(out / 'cross_language_coverage.png', dpi=220)
    plt.close(fig)
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/paired_analysis')
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20261002)
    analyze(p.parse_args())
