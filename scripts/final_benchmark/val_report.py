pass
from collections import Counter
import json
from .api import read, save
from .pipeline import DEFAULT_OUT, LANGUAGES

def main():
    dest = DEFAULT_OUT / 'validation_release'
    rows = [json.loads(x) for x in (dest / 'val.candidates.jsonl').read_text().splitlines()]
    base = {r['case_id']: r for r in rows}
    reviews = {p.stem: read(p) for p in (dest / 'reviews').glob('*.json') if p.stem in base}

    def fraction(n, d):
        return {'passed': n, 'total': d, 'rate': n / d if d else None}
    metrics = {}
    for name in ('source_fidelity', 'answer_preservation', 'normal_qa'):
        metrics[name] = fraction(sum((r.get(name) == 'pass' for r in reviews.values())), len(base))
    metrics['translation_equivalence'] = fraction(sum((v.get('translation') == 'pass' and v.get('answer_equivalence') == 'pass' for r in reviews.values() for v in r.get('languages', {}).values())), len(base) * len(LANGUAGES))
    metrics['API_render_readability'] = fraction(sum((v.get('render_readability') == 'pass' for r in reviews.values() for v in r.get('languages', {}).values())), len(base) * len(LANGUAGES))
    metrics['reference_binding'] = fraction(sum((not set(r['audit']['mechanical_issues']) & {'query_binding', 'answer_binding', 'unbound_reference'} for r in rows)), len(rows))
    metrics['mechanical_validity'] = fraction(sum((not r['audit']['mechanical_issues'] for r in rows)), len(rows))
    accepted = [r for r in rows if r['audit']['status'] == 'accepted']
    metrics['admission'] = fraction(len(accepted), len(rows))
    images = {r['image_path']: r for r in rows}
    metrics['render_integrity'] = fraction(sum((not set(r['audit']['mechanical_issues']) & {'image_hash', 'render_geometry_or_glyphs'} for r in images.values())), len(images))

    def diversity(rs):
        bs = {r['base_id']: r for r in rs}
        return {'samples': len(rs), 'base_cases': len(bs), 'distinct_images': len({r['image_path'] for r in rs}), 'sources_base': dict(Counter((r['source'] for r in bs.values()))), 'difficulty_base': dict(Counter((r['difficulty']['label'] for r in bs.values()))), 'chart_types_base_multilabel': dict(Counter((t for r in bs.values() for t in r['chart_types']))), 'table_types_base': dict(Counter((r['table_structure']['type'] for r in bs.values() if r['table_structure']))), 'reasoning_operations_base_multilabel': dict(Counter((t for r in bs.values() for t in r['features'].get('reasoning_operations', [])))), 'query_languages_samples': dict(Counter((r['query_language'] for r in rs))), 'image_languages_samples': dict(Counter((r['image_language'] for r in rs))), 'configurations_samples': dict(Counter((r['configuration'] for r in rs)))}
    report = {'metrics': metrics, 'candidate_diversity': diversity(rows), 'admitted_diversity': diversity(accepted), 'review_coverage': fraction(len(reviews), len(base)), 'curation_issue_counts': dict(Counter((x for r in rows for x in r['audit'].get('curation_issues', [])))), 'interpretation': 'Automated same-provider audit. Human verification not performed. Pending/uncertain are not passes. Variants are correlated.'}
    attempts = [read(p) for stage in (DEFAULT_OUT / 'api').iterdir() if stage.name == 'val_multilingual_audit_v1' or stage.name.startswith('val_batch_') for p in stage.glob('*/*/attempt_*.json')]
    report['audit_API_usage'] = {'attempts': len(attempts), 'statuses': dict(Counter((x.get('status', 'unknown') for x in attempts))), 'observed_tokens': {k: sum(((x.get('usage') or {}).get(k, 0) for x in attempts)) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}, 'attempts_with_usage': sum((bool(x.get('usage')) for x in attempts)), 'scope': 'this validation audit only; missing usage is unknown, not zero'}
    if (dest / 'code_invariance.json').exists():
        report['code_invariance'] = read(dest / 'code_invariance.json')
    save(dest / 'quality_metrics.json', report)
    issue_rows = []
    for (cid, row) in base.items():
        review = reviews.get(cid, {})
        variants = [r for r in rows if r['case_id'] == cid]
        categories = []
        if not review:
            categories.append('audit_pending')
        for field in ('source_fidelity', 'answer_preservation', 'normal_qa'):
            if review and review.get(field) != 'pass':
                categories.append(field + '_' + review.get(field, 'pending'))
        languages = {l: {k: v.get(k, 'pending') for k in ('translation', 'answer_equivalence', 'render_readability')} for (l, v) in review.get('languages', {}).items() if any((v.get(k) != 'pass' for k in ('translation', 'answer_equivalence', 'render_readability')))}
        if languages:
            categories.append('language_quality_issues')
        curation = sorted({x for r in variants for x in r['audit'].get('curation_issues', [])})
        if curation:
            categories.append('answer_or_source_curation')
        if any((r['audit'].get('localization_flags') and any(r['audit']['localization_flags'].values()) for r in variants)):
            categories.append('localization_flags_require_adjudication')
        if review.get('critical_issues'):
            categories.append('critical_issues')
        issue_rows.append({'case_id': cid, 'source': row['source'], 'categories': categories, 'review_available': bool(review), 'accepted_variants': sum((r['audit']['status'] == 'accepted' for r in variants)), 'total_variants': len(variants), 'language_checks_needing_attention': languages, 'critical_issues': review.get('critical_issues', []), 'curation_issues': curation, 'source_evidence': review.get('source_evidence'), 'review_path': f'reviews/{cid}.json' if review else None})
    (dest / 'quality_issues_by_case.jsonl').write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in issue_rows)))
    save(dest / 'quality_issue_summary.json', {'base_cases': len(issue_rows), 'category_counts_multilabel': dict(Counter((t for r in issue_rows for t in r['categories']))), 'note': 'Overlapping categories counted per source question; a pending audit is not a failed audit.'})
    text = ['# Validation audit results', '', report['interpretation'], '', '| Metric | Passed / Total | Rate |', '|---|---:|---:|']
    for (k, v) in metrics.items():
        text.append(f"| {k} | {v['passed']} / {v['total']} | {100 * v['rate']:.2f}% |")
    text += ['', '## Diversity by source question', '', '```json', json.dumps({'candidates': report['candidate_diversity'], 'admitted': report['admitted_diversity']}, ensure_ascii=False, indent=2), '```']
    (dest / 'AUDIT_RESULTS.md').write_text('\n'.join(text) + '\n')
    screened = len(accepted)
    kept = len({r['base_id'] for r in accepted})
    latex = '\\paragraph{Validation screening.} ' + f'We applied deterministic checks to {len(rows):,} configurations derived from {len(base)} source questions and completed automated semantic audits for {len(reviews)} source questions. ' + f'The screened validation subset contains {screened:,} configurations from {kept} source questions. ' + 'The audit assessed source fidelity, reference-answer preservation, QA suitability, translation equivalence, and rendered readability. Uncertain or failed configurations were retained separately. Difficulty and chart-type labels are automated annotations, not human ground truth. ' + 'All language variants retain their source figure/document grouping. Automated review used the configured generation provider and does not constitute independent human verification.' + '\n\n' + '\\paragraph{Evaluation.} We use Unicode-NFC and outer-whitespace-normalized exact match as the baseline accuracy metric, reporting micro accuracy and macro averages by source, base question, and language. We report matched monolingual-minus-cross-language accuracy gaps for fixed visual languages. We obtain 95\\% percentile confidence intervals using 2,000 bootstrap resamples of connected source figure/document groups (seed 42), preserving dependencies among language variants. Numeric tolerance and language-specific answer normalization require separately declared per-case rules and are not inferred from model outputs.' + '\n'
    (dest / 'appendix_results.tex').write_text(latex)
    print(json.dumps({'metrics': metrics, 'reviewed': len(reviews)}))
if __name__ == '__main__':
    main()
