pass
import argparse
import fcntl
import re
from pathlib import Path
from scripts.final_benchmark.api import read, save, now
from scripts.evaluation.score import LANGUAGES, latex_escape
DISPLAY = {'gpt-6-astra': 'GPT-6-Astra', 'gpt-5.6-sol': 'GPT-5.6-Sol', 'google/gemini-3.8-flash': 'Gemini-3.8-Flash'}

def update_row(text, model, values):
    if len(values) != 27:
        raise ValueError('Expected 27 metric columns')
    label = latex_escape(DISPLAY.get(model, model))
    prefix = '\\shortstack[l]{' + label + '}'
    cells = ['\\textemdash' if v is None else f'{v:.1f}' for v in values]
    row = '    ' + prefix + '\n      & ' + ' & '.join(cells[:2]) + '\n'
    for start in range(2, 26, 4):
        row += '      & ' + ' & '.join(cells[start:start + 4]) + '\n'
    row += '      & ' + cells[-1] + ' \\\\' + '\n'
    pattern = re.compile('(?m)^ *' + re.escape(prefix) + '.*?\\\\\\\\\\n', re.S)
    if len(pattern.findall(text)) > 1:
        raise ValueError('Duplicate model rows')
    if pattern.search(text):
        return pattern.sub(lambda _: row, text, count=1)
    marker = '    ' + '\\bottomrule'
    if text.count(marker) != 1:
        raise ValueError('Expected a single table body')
    return text.replace(marker, row + marker, 1)

def publish(judge_dir, table, expected_judge=None):
    (judge_dir, table) = (Path(judge_dir), Path(table))
    progress = read(judge_dir / 'progress.json')
    manifest = read(judge_dir / 'manifest.json')
    if progress['state'] != 'finished' or progress['remaining'] != 0:
        raise ValueError('Judge not complete')
    if manifest['protocol'] != 'strict_match_then_text_equivalence_v2':
        raise ValueError('Unsupported Judge protocol')
    report = read(judge_dir / 'scores.json')
    scores = report['cohorts']['all']
    if expected_judge and report['judge_model'] != expected_judge:
        raise ValueError('Unexpected Judge model')
    if manifest['judge_model'] != report['judge_model']:
        raise ValueError('Judge model mismatch')
    if scores['n'] != progress['expected'] or scores['covered_configurations'] != 70:
        raise ValueError('Incomplete coverage')
    values = [scores['XQA']['ZH'], scores['XQA']['EN']] + [scores['LQA'][l.upper()] for l in LANGUAGES] + [scores['AVG']]
    if any((v is None or not 0 <= v <= 100 for v in values)):
        raise ValueError('Invalid accuracy')
    with table.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        updated = update_row(table.read_text(), report['model'], values)
        temp = table.with_suffix('.tmp')
        temp.write_text(updated)
        temp.replace(table)
    save(judge_dir / 'table_publication.json', {'updated_at': now(), 'model': report['model'], 'judge_model': report['judge_model'], 'table': str(table.resolve()), 'scores_source': str((judge_dir / 'scores.json').resolve()), 'values': values})
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--judge-dir', type=Path, required=True)
    p.add_argument('--table', type=Path, required=True)
    p.add_argument('--expected-judge')
    a = p.parse_args()
    publish(a.judge_dir, a.table, a.expected_judge)
