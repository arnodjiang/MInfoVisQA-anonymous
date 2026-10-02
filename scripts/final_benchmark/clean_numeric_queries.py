pass
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from .api import digest, read, save
from . import export
from .languages24 import activate
from .query_policy import is_plain_number

def clean(out):
    activate()
    names = ['benchmark.jsonl', 'benchmark.api_reviewed.jsonl', 'validation_release/val.candidates.jsonl', 'validation_release/val.jsonl', 'validation_release/val.needs_review.jsonl']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = out / 'cleaning_history' / ('numeric_reply_' + stamp)
    updates = []
    counts = {}
    for name in names:
        path = out / name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        changed = 0
        field = 'question' if name.startswith('benchmark') else 'query'
        for row in rows:
            suffix = '\n' + export.REPLY[row['answer_language']]
            if is_plain_number(row['answer']) and row[field].endswith(suffix):
                row[field] = row[field][:-len(suffix)]
                if field == 'question':
                    assert row[field] == row['question_without_instruction']
                    row['query_revision'] = digest([row['variant_id'], row[field]])
                changed += 1
        counts[name] = changed
        if changed:
            updates.append((path, rows))
    if not updates:
        print(json.dumps({'changed': counts, 'already_clean': True}))
        return
    for (path, _) in updates:
        target = backup / path.relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for (path, rows) in updates:
        temp = path.with_suffix('.tmp')
        temp.write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in rows)), encoding='utf-8')
        temp.replace(path)
    report = {'rule': 'plain ASCII numeric answers omit the exact generated language suffix', 'changed': counts, 'backup': str(backup.relative_to(out)), 'source_qa_answers_images_and_audits_unchanged': True}
    save(backup / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    clean(parser.parse_args().output.resolve())
if __name__ == '__main__':
    main()
