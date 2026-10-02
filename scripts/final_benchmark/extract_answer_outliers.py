pass
import argparse
import json
from pathlib import Path
from .api import read
from .pipeline import DEFAULT_OUT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(DEFAULT_OUT / 'validation_release' / 'original_answer_outliers.jsonl'))
    args = parser.parse_args()
    out = DEFAULT_OUT
    records = []
    for folder in sorted((out / 'cases').iterdir()):
        if not folder.is_dir() or not (folder / 'qa.json').exists():
            continue
        qa = read(folder / 'qa.json')
        if qa.get('answer_type') not in ('long_text', 'structured'):
            continue
        source_doc = read(folder / 'source.json')
        candidate = source_doc['candidate']
        spec = read(folder / 'spec.json')
        item = {'case_id': folder.name, 'base_id': source_doc.get('identity', {}).get('base_id'), 'answer_type': qa['answer_type'], 'source_dataset': candidate.get('source'), 'source_file': candidate.get('file'), 'source_row_index': candidate.get('row_index'), 'source_split': candidate.get('split'), 'source_slot': candidate.get('slot'), 'stratum': candidate.get('stratum'), 'task_hint': candidate.get('task_hint'), 'task_tags': qa.get('task_tags', []), 'original_question': candidate.get('question'), 'original_answer': candidate.get('answer'), 'raw_record': candidate.get('raw_record'), 'normalized_question': qa.get('question'), 'normalized_answer': qa.get('answer'), 'answer_template': qa.get('answer_template'), 'normalization_notes': qa.get('normalization_notes', []), 'review_flags': qa.get('review_flags', []), 'visual_kind': spec.get('kind'), 'original_image_path': f'../cases/{folder.name}/original/original.jpg', 'reconstructed_image_path': f'../cases/{folder.name}/images/en.png', 'spec_path': f'../cases/{folder.name}/spec.json', 'qa_path': f'../cases/{folder.name}/qa.json'}
        records.append(item)
    records.sort(key=lambda x: (x['answer_type'], x['source_row_index'] or -1, x['case_id']))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in records)))
    summary = {'count': len(records), 'by_answer_type': {kind: sum((x['answer_type'] == kind for x in records)) for kind in ('long_text', 'structured')}, 'by_source_dataset': {source: sum((x['source_dataset'] == source for x in records)) for source in sorted({x['source_dataset'] for x in records})}, 'records': [{'case_id': x['case_id'], 'answer_type': x['answer_type'], 'source_dataset': x['source_dataset'], 'source_row_index': x['source_row_index']} for x in records]}
    path.with_name('original_answer_outliers.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
if __name__ == '__main__':
    main()
