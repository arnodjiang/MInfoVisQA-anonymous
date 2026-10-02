pass
import argparse
from collections import Counter, defaultdict
from html.parser import HTMLParser
import json
from pathlib import Path
import pyarrow.parquet as pq
from scripts.final_benchmark.api import digest, read, save
from scripts.final_benchmark.provenance import source_binding
from scripts.final_benchmark.restore_context import source_paragraphs
ROOT = Path(__file__).resolve().parents[2]

class NonTableText(HTMLParser):

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.skip = 0
        self.in_caption = False
        self.outside = []
        self.caption = []

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.depth += 1
        if tag in ('style', 'script'):
            self.skip += 1
        if tag == 'caption':
            self.in_caption = True

    def handle_endtag(self, tag):
        if tag == 'table':
            self.depth -= 1
        if tag in ('style', 'script'):
            self.skip -= 1
        if tag == 'caption':
            self.in_caption = False

    def handle_data(self, text):
        if text.strip() and (not self.skip):
            if not self.depth:
                self.outside.append(text.strip())
            if self.in_caption:
                self.caption.append(text.strip())

def audit(dataset, output):
    dataset = Path(dataset).resolve()
    (selected, by_file) = ({}, defaultdict(dict))
    for path in sorted((dataset / 'cases').glob('*/source.json')):
        source = read(path)
        candidate = source['candidate']
        binding = source_binding(path.parent)
        paragraphs = source_paragraphs(source)
        record = {'case_id': candidate['id'], 'source': candidate['source'], 'base_id': binding['base_id'], 'file': candidate['file'], 'row_index': candidate['row_index'], 'raw_fields': sorted(candidate['raw_record']), 'context_fields': sorted({p['source_field'] for p in paragraphs}), 'context_paragraphs': len(paragraphs), 'context_words': len(' '.join((p['text'] for p in paragraphs)).split()), 'source_text_sha256': digest(paragraphs) if paragraphs else None, 'requires_context_restoration': bool(paragraphs)}
        if candidate['source'] == 'TableVQA-Bench':
            parser = NonTableText()
            parser.feed(candidate['raw_record'].get('text_html_table') or '')
            record['html_outside_table_text'] = parser.outside
            record['html_caption'] = ' '.join(parser.caption)
            labels = read(path.parent / 'spec.json')['labels']
            record['caption_already_in_render_labels'] = not parser.caption or record['html_caption'] in labels.values()
        selected[candidate['id']] = record
        by_file[candidate['file']][candidate['row_index']] = candidate
    inventory = []
    for directory in sorted((ROOT / 'data/raw').iterdir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob('*.parquet')):
            parquet = pq.ParquetFile(path)
            relative = str(path.relative_to(ROOT))
            stats = {'file': relative, 'source': directory.name, 'rows': parquet.metadata.num_rows, 'columns': parquet.schema_arrow.names, 'rows_with_allowed_context': 0, 'html_rows_with_outside_text': 0}
            columns = [k for k in parquet.schema_arrow.names if k != 'image']
            index = 0
            for batch in parquet.iter_batches(batch_size=512, columns=columns):
                for raw in batch.to_pylist():
                    c = {'source': directory.name, 'raw_record': raw}
                    stats['rows_with_allowed_context'] += bool(source_paragraphs({'candidate': c}))
                    if directory.name == 'TableVQA-Bench':
                        parser = NonTableText()
                        parser.feed(raw.get('text_html_table') or '')
                        stats['html_rows_with_outside_text'] += bool(parser.outside)
                    if index in by_file.get(relative, {}):
                        candidate = by_file[relative][index]
                        if any((raw.get(k) != v for (k, v) in candidate['raw_record'].items())):
                            raise ValueError('raw_source_row_mismatch:' + candidate['id'])
                        selected[candidate['id']]['parquet_fields_verified'] = True
                    index += 1
            inventory.append(stats)
    if not all((r.get('parquet_fields_verified') for r in selected.values())):
        raise ValueError('unverified_source_rows')
    if any((r.get('html_outside_table_text') or not r.get('caption_already_in_render_labels', True) for r in selected.values())):
        raise ValueError('selected_html_context_requires_review')
    summary = []
    for source in sorted({r['source'] for r in selected.values()}):
        rows = [r for r in selected.values() if r['source'] == source]
        summary.append({'source': source, 'selected': len(rows), 'with_context': sum((r['requires_context_restoration'] for r in rows)), 'affected_configurations': 70 * sum((r['requires_context_restoration'] for r in rows))})
    report = {'scope': 'Downloaded files and current 128 selected QA; no external article retrieval.', 'dataset': str(dataset), 'summary': summary, 'inventory': inventory, 'selected': list(selected.values()), 'excluded_fields': {'ChartQAPro': ['Year', 'Answer', 'other conversation answers'], 'CharXiv': ['descriptive_a*', 'reasoning_a', 'year (provenance)'], 'TableVQA-Bench': ['text_html_table', 'text_markdown_table', 'gt'], 'MMTU': ['qa', 'label', 'gold_inds', 'steps', 'program', 'model_input']}}
    save(output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.dataset, args.output)
