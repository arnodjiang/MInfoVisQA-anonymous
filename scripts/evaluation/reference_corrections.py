pass
from pathlib import Path
from scripts.final_benchmark.api import read, digest
DEFAULT = Path(__file__).resolve().parents[2] / 'configs/reference_corrections.json'

def load(path=DEFAULT):
    rows = read(path)['corrections']
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate_reference_correction')
    return {r['case_id']: r for r in rows}

def corrected_reference(row, corrections):
    c = corrections.get(row.get('case_id', row.get('id')))
    if not c:
        return (row['answer'], None)
    provenance = row['provenance']
    upstream = row.get('upstream_original_answer', row.get('original_answer', row.get('upstream_source_answer', row.get('source_answer'))))
    source_matches = upstream == c['original_answer']
    if 'upstream_answer_sha256' in row:
        source_matches = row['upstream_answer_sha256'] == digest(c['original_answer']) and row['answer'] == c['corrected_answer']
    if row['source'] != c['source'] or provenance.get('file') != c['file'] or provenance.get('row_index') != c['row_index'] or (not source_matches):
        raise ValueError('reference_correction_source_mismatch')
    return (c['corrected_answer'], c['id'])
