pass
import csv, json, hashlib
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/benchmark/validation_release/val.candidates.jsonl'
OUT = ROOT / 'outputs/figures'
TEXT_NUMERIC = {'3cb6452a2a243405fae2': 'Scale word billion requires translation.', '54e1b09069328391f913': 'Scale word million requires translation.', '14a8f81e18e712798c54': 'Reference includes a verbal explanation and category name.', '67889af29211ab1128fd': 'Reference includes the words All and grape varieties.', '72535a8c0ee6664fabd0': 'Reference is a sentence with a verbal unit and approximation.'}
FIXED_OTHER = {'3d60f3188bd83b417d5e': 'Fixed experiment identifier SA-QID-dev.', 'ef91d5570ddf4f10896c': 'Mathematical parameter and numeric value.', 'ec3da0f1c31be820cc61': 'Fixed acronym EBP.', '659d4312e8d18271430f': 'Fixed identifier 04_MSY.', 'a97c3c13b5f40162dd14': 'Fixed algorithm identifier Root-MUSIC.', 'ce3cd691bfbeb2813fd9': 'Numeric angle list, with degree symbols.', '0c515d14c641ec4b6d5f': 'Mathematical symbol omega_z.', 'cfadf691562200d04818': 'Numeric range.', 'd257d2ff1f5867cfaa3f': 'Fixed method identifier HLF-SZO.', 'c956813ef506d7ebf442': 'Fixed acronym UKBB ICA.', '8d37c5749eaa9ab6d4ff': 'Mathematical symbol alpha.', 'afaaeebc1399e1b983e1': 'Numeric date representation, no lexical translation.'}

def main():
    rows = [json.loads(s) for s in SOURCE.read_text().splitlines()]
    seeds = [r for r in rows if r['image_language'] == r['query_language'] == 'en']
    assert len(seeds) == len({r['case_id'] for r in seeds}) == 128
    assert set(TEXT_NUMERIC) | set(FIXED_OTHER) <= set((r['case_id'] for r in seeds))
    result = []
    for r in seeds:
        cid = r['case_id']
        stable = r['answer_type'] == 'numeric'
        reason = 'Numeric or mathematical reference without lexical content.' if stable else 'Contains words or names subject to translation/transliteration.'
        if cid in TEXT_NUMERIC:
            stable = False
            reason = TEXT_NUMERIC[cid]
        if cid in FIXED_OTHER:
            stable = True
            reason = FIXED_OTHER[cid]
        result.append(dict(case_id=cid, source=r['source'], answer=r['answer'], original_answer_type=r['answer_type'], category='Translation-invariant' if stable else 'Language-bearing', reason=reason))
    OUT.mkdir(exist_ok=True, parents=True)
    with (OUT / 'answer_translation_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    data = dict(n=len(seeds), source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), unit='Unique seed QA, English reference; each seed counted once.', sources=dict(Counter((r['source'] for r in seeds))), answers=dict(Counter((r['category'] for r in result))), policy='Translation-invariant includes numbers, numeric ranges/dates, symbols and fixed identifiers. Language-bearing includes lexical units, explanations, and names subject to translation/transliteration. Formatting may still differ. This is a reference-content classification, not a claim that every translation string is identical.', metadata_answer_types=dict(Counter((r['answer_type'] for r in seeds))))
    (OUT / 'seed_composition.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(data, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
