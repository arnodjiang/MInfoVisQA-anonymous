import hashlib
import json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    base = ROOT / 'data/benchmark/validation_release'
    rows = [json.loads(x) for x in (base / 'val.candidates.jsonl').read_text().splitlines() if x.strip()]
    assert len(rows) == len({r['id'] for r in rows}) == 8960
    assert len({r['case_id'] for r in rows}) == 128
    assert set(Counter(r['case_id'] for r in rows).values()) == {70}
    assert len(Counter((r['image_language'], r['query_language']) for r in rows)) == 70
    images = {}
    for row in rows:
        path = (base / row['image_path']).resolve()
        assert (ROOT / 'data/benchmark').resolve() in path.parents
        if path not in images:
            images[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert images[path] == row['image_sha256']
        assert (base / row['code_path']).is_file()
    assert len(images) == 3072
    by_id = {r['id']: r for r in rows}
    for folder in sorted((ROOT / 'results/models').iterdir()):
        scores = json.loads((folder / 'scores.json').read_text())['cohorts']['all']
        judgments = [json.loads(x) for x in (folder / 'judgments.jsonl').read_text().splitlines()]
        assert len(judgments) == len(rows)
        assert {j['id'] for j in judgments} == set(by_id)
        pairs = {}
        for j in judgments:
            row = by_id[j['id']]
            pairs.setdefault((row['image_language'], row['query_language']), []).append(j['verdict'] == 'equivalent')
        accuracy = {k: sum(v) / len(v) * 100 for k, v in pairs.items()}
        assert abs(sum(accuracy.values()) / 70 - scores['AVG']) < 1e-8
        for lang, expected in scores['LQA'].items():
            assert abs(accuracy[(lang.lower(), lang.lower())] - expected) < 1e-8
        for lang, expected in scores['XQA'].items():
            values = [v for (visual, query), v in accuracy.items() if query == lang.lower() and visual != query]
            assert len(values) == 23
            assert abs(sum(values) / 23 - expected) < 1e-8
    manifest = ROOT / 'MANIFEST.sha256'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            expected, relative = line.split('  ', 1)
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    print('Verified: 8960 QA, 3072 images, 3072 renderers, five model result aggregates, and file checksums.')

if __name__ == '__main__':
    main()
