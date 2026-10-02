pass
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from profile_and_sample import OUT, ROOT

def main():
    records = [json.loads(line) for line in (OUT / 'candidates.jsonl').read_text().splitlines()]
    assets = OUT / 'assets'
    assets.mkdir(exist_ok=True)
    byfile = defaultdict(dict)
    index = {}
    for r in records:
        folder = assets / r['id']
        folder.mkdir(exist_ok=True)
        raw = r['raw_record']
        info = {'id': r['id'], 'source': r['source'], 'text_files': [], 'image': None, 'text_status': 'no_structured_visual_text_in_download'}
        texts = {}
        if r['source'] == 'TableVQA-Bench':
            texts = {'table.md': raw.get('text_markdown_table'), 'table.html': raw.get('text_html_table')}
        elif r['source'] == 'MMTU':
            texts = {'original_prompt.txt': raw['prompt']}
            spans = [{'start': m.start(), 'end': m.end(), 'text': m.group(0)} for m in re.finditer('(?m)(?:^[ \\t]*\\|[^\\n]*(?:\\n|$))+', raw['prompt'])]
            (folder / 'table_spans.json').write_text(json.dumps(spans, ensure_ascii=False, indent=2))
            for (i, span) in enumerate(spans):
                texts[f'table_{i:02d}.md'] = span['text']
            info['table_span_count'] = len(spans)
            info['rendering_status'] = 'required_before_visual_evaluation'
        for (filename, text) in texts.items():
            if text is not None:
                (folder / filename).write_text(text)
                info['text_files'].append(str((folder / filename).relative_to(ROOT)))
        if info['text_files']:
            info['text_status'] = 'original_characters_preserved'
        index[r['id']] = info
        if r['source'] != 'MMTU':
            byfile[r['file']][r['row_index']] = r
    tiles = []
    counts = Counter()
    for (file, wanted) in byfile.items():
        row_index = 0
        for batch in pq.ParquetFile(ROOT / file).iter_batches(batch_size=16, columns=['image']):
            for raw in batch.column(0).to_pylist():
                if row_index in wanted:
                    r = wanted[row_index]
                    blob = raw.get('bytes') if isinstance(raw, dict) else raw
                    with Image.open(io.BytesIO(blob)) as source:
                        ext = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}.get(source.format, 'img')
                        dest = assets / r['id'] / ('original.' + ext)
                        dest.write_bytes(blob)
                        index[r['id']].update(image=str(dest.relative_to(ROOT)), width=source.width, height=source.height, image_sha256=hashlib.sha256(blob).hexdigest())
                        if counts[r['source']] < 2:
                            counts[r['source']] += 1
                            im = source.convert('RGB')
                            im.thumbnail((580, 350))
                            tile = Image.new('RGB', (600, 410), '#eeeeee')
                            tile.paste(im, ((600 - im.width) // 2, 30))
                            draw = ImageDraw.Draw(tile)
                            draw.text((10, 5), r['source'] + ' ' + r['id'], fill='black')
                            draw.text((10, 385), r['question'][:90], fill='black')
                            tiles.append(tile)
                row_index += 1
    sheet = Image.new('RGB', (1200, 410 * ((len(tiles) + 1) // 2)), 'white')
    for (i, tile) in enumerate(tiles):
        sheet.paste(tile, (i % 2 * 600, i // 2 * 410))
    sheet.save(OUT / 'visual_contact_sheet.jpg')
    hashes = Counter((x['image_sha256'] for x in index.values() if 'image_sha256' in x))
    result = {'records': list(index.values()), 'duplicate_image_byte_hashes': {k: v for (k, v) in hashes.items() if v > 1}}
    (OUT / 'asset_index.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    review = ['# 普通 QA 候选复核表', '', '全部为待复核候选；task_hint 不是人工确认标签。', '']
    for r in records:
        info = index[r['id']]
        review.extend([f"## {r['source']} / {r['id']}", '', f"- 原生分层：{r['stratum']}", f"- 任务初判：{r['task_hint']}", f"- 问题：{r['question']}", f"- 原始答案：{json.dumps(r['answer'], ensure_ascii=False)}", f"- 原始位置：`{r['file']}`，零起始行号 {r['row_index']}", f"- 表格文字状态：{info['text_status']}", ''])
        if info['image']:
            review.extend([f"![原图]({ROOT / info['image']})", ''])
    (OUT / 'review.md').write_text('\n'.join(review))
    print(json.dumps({'images_exported': sum((bool(i['image']) for i in index.values())), 'text_records': sum((bool(i['text_files']) for i in index.values())), 'exact_image_duplicates': sum((v - 1 for v in hashes.values())), 'MMTU_table_spans': [i['table_span_count'] for i in index.values() if i['source'] == 'MMTU']}))
if __name__ == '__main__':
    main()
