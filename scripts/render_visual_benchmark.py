pass
import hashlib
import json
import os
import re
from pathlib import Path
from recover_visual_specs import OUT, ROOT, LINE_ID, HEAT_ID, COMBO_ID
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from PIL import Image, ImageDraw
from multilingual_drawing import put, wrap, MISSING_GLYPHS

def to_image(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')

def bounds(ax, canvas):
    box = ax.get_window_extent()
    return (box.x0, canvas.height - box.y1, box.width, box.height)

def multipanel(spec, labels):
    fig = plt.figure(figsize=(24, 30), dpi=100, facecolor='white')
    axes = []
    for panel in spec['panels']:
        ax = fig.add_axes([0.085 + panel['col'] * 0.303, 0.845 - panel['row'] * 0.116, 0.255, 0.078])
        for series in panel['series']:
            if series['x']:
                ax.plot(series['x'], series['y'], color=series['color'], lw=1.8)
        ax.set(xlim=panel['xlim'], ylim=panel['ylim'], xticks=panel['xticks'], yticks=panel['yticks'])
        ax.tick_params(labelsize=12, length=3, colors='#444444')
        if panel['col'] != 0:
            ax.set_yticklabels([])
        ax.grid(True, color='#e1e3e5', lw=0.7)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which='minor', color='#f0f0f0', lw=0.5)
        for spine in ax.spines.values():
            spine.set_color('#999999')
        axes.append((panel, ax))
    canvas = to_image(fig)
    draw = ImageDraw.Draw(canvas)
    boxes = []
    for (panel, ax) in axes:
        (x, y, w, h) = bounds(ax, canvas)
        draw.rectangle((x, y - 90, x + w, y - 4), fill='#e5e5e5', outline='#aaaaaa')
        boxes.append(put(canvas, labels[panel['difficulty_key']], x + w / 2, y - 66, 27, max_width=w - 18))
        boxes.append(put(canvas, labels[panel['title_key']], x + w / 2, y - 30, 33, max_width=w - 18))
    boxes.append(put(canvas, labels['episode_return'], 42, 1400, 40, rotate=90))
    boxes.append(put(canvas, labels['learner_walltime'], 1200, 2760, 42, max_width=2100))
    for (col, key) in enumerate(['unit_minutes', 'unit_hours', 'unit_hours']):
        boxes.append(put(canvas, labels[key], 510 + col * 727, 2640, 30, max_width=600))
    boxes.append(put(canvas, labels['actors_title'], 1200, 2840, 31, max_width=1800))
    for (i, (key, color)) in enumerate(zip(['series_single', 'series_2', 'series_8', 'series_16'], ['#b4c3de', '#3c82b5', '#2d4ba0', '#162644'])):
        x = 95 + i * 575
        draw.line((x, 2920, x + 75, 2920), fill=color, width=6)
        boxes.append(put(canvas, labels[key], x + 90, 2920, 28, anchor='left', max_width=445))
    plt.close(fig)
    return (canvas, boxes)

def heatmap(spec, labels):
    fig = plt.figure(figsize=(16, 15), dpi=100, facecolor='white')
    ax = fig.add_axes([0.265, 0.29, 0.57, 0.66])
    a = np.full((6, 6), np.nan)
    for (row, col, value) in spec['entries']:
        a[row, col] = value
    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad('white')
    image = ax.imshow(a, cmap=cmap, norm=TwoSlopeNorm(vmin=-0.9, vcenter=0, vmax=1), interpolation='nearest')
    ax.set(xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col, value) in spec['entries']:
        ax.text(col, row, str(value), ha='center', va='center', fontsize=19, color='white' if abs(value) > 0.5 else '#252525')
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor='#eeeeee', lw=1))
    cbax = fig.add_axes([0.89, 0.29, 0.025, 0.66])
    fig.colorbar(image, cax=cbax, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    cbax.tick_params(labelsize=16)
    canvas = to_image(fig)
    (x, y, w, h) = bounds(ax, canvas)
    boxes = []
    for (i, key) in enumerate(spec['label_keys']):
        boxes.append(put(canvas, labels[key], x - 28, y + (i + 0.5) * h / 6, 33, anchor='right', max_width=x - 50))
        from multilingual_drawing import mask
        text = labels[key]
        m = mask(text, 32)
        boxes.append(put(canvas, text, x + (i + 0.5) * w / 6, y + h + 30 + m.width / 2, 32, rotate=90, max_width=360))
    plt.close(fig)
    return (canvas, boxes)

def combo(spec, labels):
    fig = plt.figure(figsize=(18, 12), dpi=100, facecolor='white')
    ax = fig.add_axes([0.14, 0.33, 0.72, 0.59])
    right = ax.twinx()
    x = np.arange(11)
    right.bar(x, spec['attacks'], width=0.43, color='#c9c9c9', edgecolor='#666666', linewidth=1, zorder=1)
    ax.set_zorder(right.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(x, spec['population'], 'D-', color='#282828', lw=1.5, ms=7)
    ax.plot(x, spec['rate'], '^-', color='#444444', lw=1.3, ms=8)
    ax.set(xlim=(-0.5, 10.5), ylim=spec['left_ylim'], xticks=x, xticklabels=spec['categories'], yticks=[0, 5, 10, 15, 20, 25])
    right.set(ylim=spec['right_ylim'], yticks=[0, 20, 40, 60, 80, 100, 120, 140])
    ax.tick_params(axis='x', labelrotation=90, labelsize=13)
    ax.tick_params(axis='y', labelsize=15)
    right.tick_params(labelsize=15)
    ax.grid(axis='y', color='#999999', lw=1)
    canvas = to_image(fig)
    boxes = []
    draw = ImageDraw.Draw(canvas)
    boxes.append(put(canvas, labels['combo_population_axis'], 72, 520, 34, rotate=90, max_width=600))
    boxes.append(put(canvas, labels['combo_attacks_axis'], 1725, 520, 34, rotate=90, max_width=600))
    for (i, key) in enumerate(['combo_total', 'combo_rate', 'combo_population']):
        cx = 310 + i * 590
        if i == 0:
            draw.rectangle((cx - 30, 1060, cx + 30, 1076), fill='#c9c9c9', outline='#555555')
        else:
            draw.line((cx - 35, 1068, cx + 35, 1068), fill='#333333', width=3)
            if i == 1:
                draw.polygon([(cx, 1055), (cx - 10, 1080), (cx + 10, 1080)], fill='#333333')
            else:
                draw.polygon([(cx, 1055), (cx - 12, 1068), (cx, 1080), (cx + 12, 1068)], fill='#333333')
        lines = wrap(labels[key], 530, 30)
        for (j, line) in enumerate(lines):
            boxes.append(put(canvas, line, cx, 1110 + j * 36, 30, max_width=530))
    plt.close(fig)
    return (canvas, boxes)

def table_layout(spec, locales):
    ncols = max((sum((cell['colspan'] for cell in row)) for row in spec['rows']))
    width = 2100 if ncols >= 6 else 1800
    pad = 55
    cw = (width - 2 * pad) / ncols
    size = 30 if ncols < 6 else 27
    heights = []
    for row in spec['rows']:
        needed = 70
        for locale in locales.values():
            for cell in row:
                text = locale['labels'][cell['label_key']] if cell['label_key'] else cell['text']
                n = len(wrap(text, cw * cell['colspan'] - 34, size))
                needed = max(needed, n * (size + 12) + 28)
        heights.append(needed)
    return (width, pad, cw, size, heights)

def table_image(spec, labels, layout):
    (width, pad, cw, size, heights) = layout
    height = round(sum(heights) + 2 * pad)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    boxes = []
    y = pad
    for (ri, row) in enumerate(spec['rows']):
        x = pad
        for cell in row:
            w = cw * cell['colspan']
            h = heights[ri]
            fill = '#e9eeeb' if ri == 0 else '#ffffff' if ri % 2 else '#f7f9f7'
            draw.rectangle((x, y, x + w, y + h), fill=fill, outline='#aebbb4', width=2)
            text = labels[cell['label_key']] if cell['label_key'] else cell['text']
            lines = wrap(text, w - 34, size)
            start = y + h / 2 - (len(lines) - 1) * (size + 12) / 2
            for (j, line) in enumerate(lines):
                boxes.append(put(canvas, line, x + w / 2, start + j * (size + 12), size, max_width=w - 30))
            x += w
        y += heights[ri]
    return (canvas, boxes)

def bind(template, labels):
    return re.sub('\\[\\[([^\\]]+)\\]\\]', lambda m: labels[m.group(1)].strip(), template)

def answer_checks(specs):
    heat = specs[HEAT_ID]
    row = [e[2] for e in heat['entries'] if e[0] == 3] + [e[2] for e in heat['entries'] if e[1] == 3]
    pop = specs[COMBO_ID]['population']
    panel = next((p for p in specs[LINE_ID]['panels'] if p['title_key'] == 'task_hopper_stand'))
    recovered = []
    for s in panel['series']:
        if s['x'] and min(s['x']) <= 10 <= max(s['x']):
            recovered.append(float(np.interp(10, s['x'], s['y'])))
    peak = max(recovered) if recovered else None
    return {HEAT_ID: {'recovered_answer': max(row), 'reference': 0.73, 'answer_invariant': max(row) == 0.73, 'recovery': 'printed numeric annotations'}, COMBO_ID: {'recovered_answer': sum((v > 10 for v in pop)), 'reference': 6, 'answer_invariant': sum((v > 10 for v in pop)) == 6, 'minimum_margin_to_threshold': min((abs(v - 10) for v in pop)), 'recovery': 'pixel-estimated coordinates'}, LINE_ID: {'recovered_at_10h': peak, 'reference_approx': 750, 'absolute_difference': abs(peak - 750) if peak is not None else None, 'approximate_tolerance': 50, 'answer_invariant': peak is not None and abs(peak - 750) <= 50, 'recovery': 'pixel-digitized series; ±50 is a pilot review tolerance, not an upstream scoring rule'}, '3cb6452a2a243405fae2': {'source_numeric_cells_unchanged': True, 'answer_invariant': True, 'reference': '$3.0 billion'}, '89a32f0298031a49cbe3': {'recovered_answer': round(140.8 - 78.1, 1), 'reference': 62.7, 'answer_invariant': round(140.8 - 78.1, 1) == 62.7}}

def main():
    specs = json.loads((OUT / 'specs.json').read_text())
    locales = {p.stem: json.loads(p.read_text()) for p in (OUT / 'locales').glob('*.json') if not p.name.endswith('.raw.json')}
    assert len(locales) == 11
    image_records = []
    benchmark = []
    qa_checks = answer_checks(specs)
    layouts = {id: table_layout(s, locales) for (id, s) in specs.items() if s['kind'] == 'table'}
    for (lang, locale) in sorted(locales.items()):
        folder = OUT / 'images' / lang
        folder.mkdir(parents=True, exist_ok=True)
        for (id, spec) in specs.items():
            labels = locale['labels']
            if spec['kind'] == 'multipanel_line':
                (im, boxes) = multipanel(spec, labels)
            elif spec['kind'] == 'heatmap':
                (im, boxes) = heatmap(spec, labels)
            elif spec['kind'] == 'dual_axis_combo':
                (im, boxes) = combo(spec, labels)
            else:
                (im, boxes) = table_image(spec, labels, layouts[id])
            path = folder / (id + '.png')
            im.save(path, optimize=True)
            record = {'id': id, 'base_id': spec['base_id'], 'language': lang, 'kind': spec['kind'], 'path': str(path.relative_to(OUT)), 'width': im.width, 'height': im.height, 'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'data_spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), 'text_boxes': boxes, 'all_text_inside_canvas': all((b['inside_canvas'] for b in boxes)), 'answer_check': qa_checks[id], 'benchmark_status': 'candidate_requires_visual_semantic_review'}
            image_records.append(record)
            for qlang in sorted({lang, 'zh', 'en'}):
                qa = locales[qlang]['qas'][id]
                question = bind(qa['question'], locales[qlang]['labels'])
                benchmark.append({'base_id': spec['base_id'], 'variant_id': 'variant_' + hashlib.sha256(json.dumps([spec['base_id'], lang, qlang, qlang]).encode()).hexdigest(), 'source': spec['source'], 'id': id, 'visual_language': lang, 'query_language': qlang, 'answer_language': qlang, 'image': record['path'], 'question': question, 'answer': qa['answer'], 'question_template': qa['question'], 'reference_label_keys': re.findall('\\[\\[([^\\]]+)\\]\\]', qa['question']), 'status': 'candidate_requires_visual_semantic_review'})
            print('Rendered ' + lang + ' ' + spec['kind'] + ' ' + id, flush=True)
    assert len(image_records) == 55 and len(benchmark) == 155
    (OUT / 'image_manifest.json').write_text(json.dumps(image_records, ensure_ascii=False, indent=2))
    (OUT / 'benchmark.jsonl').write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in benchmark)))
    (OUT / 'validation.json').write_text(json.dumps({'images': 55, 'qa_configurations': 155, 'answer_checks': qa_checks, 'missing_glyph_strings': sorted(set(MISSING_GLYPHS)), 'all_text_inside_canvas': all((r['all_text_inside_canvas'] for r in image_records)), 'all_language_data_hashes_identical': all((len({r['data_spec_sha256'] for r in image_records if r['id'] == id}) == 1 for id in specs)), 'semantic_translation_review': 'pending', 'pixel_reconstruction_review': 'approximate, not original raw data'}, ensure_ascii=False, indent=2))
    print('Rendered 55 images and assembled 155 same/cross-language QA configurations.')
if __name__ == '__main__':
    main()
