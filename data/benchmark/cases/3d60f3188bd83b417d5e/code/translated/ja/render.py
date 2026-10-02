import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--font', default='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
args = parser.parse_args()
os.environ['MVISQA_FONT'] = args.font
os.environ.setdefault('MPLCONFIGDIR', str(Path(args.output).resolve().parent / '.matplotlib_cache'))
pass
from functools import lru_cache
import freetype
import numpy as np
import uharfbuzz as hb
from PIL import Image
import unicodedata
from bidi import algorithm as bidi_algorithm
import os
FONT = os.environ.get('MVISQA_FONT', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
FONT_BYTES = open(FONT, 'rb').read()
HB_FACE = hb.Face(FONT_BYTES)
MISSING_GLYPHS = []
USED_FONTS = set()
FALLBACK_FONTS = [p for p in os.environ.get('MVISQA_FALLBACK_FONTS', '/System/Library/Fonts/KohinoorBangla.ttc:/System/Library/Fonts/GeezaPro.ttc:/System/Library/Fonts/KohinoorTelugu.ttc').split(os.pathsep) if os.path.isfile(p)]

@lru_cache(maxsize=32)
def font_face(path):
    return hb.Face(open(path, 'rb').read())

@lru_cache(maxsize=16384)
def supports(path, text):
    face = freetype.Face(path)
    return all((face.get_char_index(ord(c)) or unicodedata.category(c) == 'Cf' for c in text))

def font_runs(text, direction):
    if supports(FONT, text):
        return [(text, FONT)]
    groups = []
    for char in text:
        name = unicodedata.name(char, '')
        script = next((s for s in ('ARABIC', 'BENGALI', 'TELUGU', 'TAMIL', 'DEVANAGARI', 'THAI') if name.startswith(s)), 'common')
        if unicodedata.category(char) == 'Cf' and groups:
            script = groups[-1][0]
        if groups and groups[-1][0] == script:
            groups[-1][1] += char
        else:
            groups.append([script, char])
    result = []
    for (_, part) in groups:
        path = next((p for p in [FONT] + FALLBACK_FONTS if supports(p, part)), FONT)
        result.append((part, path))
    return result[::-1] if direction == 'rtl' else result

def bidi_runs(text):
    pass
    if not any((unicodedata.bidirectional(c) in ('R', 'AL') for c in text)):
        return [(text, 'ltr')]
    storage = bidi_algorithm.get_empty_storage()
    storage['base_level'] = bidi_algorithm.get_base_level(text)
    storage['base_dir'] = ('L', 'R')[storage['base_level']]
    bidi_algorithm.get_embedding_levels(text, storage)
    bidi_algorithm.explicit_embed_and_overrides(storage, False)
    bidi_algorithm.resolve_weak_types(storage, False)
    bidi_algorithm.resolve_neutral_types(storage, False)
    bidi_algorithm.resolve_implicit_levels(storage, False)
    bidi_algorithm.reorder_resolved_levels(storage, False)
    groups = []
    for char in storage['chars']:
        level = char['level']
        if groups and groups[-1][0] == level:
            groups[-1][1].append(char['ch'])
        else:
            groups.append((level, [char['ch']]))
    return [(''.join(chars[::-1] if level % 2 else chars), 'rtl' if level % 2 else 'ltr') for (level, chars) in groups]

@lru_cache(maxsize=4096)
def mask(text, size):
    size = int(size)
    if not str(text):
        return Image.new('L', (1, max(1, size)), 0)
    text = str(text).replace('\t', '    ')
    if '\n' in text:
        lines = [mask(line, size) for line in text.split('\n')]
        spacing = max(size + 4, max((m.height for m in lines)) + 4)
        result = Image.new('L', (max((m.width for m in lines)), spacing * len(lines)), 0)
        for (i, line) in enumerate(lines):
            result.paste(line, ((result.width - line.width) // 2, i * spacing))
        return result
    parts = []
    penx = 0
    peny = 0
    shaped_runs = [(part, direction, path) for (run, direction) in bidi_runs(text) for (part, path) in font_runs(run, direction)]
    for (run, direction, path) in shaped_runs:
        USED_FONTS.add(path)
        font = hb.Font(font_face(path))
        font.scale = (size * 64, size * 64)
        hb.ot_font_set_funcs(font)
        ft = freetype.Face(path)
        ft.set_char_size(size * 64)
        buf = hb.Buffer()
        buf.add_str(run)
        buf.guess_segment_properties()
        buf.direction = direction
        hb.shape(font, buf)
        for (info, pos) in zip(buf.glyph_infos, buf.glyph_positions):
            if info.codepoint == 0:
                MISSING_GLYPHS.append(str(text))
            ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            bitmap = ft.glyph.bitmap
            if bitmap.width and bitmap.rows:
                a = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, abs(bitmap.pitch))[:, :bitmap.width]
                x = round(penx + pos.x_offset / 64 + ft.glyph.bitmap_left)
                y = round(-peny - pos.y_offset / 64 - ft.glyph.bitmap_top)
                parts.append((x, y, Image.fromarray(a, 'L')))
            penx += pos.x_advance / 64
            peny += pos.y_advance / 64
    if not parts:
        return Image.new('L', (max(1, round(abs(penx))), max(1, size)), 0)
    left = min((p[0] for p in parts))
    top = min((p[1] for p in parts))
    right = max((p[0] + p[2].width for p in parts))
    bottom = max((p[1] + p[2].height for p in parts))
    result = Image.new('L', (max(1, right - left), max(1, bottom - top)), 0)
    for (x, y, im) in parts:
        from PIL import ImageChops
        patch = result.crop((x - left, y - top, x - left + im.width, y - top + im.height))
        result.paste(ImageChops.lighter(patch, im), (x - left, y - top))
    return result

def put(canvas, text, x, y, size=30, color='#202626', anchor='center', max_width=None, rotate=0):
    text = str(text)
    m = mask(text, size)
    while max_width and m.width > max_width and (size > 13):
        size -= 1
        m = mask(text, size)
    if rotate:
        m = m.rotate(rotate, expand=True)
    rgba = Image.new('RGBA', m.size, color)
    rgba.putalpha(m)
    left = round(x - (m.width / 2 if anchor == 'center' else m.width if anchor == 'right' else 0))
    top = round(y - m.height / 2)
    canvas.paste(rgba, (left, top), rgba)
    return {'text': text, 'font_size': size, 'box': [left, top, left + m.width, top + m.height], 'inside_canvas': left >= 0 and top >= 0 and (left + m.width <= canvas.width) and (top + m.height <= canvas.height)}

def text_clusters(text):
    pass
    clusters = []
    join_next = False
    for char in str(text):
        mark = unicodedata.category(char).startswith('M')
        joiner = char in ('\u200c', '\u200d')
        if clusters and (mark or joiner or join_next):
            clusters[-1] += char
        else:
            clusters.append(char)
        name = unicodedata.name(char, '')
        join_next = joiner or 'VIRAMA' in name or char in 'เแโใไ'
    return clusters

def wrap(text, max_width, size):
    text = str(text).strip()
    if not text:
        return ['']
    if '\n' in text:
        return [line for paragraph in text.split('\n') for line in wrap(paragraph, max_width, size)]
    words = text.split(' ') if ' ' in text else text_clusters(text)
    join = ' ' if ' ' in text else ''
    lines = []
    current = ''
    for word in words:
        if mask(word, size).width > max_width:
            if current:
                lines.append(current)
                current = ''
            chunk = ''
            for char in text_clusters(word):
                candidate = chunk + char
                if chunk and mask(candidate, size).width > max_width:
                    lines.append(chunk)
                    chunk = char
                else:
                    chunk = candidate
            if chunk:
                current = chunk
            continue
        candidate = current + join + word if current else word
        if current and mask(candidate, size).width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
pass
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Patch
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw
FONT_HELPER_BOUNDARY = True

def finish(fig, labels, placements):
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    boxes = []
    for p in placements:
        key = p['key']
        text = labels[key]
        size = min(48, max(10, int(p.get('size', 24))))
        width = max(15, float(p.get('max_width', 0.3)) * image.width)
        while mask(text, size).width > width and size > 10:
            size -= 1
        if mask(text, size).width > image.width - 6:
            text = '\n'.join(wrap(text, image.width - 12, size))
        x = float(p['x']) * image.width
        y = float(p['y']) * image.height
        rotation = float(p.get('rotation', 0))
        anchor = p.get('anchor', 'center')
        if anchor not in ('left', 'center', 'right'):
            anchor = 'center'
        m = mask(text, size)
        if rotation:
            m = m.rotate(rotation, expand=True)
        left_extent = m.width / 2 if anchor == 'center' else m.width if anchor == 'right' else 0
        new_x = max(left_extent + 3, min(x, image.width - (m.width - left_extent) - 3))
        new_y = max(m.height / 2 + 3, min(y, image.height - m.height / 2 - 3))
        box = put(image, text, new_x, new_y, size, color=p.get('color', '#202626'), anchor=anchor, rotate=rotation)
        box.update(label_key=key, requested_position=[x, y], placement_adjusted=abs(new_x - x) > 1 or abs(new_y - y) > 1)
        boxes.append(box)
    plt.close(fig)
    return (image, boxes)

def table_geometry(data):
    occupied = set()
    positioned = []
    ncols = 0
    for (ri, row) in enumerate(data['rows']):
        col = 0
        for cell in row:
            while (ri, col) in occupied:
                col += 1
            (rs, cs) = (int(cell.get('rowspan', 1)), int(cell.get('colspan', 1)))
            if min(rs, cs) < 1 or ri + rs > len(data['rows']):
                raise ValueError('invalid_cell_span')
            for r in range(ri, ri + rs):
                for c in range(col, col + cs):
                    if (r, c) in occupied:
                        raise ValueError('overlapping_cells')
                    occupied.add((r, c))
            positioned.append((ri, col, rs, cs, cell))
            col += cs
            ncols = max(ncols, col)
    if not ncols or len(occupied) != len(data['rows']) * ncols:
        raise ValueError('ragged_table')
    return (positioned, ncols)

def table_layout(data, locales):
    (positioned, ncols) = table_geometry(data)
    width = max(1500, ncols * 280)
    padding = 45
    cw = (width - padding * 2) / ncols
    size = 27
    heights = [62] * len(data['rows'])
    for labels in locales:
        for (r, c, rs, cs, cell) in positioned:
            text = labels[cell['label_key']] if cell.get('label_key') else cell['text']
            count = len(wrap(text, cw * cs - 32, size))
            needed = count * (size + 12) + 28
            deficit = max(0, needed - sum(heights[r:r + rs]))
            heights[r + rs - 1] += deficit
    return {'width': width, 'padding': padding, 'cell_width': cw, 'font_size': size, 'heights': heights}

def render_table(data, labels):
    (positioned, ncols) = table_geometry(data)
    cfg = data['layout']
    (pad, cw, size, heights) = (cfg['padding'], cfg['cell_width'], cfg['font_size'], cfg['heights'])
    image = Image.new('RGB', (cfg['width'], int(sum(heights) + 2 * pad)), 'white')
    draw = ImageDraw.Draw(image)
    boxes = []
    for (r, c, rs, cs, cell) in positioned:
        (x, y) = (pad + c * cw, pad + sum(heights[:r]))
        (w, h) = (cw * cs, sum(heights[r:r + rs]))
        draw.rectangle((x, y, x + w, y + h), fill=cell.get('background', '#edf2ee' if r == 0 else '#ffffff'), outline='#a4b3a9', width=2)
        text = labels[cell['label_key']] if cell.get('label_key') else cell['text']
        lines = wrap(text, w - 32, size)
        start = y + h / 2 - (len(lines) - 1) * (size + 12) / 2
        for (j, line) in enumerate(lines):
            fitted = size
            while mask(line, fitted).width > w - 28 and fitted > 8:
                fitted -= 1
            box = put(image, line, x + w / 2, start + j * (size + 12), fitted)
            (a, b, cc, d) = box['box']
            box.update(label_key=cell.get('label_key'), cell=[r, c], inside_cell=a >= x and b >= y and (cc <= x + w) and (d <= y + h))
            boxes.append(box)
    return (image, boxes)

def render(data, labels):
    fig = plt.figure(figsize=(data['width'] / 100, data['height'] / 100), dpi=100)
    placements = []
    lefts = [0.055, 0.548]
    width = 0.436
    height = 0.124
    for (i, p) in enumerate(data['panels']):
        row = i // 2
        left = lefts[i % 2]
        top = 0.015 + row * 0.1668
        ax = fig.add_axes([left, 1 - top - height, width, height])
        ax.fill_between(p['x'], p['y'], 0, step='post', color=data['color'], linewidth=0)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['%.1f' % v for v in data['yticks']])
        ax.tick_params(axis='both', labelsize=11, pad=3, length=2, color='#888888')
        for s in ax.spines.values():
            s.set_color('#808080')
            s.set_linewidth(1)
        placements.append({'key': p['title'], 'x': left + width / 2, 'y': top - 0.007, 'size': 24, 'max_width': width, 'anchor': 'center'})
        placements.append({'key': 'steps', 'x': left + width / 2, 'y': top + height + 0.019, 'size': 17, 'max_width': 0.2, 'anchor': 'center'})
        placements.append({'key': p['average'], 'x': left + width * 0.64, 'y': top + height * 0.88, 'size': 29, 'max_width': width * 0.59, 'anchor': 'center', 'color': 'white'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_d9927172b36ddfec42f7fadf69948f423ee0fa832c5666bb36d65a1c356506ca'
LANGUAGE = 'ja'
DATA = {'width': 900, 'height': 2075, 'xlim': [-2, 53], 'ylim': [0, 1], 'xticks': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1], 'color': '#1f77b4', 'panels': [{'title': 'p0', 'average': 'a0', 'x': [0, 1, 50], 'y': [0, 1, 1]}, {'title': 'p1', 'average': 'a1', 'x': [0, 2, 50], 'y': [0, 1, 1]}, {'title': 'p2', 'average': 'a2', 'x': [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 42, 45, 48, 51], 'y': [0, 0, 0.05, 0.17, 0.3, 0.39, 0.42, 0.43, 0.44, 0.47, 0.51, 0.58, 0.66, 0.73, 0.77, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.965, 0.97, 0.972, 0.976, 0.98, 0.98]}, {'title': 'p3', 'average': 'a3', 'x': [0, 3, 4, 5, 6, 7, 8, 9, 50], 'y': [0, 0, 0.14, 0.39, 0.64, 0.89, 0.97, 1, 1]}, {'title': 'p4', 'average': 'a4', 'x': [0, 2, 50], 'y': [0, 1, 1]}, {'title': 'p5', 'average': 'a5', 'x': [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28, 32, 36, 40, 44, 50], 'y': [0, 0, 0.12, 0.28, 0.44, 0.55, 0.63, 0.69, 0.75, 0.8, 0.83, 0.86, 0.88, 0.9, 0.915, 0.94, 0.95, 0.96, 0.97, 0.977, 0.984, 0.988, 0.993, 0.998, 1]}, {'title': 'p6', 'average': 'a6', 'x': [0, 4, 5, 6, 7, 8, 9, 10, 11, 50], 'y': [0, 0.64, 0.7, 0.94, 0.975, 0.991, 0.997, 1, 1, 1]}, {'title': 'p7', 'average': 'a7', 'x': [0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 46, 48, 50], 'y': [0, 0, 0.025, 0.04, 0.055, 0.1, 0.2, 0.43, 0.49, 0.54, 0.58, 0.61, 0.63, 0.66, 0.68, 0.7, 0.72, 0.74, 0.765, 0.79, 0.81, 0.83, 0.845, 0.86, 0.875, 0.89, 0.903, 0.915, 0.925, 0.936, 0.947, 0.958, 0.964, 0.97, 0.983]}, {'title': 'p8', 'average': 'a8', 'x': [0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 35, 38, 41, 44, 47, 50], 'y': [0, 0.55, 0.71, 0.77, 0.81, 0.835, 0.85, 0.86, 0.875, 0.884, 0.89, 0.904, 0.915, 0.925, 0.935, 0.943, 0.947, 0.951, 0.955, 0.96]}, {'title': 'p9', 'average': 'a9', 'x': [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], 'y': [0, 0, 0.05, 0.08, 0.11, 0.14, 0.18, 0.25, 0.29, 0.35, 0.39, 0.46, 0.49, 0.54, 0.59, 0.66, 0.7, 0.74, 0.77, 0.79, 0.83, 0.85, 0.875, 0.892, 0.906, 0.925, 0.935, 0.948, 0.96, 0.966, 0.971, 0.98, 0.984]}, {'title': 'p10', 'average': 'a10', 'x': [0, 13, 14, 51], 'y': [0, 0.88, 1, 1]}, {'title': 'p11', 'average': 'a11', 'x': [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 35, 38, 41, 45, 50], 'y': [0, 0, 0.13, 0.25, 0.5, 0.59, 0.67, 0.72, 0.81, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95, 0.958, 0.966, 0.973, 0.981, 0.987, 0.99, 0.993, 0.996]}]}
LABELS = {'p0': 'PTA-カーソル-開発セット', 'p1': 'PTA-ボタン-開発セット', 'p2': 'PTA-テキスト-開発セット', 'p3': 'PTA-領域-開発セット', 'p4': 'MNLI-開発セット-BUI', 'p5': 'SQuAD-開発セット-BUI', 'p6': 'VQA-開発セット-BUI', 'p7': 'SA-開発セット', 'p8': 'SA-QID-開発セット', 'p9': 'SA-Q-開発セット', 'p10': 'SA-H-開発セット', 'p11': 'SA-A-開発セット', 'steps': 'ステップ数', 'a0': '平均: 1.00', 'a1': '平均: 2.00', 'a2': '平均: 17.72', 'a3': '平均: 5.97', 'a4': '平均: 2.00', 'a5': '平均: 9.23', 'a6': '平均: 5.82', 'a7': '平均: 24.03', 'a8': '平均: 22.98', 'a9': '平均: 30.75', 'a10': '平均: 13.12', 'a11': '平均: 19.36'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
