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
    (W, H) = data['canvas']
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    placements = [{'key': 'title', 'x': 0.5, 'y': 0.067, 'size': 32, 'max_width': 0.94, 'anchor': 'center'}, {'key': 'subtitle', 'x': 0.5, 'y': 0.181, 'size': 18, 'max_width': 0.9, 'anchor': 'center'}]
    c = data['colors']
    days = np.arange(data['xlim'][1] + 1)
    anchors = np.array(data['series_anchor_days'])
    ripple = np.resize(np.array(data['band_ripple']), len(days))
    for (i, box) in enumerate(data['axes']):
        ax = fig.add_axes(box)
        mean = np.interp(days, anchors, data['common_mean'] + data['future_means'][i])
        lo = np.interp(days, anchors, data['early_lower'] + data['future_lower'][i])
        hi = np.interp(days, anchors, data['early_upper'] + data['future_upper'][i])
        hi = np.maximum(mean, hi + ripple)
        lo = np.maximum(0, lo + ripple * 0.3)
        ax.fill_between(days, lo, hi, color=c['outer'], linewidth=0, alpha=0.9)
        ax.fill_between(days, mean - (mean - lo) * 0.52, mean + (hi - mean) * 0.46, color=c['inner'], linewidth=0, alpha=0.85)
        ax.plot(days, mean, color=c['line'], linewidth=2.2, zorder=3)
        ax.plot(data['observed_days'], data['observed_cases'], color=c['observed'], linewidth=0.65, marker='o', markersize=2.7, markerfacecolor='white', markeredgecolor=c['observed'], markeredgewidth=0.4, zorder=4)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['month_positions'])
        ax.set_xticklabels([])
        ax.set_yticks(data['yticks'])
        ax.tick_params(axis='both', length=0, labelsize=8, colors='#666666', pad=4)
        ax.spines['top'].set_visible(False)
        for edge in ['left', 'bottom', 'right']:
            ax.spines[edge].set_color(c['axis'])
            ax.spines[edge].set_linewidth(0.8)
        (l, b, w, h) = box
        placements.append({'key': 'cases', 'x': l - 0.03, 'y': 1 - b - h + 0.075, 'size': 14, 'max_width': 0.12, 'rotation': 90, 'anchor': 'center'})
        placements.append({'key': 'contact', 'x': l + w * 0.57, 'y': 1 - b - h + 0.043, 'size': 18, 'max_width': w * 0.62, 'anchor': 'center'})
        placements.append({'key': data['scenario_keys'][i], 'x': l + w * 0.57, 'y': 1 - b - h + 0.082, 'size': 18, 'max_width': w * 0.62, 'anchor': 'center'})
        for (x, key) in zip(data['month_positions'], data['month_keys']):
            placements.append({'key': key, 'x': l + w * x / data['xlim'][1], 'y': 1 - b + 0.021, 'size': 10, 'max_width': 0.052, 'anchor': 'center'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_e6179801aab93c4905606c2f8ca71a873959855a93483a435325d084dcf271a5'
LANGUAGE = 'fr'
DATA = {'canvas': [1100, 550], 'axes': [[0.074, 0.431, 0.388, 0.294], [0.55, 0.431, 0.388, 0.294], [0.074, 0.07, 0.388, 0.294], [0.55, 0.07, 0.388, 0.294]], 'xlim': [0, 166], 'ylim': [0, 68], 'month_positions': [0, 31, 61, 92, 122, 153], 'month_keys': ['mar', 'apr', 'may', 'jun', 'jul', 'aug'], 'yticks': [0, 25, 50], 'scenario_keys': ['scenario_50', 'scenario_60', 'scenario_70', 'scenario_80'], 'anchor_days': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 166], 'series_anchor_days': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 136, 140, 144, 148, 152, 156, 160, 166], 'common_mean': [0.5, 1.5, 3, 6, 14, 24, 27, 26, 23, 20, 17, 14, 12, 11, 10, 8, 6.8, 5.7, 4.8, 4.5, 4.5, 4.9, 5.2, 5.4, 5.7, 6, 6.2, 6.4, 6.8, 7.2, 8, 8.6, 9.2], 'future_means': [[10, 10.8, 11.8, 12.2, 12.1, 11.6, 10.9, 10], [9.8, 10.4, 11.2, 12, 12.8, 13.5, 14.2, 15.8], [10.2, 11, 12.4, 14.5, 17, 20.2, 23.5, 28], [10.2, 11.1, 12.9, 15.9, 20.8, 28, 36.5, 47]], 'early_lower': [0, 0, 0.3, 1, 3, 10, 12, 11, 10, 8, 6, 4, 3, 3, 2, 1.5, 1, 0.6, 0.4, 0.4, 0.6, 0.7, 1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 2.2, 2.4, 2.7], 'early_upper': [3, 5, 8, 14, 29, 43, 49, 47, 42, 33, 27, 25, 22, 21, 22, 18, 14, 11, 10, 9, 10, 11, 11, 11.5, 12, 13, 13, 14, 14.5, 15, 16, 17.5, 18], 'future_lower': [[3, 3.3, 3.8, 4.1, 4, 3.7, 3.4, 3], [3.2, 3.5, 3.8, 4.1, 4.5, 4.8, 5.1, 5.5], [3.5, 4, 4.5, 5, 6, 7.7, 9.1, 10.9], [3.2, 3.7, 4.4, 5.7, 7.8, 11, 15.5, 21]], 'future_upper': [[20, 22, 24, 25, 24, 23, 22, 20], [20, 21.5, 23, 24, 26, 27.5, 29, 31], [21, 23, 26, 29, 34, 40, 46, 54], [21, 22, 25, 30, 39, 50, 68, 82]], 'observed_days': [0, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 119, 120, 121, 123, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136], 'observed_cases': [0, 1, 3, 6, 2, 4, 5, 3, 4, 7, 10, 18, 25, 17, 12, 25, 33, 26, 35, 44, 39, 48, 27, 24, 21, 53, 26, 19, 22, 10, 19, 16, 12, 7, 26, 21, 11, 16, 5, 12, 18, 8, 19, 13, 10, 5, 15, 8, 7, 17, 12, 4, 8, 9, 23, 6, 13, 10, 6, 20, 9, 8, 3, 13, 6, 11, 2, 4, 4, 1, 6, 2, 5, 3, 6, 9, 5, 6, 1, 4, 8, 12, 4, 6, 4, 14, 4, 15, 3, 5, 9, 4, 8, 7, 10, 5, 9, 6, 9, 7, 2, 4, 8, 13, 4, 7, 2, 11, 9, 15, 7, 5, 9, 7, 9, 9, 7, 8, 17, 11, 19, 7, 5, 9, 9, 10], 'band_ripple': [0, 0.4, -0.5, 0.7, -0.3, 0.2, -0.8, 0.5, 0.1, -0.4, 0.6, -0.2], 'colors': {'line': '#f32e5b', 'outer': '#fac5d1', 'inner': '#fba6bc', 'observed': '#555555', 'axis': '#b6b6b6'}}
LABELS = {'title': 'Modélisation compartimentale dynamique : scénarios', 'subtitle': 'Les scénarios de notre modèle illustrent l’importance de réduire les contacts infectieux et de diminuer le risque par d’autres moyens (p. ex., port du masque, hygiène des mains). Un moindre respect de ces recommandations pourrait entraîner un rebond des nouveaux cas.', 'contact': 'Taux de contact à', 'scenario_50': '50% de la normale', 'scenario_60': '60% de la normale', 'scenario_70': '70% de la normale', 'scenario_80': '80% de la normale', 'cases': 'Cas', 'mar': 'Mars', 'apr': 'Avr.', 'may': 'Mai', 'jun': 'Juin', 'jul': 'Juil.', 'aug': 'Août'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
