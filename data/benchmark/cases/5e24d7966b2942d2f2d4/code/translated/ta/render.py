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
    fig.patch.set_facecolor('white')
    placements = [{'key': 'title', 'x': 0.022, 'y': 0.018, 'size': 26, 'max_width': 0.966, 'anchor': 'left'}]
    c = data['colors']
    for (p, pos) in zip(data['panels'], data['panel_positions']):
        ax = fig.add_axes(pos)
        ax.set_facecolor(c['background'])
        ax.set_xlim(p['xlim'])
        ax.set_ylim(p['ylim'])
        ax.set_xticks(p['xticks'])
        ax.set_yticks(p['yticks'])
        ax.set_xticks(p['xminorticks'], minor=True)
        ax.set_yticks(p['yminorticks'], minor=True)
        ax.set_xticklabels([p['xformat'] % v for v in p['xticks']])
        ax.set_yticklabels([p['yformat'] % v for v in p['yticks']])
        ax.grid(True, which='major', color=c['grid'], linewidth=1.7)
        ax.grid(True, which='minor', color=c['grid'], linewidth=0.8, alpha=0.65)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14, color='#444444', labelcolor='#555555', length=4, width=1)
        ax.tick_params(which='minor', length=0)
        x = np.array(p['x'])
        y = np.array(p['y'])
        slopes = np.diff(y) / np.diff(x)
        m = np.zeros(len(x))
        m[0] = slopes[0]
        m[-1] = slopes[-1]
        for j in range(1, len(x) - 1):
            if slopes[j - 1] * slopes[j] > 0:
                h0 = x[j] - x[j - 1]
                h1 = x[j + 1] - x[j]
                m[j] = 3 * (h0 + h1) / ((2 * h1 + h0) / slopes[j - 1] + (h1 + 2 * h0) / slopes[j])
        xx = []
        yy = []
        for j in range(len(x) - 1):
            t = np.linspace(0, 1, 12)
            h = x[j + 1] - x[j]
            z = (2 * t ** 3 - 3 * t ** 2 + 1) * y[j] + (t ** 3 - 2 * t ** 2 + t) * h * m[j] + (-2 * t ** 3 + 3 * t ** 2) * y[j + 1] + (t ** 3 - t ** 2) * h * m[j + 1]
            xx.extend(x[j] + t * h)
            yy.extend(z)
        ax.plot(xx, yy, color=c['curve'], linewidth=1.8, zorder=3)
        (a, b) = p['interval']
        v = p['interval_height']
        ax.plot([a, b], [v, v], color=c['interval'], linewidth=3, zorder=4)
        ax.plot([a, a], [0, v], color=c['endpoints'], linewidth=3, zorder=5)
        ax.plot([b, b], [0, v], color=c['endpoints'], linewidth=3, zorder=5)
        (l, bt, w, h) = pos
        placements.extend([{'key': p['key'], 'x': l, 'y': 1 - bt - h - 0.028, 'size': 28, 'max_width': w, 'anchor': 'left'}, {'key': 'parameter', 'x': l + w / 2, 'y': 1 - bt + 0.045, 'size': 24, 'max_width': w, 'anchor': 'center'}, {'key': 'density', 'x': l - 0.055 if l > 0.07 else l - 0.04, 'y': 1 - bt - h / 2, 'size': 23, 'max_width': h, 'rotation': 90, 'anchor': 'center'}])
    return finish(fig, labels, placements)
BASE_ID = 'qa_670f35c763c1d2173d424154187306cf3b650545a547a256e87be2176903bb2f'
LANGUAGE = 'ta'
DATA = {'canvas': [1040, 1040], 'panel_positions': [[0.078, 0.708, 0.411, 0.205], [0.57, 0.708, 0.419, 0.205], [0.078, 0.383, 0.411, 0.205], [0.57, 0.383, 0.419, 0.205], [0.059, 0.058, 0.43, 0.205], [0.57, 0.058, 0.419, 0.205]], 'panels': [{'key': 'intercept', 'x': [-0.7, -0.62, -0.54, -0.46, -0.38, -0.32, -0.27, -0.22, -0.17, -0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.7], 'y': [0, 0, 0, 0, 0.01, 0.035, 0.08, 0.16, 0.25, 0.34, 0.5, 0.69, 0.99, 1.29, 1.56, 1.88, 2.12, 2.14, 2.1, 2.03, 1.92, 1.65, 1.4, 1.17, 0.94, 0.72, 0.48, 0.3, 0.19, 0.1, 0.07], 'xlim': [-0.77, 0.77], 'ylim': [-0.11, 2.24], 'xticks': [-0.4, 0, 0.4], 'xminorticks': [-0.6, -0.2, 0.2, 0.6], 'yticks': [0, 0.5, 1, 1.5, 2], 'yminorticks': [0.25, 0.75, 1.25, 1.75], 'xformat': '%.1f', 'yformat': '%.1f', 'interval': [-0.103, 0.609], 'interval_height': 0.35}, {'key': 'temperature', 'x': [-0.03, -0.027, -0.024, -0.021, -0.0185, -0.016, -0.014, -0.012, -0.01, -0.008, -0.006, -0.0045, -0.003, -0.0015, 0, 0.0015, 0.003, 0.0045, 0.006, 0.008, 0.01, 0.012, 0.014, 0.017, 0.02, 0.024, 0.027, 0.03], 'y': [0, 0.4, 1.8, 4.5, 8, 13, 18, 26, 35, 43, 49, 50.6, 49.2, 47.8, 47.2, 45.2, 38, 31, 26, 18, 13, 8, 5, 1.8, 0.7, 0.1, 0.1, 0], 'xlim': [-0.033, 0.033], 'ylim': [-2.5, 53], 'xticks': [-0.02, 0, 0.02], 'xminorticks': [-0.03, -0.01, 0.01, 0.03], 'yticks': [0, 10, 20, 30, 40, 50], 'yminorticks': [5, 15, 25, 35, 45], 'xformat': '%.2f', 'yformat': '%.0f', 'interval': [-0.0187, 0.0118], 'interval_height': 7}, {'key': 'humidity', 'x': [-0.01, -0.0095, -0.009, -0.0085, -0.008, -0.0075, -0.007, -0.0065, -0.006, -0.0055, -0.005, -0.0045, -0.004, -0.0035, -0.003, -0.0025, -0.002, -0.0015, -0.001, -0.0005, 0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01], 'y': [0, 1, 4, 9, 17, 29, 52, 84, 124, 165, 203, 225, 231, 229, 205, 167, 122, 83, 51, 29, 15, 8, 4, 1.5, 0, 0, 0, 0, 0], 'xlim': [-0.011, 0.011], 'ylim': [-12, 242], 'xticks': [-0.01, -0.005, 0, 0.005, 0.01], 'xminorticks': [-0.0075, -0.0025, 0.0025, 0.0075], 'yticks': [0, 50, 100, 150, 200], 'yminorticks': [25, 75, 125, 175, 225], 'xformat': '%.3f', 'yformat': '%.0f', 'interval': [-0.00735, -0.0009], 'interval_height': 31}, {'key': 'wind', 'x': [-0.1, -0.092, -0.084, -0.078, -0.072, -0.066, -0.06, -0.054, -0.048, -0.042, -0.036, -0.03, -0.024, -0.018, -0.012, -0.006, 0, 0.006, 0.012, 0.018, 0.024, 0.03, 0.036, 0.042, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], 'y': [0.15, 0.7, 1.25, 2.2, 3, 3.9, 5.5, 7.2, 9, 10.8, 12.2, 13.2, 13.7, 13.9, 13.4, 12.7, 11.6, 9.9, 7.9, 6.2, 4.6, 3.2, 2.25, 1.6, 0.9, 0.5, 0.1, 0.1, 0, 0], 'xlim': [-0.11, 0.11], 'ylim': [-0.7, 14.5], 'xticks': [-0.1, -0.05, 0, 0.05, 0.1], 'xminorticks': [-0.075, -0.025, 0.025, 0.075], 'yticks': [0, 5, 10], 'yminorticks': [2.5, 7.5, 12.5], 'xformat': '%.2f', 'yformat': '%.0f', 'interval': [-0.077, 0.036], 'interval_height': 2.15}, {'key': 'solar', 'x': [-0.1, -0.092, -0.084, -0.076, -0.068, -0.06, -0.052, -0.044, -0.036, -0.028, -0.02, -0.012, -0.006, 0, 0.008, 0.016, 0.024, 0.032, 0.04, 0.048, 0.056, 0.064, 0.072, 0.08, 0.088, 0.096, 0.1], 'y': [0.65, 1.4, 2.3, 3.25, 4.3, 5.5, 6.9, 8.15, 9.25, 10.1, 10.5, 10.65, 10.25, 9.45, 8.85, 8.05, 6.55, 4.95, 3.65, 2.6, 1.85, 1.25, 0.9, 0.58, 0.31, 0.22, 0.15], 'xlim': [-0.11, 0.11], 'ylim': [-0.5, 11.2], 'xticks': [-0.1, -0.05, 0, 0.05, 0.1], 'xminorticks': [-0.075, -0.025, 0.025, 0.075], 'yticks': [0, 3, 6, 9], 'yminorticks': [1.5, 4.5, 7.5, 10.5], 'xformat': '%.2f', 'yformat': '%.0f', 'interval': [-0.092, 0.058], 'interval_height': 1.5}, {'key': 'precipitation', 'x': [-0.05, -0.045, -0.04, -0.035, -0.03, -0.025, -0.02, -0.016, -0.012, -0.008, -0.004, 0, 0.003, 0.006, 0.009, 0.012, 0.016, 0.02, 0.024, 0.028, 0.032, 0.036, 0.04, 0.045, 0.05], 'y': [0.15, 0.4, 0.85, 1.7, 3, 4.9, 8.6, 11.7, 15.1, 18.5, 20.8, 22.8, 23.6, 23.6, 23.3, 22.2, 19.9, 17.1, 13.5, 10.5, 7, 4.6, 2.7, 1.3, 0.3], 'xlim': [-0.055, 0.055], 'ylim': [-1.1, 25], 'xticks': [-0.05, -0.025, 0, 0.025, 0.05], 'xminorticks': [-0.0375, -0.0125, 0.0125, 0.0375], 'yticks': [0, 5, 10, 15, 20], 'yminorticks': [2.5, 7.5, 12.5, 17.5, 22.5], 'xformat': '%.3f', 'yformat': '%.0f', 'interval': [-0.0275, 0.037], 'interval_height': 3.1}], 'colors': {'background': '#EBEBEB', 'grid': '#FFFFFF', 'curve': '#242424', 'interval': '#F8766D', 'endpoints': '#00BFC4'}}
LABELS = {'title': '95% நம்பக இடைவெளிகளுடன் வானிலை மற்றும் இடைமறிப்பு அளவுருக்களின் பின்நிகழ்தகவு பரவல்கள்', 'intercept': 'இடைமறிப்பு', 'temperature': 'வெப்பநிலை', 'humidity': 'ஒப்பு ஈரப்பதம்', 'wind': 'காற்றின் திசைவேகம்', 'solar': 'சூரியக் கதிர்வீச்சு', 'precipitation': 'மழைப்பொழிவு', 'density': 'அடர்த்தி', 'parameter': 'அளவுரு மதிப்பு'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
