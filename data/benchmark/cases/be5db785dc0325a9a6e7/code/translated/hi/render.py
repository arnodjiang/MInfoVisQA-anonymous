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
    ax = fig.add_axes(data['main_axes'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(data['x_limits'])
    ax.set_ylim(data['y_limits'])
    ax.plot(data['distance'], data['lineages'], color=data['colors'][0], lw=4.5)
    ax.plot(data['reference_x'], data['reference_y'], color=data['colors'][1], lw=4, ls='--')
    ax.set_xticks(data['x_ticks'])
    ax.set_yticks(data['y_ticks'])
    ax.tick_params(which='major', labelsize=25, width=2, length=7, pad=6)
    ax.tick_params(which='minor', width=1.8, length=6)
    for sp in ax.spines.values():
        sp.set_color('#555555')
        sp.set_linewidth(0.8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('serif')
    placements = [{'key': 'x_axis', 'x': 0.551, 'y': 0.951, 'size': 40, 'max_width': 0.76, 'anchor': 'center'}, {'key': 'y_axis', 'x': 0.027, 'y': 0.44, 'size': 40, 'max_width': 0.73, 'rotation': 90, 'anchor': 'center'}, {'key': 'data', 'x': 0.862, 'y': 0.071, 'size': 38, 'max_width': 0.12, 'anchor': 'left'}]
    fig.text(0.862, 0.865, labels['power'], fontsize=25.2, ha='left', va='center', math_fontfamily='cm')
    for (yy, col, dash) in [(0.93, data['colors'][0], '-'), (0.854, data['colors'][1], '--')]:
        ax.plot([0.762, 0.859], [yy, yy], transform=ax.transAxes, color=col, lw=4.3, ls=dash, clip_on=False)
    inset = fig.add_axes(data['inset_axes'])
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_xticks([])
    inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_linewidth(1.5)
    for (k, path) in enumerate(data['tree_paths']):
        p = np.array(path)
        xx = []
        yy = []
        for i in range(len(p) - 1):
            (a, b) = (p[i], p[i + 1])
            count = max(2, int((b[0] - a[0]) * 190))
            for j in range(count):
                t = j / count
                xx.append(a[0] + t * (b[0] - a[0]))
                yy.append(a[1] + t * (b[1] - a[1]) + data['jitter'][(j + i + k) % len(data['jitter'])])
        xx.append(p[-1, 0])
        yy.append(p[-1, 1])
        inset.plot(xx, yy, color='#222222', lw=1.5, drawstyle='steps-mid')
    blue = data['colors'][2]
    for (x, ys) in data['sample_columns']:
        inset.plot([x, x], [min(ys), max(ys)], color=blue, lw=1.25, alpha=0.9)
        inset.scatter([x] * len(ys), ys, s=35, c=blue, edgecolors='white', linewidths=0.6, zorder=5)
    for (start, end, y) in data['arrows']:
        inset.annotate('', xy=(end, y), xytext=(start, y), arrowprops={'arrowstyle': '<->', 'color': blue, 'lw': 2, 'mutation_scale': 19})
    bx = np.array(data['brace_x'])
    by = np.array(data['brace_y'])
    tt = np.linspace(0, len(bx) - 1, 160)
    inset.plot(np.interp(tt, np.arange(len(bx)), bx), np.interp(tt, np.arange(len(by)), by), color=blue, lw=1.8, clip_on=False)
    placements.append({'key': 'n', 'x': 0.676, 'y': 0.625, 'size': 34, 'max_width': 0.05, 'anchor': 'center'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_555f0b00e2732d35dd4bf32cad9d8c8dfd038d98dc385d2f3d41690de9559127'
LANGUAGE = 'hi'
DATA = {'canvas': [1024, 700], 'main_axes': [0.135, 0.142, 0.834, 0.838], 'x_limits': [0.5, 10000], 'y_limits': [0.1, 500], 'x_ticks': [1, 10, 100, 1000, 10000], 'y_ticks': [0.1, 1, 10, 100], 'distance': [0.5, 0.7, 1, 1.25, 1.6, 2, 2.6, 3.4, 4.5, 6, 8, 10, 14, 20, 28, 40, 56, 80, 110, 160, 220, 320, 450, 650, 900, 1200, 1600, 2000, 2500, 3100, 3800, 4600, 5500, 6500, 7500, 8600], 'lineages': [211, 211, 211, 196, 180, 162, 144, 126, 108, 91, 76, 65, 53, 42, 33, 25.6, 20.1, 15.5, 12.1, 9.2, 7.25, 5.55, 4.3, 3.25, 2.54, 2.02, 1.59, 1.27, 1.01, 0.75, 0.55, 0.39, 0.27, 0.183, 0.127, 0.09], 'reference_x': [1, 4000], 'reference_y': [300, 1.19055], 'colors': ['#227aad', '#ff8926', '#5c9fc7'], 'inset_axes': [0.177, 0.223, 0.44, 0.295], 'tree_paths': [[[0, 0.6], [0.03, 0.6], [0.065, 0.58], [0.1, 0.59], [0.13, 0.61], [0.17, 0.585], [0.2, 0.55], [0.23, 0.57], [0.25, 0.54]], [[0.25, 0.54], [0.29, 0.55], [0.32, 0.555], [0.35, 0.58], [0.375, 0.61], [0.4, 0.58], [0.44, 0.56], [0.465, 0.6], [0.49, 0.65], [0.51, 0.64], [0.535, 0.66], [0.55, 0.6], [0.59, 0.6], [0.62, 0.635], [0.65, 0.65], [0.675, 0.655], [0.69, 0.68], [0.72, 0.67]], [[0.25, 0.54], [0.27, 0.515], [0.3, 0.49], [0.325, 0.46], [0.35, 0.4], [0.38, 0.39], [0.405, 0.4], [0.43, 0.34], [0.46, 0.285], [0.49, 0.275], [0.52, 0.25], [0.55, 0.22], [0.58, 0.17], [0.615, 0.155], [0.64, 0.16], [0.67, 0.13], [0.695, 0.125], [0.72, 0.105]], [[0.69, 0.68], [0.71, 0.73], [0.735, 0.79], [0.735, 0.84], [0.765, 0.855], [0.8, 0.835], [0.83, 0.83], [0.85, 0.86]], [[0.72, 0.67], [0.74, 0.69], [0.775, 0.69], [0.8, 0.71], [0.82, 0.755], [0.835, 0.76], [0.85, 0.82]], [[0.72, 0.67], [0.72, 0.59], [0.75, 0.59], [0.77, 0.63], [0.8, 0.655], [0.825, 0.69], [0.85, 0.69]], [[0.72, 0.59], [0.745, 0.555], [0.77, 0.47], [0.79, 0.475], [0.805, 0.435], [0.83, 0.415], [0.85, 0.415]], [[0.85, 0.86], [0.88, 0.9], [0.905, 0.95], [0.92, 0.95], [0.935, 0.91], [0.96, 0.93], [0.985, 0.97], [1, 0.98]], [[0.85, 0.82], [0.875, 0.835], [0.89, 0.87], [0.915, 0.865], [0.935, 0.83], [0.95, 0.835], [0.97, 0.875], [1, 0.9]], [[0.85, 0.82], [0.88, 0.815], [0.905, 0.79], [0.93, 0.77], [0.95, 0.77], [0.98, 0.79], [1, 0.805]], [[0.85, 0.69], [0.865, 0.66], [0.89, 0.645], [0.915, 0.665], [0.94, 0.675], [0.95, 0.7], [0.98, 0.7], [1, 0.72]], [[0.85, 0.415], [0.87, 0.37], [0.895, 0.385], [0.91, 0.36], [0.935, 0.38], [0.95, 0.385], [0.975, 0.375], [1, 0.41]], [[0.79, 0.475], [0.815, 0.51], [0.84, 0.51], [0.85, 0.52], [0.88, 0.52], [0.895, 0.515], [0.905, 0.54], [0.92, 0.525], [0.94, 0.535], [0.95, 0.52], [0.975, 0.52], [1, 0.55]], [[0.895, 0.385], [0.9, 0.345], [0.925, 0.34], [0.94, 0.3], [0.95, 0.29], [0.975, 0.295], [1, 0.27]], [[0.72, 0.105], [0.76, 0.13], [0.79, 0.145], [0.815, 0.125], [0.835, 0.135], [0.85, 0.105], [0.885, 0.105], [0.9, 0.125], [0.92, 0.11], [0.94, 0.15], [0.955, 0.155], [0.975, 0.19], [1, 0.18]], [[0.94, 0.15], [0.945, 0.11], [0.965, 0.105], [0.98, 0.08], [1, 0.085]], [[0.96, 0.93], [0.975, 0.915], [1, 0.94]], [[0.97, 0.875], [0.98, 0.84], [1, 0.855]], [[0.95, 0.77], [0.965, 0.74], [0.99, 0.75], [1, 0.775]], [[0.98, 0.7], [0.98, 0.66], [1, 0.665]], [[0.95, 0.7], [0.96, 0.61], [0.98, 0.63], [1, 0.6]], [[0.975, 0.52], [0.975, 0.57], [0.99, 0.57], [1, 0.585]], [[0.975, 0.52], [0.985, 0.475], [1, 0.485]], [[0.975, 0.375], [0.98, 0.345], [1, 0.355]], [[0.975, 0.295], [0.98, 0.32], [1, 0.325]], [[0.975, 0.19], [0.98, 0.23], [1, 0.225]], [[0.985, 0.475], [0.99, 0.44], [1, 0.455]], [[0.98, 0.23], [0.99, 0.25], [1, 0.25]]], 'jitter': [0, 0.006, -0.005, 0.003, -0.008, 0.002, 0.007, -0.003], 'sample_columns': [[0.72, [0.105, 0.59, 0.68, 0.75]], [0.85, [0.105, 0.415, 0.515, 0.69, 0.82, 0.86]], [0.955, [0.12, 0.18, 0.29, 0.38, 0.52, 0.7, 0.77, 0.83, 0.925]]], 'arrows': [[0.72, 1, 0.225], [0.85, 1, 0.59], [0.955, 1, 0.815]], 'brace_x': [1.015, 1.035, 1.05, 1.058, 1.061, 1.062, 1.065, 1.075, 1.091, 1.075, 1.065, 1.062, 1.061, 1.058, 1.05, 1.035, 1.015], 'brace_y': [0.985, 0.978, 0.947, 0.88, 0.77, 0.66, 0.575, 0.53, 0.51, 0.49, 0.45, 0.36, 0.24, 0.14, 0.065, 0.025, 0.015]}
LABELS = {'x_axis': 'किनारे से दूरी, s', 'y_axis': 'वंशावलियों की संख्या, l', 'data': 'डेटा', 'power': '$s^{-2/3}$', 'n': 'n'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
