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

    def panel_00(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.088, 0.108, 0.875, 0.836])
        placements = []
        x = data['quarters']
        ax.fill_between(x, data['outer_lower'], data['outer_upper'], color='#e6e6e6', linewidth=0)
        ax.fill_between(x, data['inner_lower'], data['inner_upper'], color='#bfbfbf', linewidth=0)
        ax.axhline(data['reference'], color='#858585', lw=1.5)
        ax.plot(x, data['line_black'], color='black', lw=3.6, solid_capstyle='round')
        ax.plot(x, data['line_red'], color='#ce2029', lw=3.5, ls=(0, (8, 5)))
        ax.set_xlim(data['x_limits'])
        ax.set_ylim(data['y_limits'])
        ax.set_xticks(data['x_ticks'])
        ax.set_yticks(data['y_ticks'])
        ax.tick_params(axis='both', direction='in', top=True, right=True, labelsize=22, length=7, width=1, color='#666666', labelcolor='#404040', pad=7)
        for spine in ax.spines.values():
            spine.set_color('#777777')
            spine.set_linewidth(1.5)
        placements.extend([{'key': 'title', 'x': 0.524, 'y': 0.027, 'size': 32, 'max_width': 0.94, 'anchor': 'center'}, {'key': 'x_axis', 'x': 0.524, 'y': 0.966, 'size': 30, 'max_width': 0.65, 'anchor': 'center'}, {'key': 'y_axis', 'x': 0.025, 'y': 0.478, 'size': 30, 'max_width': 0.55, 'rotation': 90, 'anchor': 'center'}])
        return finish(fig, labels, placements)

    def panel_01(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        c = data['colors']
        x = data['x']
        ax.fill_between(x, data['outer_low'], data['outer_high'], color=c['outer'], linewidth=0)
        ax.fill_between(x, data['inner_low'], data['inner_high'], color=c['inner'], linewidth=0)
        ax.axhline(0, color=c['frame'], linewidth=1.5)
        ax.plot(x, data['black'], color=c['black'], linewidth=3.6, solid_joinstyle='round')
        ax.plot(x, data['red'], color=c['red'], linewidth=3.5, linestyle=(0, (9, 6)))
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.tick_params(axis='both', which='major', direction='in', top=False, right=True, length=9, width=1, color=c['frame'], labelsize=23, pad=8)
        for s in ax.spines.values():
            s.set_color(c['frame'])
            s.set_linewidth(1.5)
        placements = [{'key': 'title', 'x': 0.537, 'y': 0.038, 'size': 34, 'max_width': 0.91, 'anchor': 'center'}, {'key': 'ylabel', 'x': 0.033, 'y': 0.498, 'size': 30, 'max_width': 0.5, 'rotation': 90, 'anchor': 'center'}, {'key': 'xlabel', 'x': 0.536, 'y': 0.967, 'size': 30, 'max_width': 0.65, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_02(data, labels):
        (W, H) = (1000, 980)
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.105, 0.12, 0.86, 0.775])
        c = data['colors']
        x = data['x']
        ax.fill_between(x, data['outer_low'], data['outer_high'], color=c['outer'], linewidth=0)
        ax.fill_between(x, data['inner_low'], data['inner_high'], color=c['inner'], linewidth=0)
        ax.axhline(data['reference'], color=c['axis'], linewidth=1.8)
        ax.plot(x, data['black'], color=c['black'], linewidth=3.4)
        ax.plot(x, data['red'], color=c['red'], linewidth=3.2, linestyle=(0, (7, 5)))
        ax.set_xlim(data['x_lim'])
        ax.set_ylim(data['y_lim'])
        ax.set_xticks(data['x_ticks'])
        ax.set_yticks(data['y_ticks'])
        ax.tick_params(axis='both', labelsize=22, direction='in', length=7, width=1, color=c['axis'], labelcolor='#555555', top=True, right=True, pad=8)
        for s in ax.spines.values():
            s.set_color(c['axis'])
            s.set_linewidth(1.4)
        placements = [{'key': 'title', 'x': 0.53, 'y': 0.048, 'size': 30, 'max_width': 0.94, 'anchor': 'center', 'rotation': 0}, {'key': 'x_axis', 'x': 0.535, 'y': 0.955, 'size': 29, 'max_width': 0.7, 'anchor': 'center', 'rotation': 0}, {'key': 'y_axis', 'x': 0.025, 'y': 0.5, 'size': 29, 'max_width': 0.65, 'anchor': 'center', 'rotation': 90}]
        return finish(fig, labels, placements)

    def panel_03(data, labels):
        fig = plt.figure(figsize=(9.6, 9.8), dpi=100)
        ax = fig.add_axes([0.095, 0.096, 0.88, 0.774])
        x = data['quarters']
        ax.fill_between(x, data['outer_lower'], data['outer_upper'], color='#e5e5e5', linewidth=0)
        ax.fill_between(x, data['inner_lower'], data['inner_upper'], color='#bcbcbc', linewidth=0)
        ax.axhline(data['reference'], color='#777777', lw=1.5)
        ax.plot(x, data['response'], color='black', lw=3.7, solid_joinstyle='round')
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['x_ticks'])
        ax.set_yticks(data['y_ticks'])
        ax.tick_params(axis='both', direction='in', top=True, right=True, length=7, width=1, color='#777777', labelcolor='#444444', labelsize=21, pad=6)
        for s in ax.spines.values():
            s.set_color('#777777')
            s.set_linewidth(1.5)
        placements = [{'key': 'title', 'x': 0.535, 'y': 0.06, 'size': 32, 'max_width': 0.91, 'anchor': 'center'}, {'key': 'ylabel', 'x': 0.031, 'y': 0.49, 'size': 32, 'max_width': 0.4, 'rotation': 90, 'anchor': 'center'}, {'key': 'xlabel', 'x': 0.535, 'y': 0.965, 'size': 32, 'max_width': 0.7, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_04(data, labels):
        fig = plt.figure(figsize=(10, 10.2), dpi=100)
        ax = fig.add_axes([0.105, 0.105, 0.86, 0.778])
        x = data['quarters']
        ax.fill_between(x, data['outer_low'], data['outer_high'], color='#e5e5e5', linewidth=0)
        ax.fill_between(x, data['inner_low'], data['inner_high'], color='#bfbfbf', linewidth=0)
        ax.axhline(data['reference'], color='#888888', linewidth=1.5)
        ax.plot(x, data['black'], color='black', linewidth=2.7)
        ax.plot(x, data['red'], color='#bc161b', linewidth=2.6, linestyle=(0, (7, 5)))
        ax.set_xlim(data['x_lim'])
        ax.set_ylim(data['y_lim'])
        ax.set_xticks(data['x_ticks'])
        ax.set_yticks(data['y_ticks'])
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=7, width=1, color='#777777', labelsize=22, labelcolor='#444444', pad=7)
        for spine in ax.spines.values():
            spine.set_color('#777777')
            spine.set_linewidth(1.3)
        placements = [{'key': 'title_1', 'x': 0.525, 'y': 0.027, 'size': 32, 'max_width': 0.94, 'anchor': 'center'}, {'key': 'title_2', 'x': 0.525, 'y': 0.064, 'size': 32, 'max_width': 0.94, 'anchor': 'center'}, {'key': 'title_3', 'x': 0.525, 'y': 0.101, 'size': 32, 'max_width': 0.98, 'anchor': 'center'}, {'key': 'y_axis', 'x': 0.027, 'y': 0.506, 'size': 30, 'max_width': 0.6, 'rotation': 90, 'anchor': 'center'}, {'key': 'x_axis', 'x': 0.535, 'y': 0.965, 'size': 30, 'max_width': 0.65, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_05(data, labels):
        fig = plt.figure(figsize=(10, 10.2), dpi=100)
        ax = fig.add_axes([0.095, 0.092, 0.882, 0.779])
        x = data['x']
        ax.fill_between(x, data['outer_low'], data['outer_high'], color='#e5e5e5', linewidth=0)
        ax.fill_between(x, data['inner_low'], data['inner_high'], color='#c3c3c3', linewidth=0)
        ax.axhline(data['reference'], color='#888888', linewidth=1.3)
        ax.plot(x, data['black'], color='black', linewidth=3.7, solid_capstyle='butt')
        ax.plot(x, data['red'], color='#ba292c', linewidth=3.7, linestyle=(0, (6, 4)), dash_capstyle='butt')
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.tick_params(axis='both', labelsize=22, direction='in', length=10, width=1, color='#777777', labelcolor='#404040', pad=8, top=False, right=True)
        for s in ax.spines.values():
            s.set_color('#777777')
            s.set_linewidth(1.4)
        placements = [{'key': 'title', 'x': 0.54, 'y': 0.063, 'size': 34, 'max_width': 0.91, 'anchor': 'center'}, {'key': 'xlabel', 'x': 0.54, 'y': 0.974, 'size': 34, 'max_width': 0.7, 'anchor': 'center'}, {'key': 'ylabel', 'x': 0.031, 'y': 0.52, 'size': 32, 'max_width': 0.55, 'rotation': 90, 'anchor': 'center'}]
        return finish(fig, labels, placements)
    functions = [panel_00, panel_01, panel_02, panel_03, panel_04, panel_05]
    (width, height) = data['canvas']
    image = Image.new('RGB', (width, height), 'white')
    boxes = []
    for (i, panel) in enumerate(data['panels']):
        local = {k: labels[v] for (k, v) in panel['label_map'].items()}
        (part, part_boxes) = functions[i](panel['data'], local)
        (left, top, right, bottom) = panel['bbox']
        (x, y) = (round(left * width), round(top * height))
        (w, h) = (round((right - left) * width), round((bottom - top) * height))
        (sx, sy) = (w / part.width, h / part.height)
        image.paste(part.resize((w, h), Image.Resampling.LANCZOS), (x, y))
        for box in part_boxes:
            b = dict(box)
            (a, bb, c, d) = b['box']
            b['box'] = [round(x + a * sx), round(y + bb * sy), round(x + c * sx), round(y + d * sy)]
            b['label_key'] = panel['label_map'].get(b.get('label_key'), b.get('label_key'))
            b['effective_font_size'] = b.get('font_size', 0) * min(sx, sy)
            b['inside_canvas'] = b['box'][0] >= 0 and b['box'][1] >= 0 and (b['box'][2] <= width) and (b['box'][3] <= height)
            boxes.append(b)
    for p in data['global_placements']:
        b = put(image, labels[p['key']], p['x'] * width, p['y'] * height, p.get('size', 30), max_width=p.get('max_width', 0.8) * width)
        b['label_key'] = p['key']
        boxes.append(b)
    return (image, boxes)
BASE_ID = 'qa_6a900b4b53bcb526e3ab6ddc71123cc55d7a8cbcd58da6b3410ddf194948f981'
LANGUAGE = 'fa'
DATA = {'canvas': [2800, 4216], 'panels': [{'bbox': [0.0, 0.008, 0.465, 0.3], 'data': {'quarters': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'line_black': [-1.25, -1.45, -2.35, -1.48, -1.45, 0.02, 0.2, 1.03, 1.75, 2.3, 2.76, 2.94, 2.26, 2.16, 1.96, 1.87, 1.94, 0.73, 1.07, 0.78, 0.73], 'line_red': [-1.48, -1.73, -2.9, -2.53, -2.13, -0.1, 0.12, 0.19, 1.02, 1.82, 2.36, 2.75, 1.72, 1.62, 2.13, 2.11, 2.51, 1.73, 2.17, 2.05, 2.04], 'outer_lower': [-2.01, -2.63, -3.6, -3.12, -3.2, -1.91, -1.71, -1.21, -0.63, 0.52, 0.82, 0.98, 0.42, 0.24, 0.27, 0.12, 0.05, -1.3, -0.9, -1.22, -1.25], 'outer_upper': [-0.15, -0.02, -0.43, 0.8, 1.13, 2.58, 2.76, 3.43, 3.87, 4.4, 5.02, 5.16, 4.63, 4.53, 4.33, 4.11, 4.1, 3.92, 2.72, 3.22, 2.89], 'inner_lower': [-1.34, -1.98, -3.13, -2.54, -2.42, -0.98, -0.87, -0.46, 0.49, 1.34, 1.62, 1.88, 1.42, 1.2, 1.16, 0.95, 0.85, -0.26, 0.06, -0.25, -0.29], 'inner_upper': [-0.75, -0.67, -1.37, -0.49, -0.43, 1.27, 1.35, 1.95, 2.56, 3.09, 3.56, 3.9, 3.42, 3.36, 3.24, 2.98, 2.94, 2.94, 1.82, 1.78, 1.73], 'x_ticks': [0, 5, 10, 15, 20], 'y_ticks': [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 'x_limits': [0, 20], 'y_limits': [-4.4, 5.8], 'reference': 0, 'canvas': [1000, 950]}, 'label_map': {'title': 'panel_00_title', 'y_axis': 'panel_00_y_axis', 'x_axis': 'panel_00_x_axis'}, 'crop_pixel_bbox': [0, 8, 316, 307], 'crop_sha256': '4cc507b7b2b22c17fb400f45bef3da7c0c7a3f2e28cb9fa4a657f620ff5112a5', 'api_request_sha256': 'f8977a855c078539261ed365f2b0126e1737afff3f5e216c3ff20bb16a0d4d68'}, {'bbox': [0.535, 0.0, 1.0, 0.3], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'black': [-1.25, -1.55, -2.95, -2.2, -1.98, -0.05, 0.08, 0.33, 1.2, 1.93, 2.52, 2.92, 1.86, 1.8, 1.99, 1.78, 1.78, 1.4, 1.41, 1.49, 1.6], 'red': [-1.48, -1.76, -2.93, -2.49, -2.03, -0.09, 0.07, 0.27, 1.05, 1.88, 2.26, 2.66, 1.66, 1.65, 2.22, 2.16, 2.44, 1.81, 2.16, 2.06, 2.04], 'outer_low': [-2.05, -2.78, -4.32, -3.9, -3.72, -2.02, -1.96, -1.75, -1.03, -0.08, 0.46, 0.92, -0.15, -0.18, 0.01, -0.23, -0.21, -0.76, -0.74, -0.69, -0.56], 'outer_high': [-0.12, -0.13, -0.83, 0.36, 0.68, 2.65, 2.66, 3.03, 3.71, 4.24, 4.94, 5.24, 4.01, 4.03, 4.22, 3.91, 3.87, 3.49, 3.55, 3.72, 3.76], 'inner_low': [-1.77, -2.12, -3.81, -3.14, -3, -1.18, -1.04, -0.91, -0.16, 0.71, 1.25, 1.67, 0.84, 0.8, 1.08, 0.76, 0.78, 0.34, 0.35, 0.43, 0.55], 'inner_high': [-0.79, -0.99, -2.06, -1.22, -0.78, 1.17, 1.24, 1.53, 2.22, 2.88, 3.54, 3.9, 2.85, 2.82, 3.08, 2.85, 2.78, 2.45, 2.52, 2.64, 2.67], 'xticks': [0, 5, 10, 15, 20], 'yticks': [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 'xlim': [0, 20], 'ylim': [-4.4, 5.8], 'canvas': [1000, 972], 'axes': [0.094, 0.103, 0.884, 0.816], 'colors': {'outer': '#e5e5e5', 'inner': '#bdbdbd', 'black': '#111111', 'red': '#b5222b', 'frame': '#888888'}}, 'label_map': {'title': 'panel_01_title', 'ylabel': 'panel_01_ylabel', 'xlabel': 'panel_01_xlabel'}, 'crop_pixel_bbox': [364, 0, 680, 307], 'crop_sha256': '8dd3f61a58993c32b3fa813a5e3edecb53747135dcc263d28c21d3df25f0c6b2', 'api_request_sha256': '1d773660f373a1311a09e606117ebd352fce2e176153249f5fdfa1cd6464ab52'}, {'bbox': [0.0, 0.347, 0.465, 0.65], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'black': [-1.18, -1.43, -2.92, -2.08, -1.72, 0.03, 0.4, 0.6, 1.46, 2.31, 3.19, 3.63, 2.14, 2.1, 2.44, 2.17, 2.02, 1.43, 1.43, 1.05, 1.29], 'red': [-1.5, -1.94, -2.8, -2.43, -1.97, -0.05, 0.14, 0.16, 1.1, 1.99, 2.38, 2.63, 1.7, 1.6, 2.02, 2.15, 2.43, 1.79, 2.04, 2.02, 2], 'outer_low': [-1.76, -2.52, -4.24, -3.78, -3.53, -2.03, -1.67, -1.49, -0.51, 0.48, 1.28, 1.92, 0.24, 0.11, 0.51, 0.2, 0.03, -0.7, -0.7, -1.07, -0.94], 'outer_high': [0.11, 0.08, -0.96, 0.44, 0.82, 2.58, 2.73, 3.03, 3.75, 4.5, 5.33, 5.68, 4.25, 4.25, 4.57, 4.36, 4.11, 3.62, 3.54, 3.58, 3.22], 'inner_low': [-1.53, -1.96, -3.76, -3.13, -2.85, -1.33, -1.02, -0.73, 0.2, 1.14, 1.95, 2.42, 0.74, 0.7, 1.05, 0.82, 0.64, 0.35, 0.36, -0.02, 0.03], 'inner_high': [-0.62, -0.62, -1.99, -1.23, -0.83, 1.34, 1.6, 1.84, 2.39, 3.36, 4.03, 4.65, 3.11, 3.13, 3.43, 3.14, 3.06, 2.59, 2.36, 2.51, 2.24], 'x_ticks': [0, 5, 10, 15, 20], 'y_ticks': [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 'x_lim': [0, 20], 'y_lim': [-4.35, 5.8], 'reference': 0, 'colors': {'outer': '#e5e5e5', 'inner': '#bfbfbf', 'black': '#111111', 'red': '#ae3235', 'axis': '#8b8b8b'}}, 'label_map': {'title': 'panel_02_title', 'y_axis': 'panel_02_y_axis', 'x_axis': 'panel_02_x_axis'}, 'crop_pixel_bbox': [0, 355, 316, 666], 'crop_sha256': '46e8a44327a46295bee1cf02f58deaaf8bd556db5c25ecdbc5a70e9faf210c58', 'api_request_sha256': '2471bcbedcd6a2e775684868a2898c6b288d38a6cfd3bdff4e72efb7bcde1b72'}, {'bbox': [0.535, 0.335, 1.0, 0.65], 'data': {'quarters': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'response': [-1.2, -1.85, -2.85, -2.4, -1.85, -0.12, 0.17, 0.2, 1.02, 1.83, 2.22, 2.61, 1.66, 1.61, 2.12, 2.12, 2.4, 1.76, 2.1, 2, 2], 'inner_lower': [-1.8, -2.5, -3.75, -3.45, -2.98, -1.03, -0.94, -0.86, 0.02, 0.7, 1.07, 1.5, 0.59, 0.56, 1.12, 1.06, 1.29, 0.69, 1.02, 0.86, 0.98], 'inner_upper': [-0.87, -1.24, -1.89, -1.48, -0.94, 1.11, 1.36, 1.44, 2.14, 2.9, 3.29, 3.65, 2.66, 2.66, 3.19, 3.19, 3.5, 2.87, 3.16, 3.04, 3.05], 'outer_lower': [-2.31, -3.06, -4.42, -4.3, -3.85, -2.19, -1.96, -1.95, -1.06, -0.12, 0.07, 0.49, -0.45, -0.57, 0.01, -0.01, 0.39, -0.32, -0.01, -0.25, -0.16], 'outer_upper': [-0.1, -0.29, -0.82, -0.24, 0.61, 2.48, 2.73, 2.75, 3.43, 4.11, 4.69, 5.07, 4.07, 3.96, 4.39, 4.27, 4.5, 3.96, 4.39, 4.42, 4.29], 'x_ticks': [0, 5, 10, 15, 20], 'y_ticks': [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 'xlim': [0, 20], 'ylim': [-4.45, 5.8], 'reference': 0}, 'label_map': {'title': 'panel_03_title', 'ylabel': 'panel_03_ylabel', 'xlabel': 'panel_03_xlabel'}, 'crop_pixel_bbox': [364, 343, 680, 666], 'crop_sha256': '732a2433076e394b279ec2fbeb367d7c3d19db834858256baef918447ded0565', 'api_request_sha256': '2ea35462efd2607e4d6d8d20059e58196a42726d026c07be9ad0c1dbddd21b70'}, {'bbox': [0.0, 0.686, 0.465, 1.0], 'data': {'quarters': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'black': [-1.35, -2.02, -2.87, -2.63, -1.96, -0.25, 0.21, 0.23, 1.02, 1.91, 2.18, 2.75, 1.62, 1.65, 1.78, 1.78, 1.85, 1.19, 1.61, 1.6, 1.58], 'red': [-1.28, -1.95, -2.79, -2.49, -1.82, -0.21, 0.13, 0.31, 0.98, 1.83, 2.12, 2.55, 1.61, 1.75, 2.17, 2.14, 2.45, 1.78, 2.14, 2.04, 2.01], 'inner_low': [-1.95, -2.84, -3.66, -3.65, -2.85, -1.58, -1.02, -0.94, -0.17, 0.84, 0.99, 1.47, 0.49, 0.63, 0.68, 0.74, 0.76, 0.1, 0.52, 0.49, 0.43], 'inner_high': [-0.57, -0.99, -1.57, -1.24, -0.41, 0.97, 1.42, 1.41, 2.16, 3.08, 3.16, 3.76, 2.57, 2.72, 2.82, 2.84, 2.94, 2.24, 2.66, 2.67, 2.6], 'outer_low': [-2.66, -3.81, -4.36, -4.37, -3.65, -2.47, -1.88, -1.85, -1.2, -0.2, 0.15, 0.56, -0.5, -0.53, -0.36, -0.44, -0.22, -1.08, -0.75, -0.79, -0.82], 'outer_high': [0.02, -0.39, -0.71, -0.34, 0.72, 2.54, 2.77, 2.82, 3.63, 4.35, 4.56, 5.12, 4.05, 4.11, 4.08, 3.99, 4.03, 3.52, 4.03, 4.06, 3.97], 'x_ticks': [0, 5, 10, 15, 20], 'y_ticks': [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 'x_lim': [0, 20], 'y_lim': [-4.4, 5.8], 'reference': 0}, 'label_map': {'title_1': 'panel_04_title_1', 'title_2': 'panel_04_title_2', 'title_3': 'panel_04_title_3', 'y_axis': 'panel_04_y_axis', 'x_axis': 'panel_04_x_axis'}, 'crop_pixel_bbox': [0, 702, 316, 1024], 'crop_sha256': '59f009e1dcb4fa32cd750b17fbdd06fc355debb7469fecef7c4bdb59870d02b1', 'api_request_sha256': '8190f5745815c0148705f5d537791e89f97d3b9537f599714be44e173dce1685'}, {'bbox': [0.535, 0.686, 1.0, 1.0], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'black': [-1.55, -1.42, -2.38, -1.28, -1.18, 0.55, 0.05, 0.85, 1.67, 2.8, 2.7, 2.4, 1.18, 1.04, 1.82, 0.58, 1.12, 0.24, 0.76, 0.71, 0.7], 'red': [-1.5, -1.63, -2.91, -2.42, -2.04, -0.12, 0.02, 0.01, 1.05, 1.98, 2.27, 2.72, 1.62, 1.76, 2.1, 2.08, 2.43, 1.64, 2.13, 2.06, 1.96], 'outer_low': [-2.21, -2.88, -3.98, -3.14, -3.11, -1.37, -2.05, -1.11, -0.13, 0.63, 0.44, -0.02, -1.01, -1.19, -0.21, -1.79, -1.14, -2.24, -1.76, -1.92, -1.94], 'outer_high': [0.04, -0.05, 1.14, 1.43, 3.05, 2.52, 3.16, 3.88, 5.14, 5.12, 4.66, 3.73, 3.45, 3.54, 4.42, 3.03, 3.63, 2.77, 3.33, 3.3, 3.15], 'inner_low': [-1.85, -2.09, -3.36, -2.56, -2.26, -0.65, -1.1, -0.2, 0.81, 1.69, 1.47, 0.98, 0.15, -0.02, 0.78, -0.57, 0.13, -0.95, -0.41, -0.5, -0.5], 'inner_high': [-0.89, -0.68, -1.41, -0.01, 1.67, 1.12, 2.06, 2.84, 3.84, 3.78, 3.46, 3.06, 2.39, 2.04, 3.16, 1.98, 2.63, 1.48, 2.24, 2.3, 2.26], 'xticks': [0, 5, 10, 15, 20], 'yticks': [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 'xlim': [0, 20], 'ylim': [-4.4, 5.8], 'reference': 0}, 'label_map': {'title': 'panel_05_title', 'ylabel': 'panel_05_ylabel', 'xlabel': 'panel_05_xlabel'}, 'crop_pixel_bbox': [364, 702, 680, 1024], 'crop_sha256': '1f21427f32263af902d014fdf4d9777fdc81f637be9fddda3ccc1e5358f4b128', 'api_request_sha256': '393339581db4cbc30c6368db33103aabc97d1b00b6569544866a5e492dfc3dee'}], 'global_placements': []}
LABELS = {'panel_00_title': '(a) متغیرهای کنترل: وقفه\u200cهای متغیرهای کلان اقتصادی', 'panel_00_y_axis': 'درصد', 'panel_00_x_axis': 'فصل\u200cها', 'panel_01_title': '(b) متغیرهای کنترل: وقفه\u200cهای متغیرهای کلان اقتصادی\nوقفه\u200cهای موضوعات غیرمالیاتی', 'panel_01_ylabel': 'درصد', 'panel_01_xlabel': 'فصل\u200cها', 'panel_02_title': '(c) متغیرهای کنترل: وقفه\u200cهای متغیرهای کلان اقتصادی\nوقفه\u200cها و مقادیر هم\u200cزمان موضوعات غیرمالیاتی', 'panel_02_y_axis': 'درصد', 'panel_02_x_axis': 'فصل\u200cها', 'panel_03_title': '(d) متغیرهای کنترل: وقفه\u200cهای متغیرهای کلان اقتصادی\nوقفه\u200cهای موضوعات غیرمالیاتی\nوقفه\u200cهای تغییرات برون\u200cزای مالیاتی', 'panel_03_ylabel': 'درصد', 'panel_03_xlabel': 'فصل\u200cها', 'panel_04_title_1': '(e) متغیرهای کنترل: وقفه\u200cهای متغیرهای کلان اقتصادی', 'panel_04_title_2': 'وقفه\u200cهای موضوعات غیرمالیاتی', 'panel_04_title_3': 'وقفه\u200cها و مقادیر هم\u200cزمان تغییرات برون\u200cزای مالیاتی', 'panel_04_y_axis': 'درصد', 'panel_04_x_axis': 'فصل\u200cها', 'panel_05_title': '(f) متغیرهای کنترل: وقفه\u200cهای متغیرهای کلان اقتصادی\nوقفه\u200cهای موضوعات غیرمالیاتی\nمقادیر پیشرو و وقفه\u200cهای تغییرات برون\u200cزای مالیاتی', 'panel_05_ylabel': 'درصد', 'panel_05_xlabel': 'فصل\u200cها'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
