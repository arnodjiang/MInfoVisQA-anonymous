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
        ax = fig.add_axes(data['axes'])
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#cccccc', linewidth=0, zorder=1)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(31)
        ax.grid(True, color='#bcbcbc', alpha=0.6, linewidth=1.1, zorder=2)
        ax.axhline(data['reference'], color='#ff170c', linewidth=7, zorder=3)
        ax.plot(data['x'], data['response'], color='#20ec24', linewidth=8, solid_capstyle='round', zorder=4)
        ax.tick_params(axis='both', labelsize=30, length=5, width=1, color='#888888', labelcolor='#333333', pad=8, direction='in')
        for spine in ax.spines.values():
            spine.set_color('#888888')
            spine.set_linewidth(1.4)
        placements = [{'key': 'title', 'x': 0.537, 'y': 0.048, 'size': 39, 'max_width': 0.6, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_01(data, labels):
        fig = plt.figure(figsize=(9, 8.2), dpi=100)
        ax = fig.add_axes([0.155, 0.13, 0.8, 0.77])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1', '1.5'])
        ax.set_axisbelow(True)
        ax.grid(True, color='#dedede', linewidth=1.1)
        ax.fill_between(data['x'], data['band_lower'], data['band_upper'], color='#bfbfbf', alpha=0.78, linewidth=0)
        ax.axhline(data['reference'], color='#ff2929', linewidth=2.6)
        ax.plot(data['x'], data['response'], color='#16ed0c', linewidth=3.3)
        for spine in ax.spines.values():
            spine.set_color('#999999')
            spine.set_linewidth(1.1)
        ax.tick_params(axis='both', labelsize=21, length=0, pad=9, colors='#333333')
        ax.text(0, 1.015, '$\\times 10^{-3}$', transform=ax.transAxes, fontsize=20, ha='left', va='bottom')
        placements = [{'key': 'title', 'x': 0.555, 'y': 0.045, 'size': 25, 'max_width': 0.5, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_02(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        c = data['colors']
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels([format(v, '.2g') for v in data['yticks']])
        ax.set_axisbelow(True)
        ax.grid(True, color=c['grid'], linewidth=1.1)
        ax.fill_between(data['x'], data['lower'], data['upper'], color=c['band'], linewidth=0, zorder=1)
        ax.axhline(data['reference'], color=c['reference'], linewidth=2.7, zorder=2)
        ax.plot(data['x'], data['response'], color=c['curve'], linewidth=3.6, zorder=3)
        for spine in ax.spines.values():
            spine.set_color(c['frame'])
            spine.set_linewidth(1.4)
        ax.tick_params(axis='both', labelsize=26, length=0, pad=11, colors='#373737')
        placements = [{'key': 'title', 'x': 0.565, 'y': 0.055, 'size': 49, 'max_width': 0.79, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_03(data, labels):
        fig = plt.figure(figsize=(10, 9), dpi=100)
        ax = fig.add_axes([0.19, 0.13, 0.76, 0.77])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels([str(v) for v in data['yticks']])
        ax.set_axisbelow(True)
        ax.grid(True, color='#e1e1e1', linewidth=1)
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#c7c7c7', linewidth=0, zorder=2)
        ax.axhline(data['reference'], color='#ff281c', linewidth=2.5, zorder=3)
        ax.plot(data['x'], data['response'], color='#00ec13', linewidth=4, zorder=4, solid_capstyle='round')
        ax.tick_params(axis='both', labelsize=19, length=0, pad=10, colors='#333333')
        for s in ax.spines.values():
            s.set_color('#8b8b8b')
            s.set_linewidth(1.2)
        placements = [{'key': 'title', 'x': 0.57, 'y': 0.049, 'size': 27, 'max_width': 0.7, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_04(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#c9c9c9', linewidth=0, zorder=1)
        ax.axhline(data['reference'], color='#fa100e', linewidth=2.7, zorder=3)
        ax.plot(data['x'], data['response'], color='#16ed20', linewidth=3.2, zorder=4)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3), useMathText=True)
        ax.tick_params(axis='both', labelsize=23, length=0, pad=9, colors='#454545')
        ax.yaxis.get_offset_text().set_fontsize(22)
        ax.grid(True, color='#e8e8e8', linewidth=0.8, zorder=2)
        for spine in ax.spines.values():
            spine.set_color('#888888')
            spine.set_linewidth(1)
        placements = [{'key': 'title', 'x': 0.55, 'y': 0.046, 'size': 29, 'max_width': 0.74, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_05(data, labels):
        fig = plt.figure(figsize=(9, 8.65), dpi=100)
        ax = fig.add_axes([0.145, 0.155, 0.822, 0.751])
        c = data['colors']
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['-0.3', '-0.2', '-0.1', '0', '0.1'])
        ax.grid(True, color=c['grid'], linewidth=1.6)
        ax.set_axisbelow(True)
        ax.fill_between(data['x'], data['band_low'], data['band_high'], color=c['band'], alpha=0.65, linewidth=0)
        ax.axhline(data['reference'], color=c['reference'], linewidth=5, zorder=3)
        ax.plot(data['x'], data['response'], color=c['curve'], linewidth=5, solid_capstyle='round', zorder=4)
        ax.tick_params(axis='both', labelsize=30, length=0, pad=12, colors='#444444')
        for spine in ax.spines.values():
            spine.set_color(c['spine'])
            spine.set_linewidth(2)
        placements = [{'key': 'title', 'x': 0.556, 'y': 0.05, 'size': 49, 'max_width': 0.8, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_06(data, labels):
        fig = plt.figure(figsize=(data['canvas'][0] / 100, data['canvas'][1] / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(data['ytick_text'])
        ax.set_axisbelow(True)
        ax.grid(True, color='#e8e8e8', linewidth=1.2)
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#cecece', linewidth=0, zorder=2)
        ax.axhline(data['reference'], color='#ff634f', linewidth=7, zorder=3)
        ax.plot(data['x'], data['response'], color='#24ed00', linewidth=7, solid_capstyle='round', zorder=4)
        ax.tick_params(axis='both', labelsize=29, length=0, pad=12, colors='#444444')
        for spine in ax.spines.values():
            spine.set_color('#999999')
            spine.set_linewidth(2)
        placements = [{'key': 'title', 'x': 0.57, 'y': 0.047, 'size': 43, 'max_width': 0.78, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_07(data, labels):
        fig = plt.figure(figsize=(9, 7.43), dpi=100)
        ax = fig.add_axes([0.14, 0.17, 0.7, 0.74])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_xticks(data['minor_xticks'], minor=True)
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['-0.1', '-0.08', '-0.06', '-0.04', '-0.02', '0'])
        ax.set_axisbelow(True)
        ax.grid(True, color='#e9e9e9', linewidth=1.5)
        ax.grid(True, axis='x', which='minor', color='#f0f0f0', linewidth=1)
        ax.fill_between(data['x'], data['band_lower'], data['band_upper'], color='#c9cbca', alpha=0.9, linewidth=0, zorder=2)
        ax.plot(data['reference_x'], data['reference'], color='#ff2020', linewidth=5, zorder=3)
        ax.plot(data['x'], data['response'], color='#35f52e', linewidth=8, alpha=0.2, zorder=4)
        ax.plot(data['x'], data['response'], color='#24eb20', linewidth=4.5, zorder=5, solid_capstyle='round')
        for s in ax.spines.values():
            s.set_color('#999999')
            s.set_linewidth(1.7)
        ax.tick_params(axis='both', which='major', labelsize=27, length=0, pad=15, colors='#333333')
        ax.tick_params(which='minor', length=0)
        placements = [{'key': 'title', 'x': 0.49, 'y': 0.047, 'size': 42, 'max_width': 0.7, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_08(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.175, 0.157, 0.73, 0.751])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels([format(v, '.2g') for v in data['yticks']])
        ax.set_axisbelow(True)
        ax.grid(True, color='#e9e9e9', linewidth=0.8)
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#c6c6c6', linewidth=0, zorder=2)
        ax.axhline(data['reference'], color='#ff2828', linewidth=1.7, zorder=3)
        ax.plot(data['x'], data['response'], color='#00e522', linewidth=3.5, zorder=4)
        for spine in ax.spines.values():
            spine.set_color('#8e8e8e')
            spine.set_linewidth(1)
        ax.tick_params(axis='both', labelsize=29, length=0, pad=12, colors='#444444')
        placements = [{'key': 'title', 'x': 0.54, 'y': 0.051, 'size': 46, 'max_width': 0.72, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_09(data, labels):
        fig = plt.figure(figsize=(10, 9), dpi=100)
        ax = fig.add_axes([0.17, 0.15, 0.77, 0.75])
        ax.set_xlim(data['x_limits'])
        ax.set_ylim(data['y_limits'])
        ax.set_xticks(data['x_ticks'])
        ax.set_yticks(data['y_ticks'])
        ax.set_yticklabels(['-0.1', '-0.05', '0', '0.05', '0.1', '0.15'])
        ax.grid(True, color='#eeeeee', linewidth=1)
        ax.set_axisbelow(True)
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#d0d0d0', linewidth=0)
        ax.plot(data['x'], data['response'], color='#54ed4d', linewidth=3.7)
        ax.axhline(data['reference'], color='#ee241c', linewidth=3)
        ax.tick_params(axis='both', labelsize=16, color='#999999', length=4, width=0.8)
        for spine in ax.spines.values():
            spine.set_color('#999999')
            spine.set_linewidth(1)
        placements = [{'key': 'title', 'x': 0.555, 'y': 0.048, 'size': 27, 'max_width': 0.83, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_10(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.115, 0.115, 0.845, 0.785])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(23)
        ax.tick_params(axis='both', labelsize=23, length=0, pad=9)
        ax.tick_params(top=True, right=True, direction='in')
        ax.set_axisbelow(True)
        ax.grid(True, color='#e3e3e3', linewidth=1.1)
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#c8c8c8', alpha=0.9, linewidth=0, zorder=2)
        ax.axhline(data['reference'], color='#ff2727', linewidth=3, zorder=3)
        ax.plot(data['x'], data['response'], color='#00f51a', linewidth=4.5, zorder=4, solid_capstyle='round')
        for spine in ax.spines.values():
            spine.set_color('#999999')
            spine.set_linewidth(1.4)
        placements = [{'key': 'title', 'x': 0.54, 'y': 0.048, 'size': thirty if False else 30, 'max_width': 0.65, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_11(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.116, 0.129, 0.845, 0.773])
        ax.set_facecolor('white')
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#cccccc', linewidth=0, zorder=1)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4), useMathText=True)
        ax.grid(True, color='#bcbcbc', linewidth=0.85, alpha=0.65, zorder=2)
        ax.plot(data['x'], data['response'], color='#35f322', linewidth=7, solid_capstyle='round', zorder=3)
        ax.plot(data['reference_x'], data['reference'], color='#f31d20', linewidth=5, zorder=4)
        for spine in ax.spines.values():
            spine.set_color('#8c8c8c')
            spine.set_linewidth(1.4)
        ax.tick_params(axis='both', labelsize=32, length=0, pad=10, colors='#414141')
        ax.yaxis.get_offset_text().set_size(30)
        placements = [{'key': 'title', 'x': 0.541, 'y': 0.049, 'size': 48, 'max_width': 0.72, 'anchor': 'center', 'rotation': 0}]
        return finish(fig, labels, placements)

    def panel_12(data, labels):
        fig = plt.figure(figsize=(9, 8.25), dpi=100)
        ax = fig.add_axes([0.145, 0.115, 0.822, 0.785])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['-0.1', '0', '0.1', '0.2', '0.3'])
        ax.set_axisbelow(True)
        ax.grid(True, color='#ededed', linewidth=1.2)
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#cccccc', alpha=0.9, linewidth=0)
        ax.axhline(data['reference'], color='#ff180f', linewidth=3.5, zorder=3)
        ax.plot(data['x'], data['response'], color='#36ed24', linewidth=3.8, zorder=4)
        ax.tick_params(axis='both', labelsize=30, length=0, pad=12, colors='#454545')
        for spine in ax.spines.values():
            spine.set_color('#aaaaaa')
            spine.set_linewidth(2)
        placements = [{'key': 'title', 'x': 0.555, 'y': 0.052, 'size': 43, 'max_width': 0.6, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_13(data, labels):
        fig = plt.figure(figsize=(9, 8.2), dpi=100)
        ax = fig.add_axes([0.18, 0.13, 0.765, 0.775])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['-0.02', '-0.01', '0', '0.01', '0.02', '0.03'])
        ax.set_axisbelow(True)
        ax.grid(True, color='#eeeeee', linewidth=1.1)
        ax.fill_between(data['x'], data['lower'], data['upper'], color='#c6c6c6', linewidth=0, zorder=1)
        ax.axhline(data['reference'], color='#fa2819', linewidth=3, zorder=3)
        ax.plot(data['x'], data['response'], color='#00ed18', linewidth=3.8, zorder=4)
        for spine in ax.spines.values():
            spine.set_color('#929292')
            spine.set_linewidth(1.1)
        ax.tick_params(axis='both', labelsize=21, color='#929292', labelcolor='#444444', direction='in', length=4, pad=12, top=True, right=True)
        placements = [{'key': 'title', 'x': 0.5625, 'y': 0.048, 'size': 25, 'max_width': 0.77, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_14(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.157, 0.115, 0.8, 0.788])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['-0.3', '-0.2', '-0.1', '0', '0.1'])
        ax.set_axisbelow(True)
        ax.grid(True, color='#ececec', linewidth=1)
        x = data['x']
        ax.fill_between(x, data['gray_lower'], data['gray_upper'], color='#d6d6d6', alpha=0.85, linewidth=0)
        ax.fill_between(x, data['green_lower'], data['green_upper'], color='#71ec62', alpha=0.42, linewidth=0)
        ax.axhline(data['reference'], color='#ff493f', linewidth=2.3)
        ax.plot(x, data['response'], color='#43df31', linewidth=3.5, solid_joinstyle='round', solid_capstyle='round')
        for s in ax.spines.values():
            s.set_color('#999999')
            s.set_linewidth(1.1)
        ax.tick_params(axis='both', labelsize=31, colors='#555555', direction='in', length=5, width=0.8, pad=9, top=True, right=True)
        placements = [{'key': 'title', 'x': 0.557, 'y': 0.052, 'size': 49, 'max_width': 0.75, 'anchor': 'center', 'rotation': 0}]
        return finish(fig, labels, placements)
    functions = [panel_00, panel_01, panel_02, panel_03, panel_04, panel_05, panel_06, panel_07, panel_08, panel_09, panel_10, panel_11, panel_12, panel_13, panel_14]
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
BASE_ID = 'qa_1c16e2ffd82acf7a1e9a8fbca52c5bf0ebb3d49e503ef76a3aa568942e4cee66'
LANGUAGE = 'en'
DATA = {'canvas': [2800, 1611], 'panels': [{'bbox': [0.035, 0.078, 0.207, 0.36], 'data': {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 'response': [0.00041, 0.00049, 0.00057, 0.00062, 0.00064, 0.00062, 0.00057, 0.00049, 0.00042, 0.00035, 0.00028, 0.00021, 0.00014, 8e-05, 3e-05, -2e-05, -5.5e-05, -7.8e-05, -9e-05, -0.0001, -0.0001, -9.8e-05, -9e-05, -8e-05, -6.5e-05, -5e-05, -3.5e-05, -1.8e-05, 0, 1.8e-05, 3.8e-05, 6e-05, 8.3e-05, 0.000105, 0.000126, 0.000145, 0.00016, 0.00017, 0.00017, 0.000165], 'upper': [0.00092, 0.00118, 0.00142, 0.00159, 0.00168, 0.0017, 0.00169, 0.00167, 0.00165, 0.00162, 0.00159, 0.00156, 0.00154, 0.00152, 0.0015, 0.00149, 0.00149, 0.0015, 0.00151, 0.00152, 0.00153, 0.00154, 0.00155, 0.00156, 0.00157, 0.00159, 0.0016, 0.00162, 0.00163, 0.00165, 0.00166, 0.00167, 0.00168, 0.00168, 0.00169, 0.00169, 0.00169, 0.00169, 0.00169, 0.00168], 'lower': [2e-05, -0.0001, -0.0003, -0.00053, -0.00076, -0.00098, -0.00118, -0.00134, -0.00147, -0.00156, -0.00162, -0.00167, -0.0017, -0.00173, -0.00175, -0.00176, -0.00176, -0.00175, -0.00174, -0.00173, -0.00171, -0.00169, -0.00167, -0.00165, -0.00163, -0.00161, -0.00159, -0.00156, -0.00154, -0.00152, -0.0015, -0.00148, -0.00146, -0.00144, -0.00142, -0.00141, -0.0014, -0.0014, -0.0014, -0.00139], 'xlim': [0, 40], 'ylim': [-0.002, 0.002], 'xticks': [10, 20, 30, 40], 'yticks': [-0.002, -0.001, 0, 0.001, 0.002], 'reference': 0, 'canvas': [950, 900], 'axes': [0.12, 0.12, 0.835, 0.78]}, 'label_map': {'title': 'panel_00_title'}, 'crop_pixel_bbox': [36, 46, 212, 212], 'crop_sha256': '0436ab2ec0ca5b6aaa9146f3194d5ea0abadc8cdc211ac6b0c5c7b561ca06cc9', 'api_request_sha256': '5a4b225269aa4545873a4106f4f7bc2634a51b49bb4dfb8e70a772d8b72c3166'}, {'bbox': [0.219, 0.078, 0.397, 0.36], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36], 'response': [0.00028, 0.00036, 0.00044, 0.00052, 0.00058, 0.00062, 0.000645, 0.00065, 0.00064, 0.000625, 0.0006, 0.00055, 0.00049, 0.00044, 0.00039, 0.000345, 0.000305, 0.00027, 0.00024, 0.00022, 0.000215, 0.000205, 0.0002, 0.000195], 'band_upper': [0.00048, 0.00066, 0.00084, 0.001, 0.00112, 0.0012, 0.00127, 0.0013, 0.00132, 0.00133, 0.00134, 0.00134, 0.00134, 0.00132, 0.00128, 0.00126, 0.00125, 0.00125, 0.00124, 0.00122, 0.0012, 0.00119, 0.00118, 0.00118], 'band_lower': [3e-05, 8e-05, 0.00013, 0.00017, 0.00018, 0.00016, 0.00012, 7e-05, 2e-05, -3e-05, -9e-05, -0.00021, -0.00032, -0.00043, -0.00052, -0.0006, -0.00066, -0.00072, -0.00077, -0.00081, -0.00083, -0.00085, -0.00086, -0.00087], 'xlim': [0, 36], 'ylim': [-0.001, 0.0015], 'xticks': [10, 20, 30], 'yticks': [-0.001, -0.0005, 0, 0.0005, 0.001, 0.0015], 'reference': 0}, 'label_map': {'title': 'panel_01_title'}, 'crop_pixel_bbox': [224, 46, 407, 212], 'crop_sha256': '46958383d331ac715dfac5d06319c64ac740a0febe6231e835b339d4de7dc2ac', 'api_request_sha256': 'a73375d7a8c42fda3788e675af7550dc615df7cf1fdeb8b2a098bab3bcf532c7'}, {'bbox': [0.407, 0.078, 0.588, 0.36], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36], 'response': [0.139, 0.137, 0.141, 0.141, 0.133, 0.121, 0.108, 0.093, 0.078, 0.065, 0.053, 0.034, 0.02, 0.01, 0.004, 0.0, -0.002, -0.003, -0.003, -0.003, -0.0025, -0.002, -0.0015, -0.001], 'upper': [0.229, 0.23, 0.232, 0.234, 0.22, 0.202, 0.185, 0.168, 0.151, 0.135, 0.12, 0.097, 0.079, 0.065, 0.054, 0.045, 0.038, 0.033, 0.03, 0.028, 0.026, 0.024, 0.023, 0.022], 'lower': [0.06, 0.06, 0.06, 0.057, 0.048, 0.036, 0.023, 0.012, 0.002, -0.006, -0.014, -0.025, -0.03, -0.033, -0.034, -0.033, -0.032, -0.03, -0.028, -0.027, -0.025, -0.023, -0.022, -0.02], 'xlim': [0, 36], 'ylim': [-0.05, 0.25], 'xticks': [0, 10, 20, 30], 'yticks': [-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25], 'reference': 0, 'canvas': [1000, 900], 'axes': [0.166, 0.115, 0.799, 0.783], 'colors': {'band': '#c7c8c6', 'curve': '#54de35', 'reference': '#f02424', 'grid': '#eeeeee', 'frame': '#aaaaaa'}}, 'label_map': {'title': 'panel_02_title'}, 'crop_pixel_bbox': [417, 46, 602, 212], 'crop_sha256': 'aff24badc2bfb8ca33f40dd2e0d45bd2f323374ab34976fd3b93143836d2e7af', 'api_request_sha256': '82a125742ce4b10547ec276be59e2c59bd1d545316b98435d40eea7b09626a36'}, {'bbox': [0.594, 0.078, 0.779, 0.36], 'data': {'x': [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 37], 'response': [-0.002, -0.001, 0.0002, 0.0011, 0.0018, 0.002, 0.00195, 0.00165, 0.00125, 0.0008, 0.00035, -0.0001, -0.0005, -0.00125, -0.0017, -0.00185, -0.0019, -0.0018, -0.00155, -0.00125, -0.00095, -0.00065, -0.0004, -0.00012, 0.0001, 0.0003, 0.00045, 0.00055, 0.0006], 'upper': [0.0001, 0.002, 0.0044, 0.0062, 0.0074, 0.0081, 0.0084, 0.0085, 0.0084, 0.0081, 0.0078, 0.0074, 0.007, 0.0062, 0.0055, 0.0049, 0.0046, 0.0044, 0.00425, 0.0042, 0.0042, 0.0042, 0.0042, 0.0042, 0.0042, 0.00415, 0.0041, 0.00405, 0.004], 'lower': [-0.004, -0.0036, -0.0035, -0.0037, -0.0041, -0.0046, -0.0053, -0.006, -0.0066, -0.0072, -0.0078, -0.0083, -0.0087, -0.0092, -0.0093, -0.009, -0.0086, -0.0081, -0.0077, -0.0073, -0.0068, -0.0063, -0.0058, -0.0053, -0.0048, -0.0044, -0.0039, -0.0035, -0.0033], 'xlim': [-6, 37], 'ylim': [-0.01, 0.01], 'xticks': [0, 10, 20, 30], 'yticks': [-0.01, -0.005, 0, 0.005, 0.01], 'reference': 0}, 'label_map': {'title': 'panel_03_title'}, 'crop_pixel_bbox': [608, 46, 798, 212], 'crop_sha256': '8c43e24556a65f2ae90c7cde2dce152a28d66e4f25cc79b432287aad3cb8ae65', 'api_request_sha256': '6bc4405d5439c14f754b988b0863e3d10777d6a4454be2e010582666bc9b97c6'}, {'bbox': [0.796, 0.078, 0.972, 0.36], 'data': {'x': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38], 'response': [0.00135, 0.00148, 0.0015, 0.00143, 0.00134, 0.00122, 0.0011, 0.00099, 0.00088, 0.00077, 0.00067, 0.00058, 0.0005, 0.00043, 0.00037, 0.00031, 0.00026, 0.00022, 0.00018, 0.00015], 'upper': [0.00635, 0.00585, 0.0054, 0.00495, 0.0045, 0.00408, 0.00375, 0.00345, 0.00319, 0.00296, 0.00275, 0.00257, 0.0024, 0.00224, 0.0021, 0.00196, 0.00183, 0.00171, 0.0016, 0.00149], 'lower': [-0.004, -0.00359, -0.00319, -0.00283, -0.0025, -0.00222, -0.00196, -0.00173, -0.00154, -0.00137, -0.00122, -0.0011, -0.00099, -0.0009, -0.00082, -0.00075, -0.00069, -0.00064, -0.00059, -0.00055], 'reference': 0, 'xlim': [0, 38], 'ylim': [-0.004, 0.0067], 'xticks': [0, 10, 20, 30], 'yticks': [-0.004, -0.002, 0, 0.002, 0.004, 0.006], 'canvas': [1000, 930], 'axes': [0.14, 0.13, 0.82, 0.77]}, 'label_map': {'title': 'panel_04_title'}, 'crop_pixel_bbox': [815, 46, 995, 212], 'crop_sha256': 'c30d3790f77a85a8dbf805deadfd3d966de52f5be7661a83acb0eed2ccbf8ac1', 'api_request_sha256': '862ab25d8cdc3b46a16513f9aa3b8cdafc229077519f227d73ddd6c4a49f4cc4'}, {'bbox': [0.029, 0.389, 0.207, 0.669], 'data': {'x': [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36], 'response': [-0.136, -0.156, -0.161, -0.157, -0.151, -0.161, -0.134, -0.108, -0.086, -0.066, -0.048, -0.036, -0.025, -0.017, -0.011, -0.006, -0.003, 0, 0.003, 0.004, 0.003, 0.002, 0.001, 0.001, 0, 0, 0, -0.001], 'band_low': [-0.234, -0.251, -0.25, -0.245, -0.247, -0.238, -0.215, -0.184, -0.159, -0.135, -0.109, -0.089, -0.078, -0.071, -0.064, -0.058, -0.053, -0.047, -0.037, -0.03, -0.025, -0.023, -0.021, -0.02, -0.02, -0.021, -0.021, -0.022], 'band_high': [-0.062, -0.077, -0.076, -0.068, -0.072, -0.055, -0.032, -0.018, -0.008, 0.009, 0.017, 0.026, 0.031, 0.033, 0.034, 0.034, 0.034, 0.034, 0.033, 0.03, 0.028, 0.025, 0.024, 0.022, 0.021, 0.019, 0.017, 0.015], 'xlim': [-1, 36], 'ylim': [-0.3, 0.1], 'xticks': [0, 10, 20, 30], 'yticks': [-0.3, -0.2, -0.1, 0, 0.1], 'reference': 0, 'colors': {'curve': '#20fa16', 'band': '#bdbdbd', 'reference': '#ff2420', 'grid': '#eeeeee', 'spine': '#888888'}}, 'label_map': {'title': 'panel_05_title'}, 'crop_pixel_bbox': [30, 229, 212, 394], 'crop_sha256': 'f4cfb60d1c48485051499211c71a94fbcf324678dc2c4fd01b317816da12f9f1', 'api_request_sha256': '14e240d5854b84ef927ac1873803ac30b5e87c3ed9a6ab2d5ec1ea7460f9cf2f'}, {'bbox': [0.213, 0.389, 0.397, 0.669], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], 'response': [-0.0017, -0.0021, -0.00235, -0.0023, -0.0019, -0.0013, -0.00055, 0.0003, 0.00115, 0.002, 0.0027, 0.0033, 0.0038, 0.0042, 0.0045, 0.00465, 0.0047, 0.00465, 0.0045, 0.0043, 0.00405, 0.00375, 0.0034, 0.00305, 0.0027, 0.00235, 0.002, 0.00165, 0.0013, 0.001, 0.0007, 0.00045, 0.0002, 5e-05, -0.0001, -0.0002, -0.00025], 'upper': [0.001, 0.0013, 0.0017, 0.0022, 0.0029, 0.0037, 0.0046, 0.0056, 0.00665, 0.00765, 0.00855, 0.0092, 0.00965, 0.00995, 0.01005, 0.01005, 0.00995, 0.0098, 0.00955, 0.00925, 0.0089, 0.00855, 0.00815, 0.00775, 0.00735, 0.0069, 0.00645, 0.00605, 0.00565, 0.00525, 0.0049, 0.0046, 0.0043, 0.004, 0.0037, 0.00345, 0.0032], 'lower': [-0.006, -0.00635, -0.00665, -0.0068, -0.00665, -0.0062, -0.0056, -0.0049, -0.00425, -0.00355, -0.0029, -0.0023, -0.0018, -0.00135, -0.001, -0.00075, -0.0006, -0.0005, -0.0005, -0.00055, -0.00065, -0.0008, -0.001, -0.0012, -0.0014, -0.0016, -0.00185, -0.0021, -0.00235, -0.0026, -0.0028, -0.003, -0.00315, -0.0033, -0.00345, -0.00355, -0.00365], 'xlim': [0, 36], 'ylim': [-0.01, 0.0103], 'xticks': [10, 20, 30], 'yticks': [-0.01, -0.005, 0, 0.005, 0.01], 'ytick_text': ['-0.01', '-0.005', '0', '0.005', '0.01'], 'reference': 0, 'canvas': [945, 825], 'axes': [0.18, 0.12, 0.78, 0.78]}, 'label_map': {'title': 'panel_06_title'}, 'crop_pixel_bbox': [218, 229, 407, 394], 'crop_sha256': 'e754a2b37debb083e5e75db35e0a31b1f583a601675947bb6563f9ba20549eec', 'api_request_sha256': 'fe96974072e89ac7a3e98c4fb7dd070d3abf83e9fb9fec21f73a7800b6f227c5'}, {'bbox': [0.408, 0.389, 0.588, 0.669], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 38], 'response': [-0.052, -0.038, -0.026, -0.016, -0.01, -0.006, -0.003, 0.002, 0.005, 0.0065, 0.0066, 0.0058, 0.0047, 0.0037, 0.0025, 0.0015, 0.0009, 0.0005, 0.0004], 'band_lower': [-0.094, -0.074, -0.048, -0.028, -0.018, -0.012, -0.008, -0.003, 0.0005, 0.002, 0.0025, 0.002, 0.001, 0, -0.0015, -0.003, -0.004, -0.0045, -0.005], 'band_upper': [0.019, 0.017, 0.015, 0.013, 0.011, 0.01, 0.011, 0.015, 0.018, 0.0185, 0.018, 0.017, 0.016, 0.015, 0.012, 0.01, 0.008, 0.006, 0.005], 'reference': [0, 0], 'reference_x': [0, 38], 'xlim': [0, 38], 'ylim': [-0.105, 0.02], 'xticks': [10, 20, 30], 'yticks': [-0.1, -0.08, -0.06, -0.04, -0.02, 0], 'minor_xticks': [0, 5, 15, 25, 35]}, 'label_map': {'title': 'panel_07_title'}, 'crop_pixel_bbox': [418, 229, 602, 394], 'crop_sha256': '9961595eda931016a4fff1a7606d9f35ad6ec328da82b4b3cb584891293b02da', 'api_request_sha256': 'f4704c75c8b609bdbe9867082cd91e7c737d25f19035a80334f5f466ea91df96'}, {'bbox': [0.594, 0.389, 0.779, 0.669], 'data': {'canvas': [1000, 850], 'x': [1, 1.3, 1.6, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36], 'response': [0.029, 0.026, 0.022, 0.019, 0.016, 0.013, 0.0095, 0.007, 0.005, 0.0036, 0.0025, 0.0016, 0.0009, 0.0001, -0.0003, -0.0005, -0.0005, -0.0004, -0.0003, -0.0002, -0.0001, 0, 0, 0, 0, 0], 'upper': [0.112, 0.096, 0.082, 0.067, 0.054, 0.044, 0.03, 0.022, 0.016, 0.012, 0.009, 0.0072, 0.006, 0.0048, 0.0042, 0.004, 0.0038, 0.0036, 0.0035, 0.0034, 0.0033, 0.0032, 0.0032, 0.0031, 0.0031, 0.003], 'lower': [-0.077, -0.064, -0.052, -0.039, -0.029, -0.023, -0.016, -0.012, -0.009, -0.007, -0.0058, -0.005, -0.0045, -0.004, -0.0038, -0.0037, -0.0036, -0.0035, -0.0034, -0.0033, -0.0032, -0.0032, -0.0031, -0.0031, -0.003, -0.003], 'xlim': [1, 36], 'ylim': [-0.1, 0.15], 'xticks': [10, 20, 30], 'yticks': [-0.1, -0.05, 0, 0.05, 0.1, 0.15], 'reference': 0}, 'label_map': {'title': 'panel_08_title'}, 'crop_pixel_bbox': [608, 229, 798, 394], 'crop_sha256': 'd55cc13508917fd14ab9a7b3233b27318389b48b00e238203d74a6b1bd8546b9', 'api_request_sha256': '7e11f4226da56d79a981412f66f9310202344517568630beb6bfec79a250cbec'}, {'bbox': [0.788, 0.389, 0.972, 0.669], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38], 'response': [0.025, 0.041, 0.052, 0.039, 0.028, 0.029, 0.031, 0.033, 0.034, 0.033, 0.031, 0.024, 0.016, 0.009, 0.003, -0.001, -0.004, -0.006, -0.007, -0.007, -0.0065, -0.006, -0.005, -0.004, -0.003], 'upper': [0.137, 0.127, 0.095, 0.071, 0.063, 0.063, 0.065, 0.066, 0.065, 0.063, 0.06, 0.051, 0.042, 0.033, 0.025, 0.019, 0.015, 0.012, 0.01, 0.009, 0.009, 0.009, 0.009, 0.01, 0.011], 'lower': [-0.073, -0.043, -0.021, -0.017, -0.019, -0.014, -0.011, -0.007, -0.006, -0.005, -0.005, -0.008, -0.013, -0.018, -0.022, -0.024, -0.025, -0.025, -0.024, -0.023, -0.022, -0.021, -0.02, -0.018, -0.016], 'x_ticks': [0, 10, 20, 30], 'y_ticks': [-0.1, -0.05, 0, 0.05, 0.1, 0.15], 'x_limits': [0, 38], 'y_limits': [-0.1, 0.15], 'reference': 0}, 'label_map': {'title': 'panel_09_title'}, 'crop_pixel_bbox': [807, 229, 995, 394], 'crop_sha256': '2de7488e7ac4465e0f5b9d93f55243b0079740d2c921e5b51b71d937a1e87b93', 'api_request_sha256': '67ec3e500cc3e34563f525ecde259177135d27e180bf6e5456cf619a879090b1'}, {'bbox': [0.035, 0.698, 0.207, 0.978], 'data': {'canvas': [960, 900], 'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36], 'response': [0.0002, 0.00048, 0.00074, 0.00096, 0.00111, 0.00118, 0.00116, 0.00109, 0.00098, 0.00086, 0.00072, 0.00058, 0.00044, 0.00032, 0.00022, 0.00013, 6e-05, -1.5e-05, -5e-05, -4.5e-05, -2e-05, 1.5e-05, 5.5e-05, 9.5e-05, 0.00013, 0.00015, 0.00016], 'upper': [0.00045, 0.00096, 0.00147, 0.00191, 0.00224, 0.00248, 0.00259, 0.00262, 0.00258, 0.00249, 0.00237, 0.00225, 0.00212, 0.00199, 0.00187, 0.00175, 0.00166, 0.00151, 0.0014, 0.00132, 0.00126, 0.00122, 0.00118, 0.00115, 0.00112, 0.00109, 0.00105], 'lower': [-2e-05, -1.5e-05, -2e-05, -5.5e-05, -0.00013, -0.00027, -0.00048, -0.00073, -0.00096, -0.00117, -0.00136, -0.0015, -0.0016, -0.00167, -0.00171, -0.00172, -0.00171, -0.00166, -0.00158, -0.00147, -0.00135, -0.00123, -0.00112, -0.00102, -0.00093, -0.00085, -0.00079], 'xticks': [0, 10, 20, 30], 'yticks': [-0.002, -0.001, 0, 0.001, 0.002, 0.003], 'xlim': [0, 36], 'ylim': [-0.002, 0.003], 'reference': 0}, 'label_map': {'title': 'panel_10_title'}, 'crop_pixel_bbox': [36, 411, 212, 576], 'crop_sha256': '405fcb5039fa8f800b63796b23759dc2d82372cdf2d0fa06115423969209c660', 'api_request_sha256': '38bb35e1eb0c2f2cf219db504fe5dfbbc71e73b6472342185f9c88aeee9c6170'}, {'bbox': [0.226, 0.698, 0.397, 0.978], 'data': {'x': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40], 'response': [0.00028, 0.00031, 0.00037, 0.00046, 0.00055, 0.00062, 0.00067, 0.000705, 0.00072, 0.00072, 0.00071, 0.000695, 0.00068, 0.000655, 0.00063, 0.00061, 0.00059, 0.000575, 0.00056, 0.00055, 0.00054], 'upper': [0.00056, 0.0006, 0.0007, 0.00086, 0.00102, 0.00116, 0.00126, 0.00133, 0.00137, 0.0014, 0.00141, 0.001405, 0.0014, 0.00139, 0.00138, 0.00138, 0.001365, 0.00135, 0.00133, 0.00132, 0.00131], 'lower': [8.5e-05, 9.5e-05, 0.00013, 0.00018, 0.00021, 0.00023, 0.000235, 0.00023, 0.000215, 0.000195, 0.00018, 0.000165, 0.00015, 0.00013, 0.00011, 9e-05, 7e-05, 5.5e-05, 4e-05, 3e-05, 2e-05], 'reference': [0, 0], 'reference_x': [0, 40], 'xlim': [0, 40], 'ylim': [-2e-05, 0.0015], 'xticks': [10, 20, 30], 'yticks': [0, 0.0005, 0.001, 0.0015], 'canvas': [900, 844]}, 'label_map': {'title': 'panel_11_title'}, 'crop_pixel_bbox': [231, 411, 407, 576], 'crop_sha256': 'f23e0d0989fa670029e7c2e3da1715553660d915d42c12ed82e63044da5be19c', 'api_request_sha256': '7ad5f6baf7d75f84425e1b26106c1c3c370c4c71087abd418b7c960b946488e1'}, {'bbox': [0.412, 0.698, 0.588, 0.978], 'data': {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36], 'response': [0.076, 0.133, 0.151, 0.155, 0.146, 0.132, 0.117, 0.102, 0.087, 0.071, 0.045, 0.026, 0.013, 0.005, 0.0, -0.002, -0.003, -0.003, -0.002, -0.001, 0.0, 0.001, 0.001], 'upper': [0.183, 0.231, 0.256, 0.26, 0.249, 0.228, 0.207, 0.187, 0.169, 0.15, 0.119, 0.095, 0.076, 0.061, 0.051, 0.043, 0.037, 0.032, 0.03, 0.028, 0.027, 0.026, 0.025], 'lower': [0.042, 0.056, 0.062, 0.059, 0.049, 0.038, 0.027, 0.016, 0.006, -0.003, -0.017, -0.025, -0.03, -0.033, -0.034, -0.034, -0.033, -0.031, -0.029, -0.027, -0.025, -0.023, -0.021], 'xlim': [0, 36], 'ylim': [-0.1, 0.3], 'xticks': [10, 20, 30], 'yticks': [-0.1, 0, 0.1, 0.2, 0.3], 'reference': 0}, 'label_map': {'title': 'panel_12_title'}, 'crop_pixel_bbox': [422, 411, 602, 576], 'crop_sha256': 'd2d0ea365b7e1f826dc0a4f07a1a70bfb22992ff4239905c6262dece0ab3d34f', 'api_request_sha256': 'd2d3157bfea207bd91af5a0923cdd57f3052425cba5aba49f7099bf462d56c5c'}, {'bbox': [0.594, 0.698, 0.779, 0.978], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 37], 'response': [0.0038, 0.0038, 0.0037, 0.0037, 0.0037, 0.0038, 0.0039, 0.004, 0.004, 0.0039, 0.0037, 0.0031, 0.0026, 0.0022, 0.0019, 0.0016, 0.0014, 0.0012, 0.0011, 0.001, 0.0009, 0.0008, 0.0007, 0.0007, 0.0007], 'upper': [0.026, 0.0236, 0.0218, 0.0207, 0.0197, 0.0187, 0.0178, 0.0169, 0.0161, 0.0153, 0.0146, 0.0133, 0.0121, 0.0111, 0.0102, 0.0093, 0.0085, 0.0078, 0.0071, 0.0065, 0.006, 0.0055, 0.005, 0.0046, 0.0044], 'lower': [-0.019, -0.0174, -0.016, -0.0148, -0.0137, -0.0127, -0.0118, -0.011, -0.0103, -0.0097, -0.0091, -0.0082, -0.0075, -0.0069, -0.0063, -0.0058, -0.0053, -0.0049, -0.0045, -0.0041, -0.0038, -0.0035, -0.0032, -0.003, -0.0029], 'xlim': [0, 37], 'ylim': [-0.02, 0.03], 'xticks': [10, 20, 30], 'yticks': [-0.02, -0.01, 0, 0.01, 0.02, 0.03], 'reference': 0}, 'label_map': {'title': 'panel_13_title'}, 'crop_pixel_bbox': [608, 411, 798, 576], 'crop_sha256': '2c1e16b39ee37adfa5d33d7b308dc4420f39c28e02b19530d4fcca1a84c2f669', 'api_request_sha256': 'a7dc398d4993e90447b3a043cb20e580826c34015af9f75b29e66c6b3a714432'}, {'bbox': [0.791, 0.698, 0.972, 0.978], 'data': {'canvas': [1000, 900], 'x': [0, 0.6, 1.2, 1.8, 2.5, 3.2, 4, 5, 6, 7.5, 9, 10.5, 12, 14, 16, 18, 20, 23, 26, 29, 32, 34, 36], 'response': [-0.185, -0.158, -0.114, -0.073, -0.067, -0.076, -0.053, -0.042, -0.03, -0.02, -0.013, -0.007, -0.003, 0.002, 0.003, 0.003, 0.002, 0.001, 0, -0.001, -0.001, -0.001, 0], 'gray_lower': [-0.242, -0.223, -0.184, -0.149, -0.132, -0.129, -0.104, -0.071, -0.058, -0.04, -0.029, -0.022, -0.018, -0.015, -0.014, -0.014, -0.014, -0.014, -0.014, -0.014, -0.014, -0.013, -0.012], 'gray_upper': [-0.03, -0.008, 0.01, 0.023, 0.016, 0.004, 0.007, 0.013, 0.021, 0.028, 0.031, 0.033, 0.034, 0.034, 0.033, 0.031, 0.027, 0.022, 0.018, 0.014, 0.01, 0.008, 0.008], 'green_lower': [-0.203, -0.176, -0.133, -0.09, -0.084, -0.091, -0.066, -0.054, -0.039, -0.028, -0.02, -0.013, -0.009, -0.003, -0.002, -0.002, -0.003, -0.004, -0.005, -0.005, -0.005, -0.005, -0.004], 'green_upper': [-0.165, -0.138, -0.095, -0.055, -0.05, -0.06, -0.039, -0.029, -0.02, -0.012, -0.006, -0.001, 0.003, 0.007, 0.008, 0.008, 0.007, 0.006, 0.005, 0.003, 0.003, 0.003, 0.004], 'xlim': [0, 36], 'ylim': [-0.3, 0.1], 'xticks': [0, 10, 20, 30], 'yticks': [-0.3, -0.2, -0.1, 0, 0.1], 'reference': 0}, 'label_map': {'title': 'panel_14_title'}, 'crop_pixel_bbox': [810, 411, 995, 576], 'crop_sha256': 'c2e2e81bbc859adb458e918974d4a8f467d2db3233eeeb73a7e33f01dd62b076', 'api_request_sha256': '73e26a2ad02dc4df7a331f62158d527b3278f90ec024367aeb77adc7eda68a11'}], 'global_placements': [{'key': 'title', 'x': 0.489, 'y': 0.042, 'size': 14, 'max_width': 0.9}]}
LABELS = {'title': 'Responses to an Agg. Demand shock', 'panel_00_title': 'GDP', 'panel_01_title': 'prices', 'panel_02_title': 'interest rate', 'panel_03_title': 'inv/out', 'panel_04_title': 'stock prices', 'panel_05_title': 'spread', 'panel_06_title': 'credit/REV', 'panel_07_title': 'EBP', 'panel_08_title': 'EBP/VIX', 'panel_09_title': 'mortgage rates', 'panel_10_title': 'employment', 'panel_11_title': 'core prices', 'panel_12_title': 'ffr', 'panel_13_title': 'stock prices 2', 'panel_14_title': 'spread 2'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
