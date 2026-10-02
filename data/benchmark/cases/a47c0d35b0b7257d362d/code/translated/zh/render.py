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
        fig = plt.figure(figsize=(12, 3.3), dpi=100)
        ax = fig.add_axes([0.06, 0.205, 0.915, 0.76])
        for (s, c, f) in [('blue', '#3f5792', '#92bed5'), ('red', '#b84943', '#ffa79c')]:
            x = data['x_' + s]
            y = data['y_' + s]
            ax.fill_between(x, 6.6, y, color=f, alpha=0.85, linewidth=0)
            ax.plot(x, y, color=c, linewidth=1.25)
        ax.set_xlim(-72, 24)
        ax.set_ylim(6.6, 8.2)
        ax.set_xticks(data['xticks'])
        ax.set_xticklabels([str(v) + ':00' for v in data['xticks']], fontsize=9)
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['%.1f' % v for v in data['yticks']], fontsize=10)
        ax.grid(True, color='#dedede', alpha=0.55, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, color='#555555')
        for sp in ax.spines.values():
            sp.set_color('#555555')
            sp.set_linewidth(1)
        placements = [{'key': 'ylabel', 'x': 0.022, 'y': 0.58, 'size': 22, 'max_width': 0.64, 'rotation': 90, 'anchor': 'center'}, {'key': 'caption', 'x': 0.52, 'y': 0.925, 'size': 29, 'max_width': 0.88, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_01(data, labels):
        (W, H) = data['size']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.057, 0.218, 0.924, 0.75])
        placements = []
        for (name, line, fill) in [('blue', '#26287b', '#929cdc'), ('red', '#b72b2b', '#ee958b')]:
            ys = np.array(data[name])
            xs = data[name + '_start'] + np.arange(len(ys)) * data['step']
            if name == 'blue':
                xs = np.linspace(-72, 0, len(ys))
            xx = np.linspace(xs[0], xs[-1], len(xs) * 6)
            yy = np.interp(xx, xs, ys)
            amp = data['texture'][0] + data['texture'][1] * np.minimum(yy, 0.4)
            yy = np.maximum(0, yy + amp * (np.sin(xx * 31) + 0.55 * np.sin(xx * 53)))
            for (x, y) in data[name + '_peaks']:
                yy = np.maximum(yy, y * np.maximum(0, 1 - np.abs(xx - x) / 0.19))
            ax.fill_between(xx, 0, yy, color=fill, alpha=0.72)
            ax.plot(xx, yy, color=line, lw=1)
        ax.set_xlim(*data['xlim'])
        ax.set_ylim(*data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_xticklabels([str(v) + ':00' for v in data['xticks']], fontsize=9)
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels([format(v, '.1f') if v else '0' for v in data['yticks']], fontsize=9)
        ax.grid(True, color='#dddddd', alpha=0.6, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, color='#777777', pad=3)
        for s in ax.spines.values():
            s.set_color('#555555')
            s.set_linewidth(1)
        placements.append({'key': 'y_axis', 'x': 0.017, 'y': 0.59, 'size': 20, 'max_width': 0.72, 'rotation': 90, 'anchor': 'center'})
        placements.append({'key': 'caption', 'x': 0.52, 'y': 0.925, 'size': 26, 'max_width': 0.94, 'anchor': 'center'})
        return finish(fig, labels, placements)

    def panel_02(data, labels):
        fig = plt.figure(figsize=(10.16, 6.38), dpi=100)
        ax = fig.add_axes([0.105, 0.289, 0.873, 0.68])
        x = np.arange(len(data['categories']))
        ax.bar(x - 0.17, data['random'], width=0.25, color=data['colors'][0], zorder=3)
        ax.bar(x + 0.17, data['pd'], width=0.27, facecolor='white', edgecolor=data['colors'][1], hatch='**', linewidth=1.1, zorder=3)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_yticks(data['ticks'])
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis='both', length=0, labelsize=15, pad=6)
        for s in ax.spines.values():
            s.set_linewidth(1.4)
            s.set_color('#444444')
        placements = [{'key': 'ylabel', 'x': 0.03, 'y': 0.61, 'size': 38, 'max_width': 0.48, 'rotation': 90, 'anchor': 'center'}, {'key': 'caption', 'x': 0.54, 'y': 0.89, 'size': 49, 'max_width': 0.92, 'rotation': 0, 'anchor': 'center'}]
        for (i, key) in enumerate(data['categories']):
            px = 0.105 + 0.873 * (i - data['xlim'][0]) / (data['xlim'][1] - data['xlim'][0])
            placements.append({'key': key, 'x': px, 'y': 0.75, 'size': 21, 'max_width': 0.08, 'rotation': 0, 'anchor': 'center'})
        ax.add_patch(Rectangle((0.778, 0.785), 0.21, 0.185, transform=ax.transAxes, facecolor='white', edgecolor='#dddddd', linewidth=1, zorder=5))
        for (j, key) in enumerate(['random', 'pd']):
            yy = 0.91 - j * 0.084
            ax.add_patch(Rectangle((0.791, yy), 0.054, 0.037, transform=ax.transAxes, facecolor='black' if j == 0 else 'white', edgecolor=data['colors'][j], hatch=None if j == 0 else '**', linewidth=1, zorder=6))
            placements.append({'key': key, 'x': 0.86, 'y': 1 - (0.289 + 0.68 * (yy + 0.0185)), 'size': 25, 'max_width': 0.11, 'rotation': 0, 'anchor': 'left'})
        return finish(fig, labels, placements)

    def panel_03(data, labels):
        fig = plt.figure(figsize=(10, 5.82), dpi=100)
        ax = fig.add_axes([0.13, 0.235, 0.833, 0.728])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_xticklabels(data['xticklabels'], fontsize=15)
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['0' if v == 0 else '%.2f' % v for v in data['yticks']], fontsize=15)
        ax.grid(True, color='#ebebeb', linewidth=1.5)
        ax.set_axisbelow(True)
        for (s, c) in zip(['blue', 'red'], data['colors']):
            x = data[s + '_x']
            y = data[s + '_y']
            ax.fill_between(x, 0, y, color=c, alpha=0.25)
            ax.plot(x, y, color=c, linewidth=5, alpha=0.12)
            ax.plot(x, y, color=c, linewidth=1.9, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_color('#555555')
            spine.set_linewidth(1.7)
        ax.tick_params(length=6, width=1.2, color='#555555', pad=7)
        ax.annotate('', xy=data['arrow_end'], xytext=data['arrow_start'], arrowprops={'arrowstyle': '-|>', 'color': '#f13b18', 'lw': 2.8, 'mutation_scale': 29})
        placements = [{'key': 'volume', 'x': 0.032, 'y': 0.595, 'size': 32, 'max_width': 0.76, 'rotation': 90, 'anchor': 'center'}, {'key': 'caption', 'x': 0.54, 'y': 0.935, 'size': 49, 'max_width': 0.88, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)
    functions = [panel_00, panel_01, panel_02, panel_03]
    (width, height) = data['canvas']
    image = Image.new('RGB', (width, height), 'white')
    boxes = []
    for (i, panel) in enumerate(data['panels']):
        local = {k: labels[v] for (k, v) in panel['label_map'].items()}
        (part, part_boxes) = functions[i](panel['data'], local)
        (left, top, right, bottom) = panel['bbox']
        (x, y) = (round(left * width), round(top * height))
        (w, h) = (round((right - left) * width), round((bottom - top) * height))
        max_right = max((box['box'][2] for box in part_boxes), default=part.width)
        effective_w = w
        if max_right > 0:
            effective_w = min(w, max(1, int(w * part.width / max_right) - 8))
        (sx, sy) = (effective_w / part.width, h / part.height)
        image.paste(part.resize((effective_w, h), Image.Resampling.LANCZOS), (x, y))
        for box in part_boxes:
            b = dict(box)
            (a, bb, c, d) = b['box']
            b['box'] = [round(x + a * sx), round(y + bb * sy), round(x + c * sx), round(y + d * sy)]
            b['label_key'] = panel['label_map'].get(b.get('label_key'), b.get('label_key'))
            b['effective_font_size'] = b.get('font_size', 0) * min(sx, sy)
            b['inside_canvas'] = b['box'][0] >= 0 and b['box'][1] >= 0 and (b['box'][2] <= width) and (b['box'][3] <= height)
            boxes.append(b)
    for p in data['global_placements']:
        size = p.get('size', 30)
        max_width = min(p.get('max_width', 0.8) * width, width - 20)
        text = labels[p['key']]
        while mask(text, size).width > max_width and size > 10:
            size -= 1
        if mask(text, size).width > width - 20:
            text = '\n'.join(wrap(text, width - 20, size))
        half = mask(text, size).width / 2
        gx = min(max(p['x'] * width, half + 3), width - half - 3)
        b = put(image, text, gx, p['y'] * height, size, max_width=max_width, anchor=p.get('anchor', 'center'))
        b['label_key'] = p['key']
        boxes.append(b)
    return (image, boxes)
BASE_ID = 'qa_b55ca6b93debd445098bb345f65796fd9f24b13ad68bcda809d3888b239e50a5'
LANGUAGE = 'zh'
DATA = {'canvas': [2800, 2425], 'panels': [{'bbox': [0.0, 0.0, 1.0, 0.312], 'data': {'x_blue': [-72, -71.5, -71, -70.5, -70, -69.5, -69, -68.5, -68, -67.5, -67, -66.5, -66, -65.5, -65, -64.5, -64, -63.5, -63, -62.5, -62, -61.5, -61, -60.5, -60, -59.5, -59, -58.7, -58.5, -58.2, -58, -57.5, -57, -56.5, -56, -55.5, -55, -54.5, -54, -53.7, -53.4, -53, -52.5, -52, -51.5, -51, -50.5, -50, -49.5, -49, -48.8, -48.4, -48, -47.6, -47.2, -46.7, -46.2, -45.7, -45.2, -44.7, -44.2, -43.7, -43.2, -42.7, -42.2, -41.7, -41.2, -40.7, -40.2, -39.7, -39.2, -38.7, -38.2, -37.7, -37.2, -36.7, -36.2, -35.7, -35.2, -34.7, -34.2, -33.7, -33.2, -32.7, -32.2, -31.7, -31.2, -30.7, -30.2, -29.7, -29.2, -28.7, -28.2, -27.7, -27.2, -26.7, -26.2, -25.7, -25.2, -24.7, -24.2, -23.7, -23.2, -22.7, -22.2, -21.7, -21.2, -20.7, -20.2, -19.7, -19.2, -18.7, -18.2, -17.7, -17.2, -16.7, -16.2, -15.7, -15.2, -14.7, -14.2, -13.7, -13.2, -12.7, -12.2, -11.7, -11.2, -10.7, -10.2, -9.7, -9.2, -8.7, -8.2, -7.7, -7.2, -6.7, -6.2, -5.7, -5.2, -4.7, -4.2, -3.7, -3.2, -2.7, -2.2, -1.7, -1.2, -0.7, -0.3, 0], 'y_blue': [6.7, 6.69, 6.72, 6.7, 6.7, 6.69, 6.71, 6.7, 6.69, 6.7, 6.68, 6.69, 6.7, 6.69, 6.72, 6.73, 6.72, 6.73, 6.71, 6.7, 6.73, 6.72, 6.71, 6.73, 6.72, 6.73, 6.75, 6.78, 6.87, 6.85, 6.8, 6.78, 6.77, 6.77, 6.75, 6.77, 6.76, 6.79, 6.8, 6.82, 6.87, 6.86, 6.82, 6.83, 6.84, 6.84, 6.85, 6.87, 6.86, 6.88, 6.96, 6.95, 6.99, 6.96, 6.97, 6.94, 6.93, 6.96, 6.96, 6.95, 6.96, 6.95, 6.93, 6.95, 6.92, 6.9, 6.88, 6.9, 6.88, 6.91, 6.87, 6.87, 6.9, 6.89, 6.91, 6.91, 6.93, 6.92, 6.93, 6.9, 6.89, 6.9, 6.91, 6.9, 6.93, 6.94, 6.98, 6.98, 6.97, 6.94, 6.95, 6.93, 6.97, 7, 6.98, 6.96, 6.98, 6.92, 6.96, 6.92, 6.92, 6.88, 6.93, 6.94, 6.96, 6.98, 6.97, 6.98, 6.99, 6.98, 7.01, 7.01, 7, 7.04, 7.01, 7, 6.99, 7.02, 7.02, 7.05, 7.1, 7.11, 7.06, 7.05, 7.02, 7.04, 7.02, 7.04, 7.03, 7.06, 7.07, 7.07, 7.06, 7.08, 7.06, 7.06, 7.08, 7.06, 7.1, 7.12, 7.16, 7.17, 7.21, 7.2, 7.22, 7.25, 7.29, 7.32, 7.32, 7.34], 'x_red': [0, 0.12, 0.22, 0.35, 0.55, 0.8, 1, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.1, 5.3, 5.6, 6, 6.4, 6.8, 7.2, 7.6, 8, 8.4, 8.8, 9.2, 9.6, 10, 10.4, 10.8, 11, 11.3, 11.6, 12, 12.4, 12.8, 13.2, 13.6, 14, 14.4, 14.8, 15.2, 15.6, 16, 16.3, 16.7, 17, 17.4, 17.8, 18.2, 18.6, 19, 19.4, 19.8, 20.2, 20.6, 21, 21.4, 21.8, 22.2, 22.6, 23, 23.4, 23.7, 24], 'y_red': [7.34, 7.56, 8.16, 7.78, 7.67, 7.62, 7.6, 7.47, 7.45, 7.39, 7.38, 7.39, 7.36, 7.39, 7.37, 7.39, 7.36, 7.38, 7.4, 7.32, 7.33, 7.31, 7.31, 7.33, 7.29, 7.29, 7.24, 7.23, 7.21, 7.22, 7.2, 7.23, 7.3, 7.33, 7.35, 7.27, 7.28, 7.29, 7.26, 7.26, 7.32, 7.28, 7.27, 7.26, 7.29, 7.26, 7.27, 7.29, 7.38, 7.42, 7.4, 7.41, 7.36, 7.36, 7.33, 7.35, 7.34, 7.32, 7.36, 7.37, 7.31, 7.28, 7.31, 7.3, 7.24, 7.28, 7.27, 7.3, 7.3], 'xticks': [-72, -66, -60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0, 6, 12, 18, 24], 'yticks': [6.6, 6.8, 7, 7.2, 7.4, 7.6, 7.8, 8, 8.2]}, 'label_map': {'ylabel': 'panel_00_ylabel', 'caption': 'panel_00_caption'}, 'crop_pixel_bbox': [0, 0, 1024, 277], 'crop_sha256': '06547faec07659a6fa29b9661c5c5998849be95f44d66d35751e8f0c92dcfc7b', 'api_request_sha256': 'a5efa17e146ec41c8a186b022b3b438f6f7fd033adb71b294ca721116e35247f'}, {'bbox': [0.0, 0.317, 1.0, 0.634], 'data': {'size': [1100, 310], 'xlim': [-72, 24], 'ylim': [0, 1.4], 'xticks': [-72, -66, -60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0, 6, 12, 18, 24], 'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'blue_start': -72, 'red_start': 0, 'step': 0.5, 'blue': [0.025, 0.02, 0.033, 0.023, 0.027, 0.019, 0.023, 0.02, 0.019, 0.016, 0.015, 0.019, 0.012, 0.012, 0.009, 0.014, 0.02, 0.014, 0.027, 0.015, 0.02, 0.02, 0.015, 0.018, 0.012, 0.019, 0.015, 0.023, 0.019, 0.017, 0.058, 0.049, 0.089, 0.13, 0.2, 0.14, 0.098, 0.072, 0.115, 0.08, 0.063, 0.04, 0.06, 0.028, 0.047, 0.035, 0.068, 0.062, 0.065, 0.049, 0.06, 0.047, 0.068, 0.037, 0.059, 0.099, 0.42, 0.48, 0.35, 0.25, 0.34, 0.31, 0.14, 0.1, 0.22, 0.15, 0.15, 0.1, 0.087, 0.05, 0.047, 0.032, 0.085, 0.31, 0.22, 0.3, 0.33, 0.12, 0.17, 0.27, 0.2, 0.12, 0.2, 0.32, 0.13, 0.2, 0.18, 0.21, 0.12, 0.08, 0.2, 0.13, 0.27, 0.15, 0.19, 0.14, 0.18, 0.16, 0.15, 0.11, 0.2, 0.12, 0.17, 0.18, 0.23, 0.14, 0.19, 0.15, 0.11, 0.12, 0.09, 0.09, 0.07, 0.06, 0.1, 0.08, 0.08, 0.06, 0.12, 0.075, 0.09, 0.065, 0.08, 0.065, 0.15, 0.14, 0.08, 0.045, 0.06, 0.055, 0.05, 0.037, 0.06, 0.045, 0.036, 0.08, 0.05, 0.16, 0.13, 0.11, 0.13, 0.12, 0.09, 0.07, 0.075, 0.075, 0.07, 0.05, 0.08, 0.055, 0.07, 0.06, 0.09, 0.32, 0.31, 0.13, 0.08, 0.15, 0.1, 0.18, 0.21, 0.35, 1.35], 'red': [1.45, 0.86, 0.5, 0.33, 0.24, 0.2, 0.17, 0.15, 0.27, 0.2, 0.14, 0.13, 0.11, 0.15, 0.1, 0.13, 0.08, 0.16, 0.16, 0.1, 0.115, 0.09, 0.18, 0.08, 0.085, 0.065, 0.07, 0.055, 0.07, 0.065, 0.07, 0.08, 0.085, 0.075, 0.1, 0.08, 0.085, 0.07, 0.17, 0.14, 0.1, 0.105, 0.09, 0.08, 0.075, 0.09, 0.075, 0.065, 0.2], 'blue_peaks': [[-50.8, 0.17], [-46.7, 0.28], [-45.8, 0.73], [-45.35, 0.89], [-43.65, 0.46], [-41.9, 0.36], [-40.95, 0.51], [-38.15, 0.45], [-37.2, 0.61], [-36.45, 0.72], [-33.5, 0.43], [-30.15, 0.7], [-24.25, 0.3], [-21.9, 0.23], [-20, 0.18], [-19.4, 0.22], [-18.1, 0.21], [-16.1, 0.2], [-15.6, 0.23], [-9.35, 0.42]], 'red_peaks': [[0.16, 1.7], [0.42, 1.62], [0.72, 0.95], [1.22, 0.61], [3.55, 0.33], [5.75, 0.21], [6.55, 0.31], [9.95, 0.33], [10.75, 0.47], [13.85, 0.18], [14.7, 0.17], [18.7, 0.24], [19.65, 0.23]], 'texture': [0.008, 0.025]}, 'label_map': {'y_axis': 'panel_01_y_axis', 'caption': 'panel_01_caption'}, 'crop_pixel_bbox': [0, 281, 1024, 562], 'crop_sha256': '5b38d0a2a5e8dbfda2dc1a05eaee5ba683a094799037b6884c11ee52af373cd9', 'api_request_sha256': 'e5ff3346ffeac07e4b90ee07ebf0fffe7ce9059f21e7a08c30a9e3cc53966063'}, {'bbox': [0.0, 0.64, 0.502, 1.0], 'data': {'categories': ['h72', 'h60', 'h48', 'h36', 'h24', 'h12', 'h6', 'h3', 'h1'], 'pd': [9.2, 9.3, 7.9, 7.6, 7.0, 6.5, 5.5, 4.4, 3.3], 'random': [-0.2, -0.04, -0.04, -0.16, -0.05, -0.04, -0.04, -0.07, -0.06], 'ticks': [0, 2, 4, 6, 8, 10, 12], 'colors': ['#000000', '#ed8177'], 'xlim': [-0.6, 8.85], 'ylim': [-1, 12]}, 'label_map': {'ylabel': 'panel_02_ylabel', 'random': 'panel_02_random', 'pd': 'panel_02_pd', 'h72': 'panel_02_h72', 'h60': 'panel_02_h60', 'h48': 'panel_02_h48', 'h36': 'panel_02_h36', 'h24': 'panel_02_h24', 'h12': 'panel_02_h12', 'h6': 'panel_02_h6', 'h3': 'panel_02_h3', 'h1': 'panel_02_h1', 'caption': 'panel_02_caption'}, 'crop_pixel_bbox': [0, 568, 514, 887], 'crop_sha256': '7557d34286f98059cf39477f385970d9f1413a48e5cbe9b6deb80c8aa486adf7', 'api_request_sha256': '937a82dbb6a615c7faa5ad4f4de9b6681e0df40198c32eeab4459832889b10d6'}, {'bbox': [0.505, 0.64, 1.0, 1.0], 'data': {'xlim': [-12, 12], 'ylim': [0, 2], 'xticks': [-12, -9, -6, -3, 0, 3, 6, 9, 12], 'xticklabels': ['-12:00', '-9:00', '-6:00', '-3:00', '0:00', '3:00', '6:00', '9:00', '12:00'], 'yticks': [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], 'blue_x': [-12, -11.6, -11.45, -11.3, -10.8, -10, -9, -8, -7.2, -7, -6.8, -6.4, -6, -5.5, -5.1, -4.92, -4.83, -4.73, -4.63, -4.54, -4.4, -4.25, -4.08, -3.95, -3.6, -3.25, -3.15, -3.04, -2.94, -2.85, -2.73, -2.63, -2.52, -2.42, -2.32, -2.22, -2.1, -1.94, -1.8, -1.68, -1.55, -1.44, -1.32, -1.16, -1, -0.87, -0.76, -0.66, -0.53, -0.4, -0.25, 0, 0.5, 1, 2, 3, 4, 6, 9, 12], 'blue_y': [0.005, 0.005, 0.015, 0.004, 0.004, 0.004, 0.003, 0.004, 0.007, 0.014, 0.004, 0.007, 0.004, 0.005, 0.006, 0.007, 0.37, 0.65, 0.29, 0.065, 0.027, 0.009, 0.014, 0.006, 0.007, 0.009, 0.055, 0.011, 0.095, 0.22, 0.2, 0.055, 0.016, 0.42, 0.096, 0.021, 0.108, 0.03, 0.011, 0.016, 0.13, 0.049, 0.009, 0.006, 0.032, 0.17, 0.055, 0.012, 0.004, 0.007, 0.004, 0.004, 0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003], 'red_x': [-12, -6, -3, -1, -0.25, -0.13, -0.05, 0.02, 0.1, 0.17, 0.24, 0.32, 0.42, 0.52, 0.66, 0.78, 0.91, 1.02, 1.14, 1.28, 1.4, 1.53, 1.66, 1.81, 1.98, 2.14, 2.3, 2.44, 2.56, 2.68, 2.8, 2.96, 3.1, 3.24, 3.4, 3.54, 3.67, 3.82, 4, 4.3, 4.5, 4.9, 5.3, 5.7, 6.1, 6.5, 6.9, 7.04, 7.16, 7.26, 7.38, 7.65, 7.92, 8.2, 8.44, 8.57, 8.68, 8.8, 9.03, 9.13, 9.25, 9.36, 9.51, 9.65, 9.81, 10.1, 10.28, 10.48, 10.8, 11.06, 11.23, 11.4, 11.65, 12], 'red_y': [0, 0, 0, 0, 0.015, 0.13, 0.67, 1.87, 0.99, 0.5, 0.39, 0.11, 0.036, 0.01, 0.016, 0.098, 0.021, 0.012, 0.083, 0.019, 0.068, 0.02, 0.056, 0.009, 0.051, 0.024, 0.011, 0.007, 0.01, 0.051, 0.02, 0.011, 0.015, 0.008, 0.015, 0.028, 0.008, 0.053, 0.016, 0.01, 0.024, 0.008, 0.009, 0.013, 0.008, 0.009, 0.008, 0.015, 0.19, 0.022, 0.008, 0.014, 0.009, 0.024, 0.098, 0.031, 0.016, 0.01, 0.009, 0.16, 0.056, 0.023, 0.035, 0.013, 0.009, 0.035, 0.014, 0.008, 0.008, 0.01, 0.041, 0.02, 0.009, 0.007], 'arrow_start': [-6.4, 0.72], 'arrow_end': [-4.85, 0.27], 'colors': ['#28258c', '#bd282b']}, 'label_map': {'volume': 'panel_03_volume', 'caption': 'panel_03_caption'}, 'crop_pixel_bbox': [517, 568, 1024, 887], 'crop_sha256': 'bcd7629d04783c8629575f8f01cd9e210286d56dd4fbeb6e1028c2492d3ed05f', 'api_request_sha256': '7847488f09bf3238ba7f1560da8a2bdd989d04505e4668cf919a41b6b06bf154'}], 'global_placements': []}
LABELS = {'panel_00_ylabel': 'BTC价格 (1e-5)', 'panel_00_caption': '(a) 以BTC计价的平均价格随时间的变化', 'panel_01_y_axis': '交易量 (1e6)', 'panel_01_caption': '(b) 以目标币计量的平均交易量随时间的变化', 'panel_02_ylabel': '收益率 (%)', 'panel_02_random': '随机', 'panel_02_pd': '拉高出货', 'panel_02_h72': '72小时', 'panel_02_h60': '60小时', 'panel_02_h48': '48小时', 'panel_02_h36': '36小时', 'panel_02_h24': '24小时', 'panel_02_h12': '12小时', 'panel_02_h6': '6小时', 'panel_02_h3': '3小时', 'panel_02_h1': '1小时', 'panel_02_caption': '(c) 从拉盘前x+1小时至前1小时\n这一时间窗口内的平均收益', 'panel_03_volume': '以EDO计量的交易量 (1e5)', 'panel_03_caption': '(d) 提前拉盘示例'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
