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
    ax = fig.add_axes([0.093, 0.071, 0.698, 0.883])
    placements = []
    ax.set_xlim(data['xlim'])
    ax.set_ylim(data['ylim'])
    ax.set_xticks(data['xticks'])
    ax.set_yticks(data['yticks'])
    ax.set_xticks(data['minor_xticks'], minor=True)
    ax.set_yticks(data['minor_yticks'], minor=True)
    ax.grid(which='major', color='#e9e9e9', linewidth=1.5)
    ax.grid(which='minor', color='#eeeeee', linewidth=1)
    ax.tick_params(labelsize=12, colors='#444444')
    ax.tick_params(which='minor', length=0)
    for sp in ax.spines.values():
        sp.set_color('#555555')
        sp.set_linewidth(0.8)

    def marker(m):
        return (8, 2, 0) if m == 'star8' else (6, 1, 0) if m == 'star6' else m
    for s in data['sources']:
        ax.errorbar(s['x'], s['y'], xerr=s['xerr'], yerr=s['yerr'], fmt='none', ecolor=s['color'], elinewidth=1.4, zorder=2)
        ax.scatter([s['x']], [s['y']], marker=marker(s['marker']), s=105, facecolors='none', edgecolors=s['color'], linewidths=2, zorder=4)
        ax.scatter([s['x']], [s['y']], marker='+', s=190, c=s['color'], linewidths=1.7, zorder=4)
    for s in data['series']:
        p = np.array(s['points'])
        m = marker(s['marker'])
        if s['marker'] in ['+', 'x', 'star8']:
            ax.scatter(p[:, 0], p[:, 1], marker=m, s=s['size'], color=s['color'], linewidths=2, zorder=3)
        else:
            ax.scatter(p[:, 0], p[:, 1], marker=m, s=s['size'], facecolors='none', edgecolors=s['color'], linewidths=2, zorder=3)
        if 'overlay' in s:
            ax.scatter(p[:, 0], p[:, 1], marker=s['overlay'], s=s['size'] * 0.82, color=s['color'], linewidths=1.7, zorder=3)
    placements.extend([{'key': 'title', 'x': 0.093, 'y': 0.022, 'size': 27, 'max_width': 0.85, 'anchor': 'left'}, {'key': 'x_axis', 'x': 0.442, 'y': 0.975, 'size': 23, 'max_width': 0.53, 'anchor': 'center'}, {'key': 'y_axis', 'x': 0.029, 'y': 0.513, 'size': 23, 'max_width': 0.6, 'rotation': 90, 'anchor': 'center'}])
    leg = fig.add_axes([0.815, 0.28, 0.18, 0.43])
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.axis('off')
    for (i, s) in enumerate(data['series'] + data['sources']):
        y = 0.98 - i * 0.0802
        m = marker(s['marker'])
        color = s['color']
        if s['marker'] in ['+', 'x', 'star8']:
            leg.scatter([0.15], [y], marker=m, s=s.get('size', 105), c=color, linewidths=2)
        else:
            leg.scatter([0.15], [y], marker=m, s=s.get('size', 105), facecolors='none', edgecolors=color, linewidths=2)
        if 'overlay' in s:
            leg.scatter([0.15], [y], marker='x', s=60, c=color, linewidths=1.8)
        if i >= 8:
            leg.plot([0.08, 0.22], [y, y], color=color, lw=1.2)
            leg.plot([0.15, 0.15], [y - 0.03, y + 0.03], color=color, lw=1.2)
            leg.scatter([0.15], [y], marker='+', s=110, c=color)
        placements.append({'key': s['key'], 'x': 0.87, 'y': 1 - (0.28 + 0.43 * y), 'size': 18, 'max_width': 0.126, 'anchor': 'left'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_6c586d1de6bf9ef11bb2892e02f6b92897ce717b31e6f3067f1355f3342ed794'
LANGUAGE = 'pt'
DATA = {'canvas': [1020, 1030], 'xlim': [-31.25, -6.35], 'ylim': [5.05, 16.58], 'xticks': [-30, -25, -20, -15, -10], 'yticks': [7.5, 10, 12.5, 15], 'minor_xticks': [-27.5, -22.5, -17.5, -12.5, -7.5], 'minor_yticks': [6.25, 8.75, 11.25, 13.75, 16.25], 'series': [{'key': 'period1', 'color': '#440154', 'marker': 'o', 'size': 69, 'points': [[-14.85, 12.29], [-14.37, 12.3], [-14.48, 11.94], [-14.92, 11.79], [-15.22, 11.51], [-14.81, 11.32], [-15.17, 11.18], [-14.83, 11.09], [-11.88, 10.38], [-11.66, 10.18], [-11.24, 10.21], [-11.22, 10.53], [-10.73, 10.43], [-10.43, 10.45], [-10.85, 9.9], [-11.51, 9.34]]}, {'key': 'period2', 'color': '#482878', 'marker': '^', 'size': 119, 'points': [[-18.02, 10.96], [-17.59, 11.13], [-17.43, 10.74], [-17.12, 10.62], [-17.21, 11.13], [-17.23, 11.76], [-16.98, 11.52], [-16.56, 11.49], [-16.01, 11.49], [-16.53, 11.9], [-16.38, 12.39], [-16.23, 12.45], [-16.08, 12.33], [-16.22, 12.22], [-15.79, 12.22], [-15.72, 11.78], [-15.92, 11.71], [-15.56, 11.83], [-15.59, 11.37], [-15.84, 11.24], [-15.67, 11.46], [-15.35, 11.58], [-15.0, 11.6], [-14.85, 11.15], [-15.06, 11.0], [-15.51, 12.11], [-15.26, 12.14]]}, {'key': 'period3', 'color': '#443a78', 'marker': '+', 'size': 180, 'points': [[-22.74, 8.29], [-22.38, 9.02], [-22.04, 8.6], [-21.85, 9.19], [-22.26, 9.03], [-21.63, 7.52], [-21.35, 7.74], [-18.05, 6.48], [-22.7, 9.95], [-22.45, 10.24], [-22.12, 10.3], [-21.79, 10.23], [-21.53, 10.38], [-21.29, 10.21], [-21.39, 9.99], [-21.0, 9.9], [-20.73, 10.34], [-20.38, 10.41], [-19.85, 10.32], [-20.06, 10.59], [-19.38, 10.81], [-19.35, 10.7], [-19.07, 11.13], [-18.4, 11.37], [-17.83, 11.65], [-20.22, 11.67], [-20.42, 11.59], [-20.69, 11.29], [-20.83, 11.34], [-21.04, 11.31], [-21.24, 11.11], [-21.55, 11.12], [-21.68, 11.28], [-21.8, 11.17], [-22.05, 11.08], [-22.32, 11.1], [-22.78, 11.17], [-22.33, 11.57], [-22.84, 10.8], [-22.79, 10.71], [-22.45, 10.68], [-22.13, 10.79], [-21.89, 10.78], [-21.65, 10.87], [-21.45, 10.84], [-21.19, 10.69], [-20.89, 10.82], [-20.73, 10.66], [-20.31, 10.95], [-20.03, 10.9], [-20.13, 10.73], [-21.99, 10.54], [-21.7, 10.54], [-21.48, 10.63], [-21.3, 10.46], [-21.02, 10.44], [-20.76, 10.46], [-20.04, 11.04], [-19.07, 12.39]]}, {'key': 'period4', 'color': '#355f83', 'marker': 'x', 'size': 86, 'points': [[-16.15, 12.28], [-15.73, 12.26], [-15.13, 12.14], [-14.77, 12.16], [-14.59, 12.37], [-14.29, 11.69], [-14.68, 11.38], [-14.32, 11.08], [-13.83, 11.19], [-14.43, 10.75], [-13.92, 10.78], [-12.34, 10.96], [-12.19, 11.12], [-10.42, 10.93], [-14.73, 8.15]]}, {'key': 'period5', 'color': '#2c7181', 'marker': 'D', 'size': 78, 'points': [[-16.2, 12.99], [-16.23, 12.9], [-15.74, 13.51], [-15.6, 13.37], [-15.02, 13.36], [-15.53, 13.01], [-15.62, 12.46], [-15.42, 12.24], [-15.17, 12.41], [-15.01, 12.59], [-14.94, 12.32], [-14.83, 12.02], [-14.67, 12.44], [-14.58, 12.29], [-14.46, 12.39], [-14.18, 12.72], [-13.96, 12.76], [-15.26, 12.49], [-15.79, 11.99], [-15.48, 11.88], [-14.31, 12.07], [-14.23, 11.95], [-14.25, 11.76], [-13.32, 12.0], [-14.14, 11.49], [-13.92, 11.42], [-13.84, 11.51], [-14.75, 11.33], [-15.22, 10.57], [-13.32, 10.99], [-12.85, 11.3], [-12.99, 10.48], [-13.76, 10.67]]}, {'key': 'period6', 'color': '#238a8b', 'marker': 'v', 'size': 111, 'points': [[-21.39, 9.25], [-21.16, 11.16], [-20.89, 10.94], [-20.86, 10.78], [-20.77, 10.61], [-20.58, 10.45], [-20.47, 10.88], [-20.37, 11.13], [-20.17, 10.71], [-20.02, 10.44], [-19.69, 11.13], [-19.35, 10.62], [-21.53, 10.91], [-21.23, 10.73]]}, {'key': 'period7', 'color': '#209a89', 'marker': 's', 'overlay': 'x', 'size': 76, 'points': [[-29.23, 8.03], [-28.94, 8.15], [-28.73, 8.38], [-28.69, 7.96], [-28.46, 8.16], [-28.3, 8.01], [-28.14, 8.06], [-28.03, 8.12], [-27.83, 8.04], [-27.49, 8.23], [-27.74, 8.59], [-27.75, 8.94], [-27.72, 9.36], [-27.66, 9.48], [-27.47, 9.49], [-27.44, 9.4], [-27.29, 9.67], [-27.78, 7.58], [-27.45, 7.65], [-26.81, 8.07], [-26.71, 8.32], [-26.62, 8.86], [-26.02, 9.11], [-25.96, 9.56], [-26.01, 9.78], [-25.44, 8.52], [-25.39, 8.61], [-25.33, 9.6], [-24.79, 9.49], [-24.87, 9.83], [-24.62, 9.81], [-17.35, 10.95], [-14.39, 9.38]]}, {'key': 'period8', 'color': '#35ad87', 'marker': 'star8', 'size': 163, 'points': [[-29.4, 8.34], [-29.29, 7.13], [-29.05, 7.32], [-28.75, 7.25], [-28.59, 7.22], [-28.61, 7.42], [-28.45, 7.74], [-28.34, 7.99], [-28.45, 8.21], [-28.83, 8.29], [-28.91, 8.07], [-29.17, 7.99], [-29.2, 8.8], [-28.35, 8.91], [-27.96, 8.88], [-27.74, 8.68], [-27.76, 9.36], [-27.71, 9.62], [-27.57, 9.49], [-27.37, 8.34], [-27.33, 7.74], [-27.35, 7.55], [-27.32, 7.32], [-28.44, 7.48], [-26.5, 8.77], [-26.26, 8.21], [-26.02, 10.05], [-26.03, 7.36], [-26.11, 7.3], [-24.36, 8.85], [-22.59, 9.12]]}], 'sources': [{'key': 'zostera', 'color': '#57b978', 'marker': 'D', 'x': -9.53, 'y': 10.03, 'xerr': 1.4, 'yerr': 1.63}, {'key': 'grass', 'color': '#86c54e', 'marker': 'o', 'x': -29.27, 'y': 7.97, 'xerr': 0.91, 'yerr': 2.38}, {'key': 'ulactuca', 'color': '#b3cf42', 'marker': 'star6', 'x': -9.54, 'y': 14.72, 'xerr': 2.05, 'yerr': 1.34}, {'key': 'enteromorpha', 'color': '#eee83c', 'marker': 's', 'x': -12.43, 'y': 13.36, 'xerr': 1.32, 'yerr': 1.12}]}
LABELS = {'title': 'Gráfico do espaço isotópico dos dados de gansos de Inger et al.', 'x_axis': 'δ¹³C (‰)', 'y_axis': 'δ¹⁵N (‰)', 'period1': 'Gansos: período 1', 'period2': 'Gansos: período 2', 'period3': 'Gansos: período 3', 'period4': 'Gansos: período 4', 'period5': 'Gansos: período 5', 'period6': 'Gansos: período 6', 'period7': 'Gansos: período 7', 'period8': 'Gansos: período 8', 'zostera': 'Zostera', 'grass': 'Erva', 'ulactuca': 'U.lactuca', 'enteromorpha': 'Enteromorpha'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
