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
        fig = plt.figure(figsize=(9.3, 5.64), dpi=100)
        ax = fig.add_axes([0.147, 0.15, 0.846, 0.792])
        for (j, s) in enumerate(data['series']):
            x = np.array(s['x'])
            y = np.array(s['y'])
            xx = []
            yy = []
            for i in range(len(x) - 1):
                for t in [0, 0.25, 0.5, 0.75]:
                    xx.append(x[i] + t * (x[i + 1] - x[i]))
                    yy.append(y[i] + t * (y[i + 1] - y[i]) + 0.0018 * math.sin((i * 4 + t * 4) * 2.3 + j) * math.sin(math.pi * t))
            xx.append(x[-1])
            yy.append(y[-1])
            ax.plot(xx, yy, color=s['color'], lw=1.6)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_xticklabels(['0', '0.05', '0.1', '0.15'], fontfamily='serif', fontsize=23)
        ax.set_yticklabels(['0', '0.1', '0.2', '0.3'], fontfamily='serif', fontsize=23)
        ax.grid(True, color='#d4d4d4', linewidth=2)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=15)
        for spine in ax.spines.values():
            spine.set_color('#606060')
            spine.set_linewidth(2)
        return finish(fig, labels, [])

    def panel_01(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.13, 0.16, 0.82, 0.79])
        placements = []
        for (y, c, a, lw) in zip(data['series'], data['colors'], data['alpha'], data['line_width']):
            ax.plot(data['x'], y, color=c, alpha=a, linewidth=lw * 2.1)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.grid(True, color='#cfcfcf', linewidth=1.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0, labelsize=32, pad=10)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('serif')
        for s in ax.spines.values():
            s.set_color('#444444')
            s.set_linewidth(1.4)
        return finish(fig, labels, placements)

    def panel_02(data, labels):
        fig = plt.figure(figsize=(data['canvas'][0] / 100, data['canvas'][1] / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.grid(True, color='#d2d2d2', linewidth=2)
        ax.set_axisbelow(True)
        for s in data['series']:
            ax.plot(data['x'], s['y'], color=s['color'], linewidth=2.1)
        ax.tick_params(axis='both', which='both', length=0, labelsize=49, pad=12)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('serif')
        for spine in ax.spines.values():
            spine.set_color('#535353')
            spine.set_linewidth(2.5)
        return finish(fig, labels, [])

    def panel_03(data, labels):
        fig = plt.figure(figsize=(9.3, 5.34), dpi=100)
        ax = fig.add_axes([0.146, 0.155, 0.846, 0.833])
        for (i, s) in enumerate(data['curves']):
            x = np.array(s['x'])
            y = np.array(data['levels']) + np.array(s.get('dy', [0] * len(x)))
            xx = np.linspace(x[0], x[-1], 180)
            yy = np.interp(xx, x, y)
            taper = np.minimum(1, (x[-1] - xx) / 0.008)
            yy += taper * (0.0017 * np.sin(np.arange(180) * 1.14 + i * 1.8) + 0.0011 * np.sin(np.arange(180) * 0.39 + i))
            ax.plot(xx, yy, color=s['c'], lw=1.6, alpha=0.95)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_xticklabels(['0', '0.03', '0.06', '0.09'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['0', '0.1', '0.2'])
        ax.tick_params(axis='both', length=0, pad=17, labelsize=43)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('serif')
        ax.grid(True, color='#cfcfcf', linewidth=2)
        for sp in ax.spines.values():
            sp.set_linewidth(1.7)
            sp.set_color('#444444')
        return finish(fig, labels, [])

    def panel_04(data, labels):
        fig = plt.figure(figsize=(10, 5.56), dpi=100)
        ax = fig.add_axes([0.13, 0.16, 0.82, 0.83])
        for s in data['series']:
            ax.plot(s['x'], s['y'], color=s['color'], lw=1.45)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['0', '0.5', '1', '1.5', '2'], fontfamily='serif')
        ax.tick_params(axis='both', labelsize=23, length=0, pad=11)
        for t in ax.get_xticklabels():
            t.set_fontfamily('serif')
        ax.grid(True, color='#c8c8c8', lw=1.2)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color('#555555')
            sp.set_linewidth(1)
        return finish(fig, labels, [])

    def panel_05(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        placements = []
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.grid(True, color='#cccccc', linewidth=1.5, alpha=0.8)
        ax.set_axisbelow(True)
        for i in [6, 1, 2, 0, 4, 5, 3]:
            ax.plot(data['x'], np.array(data['base']) + np.array(data['offsets'][i]), color=data['colors'][i], linewidth=1.5, alpha=0.94)
        ax.tick_params(axis='both', which='major', labelsize=48, length=0, pad=12)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('STIXGeneral')
        for s in ax.spines.values():
            s.set_color('#444444')
            s.set_linewidth(2)
        return finish(fig, labels, placements)

    def panel_06(data, labels):
        fig = plt.figure(figsize=(10, 6), dpi=100)
        ax = fig.add_axes([0.145, 0.16, 0.842, 0.825])
        for (i, (path, c)) in enumerate(zip(data['paths'], data['colors'])):
            p = np.array(path)
            x = np.linspace(p[0, 0], p[-1, 0], int(data['noise'][2]))
            y = np.interp(x, p[:, 0], p[:, 1])
            t = np.arange(len(x))
            y += data['noise'][0] * np.sin(t * data['noise'][3] + i) + data['noise'][1] * np.sin(t * data['noise'][4] + i * 2)
            ax.plot(x, y, color=c, lw=1.5)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_xticklabels(['0', '0.02', '0.04', '0.06', '0.08'], fontfamily='serif', fontsize=23)
        ax.set_yticklabels(['0', '0.1', '0.2'], fontfamily='serif', fontsize=23)
        ax.grid(True, color='#bcbcbc', linewidth=1)
        ax.tick_params(length=0, pad=9)
        for s in ax.spines.values():
            s.set_linewidth(1.2)
            s.set_color('#333333')
        return finish(fig, labels, [])

    def panel_07(data, labels):
        fig = plt.figure(figsize=(9.6, 5.37), dpi=100)
        ax = fig.add_axes([0.13, 0.155, 0.82, 0.83])
        for (i, (t, c)) in enumerate(zip(data['traces'], data['colors'])):
            a = np.array(t)
            x = np.linspace(a[0, 0], a[-1, 0], 240)
            y = np.interp(x, a[:, 0], a[:, 1])
            w = np.minimum(1, (x[-1] - x) / 0.012)
            y += w * (0.004 * np.sin(x * 1060 + i * 2) + 0.003 * np.sin(x * 457 + i) + 0.002 * np.sin(x * 2180 + i))
            ax.plot(x, y, color=c, lw=1.9 if i < 3 else 1.5, alpha=0.8)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['x_ticks'])
        ax.set_yticks(data['y_ticks'])
        ax.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4'])
        ax.set_yticklabels(['0', '0.1', '0.2', '0.3'])
        ax.grid(True, color='#d0d0d0', lw=1.7)
        ax.tick_params(length=0, labelsize=43, pad=14)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('serif')
        for s in ax.spines.values():
            s.set_linewidth(1.7)
        return finish(fig, labels, [])

    def panel_08(data, labels):
        fig = plt.figure(figsize=(9, 5.3), dpi=100)
        ax = fig.add_axes([0.096, 0.157, 0.862, 0.827])
        for (y, c) in zip(data['series'], data['colors']):
            ax.plot(data['x'], y, color=c, linewidth=1.65)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_axisbelow(True)
        ax.grid(True, color='#bdbdbd', linewidth=1.4, alpha=0.65)
        ax.tick_params(axis='both', length=0, pad=13, labelsize=43)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('serif')
        for s in ax.spines.values():
            s.set_color('#555555')
            s.set_linewidth(1.5)
        return finish(fig, labels, [])

    def panel_09(data, labels):
        (W, H) = data['size']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        placements = []
        for (j, (color, end, prefix, bottom)) in enumerate(data['series']):
            f = np.array(data['tail_fraction'])
            x = np.r_[data['prefix_x'], 0.03 + (end - 0.03) * f]
            y = np.r_[prefix, prefix[-1] - (prefix[-1] - bottom) * np.array(data['tail_drop'])]
            xx = []
            yy = []
            for k in range(len(x) - 1):
                for q in range(4):
                    t = q / 4
                    xx.append(x[k] + (x[k + 1] - x[k]) * t)
                    yy.append(y[k] + (y[k + 1] - y[k]) * t + data['wiggle'][(k * 3 + q + j) % 10] * (1 if q else 0))
            xx.append(x[-1])
            yy.append(y[-1])
            ax.plot(xx, yy, color=color, lw=1.25)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_xticklabels(['0', '0.02', '0.04', '0.06'])
        ax.set_yticklabels(['0', '0.1', '0.2'])
        ax.tick_params(length=0, pad=17, labelsize=25)
        ax.grid(True, color='#cccccc', lw=1.3)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color('#444444')
            s.set_linewidth(1.8)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('serif')
        return finish(fig, labels, placements)

    def panel_10(data, labels):
        (W, H) = data['size']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        placements = []
        for (i, (end, c, ys)) in enumerate(data['series']):
            xp = np.array(data['fractions']) * end
            x = np.linspace(0, end, 210)
            y = np.interp(x, xp, ys)
            (a, b, d) = data['wiggle']
            y = y + a * (np.sin(x * b + i * 2) + 0.5 * np.sin(x * d + i))
            ax.plot(x, y, color=c, lw=data['linewidth'])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_xticklabels(['0', '0.05', '0.1'])
        ax.set_yticklabels(['0', '0.1', '0.2', '0.3'])
        ax.tick_params(length=0, pad=17, labelsize=39)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('serif')
        ax.grid(True, color='#bcbcbc', linewidth=1)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color('#444444')
            s.set_linewidth(1.5)
        return finish(fig, labels, placements)

    def panel_11(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.grid(True, color=data['grid_color'], linewidth=1.9, alpha=0.8)
        ax.set_axisbelow(True)
        for (y, c) in zip(data['traces'], data['colors']):
            ax.plot(data['x'], y, color=c, lw=data['line_width'])
        for s in ax.spines.values():
            s.set_color(data['spine_color'])
            s.set_linewidth(2.5)
        ax.tick_params(axis='both', length=0, labelsize=49, pad=15)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily('serif')
        return finish(fig, labels, [])
    functions = [panel_00, panel_01, panel_02, panel_03, panel_04, panel_05, panel_06, panel_07, panel_08, panel_09, panel_10, panel_11]
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
BASE_ID = 'qa_c92c443366441a8b24ca56780aa450649b4c58456db7ccb5fe3cbd791a298485'
LANGUAGE = 'pl'
DATA = {'canvas': [2800, 2218], 'panels': [{'bbox': [0.043, 0.048, 0.346, 0.28], 'data': {'xlim': [0, 0.18], 'ylim': [0, 0.305], 'xticks': [0, 0.05, 0.1, 0.15], 'yticks': [0, 0.1, 0.2, 0.3], 'series': [{'color': '#29218a', 'x': [0, 0.01, 0.022, 0.036, 0.047, 0.057, 0.068, 0.078, 0.09, 0.102, 0.112, 0.121, 0.131, 0.141, 0.149, 0.153, 0.158, 0.162, 0.165], 'y': [0.243, 0.259, 0.264, 0.244, 0.23, 0.236, 0.246, 0.261, 0.264, 0.262, 0.272, 0.262, 0.254, 0.233, 0.191, 0.173, 0.124, 0.096, 0.047]}, {'color': '#cf3035', 'x': [0, 0.01, 0.02, 0.03, 0.04, 0.049, 0.057, 0.065, 0.078, 0.09, 0.1, 0.11, 0.114, 0.124, 0.132, 0.135, 0.138, 0.14], 'y': [0.247, 0.253, 0.258, 0.268, 0.27, 0.277, 0.27, 0.267, 0.268, 0.259, 0.254, 0.253, 0.24, 0.222, 0.195, 0.178, 0.107, 0.031]}, {'color': '#bd242c', 'x': [0, 0.011, 0.02, 0.031, 0.041, 0.052, 0.061, 0.068, 0.075, 0.078, 0.08, 0.081], 'y': [0.246, 0.255, 0.26, 0.246, 0.23, 0.219, 0.192, 0.177, 0.134, 0.096, 0.066, 0.027]}, {'color': '#294ba1', 'x': [0, 0.013, 0.022, 0.031, 0.042, 0.051, 0.06, 0.067, 0.072, 0.076, 0.079], 'y': [0.241, 0.256, 0.253, 0.249, 0.233, 0.208, 0.193, 0.175, 0.148, 0.087, 0.033]}, {'color': '#6d3480', 'x': [0, 0.008, 0.018, 0.029, 0.039, 0.048, 0.053, 0.057, 0.059, 0.061, 0.063], 'y': [0.25, 0.24, 0.235, 0.22, 0.203, 0.193, 0.174, 0.147, 0.112, 0.074, 0.027]}, {'color': '#34342b', 'x': [0, 0.012, 0.025, 0.037, 0.046, 0.054, 0.06, 0.065, 0.068, 0.071, 0.074], 'y': [0.243, 0.257, 0.259, 0.243, 0.213, 0.187, 0.17, 0.151, 0.115, 0.078, 0.034]}, {'color': '#c9b836', 'x': [0, 0.014, 0.023, 0.037, 0.05, 0.061, 0.068, 0.072, 0.076, 0.079], 'y': [0.245, 0.241, 0.252, 0.235, 0.209, 0.197, 0.187, 0.158, 0.083, 0.034]}, {'color': '#95609b', 'x': [0, 0.015, 0.03, 0.043, 0.055, 0.064, 0.072, 0.078, 0.083, 0.086, 0.088], 'y': [0.241, 0.23, 0.236, 0.225, 0.223, 0.215, 0.199, 0.186, 0.142, 0.085, 0.035]}, {'color': '#c584be', 'x': [0, 0.015, 0.028, 0.039, 0.053, 0.064, 0.077, 0.084, 0.091, 0.094, 0.098], 'y': [0.25, 0.249, 0.242, 0.226, 0.24, 0.219, 0.204, 0.189, 0.172, 0.133, 0.033]}, {'color': '#dfc9e0', 'x': [0, 0.018, 0.032, 0.048, 0.063, 0.074, 0.083, 0.088, 0.093], 'y': [0.245, 0.245, 0.23, 0.237, 0.223, 0.202, 0.164, 0.111, 0.033]}, {'color': '#05d934', 'x': [0, 0.016, 0.032, 0.046, 0.058, 0.066, 0.078, 0.087, 0.095, 0.1, 0.103, 0.107], 'y': [0.242, 0.247, 0.241, 0.253, 0.24, 0.225, 0.23, 0.214, 0.182, 0.152, 0.092, 0.034]}, {'color': '#7beb7d', 'x': [0, 0.019, 0.035, 0.049, 0.061, 0.072, 0.085, 0.095, 0.103, 0.107, 0.11], 'y': [0.248, 0.239, 0.226, 0.232, 0.237, 0.227, 0.195, 0.163, 0.145, 0.08, 0.029]}, {'color': '#c5ddaa', 'x': [0, 0.02, 0.04, 0.057, 0.073, 0.086, 0.096, 0.107, 0.113, 0.118, 0.123], 'y': [0.245, 0.243, 0.223, 0.24, 0.244, 0.21, 0.187, 0.178, 0.13, 0.095, 0.038]}, {'color': '#daa748', 'x': [0, 0.016, 0.029, 0.044, 0.058, 0.07, 0.083, 0.093, 0.101, 0.11, 0.116, 0.121, 0.124], 'y': [0.244, 0.239, 0.225, 0.207, 0.23, 0.217, 0.212, 0.19, 0.167, 0.166, 0.151, 0.087, 0.034]}, {'color': '#ead4c8', 'x': [0, 0.02, 0.039, 0.057, 0.073, 0.088, 0.098, 0.108, 0.116, 0.121, 0.129, 0.134, 0.139], 'y': [0.249, 0.25, 0.247, 0.239, 0.225, 0.203, 0.19, 0.197, 0.179, 0.151, 0.141, 0.119, 0.035]}]}, 'label_map': {}, 'crop_pixel_bbox': [44, 39, 354, 227], 'crop_sha256': 'd5b0c59fb948f21496a6b60cfa56f2c99fccd1dd6c6d30ec6e5fda0b201e735e', 'api_request_sha256': 'c5e87750dae92ecfd2f363df2241542dbcf82d37832cabb444669c5c68f10c8a'}, {'bbox': [0.349, 0.048, 0.661, 0.28], 'data': {'canvas': [960, 650], 'x': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.4, 6.6, 7, 7.5, 8, 8.5, 9, 9.5, 10], 'series': [[0.1, 0.39, 0.82, 1.18, 1.65, 2.04, 2.47, 2.86, 3.35, 3.74, 4.15, 4.57, 4.99, 5.34, 5.53, 5.83, 6.25, 6.72, 7.13, 7.57, 8.01, 8.43], [0.15, 0.59, 1.03, 1.36, 1.84, 2.2, 2.62, 3.06, 3.59, 4.01, 4.46, 4.88, 5.3, 5.65, 5.84, 6.18, 6.62, 7.03, 7.51, 7.91, 8.36, 8.77], [0.12, 0.36, 0.7, 1.11, 1.5, 1.99, 2.36, 2.86, 3.29, 3.71, 4.1, 4.6, 5.06, 5.29, 5.48, 5.75, 6.18, 6.68, 7.04, 7.48, 7.97, 8.41], [0.09, 0.43, 0.97, 1.33, 1.85, 2.25, 2.66, 3.04, 3.39, 3.82, 4.27, 4.73, 5.12, 5.41, 5.56, 5.86, 6.23, 6.57, 7.1, 7.66, 8.08, 8.52], [0.08, 0.49, 0.85, 1.29, 1.61, 2.01, 2.42, 2.91, 3.32, 3.85, 4.31, 4.77, 5.18, 5.47, 5.62, 5.97, 6.35, 6.78, 7.24, 7.64, 8.19, 8.64], [0.04, 0.4, 0.91, 1.32, 1.78, 2.14, 2.52, 2.96, 3.66, 3.97, 4.52, 4.86, 5.19, 5.55, 5.69, 6.03, 6.51, 6.93, 7.34, 7.77, 8.19, 8.65], [0.12, 0.53, 0.93, 1.42, 1.95, 2.35, 2.68, 3.12, 3.75, 4.2, 4.66, 4.98, 5.44, 5.58, 3.54, 4, 4.34, 4.67, 5.08, 5.61, 5.91, 6.32]], 'colors': ['#60449e', '#54cb48', '#464ba4', '#ec3980', '#e32224', '#b8992c', '#d5bba9'], 'alpha': [1, 1, 1, 1, 1, 1, 0.42], 'line_width': [1.2, 1.3, 1.2, 1.3, 1.2, 1.1, 1], 'xlim': [0, 10], 'ylim': [0, 9.65], 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0, 2, 4, 6, 8]}, 'label_map': {}, 'crop_pixel_bbox': [357, 39, 677, 227], 'crop_sha256': '80427baf191ff6ce3ae81f103cbd06c01e468bc0730ba85d75325a4ed7bd238f', 'api_request_sha256': '9c6cc07eb26ad6ec00e06a967fb4b3844a8ff57c12af078934ae41eeffb8bbe4'}, {'bbox': [0.662, 0.048, 0.96, 0.28], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'series': [{'color': '#2b64b2', 'y': [0, 1.92, 3.92, 6.08, 8.12, 10.19, 12.46, 14.47, 16.59, 18.65, 20.72]}, {'color': '#9b289f', 'y': [0, 2.01, 4.12, 6.21, 8.45, 10.52, 12.79, 14.84, 16.9, 18.86, 20.96]}, {'color': '#55c957', 'y': [0, 1.98, 3.84, 5.91, 7.87, 9.97, 12.09, 14.3, 16.58, 18.95, 21.13]}, {'color': '#dca63b', 'y': [0, 1.91, 3.91, 5.94, 8.01, 10.05, 12.13, 14.09, 16.06, 18.13, 20.17]}, {'color': '#ed5274', 'y': [0, 2.06, 4.14, 6.07, 8.11, 10.22, 12.28, 14.39, 16.42, 18.35, 20.49]}], 'xlim': [0, 10], 'ylim': [0, 23.3], 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0, 10, 20], 'canvas': [1000, 620], 'axes': [0.105, 0.15, 0.858, 0.79]}, 'label_map': {}, 'crop_pixel_bbox': [678, 39, 983, 227], 'crop_sha256': 'bf0e6306683fd4c251e230cc9aafd037b9fc28c5d461874e286286c7631002f1', 'api_request_sha256': '97854eb605b2cdc2feb2932b17a39dde03f8159811e455b484ad23b4ffeb8ba9'}, {'bbox': [0.043, 0.281, 0.346, 0.501], 'data': {'xlim': [0, 0.102], 'ylim': [0, 0.285], 'xticks': [0, 0.03, 0.06, 0.09], 'yticks': [0, 0.1, 0.2], 'levels': [0.248, 0.251, 0.253, 0.249, 0.245, 0.24, 0.233, 0.225, 0.214, 0.202, 0.19, 0.175, 0.151, 0.116, 0.079, 0.03], 'curves': [{'c': '#d0d0d0', 'x': [0, 0.006, 0.012, 0.019, 0.026, 0.034, 0.042, 0.05, 0.062, 0.074, 0.079, 0.084, 0.088, 0.09, 0.091, 0.093], 'dy': [0, 0, 0.001, 0.003, 0.006, 0.009, 0.008, 0.011, 0.005, 0.009, 0.005, 0.001, 0, 0, 0, 0.007]}, {'c': '#cbcbcb', 'x': [0, 0.005, 0.011, 0.016, 0.022, 0.026, 0.029, 0.032, 0.035, 0.037, 0.039, 0.04, 0.041, 0.042, 0.043, 0.0445]}, {'c': '#dbaa53', 'x': [0, 0.006, 0.013, 0.021, 0.029, 0.036, 0.043, 0.05, 0.056, 0.059, 0.063, 0.069, 0.074, 0.076, 0.078, 0.0805], 'dy': [0, 0.002, 0.003, 0.007, 0.011, 0.01, 0.01, 0.01, 0.01, 0.007, 0.004, 0, 0, 0, 0, 0.028]}, {'c': '#83af7f', 'x': [0, 0.006, 0.013, 0.02, 0.028, 0.034, 0.04, 0.047, 0.051, 0.055, 0.059, 0.063, 0.068, 0.071, 0.073, 0.075]}, {'c': '#7560a7', 'x': [0, 0.006, 0.012, 0.019, 0.025, 0.031, 0.038, 0.044, 0.049, 0.054, 0.059, 0.064, 0.069, 0.071, 0.073, 0.076]}, {'c': '#644191', 'x': [0, 0.004, 0.01, 0.017, 0.023, 0.03, 0.035, 0.041, 0.047, 0.054, 0.061, 0.065, 0.067, 0.068, 0.07, 0.0725]}, {'c': '#27559a', 'x': [0, 0.005, 0.012, 0.018, 0.026, 0.032, 0.036, 0.041, 0.047, 0.053, 0.059, 0.063, 0.065, 0.067, 0.068, 0.0695]}, {'c': '#ad60c1', 'x': [0, 0.005, 0.011, 0.017, 0.023, 0.028, 0.034, 0.04, 0.046, 0.052, 0.057, 0.061, 0.064, 0.065, 0.066, 0.067]}, {'c': '#325fa5', 'x': [0, 0.005, 0.01, 0.016, 0.022, 0.027, 0.033, 0.039, 0.044, 0.048, 0.052, 0.057, 0.06, 0.062, 0.063, 0.065]}, {'c': '#24a333', 'x': [0, 0.005, 0.009, 0.015, 0.022, 0.027, 0.034, 0.041, 0.046, 0.05, 0.055, 0.059, 0.06, 0.061, 0.062, 0.064]}, {'c': '#60af53', 'x': [0, 0.005, 0.011, 0.015, 0.021, 0.026, 0.031, 0.036, 0.041, 0.046, 0.051, 0.056, 0.058, 0.06, 0.061, 0.063]}, {'c': '#bc9132', 'x': [0, 0.004, 0.009, 0.015, 0.021, 0.026, 0.03, 0.035, 0.04, 0.044, 0.049, 0.054, 0.057, 0.058, 0.059, 0.06]}, {'c': '#cdaa73', 'x': [0, 0.004, 0.008, 0.014, 0.02, 0.025, 0.029, 0.033, 0.038, 0.042, 0.046, 0.049, 0.051, 0.053, 0.054, 0.057]}, {'c': '#c78172', 'x': [0, 0.004, 0.009, 0.014, 0.018, 0.023, 0.027, 0.031, 0.035, 0.039, 0.043, 0.046, 0.049, 0.051, 0.052, 0.053]}, {'c': '#ce5276', 'x': [0, 0.004, 0.008, 0.013, 0.017, 0.021, 0.025, 0.029, 0.033, 0.037, 0.04, 0.044, 0.046, 0.048, 0.049, 0.05]}, {'c': '#db7682', 'x': [0, 0.004, 0.008, 0.013, 0.017, 0.02, 0.024, 0.028, 0.032, 0.035, 0.038, 0.04, 0.043, 0.045, 0.047, 0.048]}]}, 'label_map': {}, 'crop_pixel_bbox': [44, 228, 354, 406], 'crop_sha256': '3360d0323bd7da008c1ddf48957ac5ee651f1bff833e4a792745cf8c5181830a', 'api_request_sha256': '21deeb1d89847caf28b9ffb669965b2da67d722be3583f521e7bc5c72318c6dd'}, {'bbox': [0.349, 0.281, 0.661, 0.501], 'data': {'xlim': [0, 8.1], 'ylim': [0, 2.18], 'xticks': [0, 2, 4, 6, 8], 'yticks': [0, 0.5, 1, 1.5, 2], 'series': [{'color': '#333333', 'x': [0, 0.08, 0.18, 0.27, 0.38, 0.5, 0.61, 0.73, 0.83, 0.95, 1.08, 1.18, 1.3, 1.43, 1.55, 1.67, 1.81, 1.92, 2.03, 2.13, 2.24, 2.35, 2.48, 2.6, 2.71, 2.82, 2.95, 3.06, 3.17, 3.28, 3.39, 3.52, 3.62, 3.73, 3.86, 3.96, 4.08, 4.2, 4.33, 4.43, 4.54, 4.65, 4.77, 4.89, 5, 5.12, 5.24, 5.35, 5.46, 5.58, 5.69, 5.79, 5.9, 6.02, 6.16, 6.3, 6.43, 6.57, 6.72, 6.87, 6.99, 7.1, 7.21, 7.3, 7.38], 'y': [0.18, 0.33, 0.31, 0.39, 0.34, 0.49, 0.43, 0.47, 0.6, 0.64, 0.55, 0.61, 0.76, 0.82, 0.71, 0.74, 0.66, 0.59, 0.7, 0.64, 0.8, 0.86, 0.89, 0.96, 1.05, 1.13, 1.11, 1.18, 1.14, 1.2, 1.13, 1.26, 1.22, 1.32, 1.21, 1.32, 1.31, 1.4, 1.4, 1.49, 1.6, 1.51, 1.62, 1.59, 1.68, 1.65, 1.78, 1.74, 1.8, 1.83, 1.98, 1.96, 1.62, 1.68, 1.76, 1.73, 1.82, 1.84, 1.96, 1.97, 1.94, 1.22, 0.59, 0.25, 0.06]}, {'color': '#e89a32', 'x': [0, 0.16, 0.32, 0.48, 0.63, 0.78, 0.93, 1.08, 1.23, 1.38, 1.53, 1.68, 1.83, 1.96, 2.08, 2.21, 2.32, 2.42, 2.53, 2.61, 2.71, 2.82, 2.9, 2.96], 'y': [0.27, 0.39, 0.43, 0.59, 0.61, 0.66, 0.73, 0.82, 0.81, 0.93, 0.86, 1.01, 0.96, 1.05, 0.96, 1.03, 1.08, 1.17, 1.08, 0.87, 0.8, 0.89, 0.82, 0.05]}, {'color': '#e0b64a', 'x': [0, 0.2, 0.4, 0.6, 0.8, 0.99, 1.13, 1.26, 1.4, 1.56, 1.68, 1.83, 1.96, 2.11, 2.24, 2.37, 2.52, 2.6], 'y': [0.21, 0.31, 0.41, 0.52, 0.65, 0.69, 0.77, 0.89, 0.84, 0.95, 1.07, 1.02, 0.99, 1.06, 1.05, 1.09, 0.94, 0.04]}, {'color': '#7b7443', 'x': [0, 0.22, 0.41, 0.57, 0.72, 0.88, 1.03, 1.18, 1.35, 1.51, 1.65, 1.79, 1.94, 2.08, 2.21, 2.26], 'y': [0.18, 0.36, 0.3, 0.43, 0.42, 0.49, 0.44, 0.56, 0.62, 0.65, 0.68, 0.61, 0.77, 0.72, 0.86, 0.04]}, {'color': '#999999', 'x': [0, 0.2, 0.38, 0.57, 0.74, 0.9, 1.07, 1.22, 1.38, 1.52, 1.67, 1.82, 1.97, 2.02], 'y': [0.17, 0.33, 0.3, 0.4, 0.42, 0.44, 0.52, 0.58, 0.54, 0.62, 0.58, 0.64, 0.7, 0.03]}, {'color': '#d65091', 'x': [0, 0.14, 0.27, 0.38, 0.51, 0.64, 0.77, 0.91, 1.07, 1.18, 1.28, 1.33], 'y': [0.23, 0.33, 0.27, 0.44, 0.41, 0.49, 0.47, 0.57, 0.52, 0.57, 0.44, 0.03]}, {'color': '#b09245', 'x': [0, 0.16, 0.3, 0.44, 0.57, 0.72, 0.87, 1.01, 1.13, 1.25, 1.4, 1.47], 'y': [0.22, 0.28, 0.43, 0.42, 0.54, 0.53, 0.61, 0.54, 0.62, 0.52, 0.48, 0.05]}, {'color': '#d52329', 'x': [0, 0.13, 0.23, 0.35, 0.48, 0.59, 0.71, 0.77], 'y': [0.22, 0.32, 0.42, 0.47, 0.44, 0.6, 0.54, 0.04]}, {'color': '#9c24ac', 'x': [0, 0.11, 0.2, 0.3, 0.42, 0.51, 0.57, 0.64], 'y': [0.13, 0.29, 0.25, 0.36, 0.29, 0.46, 0.37, 0.02]}, {'color': '#2dbb36', 'x': [0, 0.12, 0.23, 0.33, 0.43, 0.5], 'y': [0.16, 0.24, 0.36, 0.25, 0.28, 0.01]}, {'color': '#26bbce', 'x': [0, 0.08, 0.16, 0.24, 0.29], 'y': [0.12, 0.3, 0.27, 0.23, 0.02]}]}, 'label_map': {}, 'crop_pixel_bbox': [357, 228, 677, 406], 'crop_sha256': '2131ec6fa29598b10d9dada030742ba64ab36265d75ac5c8d8dc96a3ed6c1ccf', 'api_request_sha256': 'c59b7f6c0c7bee62f5ab151339bd5dab8356044a5ab9b9d20474a61796bb22fc'}, {'bbox': [0.662, 0.281, 0.96, 0.501], 'data': {'x': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10], 'base': [0, 0.85, 1.7, 2.55, 3.4, 4.25, 5.1, 5.95, 6.8, 7.65, 8.5, 9.35, 10.2, 11.05, 11.9, 12.75, 13.6, 14.45, 15.3, 16.15, 17], 'offsets': [[0, 0.03, 0.05, 0.08, 0.05, 0.12, 0.14, 0.18, 0.23, 0.29, 0.3, 0.32, 0.42, 0.42, 0.5, 0.51, 0.64, 0.65, 0.71, 0.77, 0.83], [0, 0.01, 0.1, 0.15, 0.13, 0.22, 0.33, 0.35, 0.44, 0.51, 0.58, 0.61, 0.73, 0.73, 0.85, 0.94, 1.01, 1.03, 1.23, 1.31, 1.4], [0, -0.01, 0.07, 0.05, 0.12, 0.15, 0.22, 0.24, 0.32, 0.39, 0.48, 0.55, 0.63, 0.61, 0.72, 0.74, 0.86, 0.95, 1.03, 1.08, 1.14], [0, 0.07, 0.16, 0.16, 0.21, 0.2, 0.26, 0.2, 0.17, 0.2, 0.12, 0.17, 0.17, 0.15, 0.14, 0.16, 0.13, 0.16, 0.2, 0.24, 0.3], [0, 0.02, 0.04, 0.01, -0.03, 0.06, 0.06, 0.08, 0.13, 0.08, 0.11, 0.1, 0.1, 0.1, 0.17, 0.21, 0.3, 0.35, 0.39, 0.43, 0.52], [0, -0.01, 0.01, 0.02, 0.04, 0.08, 0.05, -0.03, -0.08, -0.1, -0.12, -0.06, -0.02, -0.04, 0, 0.02, 0.07, 0.13, 0.09, 0.08, 0.13], [0, 0.05, 0.1, 0.12, 0.15, 0.19, 0.3, 0.39, 0.43, 0.43, 0.46, 0.49, 0.56, 0.72, 0.89, 1.01, 1.12, 1.21, 1.31, 1.48, 1.58]], 'colors': ['#41956b', '#e8a65e', '#86c676', '#e15968', '#5995b8', '#a047a0', '#eadba2'], 'xlim': [0, 10], 'ylim': [0, 19.7], 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0, 5, 10, 15], 'canvas': [1000, 600], 'axes': [0.098, 0.16, 0.86, 0.825]}, 'label_map': {}, 'crop_pixel_bbox': [678, 228, 983, 406], 'crop_sha256': '45e6ea70f99bddd99d6f48898cc821954863a346e3ab2f6107bd7c2ab818c7dd', 'api_request_sha256': '28ab9a021b218d0e03ce96f538343ca10222c1d014606e6adf8b18582f4893ec'}, {'bbox': [0.043, 0.502, 0.346, 0.723], 'data': {'xlim': [0, 0.088], 'ylim': [0, 0.285], 'xticks': [0, 0.02, 0.04, 0.06, 0.08], 'yticks': [0, 0.1, 0.2], 'colors': ['#666666', '#754267', '#c0a889', '#e5c870', '#dbbb88', '#a099aa', '#944294', '#7266aa', '#a94162', '#373837', '#bc5965', '#75b269', '#75da64', '#d3b04a'], 'paths': [[[0, 0.246], [0.005, 0.251], [0.012, 0.253], [0.019, 0.255], [0.027, 0.254], [0.032, 0.249], [0.04, 0.258], [0.045, 0.246], [0.05, 0.224], [0.055, 0.213], [0.06, 0.199], [0.065, 0.208], [0.069, 0.2], [0.072, 0.181], [0.075, 0.168], [0.077, 0.134], [0.0785, 0.086], [0.08, 0.039]], [[0, 0.244], [0.006, 0.252], [0.014, 0.246], [0.021, 0.239], [0.03, 0.232], [0.037, 0.235], [0.043, 0.23], [0.048, 0.216], [0.055, 0.199], [0.059, 0.194], [0.064, 0.173], [0.068, 0.144], [0.07, 0.104], [0.072, 0.043], [0.074, 0.033]], [[0.001, 0.247], [0.01, 0.249], [0.018, 0.251], [0.026, 0.238], [0.034, 0.225], [0.042, 0.214], [0.049, 0.193], [0.055, 0.164], [0.058, 0.108], [0.059, 0.063], [0.061, 0.036]], [[0.004, 0.247], [0.012, 0.239], [0.022, 0.233], [0.03, 0.216], [0.038, 0.216], [0.044, 0.193], [0.05, 0.173], [0.052, 0.144], [0.053, 0.097], [0.056, 0.034]], [[0.002, 0.247], [0.01, 0.235], [0.02, 0.221], [0.028, 0.218], [0.033, 0.207], [0.04, 0.196], [0.045, 0.178], [0.047, 0.14], [0.05, 0.114], [0.053, 0.035]], [[0.003, 0.247], [0.012, 0.246], [0.02, 0.234], [0.027, 0.221], [0.032, 0.209], [0.037, 0.196], [0.042, 0.177], [0.047, 0.144], [0.049, 0.098], [0.051, 0.032]], [[0.002, 0.248], [0.009, 0.246], [0.018, 0.233], [0.026, 0.222], [0.033, 0.212], [0.039, 0.185], [0.044, 0.156], [0.048, 0.113], [0.05, 0.032]], [[0.003, 0.248], [0.011, 0.245], [0.021, 0.233], [0.03, 0.223], [0.034, 0.202], [0.038, 0.181], [0.043, 0.169], [0.045, 0.134], [0.046, 0.084], [0.047, 0.036]], [[0.004, 0.25], [0.012, 0.238], [0.022, 0.219], [0.029, 0.201], [0.034, 0.185], [0.039, 0.172], [0.041, 0.139], [0.043, 0.08], [0.045, 0.03]], [[0.003, 0.249], [0.011, 0.238], [0.019, 0.222], [0.026, 0.201], [0.033, 0.187], [0.037, 0.176], [0.039, 0.156], [0.041, 0.11], [0.043, 0.032]], [[0.001, 0.249], [0.01, 0.236], [0.019, 0.223], [0.028, 0.214], [0.033, 0.192], [0.036, 0.175], [0.038, 0.131], [0.04, 0.082], [0.042, 0.031]], [[0.002, 0.248], [0.01, 0.235], [0.018, 0.213], [0.025, 0.199], [0.03, 0.182], [0.034, 0.159], [0.036, 0.119], [0.037, 0.077], [0.039, 0.03]], [[0.005, 0.243], [0.012, 0.23], [0.018, 0.21], [0.024, 0.193], [0.028, 0.178], [0.03, 0.148], [0.032, 0.105], [0.033, 0.065], [0.035, 0.029]], [[0.004, 0.245], [0.012, 0.237], [0.018, 0.223], [0.025, 0.207], [0.029, 0.19], [0.032, 0.168], [0.034, 0.128], [0.035, 0.078], [0.037, 0.03]]], 'noise': [0.0015, 0.0009, 183, 0.73, 2.13]}, 'label_map': {}, 'crop_pixel_bbox': [44, 407, 354, 586], 'crop_sha256': 'a598e49bb123a740f745c91958f76b78e001507f8be1d566799f85e2df3bc30f', 'api_request_sha256': 'eaf03fee216ed28fc94e7634bfa1f295d8a8bacba6a6041a1e3337cfb01ce3fd'}, {'bbox': [0.349, 0.502, 0.661, 0.723], 'data': {'x_ticks': [0, 0.1, 0.2, 0.3, 0.4], 'y_ticks': [0, 0.1, 0.2, 0.3], 'xlim': [0, 0.43], 'ylim': [0, 0.355], 'colors': ['#dbb4ae', '#ead0ce', '#ebcaca', '#cbcdcb', '#a5aca7', '#262628', '#672363', '#b02c83', '#dc1562', '#f72a59', '#2869b0', '#9e9727', '#a7be33', '#38cf36', '#b85036', '#90652b'], 'traces': [[[0, 0.245], [0.015, 0.269], [0.03, 0.283], [0.05, 0.266], [0.07, 0.267], [0.09, 0.281], [0.11, 0.291], [0.12, 0.319], [0.13, 0.311], [0.14, 0.296], [0.16, 0.302], [0.18, 0.315], [0.2, 0.31], [0.22, 0.325], [0.235, 0.303], [0.245, 0.274], [0.255, 0.244], [0.267, 0.22], [0.273, 0.15], [0.28, 0.035]], [[0, 0.255], [0.02, 0.274], [0.04, 0.281], [0.06, 0.289], [0.08, 0.268], [0.1, 0.269], [0.12, 0.277], [0.14, 0.283], [0.16, 0.287], [0.175, 0.272], [0.19, 0.274], [0.2, 0.273], [0.21, 0.226], [0.218, 0.187], [0.225, 0.134], [0.235, 0.028]], [[0, 0.25], [0.025, 0.271], [0.045, 0.28], [0.065, 0.276], [0.08, 0.266], [0.1, 0.269], [0.12, 0.267], [0.14, 0.238], [0.16, 0.225], [0.18, 0.241], [0.2, 0.229], [0.22, 0.239], [0.24, 0.235], [0.26, 0.213], [0.28, 0.231], [0.3, 0.221], [0.32, 0.235], [0.34, 0.227], [0.355, 0.208], [0.37, 0.178], [0.38, 0.129], [0.39, 0.065]], [[0, 0.251], [0.03, 0.263], [0.05, 0.259], [0.07, 0.248], [0.09, 0.253], [0.12, 0.258], [0.14, 0.279], [0.16, 0.267], [0.18, 0.248], [0.2, 0.195], [0.215, 0.173], [0.225, 0.18], [0.24, 0.159], [0.25, 0.131], [0.26, 0.035]], [[0, 0.258], [0.02, 0.269], [0.04, 0.271], [0.06, 0.253], [0.08, 0.247], [0.1, 0.225], [0.12, 0.224], [0.14, 0.202], [0.16, 0.21], [0.18, 0.17], [0.19, 0.125], [0.195, 0.025]], [[0, 0.248], [0.02, 0.259], [0.04, 0.244], [0.06, 0.256], [0.08, 0.248], [0.1, 0.239], [0.12, 0.25], [0.14, 0.258], [0.155, 0.251], [0.17, 0.24], [0.185, 0.209], [0.2, 0.168], [0.21, 0.145], [0.22, 0.122], [0.225, 0.129], [0.24, 0.094], [0.25, 0.045]], [[0, 0.246], [0.03, 0.248], [0.05, 0.227], [0.07, 0.217], [0.085, 0.191], [0.1, 0.124], [0.12, 0.123], [0.14, 0.146], [0.16, 0.157], [0.18, 0.154], [0.19, 0.164], [0.2, 0.153], [0.205, 0.104], [0.21, 0.063]], [[0, 0.24], [0.02, 0.256], [0.04, 0.23], [0.06, 0.223], [0.08, 0.218], [0.09, 0.174], [0.095, 0.128], [0.1, 0.027]], [[0, 0.253], [0.02, 0.241], [0.04, 0.25], [0.06, 0.246], [0.08, 0.234], [0.1, 0.22], [0.12, 0.234], [0.14, 0.234], [0.16, 0.214], [0.175, 0.177], [0.18, 0.133], [0.187, 0.064]], [[0, 0.246], [0.02, 0.232], [0.04, 0.221], [0.06, 0.213], [0.075, 0.207], [0.085, 0.17], [0.09, 0.109], [0.096, 0.035]], [[0, 0.252], [0.02, 0.258], [0.04, 0.245], [0.06, 0.238], [0.08, 0.23], [0.095, 0.171], [0.11, 0.114], [0.12, 0.129], [0.13, 0.074]], [[0, 0.252], [0.03, 0.25], [0.05, 0.247], [0.07, 0.236], [0.09, 0.217], [0.105, 0.156], [0.116, 0.039]], [[0, 0.253], [0.02, 0.246], [0.04, 0.229], [0.06, 0.23], [0.08, 0.231], [0.1, 0.152], [0.11, 0.106], [0.12, 0.083], [0.135, 0.083], [0.14, 0.054]], [[0, 0.25], [0.02, 0.242], [0.04, 0.21], [0.065, 0.189], [0.08, 0.207], [0.09, 0.196], [0.105, 0.153], [0.12, 0.121], [0.135, 0.077], [0.145, 0.103], [0.15, 0.063]], [[0, 0.25], [0.03, 0.246], [0.05, 0.252], [0.07, 0.235], [0.085, 0.226], [0.1, 0.197], [0.12, 0.175], [0.145, 0.154], [0.16, 0.076]], [[0, 0.255], [0.02, 0.247], [0.035, 0.212], [0.042, 0.143], [0.048, 0.045]]]}, 'label_map': {}, 'crop_pixel_bbox': [357, 407, 677, 586], 'crop_sha256': '6227937e737f41b7758be39e34d81ff3ec633aab799f912e7fc0aa817553c351', 'api_request_sha256': '39a81b023f04df979e6b2f43ff49b0dd36b067ff07cfb84bfd55c9bc3c2d6950'}, {'bbox': [0.662, 0.502, 0.96, 0.723], 'data': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'series': [[0, 1.45, 2.72, 4.01, 5.29, 6.55, 7.93, 9.12, 10.78, 12.01, 13.54], [0, 1.28, 2.7, 4.16, 5.72, 7.08, 8.36, 9.48, 10.75, 12.28, 13.91], [0, 1.36, 2.56, 3.95, 5.34, 6.94, 8.19, 9.38, 10.94, 12.57, 14.24], [0, 1.31, 2.6, 3.89, 5.5, 6.84, 8.07, 9.44, 10.72, 12.34, 13.75], [0, 1.4, 2.82, 4.05, 5.57, 6.94, 8.36, 9.78, 11.29, 12.52, 13.8], [0, 1.25, 2.59, 3.79, 5.14, 6.39, 7.73, 9.09, 10.26, 11.68, 13.05], [0, 1.2, 2.52, 3.81, 5.24, 6.7, 8.02, 9.57, 11.01, 12.38, 13.57], [0, 1.42, 3.04, 4.18, 5.72, 7.15, 8.46, 9.62, 10.59, 12.08, 13.29], [0, 1.14, 2.4, 3.69, 5.06, 6.39, 7.76, 9.1, 10.64, 11.88, 13.17]], 'colors': ['#36adba', '#246caf', '#339c48', '#dd923a', '#dc4638', '#7a4b9d', '#938645', '#dc4ba3', '#943eaa'], 'xlim': [0, 10], 'ylim': [0, 15.5], 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0, 5, 10, 15]}, 'label_map': {}, 'crop_pixel_bbox': [678, 407, 983, 586], 'crop_sha256': '5672e352558f5748626bf7932b722502240cc946393b4e8e425a090ddac324be', 'api_request_sha256': 'aaa2b70c8c1f4aab6b654d97fd8cf1ec9bc5187d9b2e0bd6f2945364d0954f4f'}, {'bbox': [0.043, 0.724, 0.346, 0.95], 'data': {'size': [1000, 620], 'axes': [0.145, 0.18, 0.825, 0.79], 'xlim': [0, 0.0655], 'ylim': [0, 0.28], 'xticks': [0, 0.02, 0.04, 0.06], 'yticks': [0, 0.1, 0.2], 'prefix_x': [0, 0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021, 0.024, 0.027, 0.03], 'tail_fraction': [0.12, 0.25, 0.38, 0.5, 0.62, 0.72, 0.81, 0.89, 0.95, 1], 'tail_drop': [0.025, 0.055, 0.09, 0.15, 0.24, 0.39, 0.58, 0.77, 0.93, 1], 'series': [['#444444', 0.035, [0.245, 0.24, 0.231, 0.227, 0.218, 0.213, 0.204, 0.197, 0.19, 0.177, 0.144], 0.032], ['#727272', 0.0364, [0.245, 0.238, 0.24, 0.231, 0.223, 0.216, 0.202, 0.201, 0.188, 0.184, 0.166], 0.027], ['#bd548c', 0.0395, [0.247, 0.249, 0.244, 0.244, 0.232, 0.224, 0.214, 0.211, 0.207, 0.196, 0.184], 0.032], ['#969b6a', 0.0418, [0.246, 0.247, 0.241, 0.239, 0.244, 0.237, 0.225, 0.22, 0.217, 0.208, 0.19], 0.032], ['#e56b80', 0.0387, [0.248, 0.245, 0.249, 0.241, 0.238, 0.231, 0.225, 0.225, 0.22, 0.207, 0.183], 0.037], ['#92ae6a', 0.0406, [0.249, 0.248, 0.247, 0.237, 0.228, 0.227, 0.22, 0.211, 0.205, 0.206, 0.193], 0.036], ['#c070af', 0.0437, [0.245, 0.246, 0.251, 0.249, 0.241, 0.24, 0.238, 0.232, 0.225, 0.226, 0.218], 0.034], ['#d49b54', 0.043, [0.247, 0.248, 0.249, 0.245, 0.242, 0.237, 0.23, 0.225, 0.224, 0.216, 0.204], 0.038], ['#9a7185', 0.047, [0.246, 0.248, 0.247, 0.251, 0.248, 0.238, 0.236, 0.229, 0.223, 0.218, 0.202], 0.028], ['#9a9a50', 0.048, [0.248, 0.253, 0.247, 0.245, 0.24, 0.234, 0.226, 0.229, 0.216, 0.209, 0.203], 0.032], ['#c17994', 0.049, [0.245, 0.249, 0.248, 0.252, 0.246, 0.24, 0.234, 0.226, 0.224, 0.22, 0.207], 0.034], ['#cd824e', 0.0494, [0.247, 0.25, 0.252, 0.251, 0.248, 0.245, 0.237, 0.23, 0.226, 0.222, 0.21], 0.03], ['#b54f54', 0.0514, [0.245, 0.253, 0.254, 0.252, 0.246, 0.246, 0.242, 0.235, 0.231, 0.226, 0.223], 0.036], ['#4c9364', 0.0475, [0.246, 0.251, 0.247, 0.244, 0.239, 0.236, 0.23, 0.223, 0.225, 0.217, 0.215], 0.025], ['#708143', 0.0559, [0.247, 0.249, 0.249, 0.243, 0.245, 0.241, 0.235, 0.232, 0.23, 0.22, 0.215], 0.035], ['#7770b2', 0.057, [0.246, 0.249, 0.248, 0.251, 0.247, 0.244, 0.253, 0.246, 0.247, 0.243, 0.237], 0.035], ['#a5dba8', 0.0582, [0.245, 0.25, 0.247, 0.246, 0.243, 0.242, 0.238, 0.235, 0.23, 0.229, 0.225], 0.034], ['#c4a280', 0.0599, [0.248, 0.25, 0.253, 0.246, 0.241, 0.239, 0.237, 0.231, 0.229, 0.23, 0.221], 0.025]], 'wiggle': [0, 0.001, -0.001, 0.0015, -0.001, 0.0005, 0, -0.001, 0.001, 0]}, 'label_map': {}, 'crop_pixel_bbox': [44, 587, 354, 770], 'crop_sha256': 'ec4ef730b673e2f5bfe8af3c5bce201fcc02018d17d90ab3f74c73502ec139a1', 'api_request_sha256': '1b977c1b7e19b865c0f65fd5e5cc11882adfb745d1ce7731e4b979f9d6043d56'}, {'bbox': [0.349, 0.724, 0.661, 0.95], 'data': {'size': [984, 549], 'axes': [0.126, 0.175, 0.802, 0.812], 'xlim': [0, 0.14], 'ylim': [0, 0.31], 'xticks': [0, 0.05, 0.1], 'yticks': [0, 0.1, 0.2, 0.3], 'fractions': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.78, 0.84, 0.89, 0.93, 0.96, 1], 'series': [[0.049, '#c9f2bd', [0.247, 0.243, 0.238, 0.229, 0.223, 0.214, 0.204, 0.193, 0.18, 0.167, 0.139, 0.088, 0.048, 0.029]], [0.054, '#68c34b', [0.248, 0.249, 0.239, 0.234, 0.225, 0.232, 0.219, 0.205, 0.196, 0.19, 0.181, 0.112, 0.057, 0.027]], [0.06, '#387b4b', [0.249, 0.25, 0.246, 0.243, 0.23, 0.224, 0.207, 0.196, 0.186, 0.174, 0.112, 0.072, 0.044, 0.032]], [0.058, '#5683ba', [0.25, 0.242, 0.238, 0.232, 0.225, 0.22, 0.218, 0.212, 0.204, 0.193, 0.189, 0.159, 0.074, 0.033]], [0.067, '#747c67', [0.247, 0.246, 0.25, 0.247, 0.242, 0.219, 0.219, 0.212, 0.21, 0.205, 0.163, 0.13, 0.078, 0.027]], [0.073, '#dcad52', [0.251, 0.253, 0.251, 0.24, 0.225, 0.218, 0.2, 0.192, 0.18, 0.133, 0.097, 0.093, 0.094, 0.096]], [0.074, '#826fab', [0.247, 0.251, 0.257, 0.263, 0.265, 0.257, 0.247, 0.223, 0.207, 0.197, 0.181, 0.133, 0.068, 0.028]], [0.077, '#b39aa1', [0.247, 0.255, 0.258, 0.261, 0.255, 0.239, 0.231, 0.221, 0.2, 0.193, 0.155, 0.121, 0.079, 0.029]], [0.081, '#807272', [0.252, 0.259, 0.266, 0.264, 0.257, 0.247, 0.232, 0.222, 0.211, 0.209, 0.2, 0.154, 0.086, 0.027]], [0.078, '#8b5d8e', [0.246, 0.245, 0.243, 0.237, 0.238, 0.237, 0.244, 0.238, 0.227, 0.209, 0.197, 0.16, 0.111, 0.028]], [0.079, '#707c96', [0.251, 0.258, 0.259, 0.254, 0.247, 0.239, 0.237, 0.235, 0.229, 0.217, 0.215, 0.19, 0.105, 0.028]], [0.075, '#be875a', [0.251, 0.252, 0.257, 0.254, 0.247, 0.232, 0.214, 0.197, 0.18, 0.156, 0.147, 0.108, 0.07, 0.029]], [0.091, '#b63a85', [0.25, 0.247, 0.249, 0.243, 0.241, 0.218, 0.214, 0.199, 0.167, 0.154, 0.158, 0.123, 0.086, 0.039]], [0.083, '#aa4084', [0.245, 0.252, 0.25, 0.25, 0.245, 0.252, 0.25, 0.238, 0.224, 0.209, 0.194, 0.165, 0.106, 0.03]], [0.1, '#e0dedc', [0.249, 0.258, 0.269, 0.27, 0.276, 0.269, 0.271, 0.264, 0.24, 0.225, 0.216, 0.199, 0.179, 0.132]], [0.104, '#b7c1b9', [0.249, 0.258, 0.268, 0.253, 0.238, 0.236, 0.218, 0.212, 0.208, 0.183, 0.169, 0.142, 0.132, 0.024]], [0.105, '#c6458a', [0.251, 0.27, 0.271, 0.265, 0.261, 0.251, 0.26, 0.265, 0.259, 0.245, 0.225, 0.2, 0.071, 0.032]], [0.113, '#c0c9c5', [0.249, 0.252, 0.253, 0.241, 0.228, 0.214, 0.207, 0.199, 0.158, 0.134, 0.115, 0.112, 0.111, 0.101]], [0.119, '#c5bba5', [0.252, 0.258, 0.267, 0.271, 0.278, 0.28, 0.279, 0.274, 0.24, 0.223, 0.217, 0.198, 0.094, 0.033]], [0.126, '#d2c3a2', [0.251, 0.261, 0.267, 0.27, 0.268, 0.272, 0.27, 0.251, 0.223, 0.211, 0.201, 0.176, 0.088, 0.026]], [0.128, '#edddba', [0.25, 0.258, 0.266, 0.276, 0.271, 0.28, 0.274, 0.257, 0.226, 0.213, 0.191, 0.132, 0.099, 0.06]]], 'wiggle': [0.0018, 1770, 3110], 'linewidth': 0.85}, 'label_map': {}, 'crop_pixel_bbox': [357, 587, 677, 770], 'crop_sha256': '24ad992842d7e853b339058f95ad922ec032cce13d5bfcc325bd5e5531b2723a', 'api_request_sha256': 'b3452b7a03da716eb19109d7ec5ae817b05373552ebd4b9833b15057109b3cc4'}, {'bbox': [0.662, 0.724, 0.96, 0.95], 'data': {'x': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10], 'traces': [[0.25, 0.8, 1.35, 1.86, 2.48, 2.9, 3.47, 3.88, 4.34, 4.86, 5.34, 5.84, 6.31, 6.81, 7.22, 7.69, 8.3, 8.81, 9.48, 9.99, 10.58], [0.3, 0.82, 1.32, 1.83, 2.27, 2.92, 3.34, 3.92, 4.63, 5.08, 5.66, 6.15, 6.73, 6.99, 7.51, 8.03, 8.62, 9.05, 9.45, 9.91, 10.43], [0.23, 0.53, 0.95, 1.3, 1.62, 2.21, 2.72, 3.28, 3.83, 4.15, 4.72, 5.09, 5.46, 6.02, 6.38, 6.79, 7.26, 7.66, 8.15, 8.56, 8.96], [0.29, 0.69, 1.26, 1.72, 2.14, 2.49, 2.96, 3.46, 3.85, 4.38, 4.76, 5.18, 5.83, 6.2, 6.67, 7.02, 7.39, 7.89, 8.35, 8.97, 9.49], [0.27, 0.72, 1.23, 1.59, 1.97, 2.49, 3.02, 3.55, 3.91, 4.34, 4.91, 5.4, 5.76, 6.3, 6.8, 7.28, 7.67, 8.07, 8.62, 9.08, 9.78], [0.25, 0.75, 1.15, 1.59, 2.03, 2.45, 2.91, 3.46, 3.79, 4.35, 4.83, 5.35, 5.91, 6.35, 6.91, 7.41, 7.92, 8.48, 8.91, 9.48, 10.05], [0.31, 0.77, 1.41, 1.88, 2.51, 3.03, 3.5, 4.02, 4.52, 5.01, 5.48, 5.94, 6.47, 6.83, 7.37, 7.72, 8.09, 8.48, 9.07, 9.65, 10.23], [0.25, 0.66, 1.18, 1.55, 2.05, 2.65, 3.27, 3.76, 4.12, 4.65, 5.06, 5.7, 6.14, 6.57, 7.03, 7.52, 8.04, 8.37, 8.78, 9.28, 9.72], [0.22, 0.71, 1.1, 1.44, 1.85, 2.23, 2.68, 3.08, 3.5, 4.03, 4.42, 4.83, 5.24, 5.68, 6.18, 6.66, 7.04, 7.45, 7.94, 8.36, 8.83]], 'colors': ['#244d83', '#48ec47', '#b4ad76', '#d74925', '#89448b', '#e06c32', '#429762', '#705441', '#ab517b'], 'xlim': [0, 10], 'ylim': [0, 11.6], 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0, 5, 10], 'canvas': [1000, 600], 'axes': [0.095, 0.177, 0.862, 0.812], 'line_width': 1.6, 'grid_color': '#c9c9c9', 'spine_color': '#494949'}, 'label_map': {}, 'crop_pixel_bbox': [678, 587, 983, 770], 'crop_sha256': '569a5eeced56dee241c299761701a9016a42ccf963b62ec0f22ca42444aef266', 'api_request_sha256': '606466f0cd7d37389d9281c22bd4d200670654541df51d7ba46a2294a3b29489'}], 'global_placements': [{'key': 'column_1', 'x': 0.216, 'y': 0.024, 'size': 24, 'max_width': 0.29}, {'key': 'column_2', 'x': 0.517, 'y': 0.024, 'size': 24, 'max_width': 0.29}, {'key': 'column_3', 'x': 0.819, 'y': 0.024, 'size': 24, 'max_width': 0.29}, {'key': 'row_1', 'x': 0.977, 'y': 0.153, 'size': 23, 'max_width': 0.18}, {'key': 'row_2', 'x': 0.977, 'y': 0.375, 'size': 23, 'max_width': 0.18}, {'key': 'row_3', 'x': 0.977, 'y': 0.597, 'size': 23, 'max_width': 0.18}, {'key': 'row_4', 'x': 0.977, 'y': 0.818, 'size': 23, 'max_width': 0.18}, {'key': 'y_axis', 'x': 0.021, 'y': 0.375, 'size': 24, 'max_width': 0.15}, {'key': 'x_axis', 'x': 0.517, 'y': 0.977, 'size': 24, 'max_width': 0.2}]}
LABELS = {'title': '', 'column_1': 'c_{tub} = 7 μM', 'column_2': 'c_{tub} = 10 μM', 'column_3': 'c_{tub} = 13 μM', 'row_1': 'k_{hydr}^{0} = 1.5 s⁻¹', 'row_2': 'k_{hydr}^{0} = 2.5 s⁻¹', 'row_3': 'k_{hydr}^{0} = 3.5 s⁻¹', 'row_4': 'k_{hydr}^{0} = 4.5 s⁻¹', 'y_axis': 'ℓ_{MT}/μM', 'x_axis': 't_{sim}/min'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
