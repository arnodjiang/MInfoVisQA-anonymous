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
    placements = []
    for (i, p) in enumerate(data['panels']):
        col = i % 2
        row = i // 2
        left = 0.052 + col * 0.503
        top = 0.029 + row * 0.335
        width = 0.329
        height = 0.265
        ax = fig.add_axes([left, 1 - top - height, width, height])
        ax.set_facecolor('#EBEBEB')
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticks(data['minor_y'], minor=True)
        ax.grid(True, color='white', linewidth=1.3)
        ax.grid(True, which='minor', color='white', linewidth=0.7)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(axis='both', labelsize=10, length=3, color='#555555', labelcolor='#444444')
        ax.tick_params(which='minor', length=0)
        ax.axhline(0, color='#555555', linewidth=0.85)
        for (j, (v, c)) in enumerate(zip(p['values'], p['colors'])):
            ax.plot(data['years'], v, color=c, linewidth=1.25)
            ax.text(2015, v[-1] + p['end_offsets'][j], str(j + 1), color=c, fontsize=13, ha='left', va='center')
        placements.extend([{'key': 'title' + str(p['g']), 'x': left, 'y': top - 0.016, 'size': 21, 'max_width': width, 'anchor': 'left'}, {'key': 'x', 'x': left + width / 2, 'y': top + height + 0.026, 'size': 17, 'max_width': width, 'anchor': 'center'}, {'key': 'y', 'x': left - 0.035, 'y': top + height / 2, 'size': 17, 'max_width': height, 'rotation': 90, 'anchor': 'center'}])
        lx = left + width + 0.025
        ly = top + height / 2 - (len(p['values']) * 0.021 + 0.026) / 2
        placements.append({'key': 'group', 'x': lx, 'y': ly, 'size': 11, 'max_width': 0.104, 'anchor': 'left'})
        for (j, c) in enumerate(p['colors']):
            yy = ly + 0.021 + j * 0.021
            fig.add_artist(Rectangle((lx, 1 - yy - 0.009), 0.025, 0.018, transform=fig.transFigure, facecolor='#F2F2F2', edgecolor='none'))
            fig.add_artist(Line2D([lx + 0.003, lx + 0.023], [1 - yy, 1 - yy], transform=fig.transFigure, color=c, linewidth=1.25))
            placements.append({'key': 'g' + str(p['g']) + 's' + str(j + 1), 'x': lx + 0.031, 'y': yy, 'size': 10.5, 'max_width': 0.081, 'anchor': 'left'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_5db1349b8f3495df8239fcc1398fa6e47c31a602a3d57a05c5a04653fa9123a9'
LANGUAGE = 'de'
DATA = {'canvas': [1100, 1347], 'years': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015], 'xlim': [2003, 2016], 'ylim': [-1.04, 1.77], 'xticks': [2004, 2006, 2008, 2010, 2012, 2014], 'yticks': [-1, 0, 1], 'minor_y': [-0.5, 0.5, 1.5], 'panels': [{'g': 4, 'colors': ['#F8766D', '#7CAE00', '#00BFC4', '#C77CFF'], 'values': [[0, 0.01, 0.03, 0.02, -0.02, -0.29, -0.81, -0.63, -0.4, -0.34, -0.2, -0.12], [0, 0.01, 0.02, 0.05, 0.03, 0.12, 0.14, 0.03, -0.07, -0.1, -0.14, -0.14], [0, 0.02, 0.06, 0.1, 0.18, 0.47, 0.74, 0.86, 1.01, 1.31, 1.43, 1.54], [0, 0.01, 0.03, 0.06, 0.11, 0.29, 0.47, 0.48, 0.48, 0.59, 0.57, 0.59]], 'end_offsets': [0.03, -0.06, 0, 0]}, {'g': 5, 'colors': ['#F8766D', '#A3A500', '#00BF7D', '#00B0F6', '#E76BF3'], 'values': [[0, 0.01, 0.04, 0.06, 0.13, 0.34, 0.62, 0.7, 0.81, 1.08, 1.23, 1.29], [0, 0, 0.02, 0.05, 0.04, 0.13, 0.12, 0.01, -0.11, -0.14, -0.18, -0.17], [0, 0.02, 0.07, 0.13, 0.25, 0.56, 0.85, 1.06, 1.25, 1.36, 0.75, -0.36], [0, 0.01, 0.02, 0.04, 0.08, 0.28, 0.44, 0.45, 0.43, 0.52, 0.5, 0.53], [0, 0, 0.01, 0, -0.03, -0.28, -0.83, -0.64, -0.42, -0.36, -0.22, -0.13]], 'end_offsets': [0, -0.07, 0.01, 0, 0.03]}, {'g': 6, 'colors': ['#F8766D', '#B79F00', '#00BA38', '#00BFC4', '#619CFF', '#F564E3'], 'values': [[0, 0.01, 0.03, 0.06, 0.14, 0.29, 0.43, 0.45, 0.43, 0.53, 0.51, 0.54], [0, 0.03, 0.07, 0.12, 0.23, 0.55, 0.84, 1.04, 1.26, 1.47, 0.91, -0.03], [0, 0, 0, 0.01, -0.01, -0.3, -0.88, -0.65, -0.42, -0.37, -0.22, -0.15], [0, 0, 0.01, 0.03, 0.04, 0.07, -0.03, -0.14, -0.19, -0.16, -0.16, -0.15], [0, 0.01, 0.02, 0.06, 0.1, 0.32, 0.54, 0.51, 0.19, -0.1, -0.28, -0.3], [0, 0.01, 0.02, 0.06, 0.16, 0.4, 0.64, 0.7, 0.79, 1.06, 1.22, 1.31]], 'end_offsets': [0, 0.04, 0.04, -0.08, -0.06, 0]}, {'g': 7, 'colors': ['#F8766D', '#C49A00', '#53B400', '#00C094', '#00B6EB', '#A58AFF', '#FB61D7'], 'values': [[0, 0.01, 0.04, 0.06, 0.15, 0.33, 0.56, 0.57, 0.59, 0.72, 0.74, 0.82], [0, 0.03, 0.06, 0.12, 0.22, 0.49, 0.67, 0.76, 0.88, 1.19, 1.46, 1.55], [0, 0.03, 0.08, 0.16, 0.3, 0.61, 0.94, 1.16, 1.38, 1.61, 1.17, 0.11], [0, 0.01, 0.01, 0.02, 0.04, 0.06, -0.04, -0.17, -0.21, -0.18, -0.18, -0.16], [0, 0, 0.01, 0.01, -0.03, -0.34, -0.88, -0.66, -0.41, -0.35, -0.21, -0.13], [0, 0, 0.01, 0.05, 0.12, 0.35, 0.56, 0.51, 0.17, -0.1, -0.25, -0.27], [0, 0.01, 0.02, 0.06, 0.06, 0.25, 0.38, 0.38, 0.36, 0.43, 0.4, 0.41]], 'end_offsets': [0, 0, 0.02, -0.04, 0.07, -0.04, 0]}, {'g': 8, 'colors': ['#F8766D', '#CD9600', '#7CAE00', '#00BE67', '#00BFC4', '#00A9FF', '#C77CFF', '#FF61CC'], 'values': [[0, 0, 0.02, 0.02, 0.05, 0.04, -0.71, -0.66, -0.53, -0.47, -0.38, -0.35], [0, 0.02, 0.05, 0.1, 0.21, 0.51, 0.66, 0.76, 0.88, 1.19, 1.46, 1.55], [0, 0.01, 0.03, 0.07, 0.15, 0.33, 0.53, 0.55, 0.59, 0.73, 0.77, 0.85], [0, 0, 0.01, 0.03, 0.09, 0.27, 0.37, 0.38, 0.36, 0.44, 0.42, 0.43], [0, 0, 0.01, 0.02, 0.04, 0.09, 0.02, -0.11, -0.17, -0.14, -0.14, -0.14], [0, 0.04, 0.08, 0.15, 0.26, 0.58, 0.86, 1.08, 1.34, 1.59, 1.12, 0.04], [0, 0, 0.01, 0.04, 0.1, 0.34, 0.55, 0.51, 0.2, -0.05, -0.26, -0.28], [0, 0, 0, 0, -0.06, -0.49, -0.9, -0.67, -0.38, -0.31, -0.17, -0.12]], 'end_offsets': [-0.04, 0, 0, 0, -0.04, 0.06, -0.02, 0.08]}, {'g': 9, 'colors': ['#F8766D', '#D39200', '#93AA00', '#00BA38', '#00C19F', '#00B9E3', '#619CFF', '#DB72FB', '#FF61C3'], 'values': [[0, 0, 0, 0.01, -0.03, -0.52, -0.91, -0.65, -0.54, -0.46, -0.25, -0.04], [0, 0.02, 0.07, 0.12, 0.25, 0.56, 0.82, 0.95, 1.23, 1.65, 1.48, 0.79], [0, 0, 0.01, 0.03, 0.05, 0.09, -0.73, -0.67, -0.53, -0.47, -0.37, -0.35], [0, 0.02, 0.04, 0.06, 0.12, 0.27, 0.36, 0.36, 0.34, 0.42, 0.39, 0.39], [0, 0.05, 0.1, 0.14, 0.29, 0.58, 0.86, 1.07, 1.22, 1.21, 0.72, -0.71], [0, 0, 0, 0.01, 0.01, -0.02, -0.1, -0.14, -0.17, -0.14, -0.14, -0.14], [0, 0, 0.01, 0.05, 0.11, 0.3, 0.53, 0.54, 0.56, 0.69, 0.71, 0.79], [0, 0, 0.01, 0.03, 0.05, 0.26, 0.47, 0.28, 0, -0.18, -0.29, -0.27], [0, 0.01, 0.04, 0.08, 0.2, 0.43, 0.61, 0.69, 0.81, 1.08, 1.3, 1.55]], 'end_offsets': [0, 0.07, -0.03, 0, 0.03, -0.01, -0.05, -0.01, 0]}]}
LABELS = {'title4': 'G = 4 Zeitprofile', 'title5': 'G = 5 Zeitprofile', 'title6': 'G = 6 Zeitprofile', 'title7': 'G = 7 Zeitprofile', 'title8': 'G = 8 Zeitprofile', 'title9': 'G = 9 Zeitprofile', 'x': 'Geschäftsjahr', 'y': 'Zeitprofilwert', 'group': 'Gruppe', 'g4s1': '1 (n=1447)', 'g4s2': '2 (n=2368)', 'g4s3': '3 (n=1358)', 'g4s4': '4 (n=4343)', 'g5s1': '1 (n=1227)', 'g5s2': '2 (n=2320)', 'g5s3': '3 (n=369)', 'g5s4': '4 (n=4159)', 'g5s5': '5 (n=1441)', 'g6s1': '1 (n=4154)', 'g6s2': '2 (n=358)', 'g6s3': '3 (n=1309)', 'g6s4': '4 (n=1987)', 'g6s5': '5 (n=527)', 'g6s6': '6 (n=1181)', 'g7s1': '1 (n=1057)', 'g7s2': '2 (n=865)', 'g7s3': '3 (n=331)', 'g7s4': '4 (n=1854)', 'g7s5': '5 (n=1298)', 'g7s6': '6 (n=508)', 'g7s7': '7 (n=2603)', 'g8s1': '1 (n=428)', 'g8s2': '2 (n=828)', 'g8s3': '3 (n=1909)', 'g8s4': '4 (n=2721)', 'g8s5': '5 (n=1807)', 'g8s6': '6 (n=332)', 'g8s7': '7 (n=487)', 'g8s8': '8 (n=1004)', 'g9s1': '1 (n=974)', 'g9s2': '2 (n=301)', 'g9s3': '3 (n=416)', 'g9s4': '4 (n=2543)', 'g9s5': '5 (n=226)', 'g9s6': '6 (n=1595)', 'g9s7': '7 (n=2047)', 'g9s8': '8 (n=661)', 'g9s9': '9 (n=753)'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
