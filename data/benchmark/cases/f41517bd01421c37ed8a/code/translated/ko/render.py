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
    (W, H) = (data['width'], data['height'])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes(data['layout']['axes'])
    ax.set_xlim(data['x_limits'])
    ax.set_ylim(data['y_limits'])
    ax.set_xticks(data['x_ticks'])
    ax.set_xticklabels(data['x_tick_labels'], fontsize=18)
    ax.set_yticks(data['y_ticks'])
    ax.set_yticklabels([str(v) + '%' for v in data['y_ticks']], fontsize=19)
    ax.tick_params(axis='both', length=6, color='#777777', pad=7)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#888888')
        ax.spines[side].set_linewidth(1.6)
    ax.grid(axis='y', color=data['layout']['grid_color'], linewidth=1)
    ax.set_axisbelow(True)
    placements = []

    def put(key, x, y, size, width, anchor='left', rotation=0):
        placements.append({'key': key, 'x': x, 'y': y, 'size': size, 'max_width': width, 'anchor': anchor, 'rotation': rotation})

    def coord(x, y):
        p = ax.transData.transform((x, y))
        return (p[0] / W, 1 - p[1] / H)
    put('title', 0.054, 0.038, 43, 0.315)
    put('period', 0.376, 0.038, 42, 0.57)
    put('subtitle', 0.057, 0.08, 38, 0.9)
    for s in data['series']:
        x = np.array(s.get('x', data['years']), dtype=float)
        y = np.array(s['values'], dtype=float)
        xx = np.linspace(x[0], x[-1], int((x[-1] - x[0]) * data['layout']['samples_per_year']) + 1)
        yy = np.interp(xx, x, y)
        pattern = np.array(data['layout']['ripple_pattern'])
        yy = yy + s['wiggle'] * pattern[np.arange(len(xx)) % len(pattern)]
        ax.plot(xx, yy, color=s['color'], linewidth=data['layout']['line_width'], solid_capstyle='round', zorder=3)
    ax.axhline(data['inflation'], color='black', linewidth=2.1, zorder=4)
    ax.add_patch(Rectangle((0, data['inflation']), 8.75, 13.4, facecolor='black', edgecolor='none', zorder=5))
    fig.canvas.draw()
    (x, y) = coord(0.3, data['inflation'] + 7.0)
    put('inflation', x, y, 18, 0.231)
    for s in data['series']:
        (x, y) = coord(data['layout']['label_x'], s['label_y'])
        put(s['key'], x, y, 20, 0.229)
    (x, y) = coord(0.75, 183)
    put('more_expensive', x, y, 25, 0.21)
    (x, y) = coord(0.75, -63)
    put('more_affordable', x, y, 25, 0.24)
    put('source', 0.12, 0.969, 25, 0.29)
    put('credit', 0.748, 0.963, 17, 0.12)
    put('logo', 0.86, 0.958, 30, 0.07)
    return finish(fig, labels, placements)
BASE_ID = 'qa_e6662b4f0eb4b989e702a2824b2d7ac8af821fcda398229f4280f72e8b5c96fd'
LANGUAGE = 'ko'
DATA = {'width': 1000, 'height': 1280, 'x_limits': [0, 30], 'y_limits': [-100, 230], 'years': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21.5], 'x_ticks': [0.5, 10.5, 20.5], 'x_tick_labels': [1997, 2007, 2017], 'y_ticks': [-80, -40, 0, 40, 80, 120, 160, 200], 'inflation': 57.4, 'series': [{'key': 'hospital', 'color': '#d50000', 'values': [0, 5, 10, 15, 22, 31, 39, 48, 60, 73, 85, 97, 109, 119, 132, 141, 153, 163, 176, 181, 196, 207, 216], 'label_y': 218, 'wiggle': 0.65}, {'key': 'textbooks', 'color': '#65090c', 'x': [0, 1, 2, 3, 4, 4.5, 4.8, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 13.5, 13.8, 14, 15, 15.5, 16, 17, 18, 18.3, 18.7, 19, 19.5, 20, 20.2, 20.5, 20.7, 21, 21.2, 21.5], 'values': [0, 5, 10, 17, 24, 32, 28, 35, 40, 41, 46, 50, 56, 53, 56, 62, 77, 91, 105, 117, 116, 121, 124, 134, 145, 151, 162, 171, 176, 177, 184, 195, 198, 192, 194, 192, 196, 195, 209], 'label_y': 200, 'wiggle': 0.7}, {'key': 'tuition', 'color': '#e00000', 'values': [0, 3, 8, 12, 17, 22, 30, 39, 48, 60, 71, 85, 101, 114, 127, 138, 150, 160, 168, 176, 182, 186, 190], 'label_y': 185, 'wiggle': 0.55}, {'key': 'childcare', 'color': '#6d1115', 'values': [0, 5, 9, 14, 19, 26, 34, 39, 46, 54, 61, 70, 77, 83, 88, 92, 97, 101, 107, 113, 118, 121, 124], 'label_y': 125, 'wiggle': 0.3}, {'key': 'medical', 'color': '#d40000', 'values': [0, 3, 7, 12, 18, 23, 27, 33, 40, 47, 57, 62, 67, 75, 80, 87, 92, 96, 100, 109, 114, 115, 120], 'label_y': 114, 'wiggle': 0.45}, {'key': 'wages', 'color': '#771015', 'values': [0, 3, 6, 10, 14, 18, 22, 26, 30, 35, 40, 46, 51, 55, 58, 60, 63, 66, 70, 74, 78, 82, 84], 'label_y': 84, 'wiggle': 0.15}, {'key': 'housing', 'color': '#d10000', 'values': [0, 2, 4, 8, 11, 13, 18, 21, 24, 28, 34, 40, 38, 39, 42, 44, 46, 49, 52, 55, 59, 64, 66], 'label_y': 69, 'wiggle': 0.13}, {'key': 'food', 'color': '#7d090c', 'values': [0, 2, 4, 7, 11, 13, 16, 20, 21, 25, 30, 39, 39, 41, 48, 50, 52, 55, 58, 59, 58, 60, 61], 'label_y': 57, 'wiggle': 0.14}, {'key': 'cars', 'color': '#2585a5', 'values': [0, -1, -2, -2, -2, -3, -4, -4, -5, -5, -6, -6, -8, -4, -4, -2, -1, 1, 1, 2, 2, 1, 0], 'label_y': 5, 'wiggle': 0.3}, {'key': 'furnishings', 'color': '#07072d', 'values': [0, 0, 0, 1, 2, 3, 2, 0, 1, 2, 2, 1, 4, 2, -1, 0, 1, 0, -1, -2, -2, -4, -3], 'label_y': -3, 'wiggle': 0.3}, {'key': 'clothing', 'color': '#00769b', 'values': [0, -4, -4, -6, -4, -8, -10, -9, -10, -11, -9, -10, -11, -10, -6, -4, -4, -3, -5, -5, -4, -6, -4], 'label_y': -12, 'wiggle': 0.55}, {'key': 'cellphone', 'color': '#090329', 'x': [0, 1, 2, 3, 4, 4.4, 5, 6, 6.3, 7, 8, 9, 10, 11, 12, 12.8, 13, 13.5, 14, 15, 16, 17, 17.8, 18.5, 19, 19.8, 20.1, 20.3, 20.4, 20.6, 21, 21.5], 'values': [0, -3, -11, -20, -30, -32, -32, -31, -32, -35, -35, -34, -35, -36, -35, -35, -37, -37, -40, -40, -41, -41, -42, -45, -44, -46, -46, -47, -51, -52, -52, -52], 'label_y': -49, 'wiggle': 0.25}, {'key': 'software', 'color': '#1684a4', 'values': [0, -2, -8, -12, -19, -23, -29, -34, -37, -41, -45, -46, -46, -50, -55, -57, -59, -61, -62, -63, -66, -67, -67], 'label_y': -65, 'wiggle': 0.45}, {'key': 'toys', 'color': '#070327', 'values': [0, -4, -12, -20, -22, -24, -29, -34, -38, -41, -44, -48, -48, -51, -54, -56, -59, -62, -64, -67, -70, -73, -74], 'label_y': -77, 'wiggle': 0.35}, {'key': 'tvs', 'color': '#2488a7', 'values': [0, -3, -12, -23, -28, -36, -46, -53, -58, -64, -73, -77, -82, -87, -89, -91, -92, -93, -94, -95, -96, -97, -97.5], 'label_y': -96, 'wiggle': 0.2}], 'layout': {'axes': [0.114, 0.102, 0.832, 0.795], 'label_x': 22.15, 'grid_color': '#d7d7d7', 'line_width': 2.25, 'samples_per_year': 12, 'ripple_pattern': [0, 0.35, -0.25, 0.7, -0.2, 0.15, -0.45, 0.35, -0.15, 0.2, -0.25, 0]}}
LABELS = {'title': '가격 변화', 'period': '(1997년 1월~2018년 6월)', 'subtitle': '미국의 주요 소비재·서비스 및 임금', 'more_expensive': '더\n비싸짐', 'more_affordable': '더\n저렴해짐', 'inflation': '전체 물가상승률 (57.4%)', 'hospital': '병원\n서비스', 'textbooks': '대학\n교재', 'tuition': '대학\n등록금', 'childcare': '보육', 'medical': '의료\n서비스', 'wages': '임금', 'housing': '주거', 'food': '식품 및\n음료', 'cars': '자동차', 'furnishings': '가정용 가구·비품', 'clothing': '의류', 'cellphone': '휴대전화\n서비스', 'software': '컴퓨터\n소프트웨어', 'toys': '장난감', 'tvs': '텔레비전', 'source': '출처: BLS', 'credit': 'Carpe Diem', 'logo': 'AEI'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
