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
        fig = plt.figure(figsize=(1100 / 100, 830 / 100), dpi=100)
        placements = []

        def text(k, x, y, s=18, w=0.2):
            placements.append(dict(key=k, x=x, y=y, size=s, max_width=w, anchor='center'))
        m = fig.add_axes([0, 0.035, 0.76, 0.955])
        m.set_xlim(0, 100)
        m.set_ylim(100, 0)
        m.axis('off')
        for (p, c) in zip(data['polygons'], data['polycolors']):
            m.add_patch(Polygon(np.array(p).reshape(-1, 2), facecolor=data['colors'][c], edgecolor='#bac4c1', linewidth=0.6))
        for (k, x, y) in data['maplabels']:
            text(k, 0.76 * x / 100, 0.01 + 0.955 * y / 100, 18, 0.13)
        text('note', 0.085, 0.906, 16, 0.15)
        text('title', 0.745, 0.025, 19, 0.46)
        a = fig.add_axes([0.52, 0.465, 0.45, 0.457])
        a.set_xlim(-53.5, 20.5)
        a.set_ylim(9.7, -0.8)
        for (i, v) in enumerate(data['bars']):
            a.barh(i, v, color=data['colors'][i], height=0.67)
            text(data['keys'][i], 0.9 if v < 0 else 0.8, 0.078 + 0.457 * (i + 0.8) / 10.5, 17, 0.15)
        a.axvline(0, color='#aaa', lw=1)
        a.set_yticks([])
        a.xaxis.tick_top()
        a.set_xticks([-50, -40, -30, -20, -10, 0, 10, 20])
        a.set_xticklabels(['−50%', '−40', '−30', '−20', '−10', '0', '+10', '+20%'], fontsize=13)
        a.tick_params(length=9, color='#888')
        for s in a.spines.values():
            s.set_visible(False)
        b = fig.add_axes([0.745, 0.055, 0.245, 0.377])
        b.set_xlim(1970, 2018)
        b.set_ylim(-710, 30)
        for (i, line) in enumerate(data['lines']):
            b.plot(data['years'], line, color=data['colors'][i], lw=3, solid_capstyle='round')
        b.set_xticks([1970, 1980, 1990, 2000, 2010])
        b.set_xticklabels(['1970', '’80', '’90', '’00', '’10'], fontsize=13)
        b.set_yticks([0])
        b.tick_params(length=9, color='#888', labelsize=13)
        for s in b.spines.values():
            s.set_visible(False)
        for (k, y) in [('l200', 0.69), ('l400', 0.793), ('l600', 0.895)]:
            text(k, 0.675, y, 18, 0.14)
        text('net', 0.69, 0.87, 18, 0.13)
        return finish(fig, labels, placements)
    functions = [panel_00]
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
BASE_ID = 'qa_43623d77110cae117e957eb0a749a2206173fb4da9ed62b0747e40153167a07f'
LANGUAGE = 'id'
DATA = {'canvas': [2800, 2057], 'panels': [{'bbox': [0.025, 0.01, 0.98, 0.99], 'data': {'colors': ['#dcd1bc', '#92aea7', '#a5ad83', '#d4dce1', '#e2e2e2', '#e5afa5', '#b3ceb9', '#ffd08b', '#a6bfd0', '#588eb7'], 'keys': ['grass', 'boreal', 'west', 'tundra', 'general', 'forest', 'east', 'arid', 'coast', 'wet'], 'bars': [-53.3, -33, -29.5, -23.4, -23.1, -18, -17.4, -17, -15.1, 13], 'years': [1970, 1973, 1976, 1979, 1982, 1985, 1988, 1991, 1994, 1997, 2000, 2003, 2006, 2009, 2013, 2017], 'lines': [[0, -80, -160, -235, -295, -350, -402, -452, -498, -540, -576, -607, -636, -662, -696, -728], [0, -6, -43, -115, -198, -272, -333, -380, -413, -440, -464, -483, -495, -503, -511, -510], [0, -10, -28, -45, -60, -74, -85, -95, -104, -112, -119, -126, -132, -136, -140, -140], [0, -5, -11, -15, -19, -23, -28, -34, -43, -49, -57, -66, -71, -76, -80, -81], [0, -45, -105, -165, -219, -265, -300, -328, -352, -374, -392, -407, -415, -423, -426, -420], [0, -22, -69, -128, -178, -220, -254, -286, -323, -358, -391, -418, -441, -460, -480, -492], [0, -20, -64, -93, -111, -121, -130, -139, -147, -152, -154, -157, -160, -164, -168, -168], [0, -3, -7, -13, -20, -25, -29, -34, -37, -39, -40, -41, -41, -40, -38, -36], [0, -1, -2, -1, 0, 0, -1, -3, -5, -8, -9, -9, -8, -8, -7, -7], [0, -4, -9, -16, -23, -27, -27, -21, -11, -8, -12, -13, -11, 0, 13, 22]], 'polygons': [[3, 18, 7, 12, 13, 9, 21, 12, 26, 20, 33, 24, 39, 18, 44, 20, 46, 7, 54, 1, 56, 3, 56, 18, 66, 23, 70, 28, 76, 26, 79, 31, 87, 36, 91, 39, 93, 44, 89, 48, 93, 51, 88, 55, 84, 57, 83, 62, 78, 66, 77, 75, 72, 83, 74, 88, 76, 91, 75, 94, 72, 92, 69, 87, 60, 88, 52, 94, 50, 96, 47, 95, 44, 90, 40, 91, 36, 86, 28, 86, 24, 83, 22, 83, 20, 77, 18, 71, 18, 65, 20, 59, 20, 51, 18, 45, 17, 38, 17, 32, 15, 28, 11, 27, 9, 25, 3, 27], [5, 19, 12, 13, 17, 17, 25, 19, 32, 27, 38, 26, 42, 33, 50, 35, 53, 43, 58, 46, 66, 48, 69, 44, 68, 39, 74, 35, 79, 36, 82, 41, 88, 38, 91, 41, 90, 47, 86, 49, 83, 56, 78, 61, 68, 65, 60, 64, 54, 62, 51, 56, 46, 55, 41, 52, 35, 49, 33, 58, 27, 53, 22, 43, 19, 38, 18, 31, 14, 27, 10, 26, 8, 23], [10, 25, 15, 26, 18, 31, 19, 38, 24, 41, 28, 46, 32, 54, 35, 64, 39, 68, 42, 87, 35, 87, 29, 83, 24, 83, 20, 77, 18, 70, 20, 59, 20, 51, 18, 44, 17, 36, 16, 30], [34, 49, 40, 51, 46, 55, 51, 57, 54, 65, 63, 69, 58, 76, 54, 83, 60, 88, 53, 94, 51, 96, 48, 95, 46, 91, 42, 90, 40, 85, 40, 78, 37, 69, 33, 65, 33, 58], [53, 60, 58, 65, 65, 65, 71, 63, 77, 59, 81, 55, 83, 57, 81, 63, 77, 68, 79, 73, 75, 78, 72, 84, 76, 92, 75, 94, 72, 92, 69, 87, 60, 89, 54, 88, 53, 81, 57, 74, 59, 69], [20, 66, 23, 70, 25, 66, 27, 60, 30, 58, 34, 65, 37, 64, 40, 69, 40, 77, 42, 88, 40, 90, 37, 87, 35, 85, 32, 86, 28, 84, 24, 83, 21, 79, 19, 73]], 'polycolors': [3, 1, 2, 0, 6, 7], 'maplabels': [['tundra', 49, 31], ['boreal', 62, 55], ['west', 23, 48], ['grass', 48, 73], ['east', 68, 79], ['arid', 27, 73]]}, 'label_map': {'title': 'panel_00_title', 'grass': 'panel_00_grass', 'boreal': 'panel_00_boreal', 'west': 'panel_00_west', 'tundra': 'panel_00_tundra', 'general': 'panel_00_general', 'forest': 'panel_00_forest', 'east': 'panel_00_east', 'arid': 'panel_00_arid', 'coast': 'panel_00_coast', 'wet': 'panel_00_wet', 'net': 'panel_00_net', 'l200': 'panel_00_l200', 'l400': 'panel_00_l400', 'l600': 'panel_00_l600', 'note': 'panel_00_note'}, 'crop_pixel_bbox': [29, 8, 1126, 836], 'crop_sha256': '468c48cf15d32264395220983cd02f8d8df9b419073ba391429a2e428407efe1', 'api_request_sha256': '41bc7f72a6a2572a177375b6623205edb984d64cf9b3d2a9541b9787e76ec093'}], 'global_placements': []}
LABELS = {'panel_00_title': 'Perubahan populasi burung sejak 1970 menurut habitat berkembang biak', 'panel_00_grass': 'Padang rumput', 'panel_00_boreal': 'Hutan boreal', 'panel_00_west': 'Hutan barat', 'panel_00_tundra': 'Tundra', 'panel_00_general': 'Generalis', 'panel_00_forest': 'Generalis hutan', 'panel_00_east': 'Hutan timur', 'panel_00_arid': 'Lahan kering', 'panel_00_coast': 'Pesisir', 'panel_00_wet': 'Lahan basah', 'panel_00_net': 'Penurunan bersih sebesar', 'panel_00_l200': '200 juta', 'panel_00_l400': '400 juta', 'panel_00_l600': '600 juta burung', 'panel_00_note': 'Catatan: Habitat lahan basah dan pesisir tidak ditampilkan. Batas habitat bersifat perkiraan.'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
