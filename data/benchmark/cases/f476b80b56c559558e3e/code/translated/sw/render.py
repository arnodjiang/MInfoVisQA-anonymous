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
SCRIPT_FONTS = {script: path for (script, path) in {'TELUGU': os.environ.get('MVISQA_TELUGU_FONT', '/System/Library/Fonts/KohinoorTelugu.ttc'), 'BENGALI': os.environ.get('MVISQA_BENGALI_FONT', '/System/Library/Fonts/KohinoorBangla.ttc')}.items() if os.path.isfile(path)}

@lru_cache(maxsize=32)
def font_face(path):
    return hb.Face(open(path, 'rb').read())

@lru_cache(maxsize=16384)
def supports(path, text):
    face = freetype.Face(path)
    return all((face.get_char_index(ord(c)) or unicodedata.category(c) == 'Cf' for c in text))

def font_runs(text, direction):
    preferred = any((unicodedata.name(c, '').startswith(s) for c in text for s in SCRIPT_FONTS))
    if supports(FONT, text) and (not preferred):
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
    for (script, part) in groups:
        candidates = ([SCRIPT_FONTS[script]] if script in SCRIPT_FONTS else []) + [FONT] + FALLBACK_FONTS
        path = next((p for p in candidates if supports(p, part)), FONT)
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
    (X, Y) = data['coordinate_extent']
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes(data['axes_bounds'])
    ax.set_xlim(0, X)
    ax.set_ylim(Y, 0)
    ax.set_axis_off()
    segments = list(data['segments'])
    for row in data['row_geometry']:
        for (x1, i, x2, j) in data['row_segment_templates']:
            segments.append([x1, row[i], x2, row[j]])
    for (x1, y1, x2, y2) in segments:
        ax.plot([x1, x2], [y1, y2], color=data['line_color'], linewidth=data['line_width'] * W / X, solid_capstyle='butt')
    for c in data['circles']:
        ax.add_patch(Circle(c['center'], c['radius'], facecolor=c['face'], edgecolor=c['edge'], linewidth=c['width'] * W / X))
    placements = []
    for (key, x, y, size, width) in data['texts']:
        fitted_size = size * W / X
        if key != 'title' and (not key.startswith('diagram_')):
            borders = [y1 for (x1, y1, x2, y2) in segments if y1 == y2 and min(x1, x2) <= x <= max(x1, x2)]
            above = [border for border in borders if border <= y]
            below = [border for border in borders if border >= y]
            if above and below:
                top = max(above)
                bottom = min(below)
                max_height = max(0, 2 * (min(y - top, bottom - y) * H / Y - 3))
                max_width = width * W / X
                fitted_size = max(10, min(48, int(fitted_size)))
                while True:
                    glyphs = mask(labels[key], fitted_size)
                    if glyphs.width <= max_width and glyphs.height <= max_height or fitted_size == 10:
                        break
                    fitted_size -= 1
        placements.append({'key': key, 'x': x / X, 'y': y / Y, 'size': fitted_size, 'max_width': width / X, 'rotation': data['text_rotation'], 'anchor': data['text_anchor']})
    p = data['page_number']
    ax.text(p['position'][0], p['position'][1], str(p['value']), ha='center', va='center', fontsize=p['size'] * W / X * 72 / 100, color=p['color'], fontfamily='serif')
    return finish(fig, labels, placements)
BASE_ID = 'qa_252aada3002dcb5f0ad197d9fa80c0a05fb6d381d17cc6f5b6d55d7b2534ffec'
LANGUAGE = 'sw'
DATA = {'canvas': [1680, 2780], 'coordinate_extent': [1700, 2812], 'axes_bounds': [0, 0, 1, 1], 'components': [{'id': 'table', 'type': 'mixed_span_table', 'bounds': [12, 285, 1554, 1414], 'columns': [12, 313, 508, 727, 922, 1145, 1299, 1554], 'header_spans': [{'label': 'h_knot', 'bounds': [12, 285, 313, 384]}, {'label': 'h_tempo', 'bounds': [313, 285, 922, 335]}, {'label': 'h_ingredients', 'bounds': [922, 285, 1554, 335]}], 'body_structure': {'left_columns_span_each_three_subrows': [0, 1, 2], 'empty_body_column': 2, 'rightmost_two_columns_open_on_first_subrow': True, 'blank_first_subrow_columns': [3, 4, 5, 6], 'blank_third_subrow_columns': [4, 5, 6], 'tool_and_technique_only_populated_in_first_group': True, 'material_blank_in_group': 6}}, {'id': 'radial', 'type': 'four_cardinal_nodes_on_circle', 'center': [850, 1970], 'radius': 207, 'bounds_including_labels': [43, 1655, 1657, 2285], 'node_order_clockwise': ['diagram_top', 'diagram_right', 'diagram_bottom', 'diagram_left'], 'nodes': [{'label': 'diagram_top', 'center': [850, 1763], 'color': '#009900', 'label_position': [850, 1675]}, {'label': 'diagram_right', 'center': [1057, 1970], 'color': '#000099', 'label_position': [1382, 1970]}, {'label': 'diagram_bottom', 'center': [850, 2177], 'color': '#ffa500', 'label_position': [850, 2265]}, {'label': 'diagram_left', 'center': [643, 1970], 'color': '#ae0000', 'label_position': [318, 1970]}], 'layout': 'Centered in the lower whitespace, with outward labels separated from the ring and markers.', 'minimum_side_label_gap': 30, 'top_bottom_label_center_offset': 88}, {'id': 'page_number', 'type': 'decorative_page_number', 'value': 1, 'position': [925, 2790]}], 'line_color': '#000000', 'line_width': 0.8, 'segments': [[12, 285, 1554, 285], [12, 285, 12, 1414], [313, 285, 313, 1414], [508, 335, 508, 1414], [727, 335, 727, 1414], [922, 285, 922, 1414], [1145, 335, 1145, 1414], [1299, 335, 1299, 384], [1554, 285, 1554, 384], [313, 335, 1554, 335], [12, 384, 1554, 384]], 'row_geometry': [[384, 434, 482, 531], [531, 581, 629, 678], [678, 728, 777, 825], [825, 875, 924, 973], [973, 1022, 1071, 1120], [1120, 1170, 1218, 1267], [1267, 1317, 1365, 1414]], 'row_segment_templates': [[727, 1, 1554, 1], [727, 2, 1554, 2], [12, 3, 1554, 3], [1299, 1, 1299, 3], [1554, 1, 1554, 3]], 'circles': [{'center': [850, 1970], 'radius': 207, 'face': 'none', 'edge': '#000000', 'width': 3.4}, {'center': [850, 1763], 'radius': 20, 'face': '#009900', 'edge': 'none', 'width': 0}, {'center': [643, 1970], 'radius': 20, 'face': '#ae0000', 'edge': 'none', 'width': 0}, {'center': [1057, 1970], 'radius': 20, 'face': '#000099', 'edge': 'none', 'width': 0}, {'center': [850, 2177], 'radius': 20, 'face': '#ffa500', 'edge': 'none', 'width': 0}], 'texts': [['title', 850, 62, 65, 1600], ['h_knot', 163, 333, 41, 280], ['h_tempo', 618, 308, 41, 580], ['h_ingredients', 1238, 308, 41, 595], ['h_slow', 411, 357, 41, 180], ['h_medium', 618, 357, 41, 205], ['h_fast', 825, 357, 41, 180], ['h_material', 1034, 357, 41, 210], ['h_tool', 1222, 357, 41, 140], ['h_technique', 1427, 357, 41, 240], ['r1_name', 163, 431, 40, 285], ['r1_primary', 411, 431, 40, 180], ['r1_second', 825, 455, 40, 183], ['r1_third', 825, 504, 40, 183], ['r1_material', 1034, 455, 40, 205], ['r1_tool', 1222, 455, 40, 140], ['r1_technique', 1427, 455, 40, 240], ['r2_name', 163, 578, 40, 285], ['r2_primary', 411, 578, 40, 180], ['r2_second', 825, 602, 40, 183], ['r2_third', 825, 651, 40, 183], ['r2_material', 1034, 602, 40, 205], ['r3_name', 163, 725, 40, 285], ['r3_primary', 411, 725, 40, 180], ['r3_second', 825, 749, 40, 183], ['r3_third', 825, 798, 40, 183], ['r3_material', 1034, 749, 40, 205], ['r4_name', 163, 872, 40, 285], ['r4_primary', 411, 872, 40, 180], ['r4_second', 825, 896, 40, 183], ['r4_third', 825, 945, 40, 183], ['r4_material', 1034, 896, 40, 205], ['r5_name', 163, 1019, 40, 285], ['r5_primary', 411, 1019, 40, 180], ['r5_second', 825, 1043, 40, 183], ['r5_third', 825, 1092, 40, 183], ['r5_material', 1034, 1043, 40, 205], ['r6_name', 163, 1166, 40, 285], ['r6_primary', 411, 1166, 40, 180], ['r6_second', 825, 1190, 40, 183], ['r6_third', 825, 1239, 40, 183], ['r7_name', 163, 1313, 40, 285], ['r7_primary', 411, 1313, 40, 180], ['r7_second', 825, 1337, 40, 183], ['r7_third', 825, 1386, 40, 183], ['r7_material', 1034, 1337, 40, 205], ['diagram_top', 850, 1675, 40, 1100], ['diagram_left', 318, 1970, 40, 550], ['diagram_right', 1382, 1970, 40, 550], ['diagram_bottom', 850, 2265, 40, 1100]], 'text_rotation': 0, 'text_anchor': 'center', 'page_number': {'value': 1, 'position': [925, 2790], 'size': 40, 'color': '#000000'}}
LABELS = {'title': 'Viambato vya Mapishi ya Mbinu za Kufunga Mafundo kwa Kasi', 'h_knot': 'Aina ya Fundo', 'h_tempo': 'Kasi', 'h_ingredients': 'Viambato', 'h_slow': 'Polepole', 'h_medium': 'Wastani', 'h_fast': 'Haraka', 'h_material': 'Nyenzo', 'h_tool': 'Zana', 'h_technique': 'Mbinu', 'r1_name': 'Fundo la Mraba', 'r1_primary': 'Polepole', 'r1_second': 'Wastani', 'r1_third': 'Haraka', 'r1_material': 'Kamba', 'r1_tool': 'Kufunga', 'r1_technique': 'Msingi', 'r2_name': 'Fundo la Rifu', 'r2_primary': 'Polepole', 'r2_second': 'Wastani', 'r2_third': 'Haraka', 'r2_material': 'Kamba', 'r3_name': 'Fundo la Umbo la 8', 'r3_primary': 'Wastani', 'r3_second': 'Polepole', 'r3_third': 'Haraka', 'r3_material': 'Kamba', 'r4_name': 'Fundo la Bolini', 'r4_primary': 'Polepole', 'r4_second': 'Wastani', 'r4_third': 'Haraka', 'r4_material': 'Kamba', 'r5_name': 'Fundo la Shiti Bendi', 'r5_primary': 'Wastani', 'r5_second': 'Polepole', 'r5_third': 'Wastani', 'r5_material': 'Kamba', 'r6_name': 'Fundo la Klovu Hichi', 'r6_primary': 'Haraka', 'r6_second': 'Polepole', 'r6_third': 'Wastani', 'r7_name': 'Fundo la Timba Hichi', 'r7_primary': 'Haraka', 'r7_second': 'Polepole', 'r7_third': 'Wastani', 'r7_material': 'Kamba', 'diagram_top': 'Kasi ya Wastani', 'diagram_left': 'Kasi ya Juu', 'diagram_right': 'Kasi ya Chini', 'diagram_bottom': 'Bila Kasi'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
