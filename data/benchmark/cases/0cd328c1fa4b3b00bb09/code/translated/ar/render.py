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
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 470)
        ax.set_ylim(467, 0)
        ax.axis('off')
        placements = []
        ax.add_patch(Rectangle((54, 41), 407, 421, facecolor='#fafbf9', edgecolor='none'))
        ax.add_patch(Polygon(data['sea'], facecolor=data['water_color'], edgecolor='none'))
        land = Polygon(data['outline'], facecolor=data['land_color'], edgecolor='#aeb1ad', linewidth=1.3)
        ax.add_patch(land)
        for i in range(4, -1, -1):
            poly = data['zone_edges'][i] + list(reversed(data['coast']))
            p = Polygon(poly, facecolor=data['risk_colors'][i], edgecolor='#9b9d82', linewidth=0.65)
            ax.add_patch(p)
            p.set_clip_path(land)
        ax.add_patch(Polygon(data['inland_risk_patch'], facecolor='#b83434', edgecolor='#967753', linewidth=0.8))
        ax.plot(*np.array(data['coast']).T, color='#a54c46', linewidth=1.3)
        for p in data['water_cuts']:
            ax.add_patch(Polygon(p, facecolor='#f7f8f5', edgecolor='none'))
        g = data['boundary_generation']
        rng = np.random.default_rng(g['seed'])
        b = g['box']
        rural = np.column_stack((rng.uniform(b[0], b[1], g['rural_count']), rng.uniform(b[2], b[3], g['rural_count'])))
        urban = rng.normal(g['urban_center'], g['urban_spread'], (g['urban_count'], 2))
        pts = np.vstack((rural, urban))
        for (k, p) in enumerate(pts):
            cell = [[b[0], b[2]], [b[1], b[2]], [b[1], b[3]], [b[0], b[3]]]
            dist = np.sum((pts - p) ** 2, axis=1)
            for j in np.argsort(dist)[1:28]:
                q = pts[j]
                n = q - p
                c = (np.dot(q, q) - np.dot(p, p)) / 2
                new = []
                for h in range(len(cell)):
                    a = np.array(cell[h - 1])
                    z = np.array(cell[h])
                    fa = np.dot(a, n) - c
                    fz = np.dot(z, n) - c
                    if (fa <= 0) != (fz <= 0):
                        new.append((a + (z - a) * fa / (fa - fz)).tolist())
                    if fz <= 0:
                        new.append(z.tolist())
                cell = new
                if not cell:
                    break
            if cell:
                patch = Polygon(cell, fill=False, edgecolor=data['boundary_color'], linewidth=0.65, alpha=0.8)
                ax.add_patch(patch)
                patch.set_clip_path(land)
        (x, y, w, h) = data['legend_box']
        ax.add_patch(Rectangle((x, y), w, h, facecolor='#f8fbfa', edgecolor='#cbd2d2', linewidth=5))
        (sx, sy, sw, sh) = data['legend_swatch']
        for (i, c) in enumerate(data['risk_colors']):
            ax.add_patch(Rectangle((sx, sy + i * sh), sw, sh, facecolor=c, edgecolor='none'))
            ax.text(sx + 32, sy + i * sh + sh / 2, str(data['risk_numbers'][i]), va='center', fontsize=15, color='#565d59')
        placements.append({'key': 'panel', 'x': 0.009, 'y': 0.026, 'size': 25, 'max_width': 0.07, 'anchor': 'left'})
        placements.append({'key': 'legend', 'x': 0.143, 'y': 0.112, 'size': 21, 'max_width': 0.34, 'anchor': 'left'})
        for (key, x, y, size) in data['geographic_labels']:
            placements.append({'key': key, 'x': x / 470, 'y': y / 467, 'size': size * 2, 'max_width': 0.16, 'anchor': 'center'})
        return finish(fig, labels, placements)

    def panel_01(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.257, 0.311, 0.717, 0.601])
        placements = []
        for s in data['series']:
            ax.fill_between(data['x'], s['lower'], s['upper'], color=s['color'], alpha=0.075, linewidth=0)
        for s in data['series']:
            ax.plot(data['x'], s['curve'], color=s['color'], lw=2.7, alpha=0.9)
            p = np.array(s['points'])
            ax.scatter(p[:, 0], p[:, 1], s=95, color=s['color'], alpha=0.2, edgecolors='none', zorder=4)
            ax.scatter(p[:, 0], p[:, 1], s=42, color=s['color'], edgecolors='none', zorder=5)
        ax.set_xlim(*data['xlim'])
        ax.set_ylim(*data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['%.2f' % v for v in data['yticks']])
        ax.tick_params(axis='both', labelsize=27, length=9, width=2, pad=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_linewidth(2.6)
        placements.extend([{'key': 'panel', 'x': 0.01, 'y': 0.027, 'size': 45, 'max_width': 0.08, 'anchor': 'left'}, {'key': 'y_axis', 'x': 0.109, 'y': 0.388, 'size': 46, 'max_width': 0.52, 'rotation': 90, 'anchor': 'center'}, {'key': 'x_axis', 'x': 0.615, 'y': 0.786, 'size': 46, 'max_width': 0.52, 'anchor': 'center'}, {'key': 'legend_title', 'x': 0.255, 'y': 0.922, 'size': 43, 'max_width': 0.23, 'anchor': 'left'}])
        for (i, s) in enumerate(data['series']):
            col = i // 2
            row = i % 2
            x = 0.5 + col * 0.128
            y = 0.868 + row * 0.055
            fig.patches.append(Rectangle((x, 1 - y - 0.049), 0.049, 0.049, transform=fig.transFigure, facecolor=s.get('legend_color', s['color']), edgecolor='none'))
            placements.append({'key': s['key'], 'x': x + 0.075, 'y': y + 0.0245, 'size': 39, 'max_width': 0.045, 'anchor': 'left'})
        return finish(fig, labels, placements)
    functions = [panel_00, panel_01]
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
BASE_ID = 'qa_3063447404749ee754e97efeed8ed771a43af0dcc5828a283a806baa7c7dc462'
LANGUAGE = 'ar'
DATA = {'canvas': [2800, 1329], 'panels': [{'bbox': [0.003, 0.016, 0.462, 0.977], 'data': {'canvas': [940, 934], 'map_extent': [0, 470, 467, 0], 'land_color': '#c6c8c5', 'water_color': '#d4dadd', 'boundary_color': '#929591', 'risk_colors': ['#d97f82', '#f2c897', '#faf8d1', '#b4dbe5', '#85b3cc'], 'risk_numbers': [1, 2, 3, 4, 5], 'outline': [[83, 231], [97, 241], [119, 241], [143, 230], [154, 240], [171, 247], [182, 258], [199, 245], [213, 228], [209, 215], [203, 199], [218, 185], [235, 177], [250, 161], [237, 154], [215, 151], [206, 143], [202, 126], [196, 115], [211, 98], [232, 78], [249, 60], [267, 65], [282, 71], [296, 74], [309, 81], [337, 70], [355, 72], [374, 60], [403, 57], [424, 52], [447, 51], [443, 64], [451, 72], [440, 83], [448, 91], [439, 103], [444, 113], [435, 121], [437, 134], [426, 146], [423, 158], [417, 165], [422, 180], [412, 190], [412, 202], [421, 211], [419, 230], [425, 239], [416, 249], [419, 261], [403, 262], [395, 274], [390, 284], [386, 293], [395, 302], [402, 314], [379, 317], [355, 326], [326, 337], [300, 349], [278, 357], [262, 369], [244, 380], [226, 394], [211, 407], [196, 424], [178, 441], [155, 455], [134, 461], [111, 458], [100, 452], [98, 443], [108, 433], [99, 419], [99, 402], [95, 390], [111, 375], [128, 381], [143, 366], [146, 350], [162, 334], [155, 324], [137, 320], [119, 309], [104, 296], [99, 276], [92, 254]], 'coast': [[99, 459], [119, 462], [138, 462], [157, 451], [178, 439], [196, 421], [216, 405], [237, 389], [257, 376], [268, 365], [282, 356], [305, 347], [328, 336], [354, 326], [378, 317], [402, 315], [395, 302], [386, 294], [392, 281], [400, 271], [404, 263], [417, 262], [422, 248], [418, 236]], 'sea': [[156, 467], [179, 446], [199, 429], [221, 410], [244, 395], [263, 379], [284, 366], [307, 354], [332, 343], [356, 332], [383, 321], [406, 309], [439, 301], [461, 299], [461, 467]], 'zone_edges': [[[99, 458], [116, 457], [137, 461], [154, 450], [170, 441], [176, 426], [186, 411], [193, 400], [207, 397], [216, 387], [229, 382], [240, 373], [236, 367], [250, 371], [262, 367], [271, 356], [279, 347], [297, 349], [307, 340], [320, 338], [332, 328], [352, 320], [359, 311], [372, 314], [370, 302], [377, 298], [369, 286], [379, 281], [377, 270], [388, 274], [395, 266], [403, 258], [418, 259], [422, 243], [416, 232]], [[101, 456], [113, 447], [126, 451], [138, 459], [148, 447], [156, 429], [171, 421], [174, 402], [187, 391], [198, 385], [205, 377], [219, 375], [231, 367], [239, 373], [248, 370], [232, 361], [226, 352], [215, 349], [220, 338], [214, 326], [223, 316], [233, 316], [240, 310], [250, 301], [256, 286], [267, 285], [273, 300], [272, 320], [261, 339], [274, 338], [289, 323], [300, 324], [304, 331], [319, 324], [328, 313], [332, 289], [350, 286], [359, 278], [367, 281], [375, 274], [373, 263], [383, 258], [390, 266], [401, 258], [411, 248], [417, 230]], [[102, 451], [110, 443], [121, 445], [133, 451], [142, 447], [151, 427], [160, 414], [164, 395], [177, 384], [180, 375], [196, 371], [204, 362], [218, 365], [230, 359], [224, 350], [216, 348], [207, 337], [211, 325], [220, 319], [227, 307], [239, 313], [247, 301], [253, 277], [257, 263], [250, 251], [251, 237], [258, 242], [263, 269], [273, 282], [280, 297], [294, 298], [300, 306], [313, 302], [324, 299], [327, 281], [339, 278], [349, 270], [356, 264], [359, 248], [367, 245], [370, 252], [377, 248], [383, 255], [395, 249], [400, 243], [414, 240], [420, 228]], [[99, 449], [108, 438], [119, 442], [129, 447], [140, 442], [149, 427], [153, 410], [162, 396], [169, 381], [179, 369], [193, 363], [203, 355], [208, 345], [204, 335], [211, 323], [207, 314], [213, 303], [221, 311], [230, 310], [237, 303], [242, 291], [244, 303], [253, 283], [252, 263], [246, 250], [245, 230], [253, 241], [259, 241], [259, 256], [269, 279], [280, 288], [300, 289], [313, 296], [321, 286], [321, 268], [330, 257], [338, 256], [341, 266], [349, 255], [350, 237], [355, 228], [355, 214], [362, 202], [359, 222], [365, 232], [369, 239], [377, 233], [391, 232], [403, 228], [411, 229], [414, 219], [410, 207], [415, 198], [417, 208], [421, 211]], [[99, 447], [103, 435], [109, 432], [114, 443], [125, 440], [136, 442], [143, 432], [150, 416], [157, 394], [165, 383], [173, 367], [187, 355], [195, 358], [200, 342], [197, 330], [207, 320], [202, 309], [211, 291], [217, 298], [220, 307], [228, 306], [235, 291], [242, 296], [241, 310], [248, 298], [248, 280], [252, 269], [247, 260], [247, 248], [242, 236], [246, 225], [250, 235], [260, 236], [264, 254], [267, 272], [275, 280], [279, 288], [298, 286], [309, 289], [317, 287], [320, 263], [329, 250], [333, 254], [339, 251], [344, 236], [332, 239], [333, 229], [345, 224], [350, 227], [348, 217], [356, 208], [357, 198], [364, 192], [364, 206], [360, 216], [366, 224], [368, 232], [379, 227], [388, 224], [399, 225], [408, 222], [410, 211], [408, 202], [413, 196], [416, 206], [422, 209]]], 'inland_risk_patch': [[248, 277], [263, 277], [269, 290], [278, 296], [266, 296], [256, 303], [251, 297]], 'water_cuts': [[[119, 462], [124, 453], [129, 455], [131, 462]], [[143, 454], [147, 446], [151, 450], [149, 459]], [[181, 428], [186, 414], [190, 412], [187, 423]], [[209, 401], [218, 389], [226, 387], [218, 397]], [[240, 382], [249, 374], [255, 375], [249, 382]]], 'boundary_generation': {'seed': 18, 'rural_count': 155, 'urban_count': 180, 'urban_center': [160, 293], 'urban_spread': [28, 23], 'box': [80, 450, 50, 463]}, 'legend_box': [73, 70, 67, 134], 'legend_swatch': [87, 82, 23, 23], 'geographic_labels': [['sugar_land', 135, 326, 7], ['houston', 260, 381, 9]]}, 'label_map': {'panel': 'panel_00_panel', 'legend': 'panel_00_legend', 'sugar_land': 'panel_00_sugar_land', 'houston': 'panel_00_houston'}, 'crop_pixel_bbox': [3, 8, 473, 475], 'crop_sha256': 'f9ed03045a01f9228946ea1affa47ff459ff5e4d101828080e6022ccb789f968', 'api_request_sha256': '6b293bebcaef73a73dc2b51b06ae60e657709bc88efb3c260a8fd57ee18c9479'}, {'bbox': [0.523, 0.016, 0.996, 0.969], 'data': {'canvas': [968, 926], 'x': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], 'xticks': [0, 1, 2, 3, 4, 5], 'yticks': [0, 0.25, 0.5, 0.75, 1], 'xlim': [-0.25, 5.23], 'ylim': [-0.05, 1.05], 'series': [{'key': 'category_0', 'color': '#fa7974', 'curve': [0.46, 0.383, 0.312, 0.253, 0.196, 0.158, 0.116, 0.087, 0.062, 0.046, 0.033], 'lower': [0.34, 0.28, 0.215, 0.16, 0.109, 0.075, 0.037, 0.024, 0.014, 0.007, 0.004], 'upper': [0.58, 0.485, 0.406, 0.343, 0.283, 0.241, 0.198, 0.154, 0.117, 0.094, 0.077], 'points': [[0, 0.5], [1, 0.2], [1, 0.01], [3, 0.2], [4, 0.12]]}, {'key': 'category_1', 'color': '#b6a20d', 'curve': [0.691, 0.612, 0.535, 0.454, 0.379, 0.313, 0.256, 0.208, 0.164, 0.121, 0.087], 'lower': [0.555, 0.477, 0.396, 0.32, 0.251, 0.203, 0.16, 0.123, 0.089, 0.06, 0.036], 'upper': [0.824, 0.749, 0.675, 0.588, 0.512, 0.433, 0.365, 0.309, 0.255, 0.212, 0.18], 'points': [[0, 0.75], [1, 0.7], [1, 0.5], [3, 0.4], [4, 0.25]]}, {'key': 'category_2', 'color': '#64bc25', 'legend_color': '#00d940', 'curve': [0.865, 0.817, 0.765, 0.694, 0.62, 0.543, 0.468, 0.393, 0.325, 0.258, 0.195], 'lower': [0.764, 0.693, 0.625, 0.542, 0.459, 0.385, 0.312, 0.247, 0.189, 0.141, 0.096], 'upper': [0.946, 0.918, 0.887, 0.831, 0.764, 0.695, 0.624, 0.553, 0.484, 0.411, 0.338], 'points': [[0, 0.85], [1, 0.95], [1, 0.89], [1, 0.75], [1, 0.63], [2, 0.69], [2, 0.6], [2, 0.55], [2, 0.46], [3, 0.66], [3, 0.55], [3, 0.33], [4, 0.5], [4, 0.09], [5, 0.12]]}, {'key': 'category_3', 'color': '#18bfb8', 'legend_color': '#00c9d5', 'curve': [0.96, 0.925, 0.89, 0.85, 0.808, 0.75, 0.688, 0.612, 0.535, 0.457, 0.385], 'lower': [0.883, 0.839, 0.791, 0.742, 0.686, 0.617, 0.548, 0.466, 0.389, 0.318, 0.248], 'upper': [0.991, 0.985, 0.976, 0.957, 0.932, 0.887, 0.837, 0.766, 0.691, 0.614, 0.536], 'points': [[0.5, 0.9], [2.5, 0.75], [4, 0.75]]}, {'key': 'category_4', 'color': '#a49ddc', 'legend_color': '#599cf2', 'curve': [0.977, 0.963, 0.949, 0.932, 0.915, 0.883, 0.85, 0.803, 0.755, 0.69, 0.62], 'lower': [0.938, 0.917, 0.901, 0.875, 0.844, 0.797, 0.752, 0.694, 0.63, 0.552, 0.48], 'upper': [0.997, 0.994, 0.987, 0.98, 0.973, 0.952, 0.932, 0.9, 0.866, 0.817, 0.766], 'points': [[0.5, 0.95], [4, 0.9]]}, {'key': 'category_5', 'color': '#ee65d0', 'curve': [0.99, 0.985, 0.98, 0.973, 0.965, 0.953, 0.94, 0.914, 0.885, 0.848, 0.81], 'lower': [0.97, 0.966, 0.957, 0.946, 0.928, 0.908, 0.885, 0.853, 0.81, 0.769, 0.725], 'upper': [1.008, 1.007, 1.004, 1.002, 0.999, 0.995, 0.989, 0.979, 0.965, 0.94, 0.915], 'points': [[0, 0.99], [1, 0.99], [3, 0.8], [4, 0.99]]}]}, 'label_map': {'panel': 'panel_01_panel', 'y_axis': 'panel_01_y_axis', 'x_axis': 'panel_01_x_axis', 'legend_title': 'panel_01_legend_title', 'category_0': 'panel_01_category_0', 'category_1': 'panel_01_category_1', 'category_2': 'panel_01_category_2', 'category_3': 'panel_01_category_3', 'category_4': 'panel_01_category_4', 'category_5': 'panel_01_category_5'}, 'crop_pixel_bbox': [536, 8, 1020, 471], 'crop_sha256': 'c87679c58a2a1eb48478b51314bdbd7f100a07cd0084587fe893143ad311532e', 'api_request_sha256': 'a25cc51968a38142cbc91595cdd6684b680ca46bae0d897bbfd69f74fc0161a4'}], 'global_placements': []}
LABELS = {'panel_00_panel': 'A', 'panel_00_legend': 'مناطق الخطر', 'panel_00_sugar_land': 'شوغر لاند', 'panel_00_houston': 'هيوستن', 'panel_01_panel': 'B', 'panel_01_y_axis': 'معدلات الإخلاء', 'panel_01_x_axis': 'مناطق الخطر', 'panel_01_legend_title': 'فئة\nالإعصار', 'panel_01_category_0': '0', 'panel_01_category_1': '1', 'panel_01_category_2': '2', 'panel_01_category_3': '3', 'panel_01_category_4': '4', 'panel_01_category_5': '5'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
