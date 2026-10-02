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
    ax = fig.add_axes(data['top_axes'])
    for (x, y) in zip(data['frequency_segments'], data['sv_segments']):
        x = np.array(x)
        y = np.array(y)
        for (off, lw) in zip(data['line_offsets'], data['line_widths']):
            ax.plot(x, y + off, color='black', lw=lw)
    ax.plot(*data['short_spike'], color='black', lw=2)
    ax.set_xlim(data['top_xlim'])
    ax.set_ylim(data['top_ylim'])
    ax.set_xticks(data['top_xticks'])
    ax.set_yticks(data['top_yticks'])
    ax.grid(True, color='#d9d9d9', lw=1)
    ax.tick_params(labelsize=23, direction='in', top=True, right=True, length=7, pad=9)
    placements.extend([{'key': 'panel_a', 'x': 0.111, 'y': 0.018, 'size': 38, 'anchor': 'left', 'max_width': 0.08}, {'key': 'frequency', 'x': 0.544, 'y': 0.5, 'size': 34, 'anchor': 'center', 'max_width': 0.55}, {'key': 'depth', 'x': 0.03, 'y': 0.631, 'size': 34, 'anchor': 'center', 'max_width': 0.34, 'rotation': 90}])
    s = data['sv_label_layout']
    fig.text(s['x'], 1 - s['symbol_y'], '$S_v$', fontsize=s['size'] * 72 / 100, rotation=90, ha='center', va='center')
    fig.text(s['x'], 1 - s['power_y'], '$\\mathrm{m}^{-1})$', fontsize=s['size'] * 72 / 100, rotation=90, ha='center', va='center')
    placements.append({'key': 'sv_units', 'x': s['x'], 'y': s['units_y'], 'size': s['size'], 'anchor': 'center', 'max_width': s['units_width'], 'rotation': 90})
    (ny, nx) = data['field_resolution']
    x = np.linspace(*data['distance_limits'], nx)
    z = np.linspace(data['depth_limits'][1], data['depth_limits'][0], ny)
    (X, Z) = np.meshgrid(x, z)
    top = np.interp(x, data['school_distance'], data['school_top'])
    bot = np.interp(x, data['school_distance'], data['school_bottom'])
    p = data['field_parameters']

    def sigmoid(v):
        return 1 / (1 + np.exp(-np.clip(v, -60, 60)))
    env = sigmoid((Z - top) / p['edge_width']) * sigmoid((bot - Z) / p['edge_width'])
    (a, b, w) = p['horizontal_edge']
    env *= sigmoid((X - a) / w) * sigmoid((b - X) / w)

    def rgb(values, lo, hi):
        t = np.clip((values - lo) / (hi - lo), 0, 1)
        pal = np.array(data['palette_rgb'])
        return np.stack([np.interp(t, data['palette_positions'], pal[:, j]) for j in range(3)], axis=-1)
    for (i, info) in enumerate(data['panel_fields']):
        rng = np.random.default_rng(info['seed'])
        noise = rng.normal(size=(ny, nx))
        mix = p['grain_mix']
        noise = mix[0] * noise + mix[1] * np.roll(noise, 1, axis=1) + mix[2] * np.roll(noise, 1, axis=0)
        field = np.full((ny, nx), info['background']) + p['noise_scale'] * noise
        field += env * (info['school_gain'] + info['vertical_gain'] * (Z - 35) / 50)
        (gain, zz, ww) = p['school_ridge']
        field += gain * env * np.exp(-((Z - zz) / ww) ** 2)
        if i == 0:
            (gain, zz, ww) = p['upper_glow']
            field += -gain * np.exp(-((Z - zz) / ww) ** 2) * (1 - env)
        else:
            (gain, zz, ww) = p['deep_glow']
            field += gain * np.exp(-((Z - zz) / ww) ** 2) * (1 - env)
        for (xx, zz, ww, gain) in data['streaks']:
            field += gain * np.exp(-((X - xx) / ww) ** 2) * np.exp(-((Z - zz) / 20) ** 2) * env
        for (xx, zz, wx, wz, gain) in data['deep_echoes']:
            field += gain * np.exp(-((X - xx) / wx) ** 2 - ((Z - zz - (X - xx) * 0.15) / wz) ** 2)
        (aa, bb, cc, dd) = p['bottom_wave']
        bottom = p['bottom_depth'] + aa * np.sin(X / bb) + cc * np.sin(X / dd)
        e = np.exp(-((Z - bottom) / p['bottom_width']) ** 2)
        field = field * (1 - e) + info['bottom'] * e
        e = np.exp(-((Z - p['surface_depth']) / p['surface_width']) ** 2)
        field = field * (1 - e) + info['surface'] * e
        colors = rgb(field, *info['range'])
        if i == 1:
            colors[field < info['white_cut']] = 1
        pos = data['image_axes'][i]
        axy = fig.add_axes(pos)
        axy.imshow(colors, extent=data['heat_extent'], origin='upper', aspect='auto', interpolation='bilinear')
        axy.set_xlim(data['distance_limits'])
        axy.set_ylim(data['depth_limits'])
        axy.set_xticks(data['distance_ticks'])
        axy.set_yticks(data['depth_ticks'])
        axy.tick_params(direction='in', labelsize=23, pad=9)
        if i == 1:
            axy.set_yticklabels([])
        for zz in data['grid_depths']:
            axy.axhline(zz, color='white', lw=1, alpha=0.5, ls=(0, (1, 5)))
        for xx in data['grid_distances']:
            axy.axvline(xx, color='white', lw=1, alpha=0.5, ls=(0, (1, 5)))
        for (xx, z1, z2) in data['vertical_markers']:
            axy.plot([xx, xx], [z1, z2], color='#3a3434', lw=0.7)
        for spine in axy.spines.values():
            spine.set_linewidth(1.2)
        cb = fig.add_axes(data['colorbar_axes'][i])
        (lo, hi) = info['range']
        ramp = np.linspace(lo, hi, ny)[:, None]
        cb.imshow(rgb(ramp, lo, hi), extent=[0, 1, lo, hi], origin='lower', aspect='auto')
        cb.set_xticks([])
        cb.set_yticks(info['ticks'])
        cb.yaxis.tick_right()
        cb.tick_params(labelsize=21, length=0, pad=5)
        placements.append({'key': 'distance', 'x': pos[0] + pos[2] / 2, 'y': 0.98, 'size': 33, 'anchor': 'center', 'max_width': 0.39})
    placements.extend([{'key': 'panel_b', 'x': 0.11, 'y': 0.503, 'size': 38, 'anchor': 'left', 'max_width': 0.08}, {'key': 'panel_c', 'x': 0.876, 'y': 0.503, 'size': 38, 'anchor': 'left', 'max_width': 0.08}])
    return finish(fig, labels, placements)
BASE_ID = 'qa_8727b271401b853d788f6bee98d1ceceae72645e29605977e2a0a33cfbed91c6'
LANGUAGE = 'de'
DATA = {'canvas': [1200, 800], 'top_axes': [0.112, 0.594, 0.865, 0.348], 'image_axes': [[0.112, 0.113, 0.345, 0.346], [0.553, 0.113, 0.345, 0.346]], 'colorbar_axes': [[0.464, 0.113, 0.022, 0.346], [0.904, 0.113, 0.022, 0.346]], 'frequency_segments': [[93, 96, 100, 103, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 136, 139, 142, 144, 147, 150, 153, 156, 159, 162, 165, 168, 170, 172, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255, 258], [282, 285, 288, 291, 294, 297, 300, 303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348, 351, 354, 357, 360, 363, 366, 369, 372, 375, 378, 381, 384, 387, 390, 393, 396, 399, 402, 405, 408, 411, 414, 417, 420, 423, 426, 429, 432, 435, 438, 441, 444, 447]], 'sv_segments': [[-55.85, -56.6, -55.02, -55.3, -54.92, -55.24, -55.04, -55.12, -54.75, -54.86, -54.94, -55.28, -55.36, -55.35, -54.6, -54.5, -54.03, -54.46, -54.33, -54.55, -54.8, -54.94, -54.76, -54.63, -53.0, -53.02, -52.75, -52.82, -52.98, -53.22, -53.51, -53.65, -53.79, -53.65, -53.31, -53.29, -53.13, -53.03, -53.04, -53.13, -53.13, -53.25, -53.09, -53.63, -53.76, -53.92, -54.13, -53.85, -53.62, -53.08, -52.32, -51.75, -51.35, -50.95, -51.17, -50.99, -50.95], [-49.47, -49.41, -49.51, -49.83, -50.15, -50.24, -50.43, -50.58, -50.75, -50.68, -51.05, -51.7, -52.02, -52.04, -51.55, -51.04, -50.67, -50.59, -50.71, -51.0, -51.03, -50.8, -50.53, -50.13, -49.85, -49.64, -49.4, -49.39, -49.27, -49.19, -49.63, -49.91, -50.19, -50.46, -49.79, -49.59, -48.76, -48.93, -51.19, -51.36, -51.16, -51.19, -50.57, -50.35, -49.9, -49.72, -49.36, -49.29, -49.08, -49.28, -48.96, -48.13, -48.79, -49.5, -50.15, -50.02]], 'line_offsets': [-0.16, 0, 0.15], 'line_widths': [1.0, 2.4, 1.0], 'short_spike': [[162, 166, 168], [-52.55, -53.8, -53.05]], 'top_xlim': [90, 450], 'top_ylim': [-56.7, -48], 'top_xticks': [100, 150, 200, 250, 300, 350, 400, 450], 'top_yticks': [-56, -54, -52, -50, -48], 'distance_limits': [0, 245], 'depth_limits': [133, 7], 'distance_ticks': [0, 100, 200], 'depth_ticks': [20, 40, 60, 80, 100, 120], 'school_distance': [0, 5, 10, 16, 23, 32, 43, 55, 68, 80, 95, 110, 128, 145, 160, 177, 193, 208, 220, 229, 237, 245], 'school_top': [29, 27, 25, 24, 28, 27, 25, 27, 27, 28, 29, 28, 30, 30, 31, 30, 33, 34, 34, 37, 48, 55], 'school_bottom': [33, 38, 47, 57, 66, 75, 78, 79, 81, 82, 83, 84, 84, 84, 83, 84, 86, 87, 87, 84, 76, 65], 'panel_fields': [{'seed': 27, 'range': [-61, -25], 'background': -58.4, 'school_gain': 18.6, 'vertical_gain': 4.4, 'surface': -26, 'bottom': -35, 'ticks': [-60, -50, -40, -30], 'white_cut': -65}, {'seed': 29, 'range': [-81, -36], 'background': -80.5, 'school_gain': 23.8, 'vertical_gain': 5.5, 'surface': -37, 'bottom': -40, 'ticks': [-80, -60, -40], 'white_cut': -80.4}], 'palette_positions': [0, 0.08, 0.2, 0.33, 0.46, 0.59, 0.72, 0.85, 0.94, 1], 'palette_rgb': [[0.99, 0.9, 0.98], [0.86, 0.71, 0.9], [0.61, 0.62, 0.8], [0.38, 0.72, 0.78], [0.24, 0.72, 0.51], [0.24, 0.68, 0.23], [0.49, 0.57, 0.12], [0.54, 0.37, 0.05], [0.47, 0.12, 0.02], [0.32, 0.01, 0.01]], 'field_resolution': [260, 480], 'field_parameters': {'edge_width': 2.2, 'horizontal_edge': [19, 227, 3.5], 'noise_scale': 1.55, 'surface_depth': 8.8, 'surface_width': 1.05, 'bottom_depth': 130, 'bottom_width': 1.3, 'bottom_wave': [0.55, 27, 0.22, 3.8], 'upper_glow': [-13, 22, 17], 'deep_glow': [4.5, 122, 15], 'school_ridge': [2.1, 76, 5.5], 'grain_mix': [0.48, 0.32, 0.2]}, 'streaks': [[12, 29, 1.8, 3.4], [34, 44, 2.5, 2.5], [67, 31, 2.3, 2.9], [108, 30, 1.8, 3.6], [135, 30, 1.7, 3.3], [178, 76, 2.8, 2.5], [199, 78, 2.0, 3.0]], 'deep_echoes': [[22, 90, 5, 2, 5], [73, 92, 5, 0.6, 8], [125, 118, 8, 1.3, 8], [196, 94, 6, 0.8, 8], [219, 125, 7, 1.1, 7]], 'vertical_markers': [[14, 93, 133], [125, 93, 133], [236, 93, 133]], 'grid_depths': [20, 40, 60, 80], 'grid_distances': [100, 200], 'heat_extent': [0, 245, 133, 7], 'sv_label_layout': {'x': 0.028, 'symbol_y': 0.369, 'units_y': 0.257, 'power_y': 0.125, 'size': 33, 'units_width': 0.145}}
LABELS = {'panel_a': 'a', 'panel_b': 'b', 'panel_c': 'c', 'frequency': 'Frequenz (kHz)', 'sv': 'Sᵥ (dB bezogen auf 1 m⁻¹)', 'sv_units': '(dB bezogen auf 1', 'depth': 'Tiefe (m)', 'distance': 'Entfernung (m)'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
