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
    t = np.arange(0, data['trace_end'] + data['sample_step'], data['sample_step'])
    for (i, p) in enumerate(data['panels']):
        b = data['axes_bottoms'][i]
        l = data['axes_left']
        w = data['axes_width']
        h = data['axes_height']
        ax = fig.add_axes([l, b, w, h])
        y = np.zeros_like(t)
        for (period, weight) in zip(data['baseline_periods'], data['baseline_weights']):
            y += p['baseline_amplitude'] * weight * np.sin(t * 2 * np.pi / period + i)
        for (center, positive, negative, decay) in p['events']:
            dt = t - center
            envelope = np.exp(-np.abs(dt) / decay)
            wave = np.cos(2 * np.pi * dt / data['oscillation_period'])
            y += envelope * np.where(wave >= 0, positive * wave, -negative * wave)
        ax.plot(t, y, color='#249bc3', linewidth=0.42)
        ax.scatter(p['traffic_time'], p['traffic_height'], s=37, facecolors='none', edgecolors='#c7ad94', linewidths=0.65, zorder=4)
        ax.set_xlim(data['x_limits'])
        ax.set_ylim(data['y_limits'])
        ax.set_xticks(data['x_ticks'])
        ax.set_yticks(data['y_ticks'])
        ax.grid(True, color='#dddddd', linewidth=0.5)
        ax.tick_params(axis='both', direction='in', top=True, right=True, length=6, width=0.5, labelsize=10, pad=7, colors='#333333')
        for spine in ax.spines.values():
            spine.set_color('#777777')
            spine.set_linewidth(0.7)
        placements.extend([{'key': p['title'], 'x': l + w / 2, 'y': 1 - b - h - 0.013, 'size': 15, 'max_width': 0.75, 'anchor': 'center'}, {'key': 'time', 'x': l + w / 2, 'y': 1 - b + 0.042, 'size': 14, 'max_width': 0.35, 'anchor': 'center'}, {'key': 'acceleration_axis', 'x': 0.039, 'y': 1 - b - h / 2, 'size': 14, 'max_width': 0.19, 'rotation': 90, 'anchor': 'center'}])
        if i == 0:
            ax.add_patch(Rectangle((0.885, 0.728), 0.109, 0.245, transform=ax.transAxes, facecolor='white', edgecolor='#777777', linewidth=0.7, zorder=5))
            ax.plot([0.891, 0.918], [0.905, 0.905], transform=ax.transAxes, color='#249bc3', linewidth=0.5, zorder=6)
            ax.scatter([0.904], [0.799], transform=ax.transAxes, s=37, facecolors='none', edgecolors='#c7ad94', linewidths=0.65, zorder=6)
            for (key, yy) in [('acceleration', 0.905), ('traffic', 0.799)]:
                placements.append({'key': key, 'x': l + w * 0.923, 'y': 1 - (b + h * yy), 'size': 12, 'max_width': 0.065, 'anchor': 'left'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_fb6b4277fc65c602b26437388fdc1fd1b9cb66c2a3046e05f036b54aeabd3a77'
LANGUAGE = 'id'
DATA = {'canvas': [1200, 900], 'x_limits': [0, 4000], 'y_limits': [-1.5, 1.5], 'x_ticks': [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], 'y_ticks': [-1, 0, 1], 'trace_end': 3600, 'sample_step': 0.25, 'oscillation_period': 2.7, 'baseline_periods': [3.7, 7.1, 17.3, 43.1], 'baseline_weights': [0.38, 0.28, 0.22, 0.12], 'axes_left': 0.072, 'axes_width': 0.879, 'axes_height': 0.17, 'axes_bottoms': [0.8, 0.548, 0.305, 0.06], 'panels': [{'title': 'panel_1', 'baseline_amplitude': 0.006, 'events': [[505, 0.018, -0.019, 9], [667, 0.1, -0.12, 12], [1048, 0.042, -0.043, 15], [1115, 0.022, -0.026, 11], [1540, 0.03, -0.033, 15], [2150, 0.022, -0.023, 7], [3447, 0.028, -0.029, 6]], 'traffic_time': [], 'traffic_height': []}, {'title': 'panel_2', 'baseline_amplitude': 0.014, 'events': [[205, 0.023, -0.03, 7], [507, 0.035, -0.04, 7], [920, 0.03, -0.032, 6], [1110, 0.43, -0.39, 7], [1140, 0.15, -0.19, 7], [1710, 0.023, -0.027, 9], [2312, 0.025, -0.028, 7], [2598, 0.095, -0.1, 6], [3216, 0.39, -0.37, 6], [3428, 0.12, -0.17, 5], [3448, 0.14, -0.17, 4], [3480, 0.095, -0.105, 8]], 'traffic_time': [1110, 3216], 'traffic_height': [0.43, 0.41]}, {'title': 'panel_3', 'baseline_amplitude': 0.022, 'events': [[218, 0.45, -0.46, 9], [328, 0.18, -0.18, 6], [348, 0.26, -0.21, 6], [559, 0.1, -0.16, 4], [575, 0.08, -0.27, 4], [752, 0.13, -0.27, 8], [899, 0.41, -0.3, 7], [1046, 0.24, -0.18, 6], [1195, 0.51, -0.29, 8], [1277, 0.13, -0.13, 15], [1485, 0.38, -0.34, 7], [1686, 0.1, -0.09, 5], [1732, 0.05, -0.08, 6], [1808, 0.08, -0.2, 5], [1882, 0.14, -0.25, 5], [1911, 0.11, -0.2, 5], [1931, 0.21, -0.26, 7], [2050, 0.41, -0.51, 8], [2155, 0.24, -0.18, 6], [2200, 0.25, -0.28, 6], [2222, 0.08, -0.13, 5], [2310, 0.08, -0.09, 7], [2351, 0.07, -0.17, 5], [2367, 0.24, -0.13, 7], [2600, 0.14, -0.15, 6], [2774, 0.08, -0.15, 4], [2808, 0.22, -0.33, 8], [2995, 0.36, -0.47, 7], [3038, 0.38, -0.54, 8], [3079, 0.81, -0.57, 9], [3402, 0.37, -0.26, 6], [3509, 0.34, -0.19, 5], [3528, 0.12, -0.2, 5]], 'traffic_time': [218, 348, 899, 1046, 1195, 1485, 1931, 2050, 2155, 2200, 2367, 2808, 2995, 3038, 3079, 3402, 3518], 'traffic_height': [0.45, 0.26, 0.41, 0.24, 0.51, 0.38, 0.21, 0.41, 0.24, 0.25, 0.24, 0.22, 0.36, 0.38, 0.81, 0.37, 0.34]}, {'title': 'panel_4', 'baseline_amplitude': 0.025, 'events': [[76, 0.26, -0.18, 5], [205, 0.04, -0.09, 5], [225, 0.07, -0.07, 5], [248, 0.14, -0.27, 5], [582, 1.27, -1.48, 8], [698, 0.65, -0.3, 7], [965, 0.44, -0.57, 8], [1022, 0.37, -0.23, 7], [1156, 0.53, -0.31, 7], [1171, 0.12, -0.22, 6], [1277, 0.17, -0.2, 6], [1300, 0.97, -0.69, 7], [1474, 0.36, -0.22, 10], [1546, 1.48, -1.24, 9], [1576, 0.56, -0.24, 7], [1626, 0.09, -0.09, 15], [1917, 1.18, -0.93, 9], [1979, 0.62, -0.37, 7], [2012, 1.11, -1.45, 7], [2112, 0.3, -0.46, 7], [2483, 0.4, -0.74, 7], [2556, 0.34, -0.52, 8], [2627, 0.55, -0.85, 6], [3158, 0.56, -0.29, 5], [3268, 0.51, -0.28, 5]], 'traffic_time': [76, 582, 698, 965, 1022, 1156, 1300, 1474, 1576, 1917, 1979, 2012, 2112, 2483, 2556, 2627, 3158, 3268], 'traffic_height': [0.26, 1.27, 0.65, 0.44, 0.37, 0.53, 0.97, 0.36, 0.56, 1.18, 0.62, 1.11, 0.3, 0.4, 0.34, 0.55, 0.56, 0.51]}]}
LABELS = {'panel_1': 'Jendela waktu 8 (24:00-1:00)', 'panel_2': 'Jendela waktu 4 (20:00-21:00)', 'panel_3': 'Jendela waktu 16 (8:00-9:00)', 'panel_4': 'Jendela waktu 15 (16:00-17:00)', 'time': 'waktu [s]', 'acceleration_axis': 'percepatan [m/s²]', 'acceleration': 'Percepatan', 'traffic': 'Lalu lintas'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
