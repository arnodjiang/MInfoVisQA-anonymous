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
    fig.patch.set_facecolor('white')
    placements = []
    e = data['extent']
    n = data['grid_size']
    t = data['texture']
    x = np.linspace(e[0], e[1], n[0])
    y = np.linspace(e[2], e[3], n[1])
    (X, Y) = np.meshgrid(x, y)
    center = np.interp(X, data['curve_x'], data['curve_y'])
    width = np.interp(X, data['curve_x'], data['half_width'])
    center = center + t['wave_amplitude'] * (np.sin(X * t['wave_frequencies'][0]) + 0.45 * np.sin(X * t['wave_frequencies'][1]))
    d = Y - center
    band = 1 / (1 + np.exp(np.clip((np.abs(d) - width) / t['edge_softness'], -60, 60)))
    cutoff = np.clip((t['tail_end'] - X) / (t['tail_end'] - t['tail_start']), 0, 1) ** t['tail_power']
    core = band * cutoff
    rng = np.random.default_rng(t['seed'])
    spots = np.zeros_like(X)
    holes = np.zeros_like(X)
    for j in range(t['spot_count']):
        sx = rng.uniform(*t['spot_x_range'])
        sy = np.interp(sx, data['curve_x'], data['curve_y']) + rng.uniform(*t['spot_y_offset_range'])
        r = rng.uniform(*t['spot_radius_range'])
        strength = rng.uniform(*t['spot_strength_range'])
        g = np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (r * r))
        spots = np.maximum(spots, g * strength)
        if j % 3 == 0:
            holes = np.maximum(holes, np.exp(-((X - sx) ** 2 + (Y - sy + 0.004) ** 2) / (r * r)) * 0.85)
    spots = spots * (1 / (1 + np.exp(np.clip((Y - 0.064) / 0.001, -60, 60))))
    left = core * (1 - holes) * 1.35
    left = np.maximum(left, spots * 1.1)
    left = np.clip(left, 0, 1)
    slope_factor = 0.22 + 0.86 / (1 + np.exp(-(X + 0.007) / 0.003))
    upper_factor = 0.78 + 0.32 * np.tanh(d / 0.003)
    left_tip = 0.85 * np.exp(-((X + 0.028) / 0.005) ** 2)
    right = core * (slope_factor * upper_factor + left_tip) * (1 - holes)
    right = np.maximum(right, spots * (0.45 + 0.65 * np.clip((0.06 - X) / 0.025, 0, 1)))
    right = np.clip(right, 0, 1)
    for (i, p) in enumerate(data['panels']):
        ax = fig.add_axes(p['position'])
        Z = (left if i == 0 else right) * p['maximum']
        Z = np.maximum(Z, t['background'] * p['maximum'])
        im = ax.imshow(Z, extent=e, origin='lower', cmap='jet', vmin=0, vmax=p['maximum'], interpolation='nearest', aspect='auto')
        ax.set_xticks(data['ticks_x'])
        ax.set_yticks(data['ticks_y'])
        ax.set_xticklabels(['−0.025', '0.000', '0.025', '0.050'], fontfamily='serif', fontsize=12)
        ax.set_yticklabels(['−0.02', '0.00', '0.02', '0.04', '0.06'], fontfamily='serif', fontsize=12)
        ax.tick_params(length=4, width=0.8, direction='out')
        for s in ax.spines.values():
            s.set_linewidth(1)
        cax = fig.add_axes(p['colorbar'])
        cb = fig.colorbar(im, cax=cax, ticks=p['color_ticks'], format=p['format'])
        cb.ax.tick_params(labelsize=12, length=4)
        for lab in cb.ax.get_yticklabels():
            lab.set_fontfamily('serif')
        (l, b, w, h) = p['position']
        placements.extend([{'key': p['title'], 'x': l + w / 2, 'y': 0.074, 'size': 22, 'max_width': 0.4, 'anchor': 'center'}, {'key': 'x_axis', 'x': l + w / 2, 'y': 0.942, 'size': 20, 'max_width': 0.15, 'anchor': 'center'}, {'key': 'y_axis', 'x': l - 0.073, 'y': 1 - b - h / 2, 'size': 20, 'max_width': 0.15, 'rotation': 90, 'anchor': 'center'}])
    return finish(fig, labels, placements)
BASE_ID = 'qa_c7f0ac4441285dc9145b2d36c1c6c6d39edf891eb8f2779e66c6d88bc564d017'
LANGUAGE = 'ru'
DATA = {'canvas': [1100, 430], 'extent': [-0.031, 0.073, -0.029, 0.073], 'grid_size': [460, 460], 'curve_x': [-0.031, -0.029, -0.027, -0.025, -0.022, -0.019, -0.016, -0.013, -0.01, -0.006, -0.002, 0.003, 0.007, 0.011, 0.016, 0.021, 0.026, 0.031, 0.036, 0.041, 0.046, 0.051, 0.056, 0.061, 0.066, 0.07, 0.073], 'curve_y': [0.042, 0.041, 0.038, 0.031, 0.019, 0.008, -0.001, -0.008, -0.013, -0.017, -0.018, -0.018, -0.018, -0.017, -0.014, -0.01, -0.005, 0.001, 0.007, 0.014, 0.022, 0.03, 0.039, 0.048, 0.057, 0.065, 0.07], 'half_width': [0.0005, 0.0013, 0.0022, 0.0033, 0.0039, 0.0041, 0.004, 0.0044, 0.0054, 0.006, 0.0065, 0.007, 0.007, 0.0068, 0.0065, 0.0063, 0.0061, 0.006, 0.0059, 0.0058, 0.0056, 0.0054, 0.005, 0.0045, 0.004, 0.0032, 0.0025], 'ticks_x': [-0.025, 0, 0.025, 0.05], 'ticks_y': [-0.02, 0, 0.02, 0.04, 0.06], 'panels': [{'title': 'residual_title', 'position': [0.112, 0.175, 0.28, 0.705], 'colorbar': [0.412, 0.175, 0.014, 0.705], 'maximum': 0.025, 'color_ticks': [0, 0.005, 0.01, 0.015, 0.02, 0.025], 'format': '%.3f'}, {'title': 'magnitude_title', 'position': [0.625, 0.175, 0.28, 0.705], 'colorbar': [0.925, 0.175, 0.014, 0.705], 'maximum': 20, 'color_ticks': [0, 5, 10, 15, 20], 'format': '%.0f'}], 'texture': {'seed': 318, 'spot_count': 240, 'spot_x_range': [0.014, 0.07], 'spot_y_offset_range': [-0.003, 0.019], 'spot_radius_range': [0.00032, 0.00105], 'spot_strength_range': [0.25, 1.0], 'tail_start': 0.027, 'tail_end': 0.07, 'tail_power': 1.35, 'edge_softness': 0.00044, 'wave_amplitude': 0.00032, 'wave_frequencies': [900, 1700], 'background': 0.0001}}
LABELS = {'residual_title': 'Норма невязки χ', 'magnitude_title': 'Величина MPR ‖ψ̂‖₂', 'x_axis': 'ξ₁', 'y_axis': 'ξ₂'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
