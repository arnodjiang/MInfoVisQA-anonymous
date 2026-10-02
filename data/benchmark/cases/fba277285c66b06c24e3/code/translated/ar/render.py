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
    fig.patch.set_facecolor(data['background'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    placements = []
    v = np.array(data['box_vertices'])
    for (i, j) in data['box_edges']:
        ax.plot(v[[i, j], 0], v[[i, j], 1], color='#a3a6b5', lw=0.55, alpha=0.5, zorder=0)

    def numeric_edge(a, b, values, dx, dy):
        a = np.array(a)
        b = np.array(b)
        for (t, value) in zip(np.linspace(0, 1, len(values)), values):
            p = a * (1 - t) + b * t
            ax.plot([p[0] - 1, p[0] + 1], [p[1] - 2, p[1] + 2], color='#a7abba', lw=0.4, alpha=0.6)
            ax.text(p[0] + dx, p[1] + dy, str(value).rstrip('0').rstrip('.') if isinstance(value, float) else str(value), color='#a9acbb', fontsize=4.5, ha='center', va='center', alpha=0.72)
    numeric_edge(v[4], v[0], data['z_ticks'], -9, 0)
    numeric_edge(v[6], v[2], data['z_ticks'], 8, 0)
    numeric_edge(v[5], v[6], data['x_ticks'], 0, 5)
    numeric_edge(v[1], v[2], data['x_ticks'], 0, -7)
    numeric_edge(v[0], v[1], data['y_ticks'], 0, -5)
    numeric_edge(v[5], v[4], data['y_ticks'], -7, 0)
    for (key, x, y) in [('x_axis', 651, 525), ('y_axis', 657, 31), ('y_axis', 180, 39), ('x_axis', 190, 511), ('z_axis', 137, 267), ('z_axis', 904, 270)]:
        placements.append({'key': key, 'x': x / W, 'y': y / H, 'size': 7, 'max_width': 0.055, 'anchor': 'center'})
    f = data['filaments']
    rng = np.random.RandomState(f['seed'])
    s = np.array(data['plume_sections'])
    fan = data['fan']
    for k in range(fan['count']):
        upper = k % 2 == 0
        xr = fan['upper_target_x'] if upper else fan['lower_target_x']
        yr = fan['upper_target_y'] if upper else fan['lower_target_y']
        t = np.linspace(0, 1, 70)
        tx = rng.uniform(*xr)
        ty = rng.uniform(*yr)
        x = fan['source'][0] + (tx - fan['source'][0]) * t + 12 * np.sin(t * np.pi) * rng.uniform(-1, 1)
        y = fan['source'][1] + (ty - fan['source'][1]) * t
        ax.plot(x, y, color=fan['color'], alpha=fan['opacity'], lw=1, zorder=2)
    for k in range(f['count']):
        start = rng.uniform(20, 490)
        length = rng.uniform(18, 145)
        y = np.linspace(start, min(start + length, 546), f['points'])
        center = np.interp(y, s[:, 0], s[:, 1])
        width = np.interp(y, s[:, 0], s[:, 2])
        u = rng.uniform(-1, 1)
        phase = rng.uniform(0, 2 * np.pi)
        x = center + width * u
        x += 13 * np.sin(y / 27 + phase) + 8 * np.sin(y / 13 + u * 7)
        for (amp, freq) in zip(f['oscillation_amplitudes'], f['oscillation_frequencies']):
            x += amp * np.sin(y / freq * 20 + phase * freq / 33)
        y = y + 5 * np.sin(x / 13 + phase)
        color = f['blue_colors'][rng.randint(len(f['blue_colors']))]
        ax.plot(x, y, color=color, alpha=f['opacity'], lw=f['linewidth'], zorder=3)
    for k in range(f['red_count']):
        cy = rng.uniform(28, 530)
        center = np.interp(cy, s[:, 0], s[:, 1])
        width = np.interp(cy, s[:, 0], s[:, 2])
        u = rng.uniform(-1, 1)
        if abs(u) < 0.28 and 170 < cy < 395:
            continue
        cx = center + width * u + 13 * np.sin(cy / 27 + u * 6)
        t = np.linspace(0, 1, 9)
        length = rng.uniform(1, 8)
        ang = rng.uniform(0, 2 * np.pi)
        x = cx + length * t * np.cos(ang) + 1.5 * np.sin(t * 4)
        y = cy + length * t * np.sin(ang)
        ax.plot(x, y, color=f['red_colors'][rng.randint(4)], lw=rng.uniform(0.35, 0.85), alpha=f['red_opacity'], zorder=4)
    for (x, y, r) in data['stars']:
        ax.add_patch(Circle((x, y), r, facecolor='#30383b', edgecolor='#626866', lw=0.5, zorder=8))
        for j in range(12, 0, -1):
            q = j / 12
            ax.add_patch(Circle((x - r * 0.2, y - r * 0.23), r * 0.8 * q, facecolor=(0.28 + 0.15 * (1 - q), 0.31 + 0.14 * (1 - q), 0.29 + 0.13 * (1 - q)), edgecolor='none', zorder=9))
    placements.extend([{'key': 'companion', 'x': 0.298, 'y': 0.479, 'size': 14, 'max_width': 0.145, 'anchor': 'right'}, {'key': 'primary', 'x': 0.669, 'y': 0.456, 'size': 14, 'max_width': 0.13, 'anchor': 'left'}])
    cb = fig.add_axes([0.761, 0.324, 0.011, 0.332])
    gradient = np.linspace(data['color_limits'][0], data['color_limits'][1], 512).reshape(-1, 1)
    cb.imshow(gradient, origin='lower', aspect='auto', cmap='coolwarm', extent=[0, 1, 0, 2000])
    cb.set_xticks([])
    cb.set_yticks(data['color_ticks'])
    cb.yaxis.tick_right()
    cb.tick_params(axis='y', labelsize=6, colors='#dadce5', length=3, width=0.4, pad=7)
    for spine in cb.spines.values():
        spine.set_visible(False)
    placements.append({'key': 'color_axis', 'x': 0.812, 'y': 0.509, 'size': 14, 'max_width': 0.31, 'rotation': 90, 'anchor': 'center'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_9a96df9bdff65eb0da44fcfe09180ca22b2cb8bc53601bd4c3ee368cea9f136e'
LANGUAGE = 'ar'
DATA = {'canvas': [1024, 576], 'background': '#53576e', 'box_vertices': [[156, 69], [282, 7], [885, 62], [638, 98], [165, 459], [285, 573], [874, 470], [635, 405]], 'box_edges': [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]], 'stars': [[321, 276, 7.4], [671, 262, 6.4]], 'color_limits': [0, 2000], 'color_ticks': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 'z_ticks': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], 'x_ticks': [-4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], 'y_ticks': [-3, -2, -1, 0, 1, 2, 3], 'plume_sections': [[22, 414, 24], [35, 455, 92], [55, 452, 123], [80, 423, 146], [110, 421, 119], [140, 429, 103], [165, 443, 116], [195, 421, 108], [225, 407, 100], [253, 387, 114], [278, 380, 111], [303, 390, 117], [329, 411, 122], [355, 432, 134], [380, 438, 135], [405, 440, 162], [431, 446, 166], [456, 459, 163], [479, 460, 162], [501, 451, 129], [524, 426, 90], [546, 385, 28]], 'filaments': {'count': 1450, 'points': 135, 'seed': 39, 'opacity': 0.115, 'linewidth': 0.55, 'blue_colors': ['#4164d5', '#537de5', '#6696ed', '#80b4f0', '#a8cced'], 'oscillation_amplitudes': [8, 4, 2.4], 'oscillation_frequencies': [33, 91, 179], 'red_count': 940, 'red_colors': ['#f62f32', '#d63149', '#ff715e', '#fa9c87'], 'red_opacity': 0.5}, 'fan': {'count': 340, 'source': [321, 277], 'upper_target_x': [360, 483], 'upper_target_y': [28, 174], 'lower_target_x': [352, 455], 'lower_target_y': [402, 516], 'color': '#557ae1', 'opacity': 0.09}}
LABELS = {'x_axis': 'المحور X', 'y_axis': 'المحور Y', 'z_axis': 'المحور Z', 'companion': 'النجم المرافق', 'primary': 'النجم الرئيسي', 'color_axis': 'max(dEdt) (GeV/ساعة)'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
