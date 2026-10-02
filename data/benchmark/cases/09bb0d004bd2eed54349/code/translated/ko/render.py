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
    placements = [{'key': 'title', 'x': 0.5, 'y': 0.022, 'size': 20, 'max_width': 0.9, 'anchor': 'center'}]
    c = data['colors']

    def model(t, p):
        z = 1 + p['growth'] * (np.asarray(t) - p['th'])
        y = np.exp(np.minimum(z, 0))
        for i in range(40):
            delta = (y + np.log(y) - z) / (1 + 1 / y)
            y = np.maximum(y - delta, y * 0.1)
        return p['qh'] * y

    def scientific(key, x, y, size, anchor='left', growth=False):
        s = labels[key]
        if growth:
            s = s.split('%')[0] + '%/'
        for (a, b) in [('ᵤ', '_{u}'), ('ₕ', '_{h}'), ('¹³', '^{13}'), ('⁷', '^{7}'), ('·', '\\cdot '), ('$', '\\$'), ('%', '\\%')]:
            s = s.replace(a, b)
        artist = fig.text(x, 1 - y, '$' + s + '$', ha=anchor, va='center', fontsize=size * 0.72, math_fontfamily='dejavusans')
        if growth:
            fig.canvas.draw()
            right = artist.get_window_extent(fig.canvas.get_renderer()).x1 / W
            placements.append({'key': 'year', 'x': right, 'y': y, 'size': size, 'max_width': 0.14, 'anchor': 'left'})
    for p in data['panels']:
        (l, b, w, h) = p['axis']
        ax = fig.add_axes(p['axis'])
        ax.set_yscale('log')
        ax.set_xlim(data['year_limits'])
        ax.set_ylim(p['limits'])
        ax.set_xticks(data['year_ticks'])
        ax.set_yticks(p['log_ticks'])
        ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12, length=4)
        ax.tick_params(which='minor', length=2)
        ax.minorticks_on()
        t = np.linspace(data['year_limits'][0], data['year_limits'][1], data['model_sample_count'])
        yd = model(p['years'], p) * np.array(p['ratios'])
        ax.plot(p['years'], yd, color=c['data'], lw=1.5, ls='-.', zorder=3)
        ax.plot(t, model(t, p), color=c['halo'], lw=3, alpha=0.45, zorder=4)
        ax.plot(t, model(t, p), color=c['model'], lw=1.3, zorder=5)
        top = ax.secondary_xaxis('top', functions=(lambda x, g=p['growth'], th=p['th']: g * (x - th), lambda x, g=p['growth'], th=p['th']: x / g + th))
        top.set_xticks(p['top_ticks'])
        top.tick_params(direction='in', labelsize=12, pad=3)
        placements.extend([{'key': p['header'], 'x': l + w / 2, 'y': 0.092, 'size': 20, 'max_width': w + 0.02, 'anchor': 'center'}, {'key': p['ylabel'], 'x': l - 0.038, 'y': 1 - b - h / 2, 'size': 18, 'max_width': 0.32, 'rotation': 90, 'anchor': 'center'}, {'key': 'year', 'x': l + w / 2, 'y': 0.971, 'size': 18, 'max_width': 0.25, 'anchor': 'center'}])
        scientific('top_axis', l + w / 2, 0.13, 17, 'center')
        for (j, key) in enumerate(['data', 'model']):
            yy = 0.916 - j * 0.037
            ax.plot([0.06, 0.165], [yy, yy], transform=ax.transAxes, color=c['data'] if j == 0 else c['model'], ls='-.' if j == 0 else '-', lw=1.5, clip_on=False)
            placements.append({'key': key, 'x': l + w * 0.183, 'y': 1 - (b + h * yy), 'size': 14, 'max_width': w * 0.34, 'anchor': 'left'})
        for (j, key) in enumerate(p['annotations']):
            scientific(key, l + w * 0.05, 1 - (b + h * (0.81 - j * 0.058)), 14, growth=j == 0)
        ia = fig.add_axes(p['inset'])
        ia.set_xlim(data['year_limits'])
        ia.set_ylim(data['ratio_limits'])
        ia.plot(p['years'], p['ratios'], color=c['halo'], lw=3, alpha=0.5)
        ia.plot(p['years'], p['ratios'], color=c['model'], lw=1.2)
        ia.set_xticks(data['inset_ticks'])
        ia.set_yticks(data['ratio_ticks'])
        ia.set_yticklabels(['%.1f' % v for v in data['ratio_ticks']])
        ia.minorticks_on()
        ia.tick_params(which='both', direction='in', top=True, right=True, labelsize=12, pad=2, length=4)
        ia.tick_params(which='minor', length=2)
        (il, ib, iw, ih) = p['inset']
        ia.plot([0.26, 0.46], [0.775, 0.775], transform=ia.transAxes, color=c['model'], lw=1.5)
        placements.append({'key': 'ratio', 'x': il + iw * 0.493, 'y': 1 - (ib + ih * 0.775), 'size': 12, 'max_width': iw * 0.48, 'anchor': 'left'})
        placements.append({'key': 'year', 'x': il + iw / 2, 'y': 1 - ib + 0.052, 'size': 18, 'max_width': iw, 'anchor': 'center'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_522a966191fea9ee1edebfff738fee7a4305b08fecdc56167f85fd15501186af'
LANGUAGE = 'ko'
DATA = {'canvas': [1200, 640], 'year_limits': [1790, 2020], 'year_ticks': [1800, 1850, 1900, 1950, 2000], 'inset_ticks': [1800, 1900, 2000], 'ratio_limits': [0, 2], 'ratio_ticks': [0, 0.5, 1, 1.5, 2], 'model_sample_count': 461, 'colors': {'model': '#c9ab24', 'halo': '#fff4aa', 'data': '#151515'}, 'panels': [{'axis': [0.05, 0.082, 0.438, 0.744], 'inset': [0.271, 0.158, 0.196, 0.224], 'header': 'population_header', 'ylabel': 'population_axis', 'annotations': ['population_growth', 'population_q', 'population_time'], 'growth': 0.034, 'qh': 96000000, 'th': 1914, 'limits': [2800000, 560000000], 'log_ticks': [10000000, 100000000], 'top_ticks': [-4, -3, -2, -1, 0, 1, 2, 3], 'years': [1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020], 'ratios': [1.061, 1.018, 1.006, 0.982, 0.971, 0.967, 0.98, 1.041, 1.018, 1.04, 1.031, 1.036, 1.019, 1.032, 1.001, 0.937, 0.932, 0.987, 1.009, 1.009, 1.008, 1.019, 1.047, 1.055]}, {'axis': [0.555, 0.082, 0.438, 0.744], 'inset': [0.775, 0.158, 0.196, 0.224], 'header': 'gdp_header', 'ylabel': 'gdp_axis', 'annotations': ['gdp_growth', 'gdp_q', 'gdp_time'], 'growth': 0.038, 'qh': 36000000000000, 'th': 2041, 'limits': [3700000000, 65000000000000], 'log_ticks': [10000000000, 100000000000, 1000000000000, 10000000000000], 'top_ticks': [-9, -8, -7, -6, -5, -4, -3, -2, -1], 'years': [1790, 1793, 1796, 1800, 1803, 1807, 1810, 1813, 1816, 1820, 1823, 1826, 1830, 1833, 1837, 1840, 1843, 1847, 1850, 1853, 1857, 1860, 1863, 1865, 1868, 1870, 1873, 1876, 1880, 1883, 1885, 1888, 1890, 1893, 1896, 1900, 1903, 1907, 1910, 1913, 1916, 1918, 1921, 1924, 1926, 1929, 1931, 1933, 1936, 1939, 1941, 1944, 1946, 1949, 1952, 1955, 1958, 1960, 1963, 1966, 1969, 1972, 1975, 1978, 1980, 1983, 1986, 1989, 1992, 1995, 1998, 2000, 2003, 2006, 2009, 2012, 2015, 2018, 2020], 'ratios': [0.83, 0.92, 1.04, 1.02, 1.03, 1.015, 0.93, 1.04, 1.035, 0.94, 0.93, 0.98, 0.95, 0.94, 1.04, 1.02, 0.93, 0.98, 1.08, 1.07, 1.17, 1.145, 1.17, 1.25, 1.105, 1.1, 1.19, 1.1, 1.31, 1.35, 1.27, 1.11, 1.24, 1.4, 1.4, 1.1, 1.26, 1.285, 1.09, 1.21, 1.13, 1.06, 0.95, 1.04, 1.01, 0.99, 0.79, 0.64, 0.72, 0.84, 1.04, 1.2, 1.0, 0.91, 0.98, 0.97, 0.92, 0.96, 1.04, 1.02, 1.05, 1.04, 1.0, 1.05, 1.025, 1.0, 1.05, 1.065, 1.05, 1.05, 1.1, 1.1, 1.015, 1.025, 0.99, 1.02, 1.055, 1.07, 1.04]}]}
LABELS = {'title': '미국 인구와 국내총생산, 1790--2020', 'population_header': '최적 적합 모형: 단일항 억제, k = 1', 'gdp_header': '최적 적합 모형: 단일항 억제, k = 1', 'top_axis': 'gᵤ(t - tₕ)', 'population_axis': '인구', 'gdp_axis': '국내총생산 ($)', 'year': '연도', 'data': '데이터', 'model': '모형', 'ratio': '데이터/모형', 'population_growth': 'gᵤ = 3.4%/년', 'population_q': 'Qₕ = 9.6·10⁷', 'population_time': 'tₕ = 1914', 'gdp_growth': 'gᵤ = 3.8%/년', 'gdp_q': 'Qₕ = $3.6·10¹³', 'gdp_time': 'tₕ = 2041'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
