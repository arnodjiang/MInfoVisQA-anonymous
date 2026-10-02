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
    placements = [{'key': 'date', 'x': 0.505, 'y': 0.032, 'size': 25, 'max_width': 0.45, 'anchor': 'center'}]
    colors = data['colors']
    x = np.array(data['hours'])
    for (i, p) in enumerate(data['panels']):
        (row, col) = divmod(i, 4)
        left = 0.071 + col * 0.233
        bottom = 0.535 if row == 0 else 0.112
        (width, height) = (0.194, 0.353)
        ax = fig.add_axes([left, bottom, width, height])
        low = np.array(p['lower'])
        high = np.array(p['upper'])
        f = np.array(p['forecast'])
        ax.fill_between(x, low, high, color=colors['interval'], alpha=0.88, linewidth=0)
        wave = np.array(data['scenario_wave'])
        for (j, offset) in enumerate(data['scenario_offsets']):
            spread = np.where(offset >= 0, high - f, f - low)
            y = f + offset * spread + np.roll(wave, j * 3) * (high - low) * 0.48
            ax.plot(x, y, color=colors['scenario'], ls='--', lw=0.65, alpha=0.8)
        ax.plot(x, p['actual'], color=colors['actual'], lw=2.1)
        ax.plot(x, f, color=colors['forecast'], lw=2.1)
        ax.set_xlim(-1, 24)
        ax.set_ylim(p['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(p['yticks'])
        ax.tick_params(axis='both', labelsize=8.5, length=2.3, pad=2, color='#777777')
        for spine in ax.spines.values():
            spine.set_color('#888888')
            spine.set_linewidth(0.8)
        placements.append({'key': p['key'], 'x': left + width / 2, 'y': 1 - bottom - height - 0.018, 'size': 16, 'max_width': 0.22, 'anchor': 'center'})
    leg = fig.add_axes([0.28, 0.012, 0.53, 0.059])
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.axis('off')
    leg.add_patch(Rectangle((0, 0.03), 1, 0.9, facecolor='white', edgecolor='#dddddd', linewidth=0.7))
    entries = [('interval', 0.012, 0.071), ('scenario', 0.26, 0.32), ('actual', 0.535, 0.598), ('forecast', 0.755, 0.818)]
    for (key, a, b) in entries:
        if key == 'interval':
            leg.add_patch(Rectangle((a, 0.31), 0.05, 0.34, facecolor=colors[key], edgecolor='none'))
        else:
            leg.plot([a, a + 0.052], [0.48, 0.48], color=colors[key], lw=0.8 if key == 'scenario' else 2.1, ls='--' if key == 'scenario' else '-')
        placements.append({'key': key, 'x': 0.28 + 0.53 * b, 'y': 0.959, 'size': 16, 'max_width': 0.11, 'anchor': 'left'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_96903d38b54e7902b140912023f2304b13015daf72e95b1dbe3d93018526a883'
LANGUAGE = 'pl'
DATA = {'canvas': [1200, 600], 'hours': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'colors': {'interval': '#add7e0', 'scenario': '#989898', 'actual': '#ff0000', 'forecast': '#0000ee'}, 'panels': [{'key': 'coast', 'ylim': [8700, 20300], 'yticks': [10000, 12000, 14000, 16000, 18000, 20000], 'forecast': [10500, 10300, 10400, 10900, 11700, 12000, 12300, 13000, 13900, 14700, 15700, 16500, 17000, 17300, 17500, 17300, 16700, 15900, 15500, 14900, 13600, 12300, 11700, 11000], 'actual': [9400, 9250, 9450, 10000, 10600, 10900, 11400, 11900, 12700, 13500, 14100, 14700, 14900, 14800, 14500, 13600, 13100, 12700, 12500, 12200, 11600, 10800, 10100, 9700], 'lower': [9700, 9600, 9900, 10100, 10700, 10800, 11700, 12300, 12900, 13600, 14400, 14900, 15300, 15400, 15300, 15200, 14700, 14200, 13800, 13100, 12300, 11500, 10700, 9900], 'upper': [11700, 11500, 11600, 12200, 12800, 12800, 13400, 14300, 15300, 16500, 17500, 17900, 18500, 18800, 18800, 18100, 17400, 16800, 16600, 15600, 14600, 13600, 12600, 12100]}, {'key': 'east', 'ylim': [850, 2650], 'yticks': [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600], 'forecast': [1190, 1175, 1185, 1230, 1330, 1430, 1510, 1590, 1690, 1830, 1990, 2110, 2180, 2210, 2220, 2180, 2150, 2100, 2040, 1950, 1700, 1540, 1410, 1320], 'actual': [1150, 1140, 1160, 1200, 1280, 1360, 1430, 1490, 1560, 1660, 1780, 1900, 1930, 1960, 1980, 1940, 1870, 1810, 1760, 1740, 1580, 1430, 1320, 1240], 'lower': [950, 945, 970, 1050, 1130, 1210, 1270, 1340, 1450, 1570, 1710, 1730, 1770, 1830, 1840, 1800, 1700, 1600, 1610, 1540, 1350, 1190, 1090, 1130], 'upper': [1450, 1430, 1460, 1500, 1630, 1700, 1760, 1800, 1940, 2140, 2310, 2420, 2470, 2500, 2550, 2520, 2540, 2420, 2360, 2290, 2130, 1870, 1650, 1580]}, {'key': 'far_west', 'ylim': [2350, 3500], 'yticks': [2600, 2800, 3000, 3200, 3400], 'forecast': [2720, 2690, 2670, 2690, 2750, 2780, 2800, 2820, 2850, 2890, 2920, 2950, 3030, 3110, 3140, 3145, 3140, 3090, 3070, 3040, 2940, 2840, 2770, 2730], 'actual': [2590, 2550, 2520, 2540, 2600, 2660, 2660, 2650, 2690, 2740, 2810, 2910, 2960, 2990, 3030, 3050, 3080, 3060, 3050, 3030, 2870, 2790, 2730, 2690], 'lower': [2440, 2440, 2420, 2410, 2520, 2540, 2540, 2640, 2620, 2610, 2700, 2800, 2870, 2840, 2820, 2800, 2750, 2710, 2710, 2680, 2610, 2470, 2470, 2440], 'upper': [3000, 2960, 2950, 2980, 3060, 3080, 3100, 3090, 3160, 3240, 3240, 3310, 3380, 3370, 3430, 3450, 3430, 3450, 3410, 3370, 3330, 3240, 3100, 3050]}, {'key': 'north', 'ylim': [530, 1620], 'yticks': [600, 800, 1000, 1200, 1400, 1600], 'forecast': [810, 790, 780, 800, 860, 900, 950, 1030, 1070, 1110, 1160, 1220, 1290, 1320, 1300, 1270, 1230, 1200, 1170, 1130, 1050, 970, 900, 865], 'actual': [700, 685, 695, 720, 775, 795, 835, 950, 1020, 1050, 1090, 1110, 1150, 1190, 1240, 1270, 1200, 1110, 1060, 1000, 920, 860, 800, 755], 'lower': [620, 595, 575, 610, 650, 665, 770, 845, 865, 970, 985, 1020, 1040, 1080, 1090, 1040, 1060, 1100, 1030, 1000, 910, 825, 640, 660], 'upper': [1060, 1000, 980, 1000, 1090, 1160, 1200, 1280, 1360, 1390, 1420, 1470, 1520, 1580, 1550, 1520, 1470, 1450, 1400, 1370, 1330, 1280, 1140, 1110]}, {'key': 'north_central', 'ylim': [7000, 20900], 'yticks': [8000, 10000, 12000, 14000, 16000, 18000, 20000], 'forecast': [9300, 9150, 9400, 10100, 11100, 11600, 12200, 12700, 13600, 14600, 15600, 16400, 17200, 17800, 17900, 17700, 17100, 16600, 16400, 15800, 14400, 13100, 12000, 11100], 'actual': [9200, 9100, 9400, 10100, 11000, 11500, 12100, 12800, 14000, 15100, 16000, 16900, 17400, 17900, 18200, 18200, 17800, 17100, 16600, 15700, 14500, 13200, 12000, 10900], 'lower': [7900, 7700, 7700, 8300, 9500, 9900, 10500, 11200, 12100, 13100, 13800, 14100, 14500, 14500, 14300, 14300, 14200, 14400, 14200, 13100, 11800, 11000, 10300, 9500], 'upper': [10800, 10600, 11000, 11600, 12800, 13500, 14200, 15000, 15900, 17400, 18400, 19000, 20200, 20200, 20200, 20100, 19800, 18900, 18900, 18500, 16900, 15400, 14500, 13400]}, {'key': 'south', 'ylim': [2370, 5700], 'yticks': [2500, 3000, 3500, 4000, 4500, 5000, 5500], 'forecast': [3060, 3020, 2980, 3050, 3260, 3480, 3590, 3720, 4000, 4270, 4510, 4740, 4890, 4970, 4990, 4900, 4740, 4550, 4490, 4400, 4110, 3740, 3420, 3210], 'actual': [2820, 2690, 2630, 2690, 2820, 2910, 2990, 3040, 3230, 3520, 3820, 4050, 4250, 4400, 4550, 4670, 4630, 4440, 4250, 4190, 3850, 3500, 3210, 3010], 'lower': [2580, 2540, 2600, 2670, 2920, 3030, 3090, 3280, 3630, 3970, 4100, 4250, 4260, 4290, 4300, 4280, 4160, 3930, 3790, 3710, 3380, 3060, 2770, 2810], 'upper': [3430, 3400, 3400, 3520, 3710, 3870, 3970, 4150, 4450, 4770, 5050, 5250, 5380, 5430, 5460, 5550, 5410, 5260, 5120, 5020, 4840, 4470, 4070, 3900]}, {'key': 'south_central', 'ylim': [3900, 11100], 'yticks': [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000], 'forecast': [5270, 5100, 5170, 5390, 5970, 6180, 6380, 6650, 7040, 7510, 8080, 8600, 9060, 9460, 9630, 9610, 9240, 8910, 8630, 8280, 7420, 6720, 6220, 5830], 'actual': [4790, 4630, 4760, 5130, 5710, 5910, 6100, 6330, 6620, 6890, 7250, 7660, 8080, 8520, 8860, 9070, 8860, 8450, 8200, 7790, 6970, 6270, 5820, 5460], 'lower': [4320, 4180, 4440, 4770, 5370, 5580, 5760, 5990, 6190, 6600, 7090, 7610, 7970, 8080, 8160, 8090, 8010, 7660, 7480, 7020, 6280, 5660, 5270, 5100], 'upper': [5850, 5740, 5880, 6170, 6700, 6870, 7000, 7300, 7780, 8340, 8980, 9460, 9950, 10290, 10520, 10620, 10800, 10720, 10160, 9920, 9060, 8060, 7510, 6930]}, {'key': 'west', 'ylim': [510, 1920], 'yticks': [600, 800, 1000, 1200, 1400, 1600, 1800], 'forecast': [900, 880, 865, 890, 965, 1000, 1020, 1060, 1080, 1140, 1200, 1250, 1290, 1330, 1370, 1380, 1350, 1320, 1300, 1290, 1210, 1120, 1030, 950], 'actual': [960, 955, 960, 980, 1050, 1090, 1100, 1140, 1150, 1180, 1230, 1280, 1390, 1490, 1530, 1540, 1540, 1510, 1480, 1450, 1360, 1230, 1160, 1080], 'lower': [630, 590, 570, 610, 700, 710, 710, 730, 760, 790, 800, 890, 1020, 1100, 1100, 1060, 1000, 1040, 1010, 1020, 950, 860, 810, 730], 'upper': [1250, 1240, 1240, 1270, 1320, 1360, 1450, 1500, 1650, 1650, 1650, 1690, 1800, 1860, 1760, 1750, 1700, 1660, 1650, 1640, 1620, 1520, 1470, 1380]}], 'xticks': [0, 5, 10, 15, 20], 'scenario_offsets': [-0.68, -0.44, -0.22, 0.06, 0.29, 0.52, 0.78], 'scenario_wave': [0.05, -0.12, 0.19, -0.18, 0.08, 0.22, -0.09, 0.15, -0.23, 0.06, 0.21, -0.14, 0.09, 0.24, -0.07, 0.12, -0.22, 0.17, -0.11, 0.26, -0.16, 0.08, -0.19, 0.13]}
LABELS = {'date': '20180521', 'coast': 'Wybrzeże', 'east': 'Wschód', 'far_west': 'Daleki zachód', 'north': 'Północ', 'north_central': 'Północny środek', 'south': 'Południe', 'south_central': 'Południowy środek', 'west': 'Zachód', 'interval': '1%-99%', 'scenario': 'scenariusz', 'actual': 'wartość rzeczywista', 'forecast': 'prognoza'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
