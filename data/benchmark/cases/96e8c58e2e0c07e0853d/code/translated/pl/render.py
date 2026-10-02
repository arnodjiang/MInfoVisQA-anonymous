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

    def text(key, x, y, size, width, anchor='left', color='#777777'):
        placements.append({'key': key, 'x': x, 'y': y, 'size': size, 'max_width': width, 'anchor': anchor, 'color': color})
    fig.patches.append(Rectangle((0, 0.9745), 1, 0.0255, transform=fig.transFigure, facecolor='#288f8d', edgecolor='none'))
    text('brand', 0.063, 0.012, 11, 0.04, color='white')
    text('tracker', 0.099, 0.014, 9, 0.07, color='white')
    for (k, x) in [('harvard', 0.826), ('brown', 0.872), ('gates', 0.907)]:
        text(k, x, 0.012, 9, 0.055, color='white')
    text('title', 0.055, 0.083, 48, 0.88, color='#063d4a')
    text('subtitle', 0.055, 0.14, 35, 0.85, color='#585a58')
    ax = fig.add_axes([0.0945, 0.472, 0.718, 0.327])
    ax.set_xlim(data['xlim'])
    ax.set_ylim(data['ylim'])
    ax.set_yticks(data['yticks'])
    ax.set_yticklabels([str(v) + '%' for v in data['yticks']], fontsize=24, color='#7b7d7b')
    ax.tick_params(axis='both', length=0)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis='y', color='#dddddd', linestyle=(0, (4, 3)), linewidth=1.6)
    for (s, c) in zip(data['series'], data['colors']):
        p = np.array(s['points'])
        x = p[:, 0]
        y = p[:, 1]
        slopes = np.gradient(y, x)
        xx = []
        yy = []
        for i in range(len(x) - 1):
            t = np.linspace(0, 1, 7, endpoint=False)
            dx = x[i + 1] - x[i]
            xx.extend(x[i] + t * dx)
            yy.extend((2 * t ** 3 - 3 * t ** 2 + 1) * y[i] + (t ** 3 - 2 * t ** 2 + t) * dx * slopes[i] + (-2 * t ** 3 + 3 * t ** 2) * y[i + 1] + (t ** 3 - t ** 2) * dx * slopes[i + 1])
        xx.append(x[-1])
        yy.append(y[-1])
        ax.plot(xx, yy, color=c, lw=6.5, solid_capstyle='round')
        ax.scatter([x[-1]], [y[-1]], s=420, color=c, edgecolors='white', linewidths=2, zorder=5)
    ax.axvline(2173, color='#164e5c', lw=2.5, linestyle=(0, (4, 3)))

    def fx(x):
        return 0.0945 + (x - 270) / 2050 * 0.718
    for (x, k) in zip(data['tick_x'], data['tick_keys']):
        text(k, fx(x), 0.547, 27, 0.09, 'center')
    for (x, kd, kt, yd, yt) in data['events']:
        xf = fx(x)
        fig.lines.append(Line2D([xf, xf], [0.799, 1 - yt + 0.012], transform=fig.transFigure, color='#aaaaaa', linestyle=(0, (6, 4)), lw=1.6))
        text(kd, xf + 0.008, yd, 26, 0.12)
        text(kt, xf + 0.008, yt, 25, 0.145)
    text('date', 0.865, 0.194, 26, 0.12, color=data['colors'][0])
    for (s, c, y) in zip(data['series'], data['colors'], [0.245, 0.316]):
        fig.text(0.865, 1 - y, str(s['end']) + '%', fontsize=31, fontweight='bold', color=c, va='center')
        text(s['key'], 0.865, y + 0.035, 27, 0.12, color=c)
    text('footnote', 0.055, 0.917, 22, 0.87)
    text('updated', 0.055, 0.955, 22, 0.85)
    return finish(fig, labels, placements)
BASE_ID = 'qa_1d570978c1205b8145a25ee3e6c51a79f1ebb779f46eb4b8fb49b412999bdb23'
LANGUAGE = 'pl'
DATA = {'canvas': [2400, 1590], 'xlim': [270, 2320], 'ylim': [-68, 16], 'yticks': [0, -20, -40, -60], 'tick_x': [270, 710, 1010, 1293, 1590, 1878, 2320], 'tick_keys': ['jan', 'mar', 'apr', 'may', 'jun', 'jul', 'aug'], 'colors': ['#10576b', '#00a999'], 'series': [{'key': 'all', 'end': -23.3, 'points': [[270, -0.6], [282, 0.1], [298, 2.3], [312, 1.3], [326, 2], [339, 0.1], [353, 1.6], [365, 2.5], [377, 0.8], [390, 0], [405, -0.5], [422, -0.2], [433, -3.6], [443, -3.4], [450, 1], [466, 0.6], [478, 0], [488, 0.8], [501, 8.5], [514, 12.5], [530, 12.8], [542, 14.6], [555, 12], [568, 8], [580, 3], [585, -0.6], [594, 5.2], [605, 5.8], [615, 4.8], [633, 6.5], [644, 5.4], [652, 0], [665, 0], [680, 1.3], [690, 2.8], [700, 5.7], [708, 2.8], [718, 6.8], [728, 8.3], [742, 6.6], [754, 4.1], [767, 3.8], [777, 6.4], [788, 1.7], [801, 1.6], [813, 3.4], [825, 2.2], [835, 1.6], [846, -1.5], [856, -2.6], [865, -5.5], [875, -7], [886, -12], [894, -15.6], [902, -19.2], [913, -20.4], [928, -24.6], [938, -29], [951, -31.4], [960, -33.4], [970, -35.1], [984, -38.9], [1000, -38.3], [1011, -39], [1020, -40.8], [1034, -42.3], [1051, -41.2], [1070, -43.3], [1082, -43.1], [1097, -40.5], [1110, -40.4], [1124, -38.6], [1134, -36.6], [1144, -33.5], [1155, -32.4], [1169, -31.4], [1181, -29], [1191, -29.5], [1200, -26], [1213, -26.8], [1230, -23.6], [1243, -23.8], [1254, -24.6], [1264, -26.3], [1275, -26.5], [1287, -29.5], [1296, -29.9], [1307, -28.9], [1318, -28.4], [1332, -26.5], [1345, -24.3], [1357, -20.7], [1366, -18.7], [1379, -18], [1390, -17.9], [1400, -18.2], [1410, -17.1], [1422, -17.8], [1437, -17.7], [1447, -17.7], [1456, -15.5], [1474, -14.8], [1485, -12], [1500, -11], [1510, -11.1], [1522, -12.7], [1540, -11.8], [1554, -11.7], [1568, -10.8], [1580, -10.7], [1590, -7.5], [1600, -6.5], [1610, -9.5], [1626, -10.7], [1638, -11.7], [1648, -10.9], [1658, -13.5], [1674, -14.1], [1685, -12.8], [1701, -13.9], [1715, -14.8], [1730, -13], [1742, -10.9], [1754, -11.8], [1767, -10.7], [1779, -10.1], [1790, -11.9], [1807, -12.4], [1820, -13.4], [1829, -13], [1840, -15], [1855, -16.7], [1868, -17.1], [1879, -19.3], [1887, -19], [1896, -22], [1905, -16.7], [1919, -17], [1928, -17.6], [1937, -18.4], [1946, -17.9], [1961, -19.3], [1976, -19.6], [1989, -21.5], [2007, -20.7], [2023, -21.3], [2038, -21.3], [2048, -20.5], [2062, -20.9], [2077, -22.4], [2087, -22], [2103, -23], [2116, -23.7], [2132, -23], [2150, -22.8], [2173, -23.3]]}, {'key': 'hospitality', 'end': -53.8, 'points': [[270, -2.3], [282, -1.8], [292, 0], [309, 1.5], [319, 0], [329, 0.4], [338, -0.7], [352, 0], [365, 2.2], [376, 0.4], [391, 1.1], [405, 1], [423, 1.7], [434, -2.3], [441, -3.1], [449, 0], [464, -1], [479, -2.6], [490, -1.7], [501, 8], [509, 14.8], [522, 14], [536, 15.2], [546, 15.2], [556, 11.8], [566, 4], [575, -1.8], [584, -3.6], [596, -2.5], [614, -2.5], [624, -1.2], [634, 0.3], [650, -2.4], [660, -1.7], [675, -1.5], [689, 0], [698, 1.9], [710, -0.2], [722, 0.6], [736, 0], [747, -1.1], [758, -3.5], [768, -4.1], [779, -1.8], [793, -2.5], [808, -2.8], [816, -4], [827, -10.3], [836, -15.9], [845, -20.6], [853, -22.7], [862, -28], [874, -34], [884, -40], [892, -46], [901, -52.5], [910, -58], [920, -61.1], [933, -62.7], [948, -65.3], [960, -66.2], [976, -66.3], [997, -67], [1012, -65.3], [1028, -65.5], [1046, -65.6], [1063, -65.3], [1074, -66.2], [1084, -66.1], [1098, -64.4], [1110, -65], [1125, -64.6], [1133, -64.5], [1145, -63], [1159, -62.7], [1174, -61.3], [1187, -60.4], [1198, -58.6], [1216, -58.8], [1228, -57.4], [1241, -55.7], [1253, -55.9], [1265, -57], [1274, -56.2], [1283, -56.8], [1293, -56.1], [1305, -53.9], [1317, -52.2], [1325, -51.4], [1335, -49.9], [1345, -49.6], [1355, -47.4], [1370, -45.1], [1380, -41.9], [1389, -41.9], [1398, -44.5], [1410, -43.5], [1425, -43.2], [1437, -43.2], [1447, -44.1], [1458, -42.6], [1468, -40], [1483, -37.5], [1494, -34.2], [1505, -32.1], [1520, -31.1], [1533, -31.3], [1545, -32.3], [1560, -32.3], [1577, -31.5], [1594, -28.4], [1606, -29.2], [1621, -30], [1634, -30.5], [1646, -29.3], [1662, -29.2], [1674, -29.6], [1688, -29], [1703, -29.1], [1721, -28.5], [1731, -29.4], [1747, -29.5], [1762, -29.2], [1774, -29.7], [1782, -28.8], [1791, -30.2], [1803, -30], [1815, -30.1], [1826, -31.8], [1836, -34.4], [1848, -39.1], [1860, -40.1], [1869, -40.4], [1878, -43], [1887, -44.1], [1898, -47.1], [1908, -47.5], [1920, -45.8], [1929, -45.8], [1937, -47.8], [1947, -47.9], [1959, -49.3], [1972, -51.5], [1986, -53.9], [1998, -55], [2010, -55.4], [2028, -55.5], [2046, -54.8], [2062, -55.1], [2078, -55.4], [2096, -54.6], [2113, -55.2], [2130, -55], [2150, -54.5], [2173, -53.8]]}], 'events': [[318, 'event1date', 'event1', 0.598, 0.624], [901, 'event2date', 'event2', 0.598, 0.624], [1016, 'event3date', 'event3', 0.672, 0.698], [1140, 'event4date', 'event4', 0.746, 0.772], [1284, 'event5date', 'event5', 0.598, 0.624], [1293, 'event6date', 'event6', 0.82, 0.846], [1829, 'event7date', 'event7', 0.598, 0.624]]}
LABELS = {'brand': 'OPPORTUNITY\nINSIGHTS', 'tracker': 'MONITOR GOSPODARCZY', 'harvard': 'UNIWERSYTET\nHARVARDA', 'brown': 'BROWN', 'gates': 'Fundacja BILLA I MELINDY\nGATESÓW', 'title': 'Procentowa zmiana przychodów małych firm*', 'subtitle': 'W Teksasie na dzień 01 sierpnia 2020 łączne przychody małych firm spadły o 23.3%\nw porównaniu ze styczniem 2020.', 'date': '01 sie 2020', 'all': 'Wszystkie', 'hospitality': 'Rekreacja i\nhotelarstwo z gastronomią', 'jan': '15 sty', 'mar': '1 mar', 'apr': '1 kwi', 'may': '1 maj', 'jun': '1 cze', 'jul': '1 lip', 'aug': '16 sie', 'event1date': '20 sty', 'event1': 'Pierwszy przypadek COVID-19 w USA', 'event2date': '21 mar', 'event2': 'Zamknięcie szkół publicznych w Teksasie', 'event3date': '02 kwi', 'event3': 'Nakaz pozostania w domu w Teksasie', 'event4date': '15 kwi', 'event4': 'Początek wypłat\nwsparcia gospodarczego', 'event5date': '30 kwi', 'event5': 'Koniec nakazu pozostania w domu\nw Teksasie', 'event6date': '01 maj', 'event6': 'Ponowne otwarcie wybranych firm (Teksas)', 'event7date': '26 cze', 'event7': 'Ponowne zamknięcie wybranych firm (Teksas)', 'footnote': '*Zmiana przychodów netto małych firm, indeksowana względem okresu 4-31 stycznia 2020 i skorygowana sezonowo. Ta seria\nopiera się na danych Womply.', 'updated': 'ostatnia aktualizacja: 14 sierpnia 2020       następna aktualizacja przewidywana: 18 sierpnia 2020'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
