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
    return render_table(data, labels)
BASE_ID = 'qa_aff23a91fba2f4881d8c13e4505564292f9ac47206247173737afeddc7d88cae'
LANGUAGE = 'zh'
DATA = {'rows': [[{'text': 'No. in\\nseason', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_000'}, {'text': 'No. in\\nseries', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Title', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Canadian airdate', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_003'}, {'text': 'US airdate', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_004'}, {'text': 'Production code', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_005'}], [{'text': '1–2', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '318–319', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Summertime', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_002'}, {'text': 'July\xa011,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_003'}, {'text': 'July\xa011,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_004'}, {'text': '1301 & 1302', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '3', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '320', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'All I Wanna Do', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_002'}, {'text': 'July\xa018,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_003'}, {'text': 'July\xa018,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_004'}, {'text': '1303', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '4', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '321', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'My Own Worst Enemy', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_002'}, {'text': 'July\xa025,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_003'}, {'text': 'July\xa025,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_004'}, {'text': '1304', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '5', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '322', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'About A Girl', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_002'}, {'text': 'August\xa01,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_003'}, {'text': 'August\xa01,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_004'}, {'text': '1305', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '6', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '323', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Cannonball', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_002'}, {'text': 'August\xa08,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_003'}, {'text': 'August\xa08,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_004'}, {'text': '1306', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '7', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '324', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Honey', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_002'}, {'text': 'August\xa015,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_003'}, {'text': 'August\xa015,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_004'}, {'text': '1307', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '8', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '325', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Young Forever', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_002'}, {'text': 'August\xa022,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_003'}, {'text': 'August\xa022,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_004'}, {'text': '1308', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '9', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '326', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'This Is How We Do It', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_002'}, {'text': 'October\xa03,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_003'}, {'text': 'October\xa03,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_004'}, {'text': '1309', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '10', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '327', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'You Got Me', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_002'}, {'text': 'October\xa010,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_003'}, {'text': 'October\xa010,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_004'}, {'text': '1310', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '11', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '328', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'You Oughta Know', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_002'}, {'text': 'October\xa017,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_003'}, {'text': 'October\xa017,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_004'}, {'text': '1311', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '12', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '329', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': "Everything You've Done Wrong", 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_002'}, {'text': 'October\xa024,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_003'}, {'text': 'October\xa024,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_004'}, {'text': '1312', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '13', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '330', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Who Do You Think You Are', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_002'}, {'text': 'October\xa031,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_003'}, {'text': 'October\xa031,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_004'}, {'text': '1313', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '14', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '331', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Barely Breathing', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_013_002'}, {'text': 'November\xa07,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_013_003'}, {'text': 'November\xa07,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_013_004'}, {'text': '1314', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '15', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '332', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Black Or White', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_014_002'}, {'text': 'November\xa014,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_014_003'}, {'text': 'November\xa014,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_014_004'}, {'text': '1315', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '16', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '333', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Spiderwebs', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_015_002'}, {'text': 'November\xa021,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_015_003'}, {'text': 'November\xa021,\xa02013', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_015_004'}, {'text': '1316', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '17', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '334', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'The World I Know', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_016_002'}, {'text': 'January\xa028,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_016_003'}, {'text': 'January\xa028,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_016_004'}, {'text': '1317', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '18', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '335', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Better Man', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_017_002'}, {'text': 'February\xa04,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_017_003'}, {'text': 'February\xa04,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_017_004'}, {'text': '1318', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '19', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '336', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Dig Me Out', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_018_002'}, {'text': 'February\xa011,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_018_003'}, {'text': 'February\xa011,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_018_004'}, {'text': '1319', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '20', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '337', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Power to the People', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_019_002'}, {'text': 'February\xa018,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_019_003'}, {'text': 'February\xa018,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_019_004'}, {'text': '1320', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '21', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '338', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'No Surprises', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_020_002'}, {'text': 'February\xa025,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_020_003'}, {'text': 'February\xa025,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_020_004'}, {'text': '1321', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '22', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '339', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Basket Case', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_021_002'}, {'text': 'March\xa04,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_021_003'}, {'text': 'March\xa04,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_021_004'}, {'text': '1322', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '23–24', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '340–341', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Unbelievable', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_022_002'}, {'text': 'March\xa011,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_022_003'}, {'text': 'March\xa011,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_022_004'}, {'text': '1323 & 1324', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '25', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '342', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': "What It's Like", 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_023_002'}, {'text': 'March\xa018,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_023_003'}, {'text': 'March\xa018,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_023_004'}, {'text': '1325', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '26', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '343', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Close to Me', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_024_002'}, {'text': 'March\xa025,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_024_003'}, {'text': 'March\xa025,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_024_004'}, {'text': '1326', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '27', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '344', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Army of Me', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_025_002'}, {'text': 'April\xa01,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_025_003'}, {'text': 'April\xa01,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_025_004'}, {'text': '1327', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '28', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '345', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Everything Is Everything', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_026_002'}, {'text': 'April\xa08,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_026_003'}, {'text': 'April\xa08,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_026_004'}, {'text': '1328', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '29', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '346', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Sparks Will Fly\xa0Part One', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_027_002'}, {'text': 'April\xa015,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_027_003'}, {'text': 'April\xa015,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_027_004'}, {'text': '1329', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '30', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '347', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Sparks Will Fly\xa0Part Two', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_028_002'}, {'text': 'April\xa022,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_028_003'}, {'text': 'April\xa022,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_028_004'}, {'text': '1330', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '31', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '348', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'You Are Not Alone', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_029_002'}, {'text': 'June\xa03,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_029_003'}, {'text': 'June\xa03,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_029_004'}, {'text': '1331', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '32', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '349', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Enjoy The Silence', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_030_002'}, {'text': 'June\xa010,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_030_003'}, {'text': 'June\xa010,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_030_004'}, {'text': '1332', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '33', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '350', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'How Bizarre', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_031_002'}, {'text': 'June\xa017,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_031_003'}, {'text': 'June\xa017,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_031_004'}, {'text': '1333', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '34', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '351', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'My Hero', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_032_002'}, {'text': 'June\xa024,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_032_003'}, {'text': 'June\xa024,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_032_004'}, {'text': '1334', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '35', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '352', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Hypnotize', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_033_002'}, {'text': 'July\xa01,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_033_003'}, {'text': 'July\xa01,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_033_004'}, {'text': '1335', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '36', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '353', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Out Of My Head', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_034_002'}, {'text': 'July\xa08,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_034_003'}, {'text': 'July\xa08,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_034_004'}, {'text': '1336', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '37', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '354', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'TBA', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_035_002'}, {'text': 'July\xa015,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_035_003'}, {'text': 'July\xa015,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_035_004'}, {'text': '1337', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '38', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '355', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'TBA', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_036_002'}, {'text': 'July\xa022,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_036_003'}, {'text': 'July\xa022,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_036_004'}, {'text': '1338', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '39', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '356', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Thunderstruck\xa0Part One', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_037_002'}, {'text': 'July\xa029,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_037_003'}, {'text': 'July\xa029,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_037_004'}, {'text': '1339', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '357', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Thundestruck\xa0Part Two', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_038_002'}, {'text': 'July\xa029,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_038_003'}, {'text': 'July\xa029,\xa02014', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_038_004'}, {'text': '1340', 'rowspan': 1, 'colspan': 1, 'label_key': None}]], 'layout': {'width': 1680, 'padding': 45, 'cell_width': 265.0, 'font_size': 27, 'heights': [106, 106, 145, 145, 106, 106, 106, 106, 106, 106, 106, 145, 184, 106, 106, 106, 106, 106, 106, 106, 106, 145, 106, 106, 106, 145, 106, 145, 145, 106, 106, 106, 106, 106, 106, 106, 106, 145, 145]}}
LABELS = {'cell_000_000': '季内集数', 'cell_000_001': '全剧集数', 'cell_000_002': '标题', 'cell_000_003': '加拿大播出日期', 'cell_000_004': '美国播出日期', 'cell_000_005': '制作代码', 'cell_001_002': '夏日时光', 'cell_001_003': '2013年7月11日', 'cell_001_004': '2013年7月11日', 'cell_002_002': '我只想做的事', 'cell_002_003': '2013年7月18日', 'cell_002_004': '2013年7月18日', 'cell_003_002': '我自己最大的敌人', 'cell_003_003': '2013年7月25日', 'cell_003_004': '2013年7月25日', 'cell_004_002': '关于一个女孩', 'cell_004_003': '2013年8月1日', 'cell_004_004': '2013年8月1日', 'cell_005_002': '炮弹', 'cell_005_003': '2013年8月8日', 'cell_005_004': '2013年8月8日', 'cell_006_002': '亲爱的', 'cell_006_003': '2013年8月15日', 'cell_006_004': '2013年8月15日', 'cell_007_002': '永远年轻', 'cell_007_003': '2013年8月22日', 'cell_007_004': '2013年8月22日', 'cell_008_002': '我们就是这样做的', 'cell_008_003': '2013年10月3日', 'cell_008_004': '2013年10月3日', 'cell_009_002': '你俘获了我', 'cell_009_003': '2013年10月10日', 'cell_009_004': '2013年10月10日', 'cell_010_002': '你该知道', 'cell_010_003': '2013年10月17日', 'cell_010_004': '2013年10月17日', 'cell_011_002': '你做错的一切', 'cell_011_003': '2013年10月24日', 'cell_011_004': '2013年10月24日', 'cell_012_002': '你以为你是谁', 'cell_012_003': '2013年10月31日', 'cell_012_004': '2013年10月31日', 'cell_013_002': '几乎无法呼吸', 'cell_013_003': '2013年11月7日', 'cell_013_004': '2013年11月7日', 'cell_014_002': '黑或白', 'cell_014_003': '2013年11月14日', 'cell_014_004': '2013年11月14日', 'cell_015_002': '蜘蛛网', 'cell_015_003': '2013年11月21日', 'cell_015_004': '2013年11月21日', 'cell_016_002': '我所知道的世界', 'cell_016_003': '2014年1月28日', 'cell_016_004': '2014年1月28日', 'cell_017_002': '更好的男人', 'cell_017_003': '2014年2月4日', 'cell_017_004': '2014年2月4日', 'cell_018_002': '把我挖出来', 'cell_018_003': '2014年2月11日', 'cell_018_004': '2014年2月11日', 'cell_019_002': '权力归人民', 'cell_019_003': '2014年2月18日', 'cell_019_004': '2014年2月18日', 'cell_020_002': '没有意外', 'cell_020_003': '2014年2月25日', 'cell_020_004': '2014年2月25日', 'cell_021_002': '精神崩溃的人', 'cell_021_003': '2014年3月4日', 'cell_021_004': '2014年3月4日', 'cell_022_002': '难以置信', 'cell_022_003': '2014年3月11日', 'cell_022_004': '2014年3月11日', 'cell_023_002': '那是什么滋味', 'cell_023_003': '2014年3月18日', 'cell_023_004': '2014年3月18日', 'cell_024_002': '靠近我', 'cell_024_003': '2014年3月25日', 'cell_024_004': '2014年3月25日', 'cell_025_002': '我一人的军队', 'cell_025_003': '2014年4月1日', 'cell_025_004': '2014年4月1日', 'cell_026_002': '一切就是一切', 'cell_026_003': '2014年4月8日', 'cell_026_004': '2014年4月8日', 'cell_027_002': '火花将迸发 上篇', 'cell_027_003': '2014年4月15日', 'cell_027_004': '2014年4月15日', 'cell_028_002': '火花将迸发 下篇', 'cell_028_003': '2014年4月22日', 'cell_028_004': '2014年4月22日', 'cell_029_002': '你并不孤单', 'cell_029_003': '2014年6月3日', 'cell_029_004': '2014年6月3日', 'cell_030_002': '享受寂静', 'cell_030_003': '2014年6月10日', 'cell_030_004': '2014年6月10日', 'cell_031_002': '多么离奇', 'cell_031_003': '2014年6月17日', 'cell_031_004': '2014年6月17日', 'cell_032_002': '我的英雄', 'cell_032_003': '2014年6月24日', 'cell_032_004': '2014年6月24日', 'cell_033_002': '催眠', 'cell_033_003': '2014年7月1日', 'cell_033_004': '2014年7月1日', 'cell_034_002': '失去理智', 'cell_034_003': '2014年7月8日', 'cell_034_004': '2014年7月8日', 'cell_035_002': '待公布', 'cell_035_003': '2014年7月15日', 'cell_035_004': '2014年7月15日', 'cell_036_002': '待公布', 'cell_036_003': '2014年7月22日', 'cell_036_004': '2014年7月22日', 'cell_037_002': '如遭雷击 上篇', 'cell_037_003': '2014年7月29日', 'cell_037_004': '2014年7月29日', 'cell_038_002': '如遭雷击 下篇', 'cell_038_003': '2014年7月29日', 'cell_038_004': '2014年7月29日'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
