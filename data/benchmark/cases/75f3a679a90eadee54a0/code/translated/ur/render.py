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
BASE_ID = 'qa_8c0b4fee9e08b1bdc5cd83feed96561745d6ee812da52e3c8214d58a36a4e82e'
LANGUAGE = 'ur'
DATA = {'rows': [[{'text': 'Year', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Name', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Year', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Name', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_003'}, {'text': 'Year', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_004'}, {'text': 'Name', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_005'}], [{'text': '1903-04', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr B. Owen- Jones', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_001'}, {'text': '1935-36', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr W.Pearce', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_003'}, {'text': '1967-68', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.F.Serfontein', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_005'}], [{'text': '1904-05', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr B. Owen- Jones', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_001'}, {'text': '1936-37', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr W.Pearce', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_003'}, {'text': '1968-69', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Ben Steyn', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_005'}], [{'text': '1905-06', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr G. Constable', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_001'}, {'text': '1937 -38', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1969-70', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1907-08', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr T.R.Ziervogel', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_001'}, {'text': '1939 -40', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr W.E.Vickers', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_003'}, {'text': '1971-72', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Chris Smith', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_005'}], [{'text': '1908-09', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr T.R.Ziervogel', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_001'}, {'text': '1940-41', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr P.Venter', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_003'}, {'text': '1972-73', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Ben Steyn', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_005'}], [{'text': '1909-10', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.Morris', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_001'}, {'text': '1941-42', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr P.Venter', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_003'}, {'text': '1973-74', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Issy Kramer', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_005'}], [{'text': '1910-11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1942-43', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr P.Venter', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_003'}, {'text': '1974-75', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1911-12', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr B.Owen- Jones', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_001'}, {'text': '1943-44', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr P.Venter', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_003'}, {'text': '1975-76', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Sakkie Blanche', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_005'}], [{'text': '1912-13', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.Johnston', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_001'}, {'text': '1944-45', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mrs E.Myer', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_003'}, {'text': '1977-78', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Sakkie Blanche', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_005'}], [{'text': '1913-14', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.Cook', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_001'}, {'text': '1945-46', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mrs E.Myer', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_003'}, {'text': '1978-79', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1914-15', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.Cook', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_001'}, {'text': '1946-47', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mrs E.Myer', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_003'}, {'text': '1979 -80', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Kobus Durand', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_005'}], [{'text': '1915-16', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr R.Champion', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_001'}, {'text': '1947-48', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr C.Chambers', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_003'}, {'text': '1980-81', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Meyer', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_005'}], [{'text': '1916-17', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr R.Champion', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_001'}, {'text': '1948-49', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mrs S.Von Wielligh', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_003'}, {'text': '1981-82', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Wiek Steyn', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_005'}], [{'text': '1917-18', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr A.Ruffels', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_014_001'}, {'text': '1949-50', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr A.J.Law', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_014_003'}, {'text': '1982-83', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Andrew Wheeler', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_014_005'}], [{'text': '1918-19', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.Campbell', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_015_001'}, {'text': '1950-51', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr P.Venter', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_015_003'}, {'text': '1983-84', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1919-20', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr B.Melman', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_016_001'}, {'text': '1951-52', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr P.Venter', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_016_003'}, {'text': '1984-85', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1920-21', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr B.Melman', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_017_001'}, {'text': '1952-53', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Vic Pretorius', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_017_003'}, {'text': '1985-86', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.Prins', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_017_005'}], [{'text': '1921-21', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr B.Melman', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_018_001'}, {'text': '1953-54', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Vic Pretorius', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_018_003'}, {'text': '1986-87', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1922-23', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.Campbell', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_019_001'}, {'text': '1954-55', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1987-88', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1923-24', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1954-56', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.H.A.Roets', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_020_003'}, {'text': '1988-89', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Beyers De Klerk', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_020_005'}], [{'text': '1924-25', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr E.Murton', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_021_001'}, {'text': '1956-57', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr P.H.Tredoux', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_021_003'}, {'text': '1989-90', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Gerrie Wolmarans', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_021_005'}], [{'text': '1925-26', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr S.Steenberg', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_022_001'}, {'text': '1957- 58', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1990-91', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Gerrie Wolmarans', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_022_005'}], [{'text': '1926-27', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1958-59', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.M.Cawood', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_023_003'}, {'text': '1991-92', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr TJ Ferreira', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_023_005'}], [{'text': '1927-28', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.Stanbury', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_024_001'}, {'text': '1959-60', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr A.P.Scribante', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_024_003'}, {'text': '1992-93', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr Gerrie Wolmarans', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_024_005'}], [{'text': '1928-29', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr E.Murton', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_025_001'}, {'text': '1960-61', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.L.Viljoen', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_025_003'}, {'text': '1993-94', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr TJ Ferreira', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_025_005'}], [{'text': '1929-30', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr K.Turner', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_026_001'}, {'text': '1961-62', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.L.Viljoen', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_026_003'}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1930-31', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr J.E.Bigwood', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_027_001'}, {'text': '1962-63', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mrs S.Von Wielligh', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_027_003'}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1931-32', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr A.Zaretsky', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_028_001'}, {'text': '1963-64', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr F.J.Van Heerden', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_028_003'}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1932-33', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr G.J.Malan', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_029_001'}, {'text': '1964-65', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1933-34', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1965-66', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '1934-34', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1966-67', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Mr H.McLennan', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_031_003'}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}]], 'layout': {'width': 1680, 'padding': 45, 'cell_width': 265.0, 'font_size': 27, 'heights': [67, 145, 145, 106, 145, 145, 106, 67, 145, 106, 106, 106, 106, 145, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 145, 145, 106, 67, 106]}}
LABELS = {'cell_000_000': 'سال', 'cell_000_001': 'نام', 'cell_000_002': 'سال', 'cell_000_003': 'نام', 'cell_000_004': 'سال', 'cell_000_005': 'نام', 'cell_001_001': 'جناب بی۔ اوون-جونز', 'cell_001_003': 'جناب ڈبلیو۔ پیئرس', 'cell_001_005': 'جناب جے۔ ایف۔ سرفونٹین', 'cell_002_001': 'جناب بی۔ اوون-جونز', 'cell_002_003': 'جناب ڈبلیو۔ پیئرس', 'cell_002_005': 'جناب بین اسٹین', 'cell_003_001': 'جناب جی۔ کانسٹیبل', 'cell_004_001': 'جناب ٹی۔ آر۔ زیئرووگل', 'cell_004_003': 'جناب ڈبلیو۔ ای۔ وکرز', 'cell_004_005': 'جناب کرس اسمتھ', 'cell_005_001': 'جناب ٹی۔ آر۔ زیئرووگل', 'cell_005_003': 'جناب پی۔ وینٹر', 'cell_005_005': 'جناب بین اسٹین', 'cell_006_001': 'جناب جے۔ مورس', 'cell_006_003': 'جناب پی۔ وینٹر', 'cell_006_005': 'جناب اِسی کریمر', 'cell_007_003': 'جناب پی۔ وینٹر', 'cell_008_001': 'جناب بی۔ اوون-جونز', 'cell_008_003': 'جناب پی۔ وینٹر', 'cell_008_005': 'جناب ساکی بلانش', 'cell_009_001': 'جناب جے۔ جانسٹن', 'cell_009_003': 'محترمہ ای۔ مائر', 'cell_009_005': 'جناب ساکی بلانش', 'cell_010_001': 'جناب جے۔ کک', 'cell_010_003': 'محترمہ ای۔ مائر', 'cell_011_001': 'جناب جے۔ کک', 'cell_011_003': 'محترمہ ای۔ مائر', 'cell_011_005': 'جناب کوبس دورانڈ', 'cell_012_001': 'جناب آر۔ چیمپیئن', 'cell_012_003': 'جناب سی۔ چیمبرز', 'cell_012_005': 'جناب میئر', 'cell_013_001': 'جناب آر۔ چیمپیئن', 'cell_013_003': 'محترمہ ایس۔ فون ویلخ', 'cell_013_005': 'جناب ویک اسٹین', 'cell_014_001': 'جناب اے۔ رفلز', 'cell_014_003': 'جناب اے۔ جے۔ لا', 'cell_014_005': 'جناب اینڈریو وہیلر', 'cell_015_001': 'جناب جے۔ کیمبل', 'cell_015_003': 'جناب پی۔ وینٹر', 'cell_016_001': 'جناب بی۔ میلمین', 'cell_016_003': 'جناب پی۔ وینٹر', 'cell_017_001': 'جناب بی۔ میلمین', 'cell_017_003': 'جناب وِک پریٹوریئس', 'cell_017_005': 'جناب جے۔ پرنس', 'cell_018_001': 'جناب بی۔ میلمین', 'cell_018_003': 'جناب وِک پریٹوریئس', 'cell_019_001': 'جناب جے۔ کیمبل', 'cell_020_003': 'جناب جے۔ ایچ۔ اے۔ روٹس', 'cell_020_005': 'جناب بائرز ڈی کلرک', 'cell_021_001': 'جناب ای۔ مرٹن', 'cell_021_003': 'جناب پی۔ ایچ۔ ٹریڈو', 'cell_021_005': 'جناب گیری وولمارانس', 'cell_022_001': 'جناب ایس۔ اسٹینبرگ', 'cell_022_005': 'جناب گیری وولمارانس', 'cell_023_003': 'جناب جے۔ ایم۔ کاوُڈ', 'cell_023_005': 'جناب ٹی جے فریرا', 'cell_024_001': 'جناب جے۔ اسٹینبری', 'cell_024_003': 'جناب اے۔ پی۔ اسکرِبانٹے', 'cell_024_005': 'جناب گیری وولمارانس', 'cell_025_001': 'جناب ای۔ مرٹن', 'cell_025_003': 'جناب جے۔ ایل۔ فلجون', 'cell_025_005': 'جناب ٹی جے فریرا', 'cell_026_001': 'جناب کے۔ ٹرنر', 'cell_026_003': 'جناب جے۔ ایل۔ فلجون', 'cell_027_001': 'جناب جے۔ ای۔ بگ وُڈ', 'cell_027_003': 'محترمہ ایس۔ فون ویلخ', 'cell_028_001': 'جناب اے۔ زاریٹسکی', 'cell_028_003': 'جناب ایف۔ جے۔ وان ہیرڈن', 'cell_029_001': 'جناب جی۔ جے۔ ملان', 'cell_031_003': 'جناب ایچ۔ میکلینن'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
