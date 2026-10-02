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
BASE_ID = 'qa_8abbfdcff209622f5ffcdd94ef4ab68d4a9d955f5e7af889b00d8870874e04f5'
LANGUAGE = 'vi'
DATA = {'rows': [[{'text': 'Statistics for airport Koltsovo[30][31][32]', 'label_key': 'table_title', 'colspan': 9, 'rowspan': 1}], [{'text': 'Year', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Totalpassengers', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_001'}, {'text': 'PassengerChange', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Domestic', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_003'}, {'text': 'International(total)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_004'}, {'text': 'International(non-CIS)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_005'}, {'text': 'CIS', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_006'}, {'text': 'AircraftLandings', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_007'}, {'text': 'Cargo(tonnes)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_008'}], [{'text': '2000', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '930 251', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+2%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '698 957', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '231 294', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '155 898', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '75 396', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '8 619', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '18 344', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2001', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 028 295', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+10,5%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '733 022', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '295 273', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '186 861', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '108 412', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '9 062', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '22 178', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2002', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 182 815', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+15,0%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '793 295', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '389 520', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '239 461', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '150 059', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '10 162', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '20 153', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2003', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 335 757', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+12,9%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '879 665', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '456 092', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '297 421', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '158 671', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '10 092', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '18 054', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2004', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 553 628', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+16,3%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '972 287', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '581 341', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '429 049', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '152 292', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '11 816', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '20 457', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2005', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 566 792', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+0,8%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 006 422', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '560 370', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '429 790', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '130 580', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '11 877', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '11 545', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2006', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 764 948', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+12,7%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 128 489', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '636 459', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '488 954', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '147 505', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '13 289', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '15 519', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2007', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '2 345 097', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+32,9%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 486 888', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '858 209', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '683 092', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '175 117', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '16 767', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '16 965', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2008', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '2 529 395', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+7,8%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 523 102', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 006 293', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '815 124', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '191 169', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '16 407', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '17 142', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2009', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '2 169 136', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '−14,2%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 290 639', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '878 497', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '727 718', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '150 779', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '13 798', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '13 585', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2010', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '2 748 919', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+26,7%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 529 245', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 219 674', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 017 509', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '202 165', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '15 989', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '22 946', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2011', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '3 355 883', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+22,1%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 856 948', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 498 935', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 184 771', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '314 164', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '20 142', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '24 890', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2012', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '3 783 069', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+12.7%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 934 016', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 849 053', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1 448 765', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '439 668', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '21 728', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '25 866', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '2013', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '4 293 002', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '+13.5%', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '2 180 227', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '2 112 775', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '25 728', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '27 800', 'colspan': 1, 'rowspan': 1, 'label_key': None}]], 'layout': {'width': 2520, 'padding': 45, 'cell_width': 270.0, 'font_size': 27, 'heights': [67, 145, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67]}}
LABELS = {'cell_000_000': 'Năm', 'cell_000_001': 'Tổng số hành khách', 'cell_000_002': 'Thay đổi số hành khách', 'cell_000_003': 'Nội địa', 'cell_000_004': 'Quốc tế (tổng cộng)', 'cell_000_005': 'Quốc tế (ngoài SNG)', 'cell_000_006': 'SNG', 'cell_000_007': 'Số lượt máy bay hạ cánh', 'cell_000_008': 'Hàng hóa (tấn)', 'table_title': 'Thống kê sân bay Koltsovo[30][31][32]'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
