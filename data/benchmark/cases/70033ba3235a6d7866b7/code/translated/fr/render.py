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
BASE_ID = 'qa_22ce0e4594adb352e5d7c71740de2d6e8c89da74470f937c3355dd44c93d3010'
LANGUAGE = 'fr'
DATA = {'rows': [[{'text': 'rank', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_000'}, {'text': 'airline / holding', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_001'}, {'text': 'passenger fleet', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_002'}, {'text': 'current destinations', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_003'}, {'text': 'alliance / association', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_004'}], [{'text': '1', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'lufthansa group', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_001'}, {'text': '627', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '283', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_004'}], [{'text': '2', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'ryanair', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_001'}, {'text': '305', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '176', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'elfaa', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_004'}], [{'text': '3', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'air france - klm', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_001'}, {'text': '621', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '246', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'skyteam', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_004'}], [{'text': '4', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'international airlines group', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_001'}, {'text': '435', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '207', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'oneworld', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_004'}], [{'text': '5', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'easyjet', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_001'}, {'text': '194', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '126', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'elfaa', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_004'}], [{'text': '6', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'turkish airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_001'}, {'text': '222', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '245', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_004'}], [{'text': '7', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'air berlin group', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_001'}, {'text': '153', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '145', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'oneworld', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_004'}], [{'text': '8', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'aeroflot group', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_001'}, {'text': '239', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '189', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'skyteam', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_004'}], [{'text': '9', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'sas group', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_001'}, {'text': '173', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '157', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_004'}], [{'text': '10', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'alitalia', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_001'}, {'text': '143', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '101', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'skyteam', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_004'}], [{'text': '11', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'norwegian air shuttle asa', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_001'}, {'text': '79', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '120', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'elfaa', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_004'}], [{'text': '12', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'pegasus airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_001'}, {'text': '42', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '70', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_004'}], [{'text': '13', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'wizz air', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_013_001'}, {'text': '45', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '83', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'elfaa', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_013_004'}], [{'text': '14', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'transaero', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_014_001'}, {'text': '93', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '113', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_014_004'}], [{'text': '15', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'tap portugal', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_015_001'}, {'text': '71', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '80', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_015_004'}], [{'text': '16', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'aer lingus', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_016_001'}, {'text': '46', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '75', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_016_004'}], [{'text': '17', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'finnair', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_017_001'}, {'text': '44', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '65', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'oneworld', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_017_004'}], [{'text': '18', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 's7', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_018_001'}, {'text': '52', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '90', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'oneworld', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_018_004'}], [{'text': '19', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'air europa', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_019_001'}, {'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '54', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'skyteam', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_019_004'}], [{'text': '20', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'utair aviation', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_020_001'}, {'text': '108', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '117', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_020_004'}], [{'text': '21', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'sunexpress', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_021_001'}, {'text': '23', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '48', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_021_004'}], [{'text': '22', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'flybe', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_022_001'}, {'text': '68', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '56', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'elfaa', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_022_004'}], [{'text': '23', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'brussels airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_023_001'}, {'text': '45', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '67', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_023_004'}], [{'text': '24', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'aegean airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_024_001'}, {'text': '29', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_024_004'}], [{'text': '25', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'monarch airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_025_001'}, {'text': '39', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '30', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_025_004'}], [{'text': '26', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'virgin atlantic', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_026_001'}, {'text': '41', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '37', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_026_004'}], [{'text': '27', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'atlasjet', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_027_001'}, {'text': '15', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '15', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_027_004'}], [{'text': '28', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'lot polish airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_028_001'}, {'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '54', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_028_004'}], [{'text': '29', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'jet2.com', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_029_001'}, {'text': '49', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '59', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'elfaa', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_029_004'}], [{'text': '30', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'meridiana fly', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_030_001'}, {'text': '18', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_030_004'}], [{'text': '31', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'ural airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_031_001'}, {'text': '29', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '66', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_031_004'}], [{'text': '32', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'czech airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_032_001'}, {'text': '25', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '49', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'skyteam', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_032_004'}], [{'text': '33', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'airbaltic', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_033_001'}, {'text': '28', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '60', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_033_004'}], [{'text': '34', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'onur air', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_034_001'}, {'text': '29', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '21', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_034_004'}], [{'text': '35', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'ukraine international airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_035_001'}, {'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '54', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_035_004'}], [{'text': '36', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'olympic air', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_036_001'}, {'text': '16', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '37', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_036_004'}], [{'text': '37', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'tarom', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_037_001'}, {'text': '23', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '48', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'skyteam', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_037_004'}], [{'text': '38', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'icelandair', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_038_001'}, {'text': '27', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '36', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_038_004'}], [{'text': '39', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'croatia airlines', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_039_001'}, {'text': '13', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_039_004'}], [{'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'air serbia', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_040_001'}, {'text': '13', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '34', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_040_004'}], [{'text': '41', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'belavia', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_041_001'}, {'text': '23', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_041_004'}], [{'text': '42', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'cyprus airways', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_042_001'}, {'text': '9', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '18', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_042_004'}], [{'text': '43', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'bulgaria air', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_043_001'}, {'text': '11', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '22', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'n / a', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_043_004'}], [{'text': '44', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'adria airways', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_044_001'}, {'text': '12', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '37', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'star alliance', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_044_004'}]], 'layout': {'width': 1500, 'padding': 45, 'cell_width': 282.0, 'font_size': 27, 'heights': [145, 106, 106, 106, 145, 106, 106, 106, 106, 106, 67, 145, 106, 106, 106, 106, 106, 67, 67, 67, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 145, 106, 67, 106, 106, 106, 106, 106, 106, 106]}}
LABELS = {'cell_000_000': 'rang', 'cell_000_001': 'compagnie aérienne / holding', 'cell_000_002': 'flotte passagers', 'cell_000_003': 'destinations actuelles', 'cell_000_004': 'alliance / association', 'cell_001_001': 'groupe Lufthansa', 'cell_001_004': 'Star Alliance', 'cell_002_001': 'Ryanair', 'cell_002_004': 'ELFAA', 'cell_003_001': 'Air France - KLM', 'cell_003_004': 'SkyTeam', 'cell_004_001': 'International Airlines Group', 'cell_004_004': 'oneworld', 'cell_005_001': 'easyJet', 'cell_005_004': 'ELFAA', 'cell_006_001': 'Turkish Airlines', 'cell_006_004': 'Star Alliance', 'cell_007_001': 'groupe Air Berlin', 'cell_007_004': 'oneworld', 'cell_008_001': 'groupe Aeroflot', 'cell_008_004': 'SkyTeam', 'cell_009_001': 'groupe SAS', 'cell_009_004': 'Star Alliance', 'cell_010_001': 'Alitalia', 'cell_010_004': 'SkyTeam', 'cell_011_001': 'Norwegian Air Shuttle ASA', 'cell_011_004': 'ELFAA', 'cell_012_001': 'Pegasus Airlines', 'cell_012_004': 's. o.', 'cell_013_001': 'Wizz Air', 'cell_013_004': 'ELFAA', 'cell_014_001': 'Transaero', 'cell_014_004': 's. o.', 'cell_015_001': 'TAP Portugal', 'cell_015_004': 'Star Alliance', 'cell_016_001': 'Aer Lingus', 'cell_016_004': 's. o.', 'cell_017_001': 'Finnair', 'cell_017_004': 'oneworld', 'cell_018_001': 'S7', 'cell_018_004': 'oneworld', 'cell_019_001': 'Air Europa', 'cell_019_004': 'SkyTeam', 'cell_020_001': 'UTair Aviation', 'cell_020_004': 's. o.', 'cell_021_001': 'SunExpress', 'cell_021_004': 's. o.', 'cell_022_001': 'Flybe', 'cell_022_004': 'ELFAA', 'cell_023_001': 'Brussels Airlines', 'cell_023_004': 'Star Alliance', 'cell_024_001': 'Aegean Airlines', 'cell_024_004': 'Star Alliance', 'cell_025_001': 'Monarch Airlines', 'cell_025_004': 's. o.', 'cell_026_001': 'Virgin Atlantic', 'cell_026_004': 's. o.', 'cell_027_001': 'Atlasjet', 'cell_027_004': 's. o.', 'cell_028_001': 'LOT Polish Airlines', 'cell_028_004': 'Star Alliance', 'cell_029_001': 'Jet2.com', 'cell_029_004': 'ELFAA', 'cell_030_001': 'Meridiana fly', 'cell_030_004': 's. o.', 'cell_031_001': 'Ural Airlines', 'cell_031_004': 's. o.', 'cell_032_001': 'Czech Airlines', 'cell_032_004': 'SkyTeam', 'cell_033_001': 'airBaltic', 'cell_033_004': 's. o.', 'cell_034_001': 'Onur Air', 'cell_034_004': 's. o.', 'cell_035_001': 'Ukraine International Airlines', 'cell_035_004': 's. o.', 'cell_036_001': 'Olympic Air', 'cell_036_004': 's. o.', 'cell_037_001': 'TAROM', 'cell_037_004': 'SkyTeam', 'cell_038_001': 'Icelandair', 'cell_038_004': 's. o.', 'cell_039_001': 'Croatia Airlines', 'cell_039_004': 'Star Alliance', 'cell_040_001': 'Air Serbia', 'cell_040_004': 's. o.', 'cell_041_001': 'Belavia', 'cell_041_004': 's. o.', 'cell_042_001': 'Cyprus Airways', 'cell_042_004': 's. o.', 'cell_043_001': 'Bulgaria Air', 'cell_043_004': 's. o.', 'cell_044_001': 'Adria Airways', 'cell_044_004': 'Star Alliance'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
