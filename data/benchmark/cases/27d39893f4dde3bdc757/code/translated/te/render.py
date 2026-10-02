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
BASE_ID = 'qa_6cbf79854e7030de4e8db78132528de50f3404fdd57f779af4c1face12b3c74c'
LANGUAGE = 'te'
DATA = {'rows': [[{'text': 'Date', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Opponent#', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Rank#', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Site', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_003'}, {'text': 'TV', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_004'}, {'text': 'Result', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_005'}, {'text': 'Attendance', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_006'}], [{'text': 'September 3', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_000'}, {'text': 'Tennessee–Chattanooga*', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_001'}, {'text': '#11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Legion Field • Birmingham, AL', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_003'}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'W 42–13', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_005'}, {'text': '82,109', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'September 10', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_000'}, {'text': 'Vanderbilt', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_001'}, {'text': '#11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Bryant–Denny Stadium • Tuscaloosa, AL', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_003'}, {'text': 'JPS', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_004'}, {'text': 'W 17–7', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_005'}, {'text': '70,123', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'September 17', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_000'}, {'text': 'at Arkansas', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_001'}, {'text': '#12', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Razorback Stadium • Fayetteville, AR', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_003'}, {'text': 'ABC', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_004'}, {'text': 'W 13–6', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_005'}, {'text': '52,089', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'September 24', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_000'}, {'text': 'Tulane*', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_001'}, {'text': '#11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Legion Field • Birmingham, AL', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_003'}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'W 20–10', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_005'}, {'text': '81,421', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'October 1', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_000'}, {'text': 'Georgia', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_001'}, {'text': '#11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Bryant–Denny Stadium • Tuscaloosa, AL', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_003'}, {'text': 'ESPN', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_004'}, {'text': 'W 29–28', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_005'}, {'text': '70,123', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'October 8', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_000'}, {'text': 'Southern Miss*', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_001'}, {'text': '#11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Bryant–Denny Stadium • Tuscaloosa, AL', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_003'}, {'text': '', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'W 14–6', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_005'}, {'text': '70,123', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'October 15', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_000'}, {'text': 'at Tennessee', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_001'}, {'text': '#10', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Neyland Stadium • Knoxville, TN (Third Saturday in October)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_003'}, {'text': 'ESPN', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_004'}, {'text': 'W 17–13', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_005'}, {'text': '96,856', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'October 22', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_000'}, {'text': 'Ole Miss', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_001'}, {'text': '#8', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Bryant–Denny Stadium • Tuscaloosa, AL (Rivalry)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_003'}, {'text': 'ABC', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_004'}, {'text': 'W 21–10', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_005'}, {'text': '70,123', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'November 5', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_000'}, {'text': 'at LSU', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_001'}, {'text': '#6', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Tiger Stadium • Baton Rouge, LA (Rivalry)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_003'}, {'text': 'ESPN', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_004'}, {'text': 'W 35–17', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_005'}, {'text': '75,453', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'November 12', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_000'}, {'text': 'at #20 Mississippi State', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_001'}, {'text': '#6', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Scott Field • Starkville, MS (Rivalry)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_003'}, {'text': 'ABC', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_004'}, {'text': 'W 29–25', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_005'}, {'text': '41,358', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'November 19', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_000'}, {'text': '#6 Auburn', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_001'}, {'text': '#4', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Legion Field • Birmingham, AL (Iron Bowl)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_003'}, {'text': 'ABC', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_004'}, {'text': 'W 21–14', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_005'}, {'text': '83,091', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'December 3', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_000'}, {'text': 'vs. #6 Florida', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_001'}, {'text': '#3', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Georgia Dome • Atlanta, GA (SEC Championship Game)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_003'}, {'text': 'ABC', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_004'}, {'text': 'L 23–24', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_005'}, {'text': '74,751', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': 'January 2, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_000'}, {'text': 'vs. #13 Ohio State*', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_001'}, {'text': '#6', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': 'Citrus Bowl • Orlando, FL (Florida Citrus Bowl)', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_003'}, {'text': 'ABC', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_004'}, {'text': 'W 24–17', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_005'}, {'text': '71,195', 'colspan': 1, 'rowspan': 1, 'label_key': None}], [{'text': '*Non-conference game. Homecoming. #Rankings from AP Poll. All times are in Central Time.', 'colspan': 7, 'rowspan': 1, 'label_key': 'cell_014_000'}]], 'layout': {'width': 1960, 'padding': 45, 'cell_width': 267.14285714285717, 'font_size': 27, 'heights': [106, 184, 184, 184, 184, 184, 184, 262, 301, 262, 262, 223, 301, 223, 106]}}
LABELS = {'cell_000_000': 'తేదీ', 'cell_000_001': 'ప్రత్యర్థి#', 'cell_000_002': 'ర్యాంకు#', 'cell_000_003': 'వేదిక', 'cell_000_004': 'టీవీ', 'cell_000_005': 'ఫలితం', 'cell_000_006': 'హాజరు', 'cell_001_000': 'సెప్టెంబర్ 3', 'cell_001_001': 'టెన్నెస్సీ–చాటనూగా*', 'cell_001_003': 'లీజియన్ ఫీల్డ్ • బర్మింగ్\u200cహామ్, AL', 'cell_001_005': 'గెలుపు 42–13', 'cell_002_000': 'సెప్టెంబర్ 10', 'cell_002_001': 'వాండర్\u200cబిల్ట్', 'cell_002_003': 'బ్రయంట్–డెన్నీ స్టేడియం • టస్కలూసా, AL', 'cell_002_004': 'JPS', 'cell_002_005': 'గెలుపు 17–7', 'cell_003_000': 'సెప్టెంబర్ 17', 'cell_003_001': 'ఆర్కన్సాస్ సొంత మైదానంలో', 'cell_003_003': 'రేజర్\u200cబ్యాక్ స్టేడియం • ఫేయెట్\u200cవిల్, AR', 'cell_003_004': 'ABC', 'cell_003_005': 'గెలుపు 13–6', 'cell_004_000': 'సెప్టెంబర్ 24', 'cell_004_001': 'టులేన్*', 'cell_004_003': 'లీజియన్ ఫీల్డ్ • బర్మింగ్\u200cహామ్, AL', 'cell_004_005': 'గెలుపు 20–10', 'cell_005_000': 'అక్టోబర్ 1', 'cell_005_001': 'జార్జియా', 'cell_005_003': 'బ్రయంట్–డెన్నీ స్టేడియం • టస్కలూసా, AL', 'cell_005_004': 'ESPN', 'cell_005_005': 'గెలుపు 29–28', 'cell_006_000': 'అక్టోబర్ 8', 'cell_006_001': 'సదరన్ మిస్*', 'cell_006_003': 'బ్రయంట్–డెన్నీ స్టేడియం • టస్కలూసా, AL', 'cell_006_005': 'గెలుపు 14–6', 'cell_007_000': 'అక్టోబర్ 15', 'cell_007_001': 'టెన్నెస్సీ సొంత మైదానంలో', 'cell_007_003': 'నీలాండ్ స్టేడియం • నాక్స్\u200cవిల్, TN (అక్టోబర్\u200cలో మూడవ శనివారం)', 'cell_007_004': 'ESPN', 'cell_007_005': 'గెలుపు 17–13', 'cell_008_000': 'అక్టోబర్ 22', 'cell_008_001': 'ఓల్ మిస్', 'cell_008_003': 'బ్రయంట్–డెన్నీ స్టేడియం • టస్కలూసా, AL (చిరకాల ప్రత్యర్థుల పోరు)', 'cell_008_004': 'ABC', 'cell_008_005': 'గెలుపు 21–10', 'cell_009_000': 'నవంబర్ 5', 'cell_009_001': 'LSU సొంత మైదానంలో', 'cell_009_003': 'టైగర్ స్టేడియం • బాటన్ రూజ్, LA (చిరకాల ప్రత్యర్థుల పోరు)', 'cell_009_004': 'ESPN', 'cell_009_005': 'గెలుపు 35–17', 'cell_010_000': 'నవంబర్ 12', 'cell_010_001': '#20 మిసిసిపీ స్టేట్ సొంత మైదానంలో', 'cell_010_003': 'స్కాట్ ఫీల్డ్ • స్టార్క్\u200cవిల్, MS (చిరకాల ప్రత్యర్థుల పోరు)', 'cell_010_004': 'ABC', 'cell_010_005': 'గెలుపు 29–25', 'cell_011_000': 'నవంబర్ 19', 'cell_011_001': '#6 ఆబర్న్', 'cell_011_003': 'లీజియన్ ఫీల్డ్ • బర్మింగ్\u200cహామ్, AL (ఐరన్ బౌల్)', 'cell_011_004': 'ABC', 'cell_011_005': 'గెలుపు 21–14', 'cell_012_000': 'డిసెంబర్ 3', 'cell_012_001': '#6 ఫ్లోరిడాతో పోరు', 'cell_012_003': 'జార్జియా డోమ్ • అట్లాంటా, GA (SEC ఛాంపియన్\u200cషిప్ మ్యాచ్)', 'cell_012_004': 'ABC', 'cell_012_005': 'ఓటమి 23–24', 'cell_013_000': 'జనవరి 2, 1995', 'cell_013_001': '#13 ఒహాయో స్టేట్\u200cతో పోరు*', 'cell_013_003': 'సిట్రస్ బౌల్ • ఓర్లాండో, FL (ఫ్లోరిడా సిట్రస్ బౌల్)', 'cell_013_004': 'ABC', 'cell_013_005': 'గెలుపు 24–17', 'cell_014_000': '*కాన్ఫరెన్స్ వెలుపలి జట్టుతో మ్యాచ్. పూర్వ విద్యార్థుల పునరాగమన వేడుక. #ర్యాంకులు AP పోల్ ప్రకారం. అన్ని సమయాలు సెంట్రల్ టైమ్\u200cలో ఉన్నాయి.'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
