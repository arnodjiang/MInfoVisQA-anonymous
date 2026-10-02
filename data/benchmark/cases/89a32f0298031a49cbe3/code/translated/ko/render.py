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
BASE_ID = 'qa_6d22f449553f66c994e17d275425a2de24430b0991b0848f6b4e0a2e14c685fe'
LANGUAGE = 'ko'
DATA = {'rows': [[{'text': '-', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Soviet Union', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Poland and Danzig', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Finland', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_003'}, {'text': 'Estonia', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_004'}, {'text': 'Latvia', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_005'}, {'text': 'Lithuania', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_006'}], [{'text': '1934', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '223.0', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '78.1', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '42.3', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '8.2', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '21.1', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '15.1', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '1935', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '201.7', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '75.5', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '41.4', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '13.0', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '31.1', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '2.0', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '1936', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '93.2', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '74.0', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '46.1', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '13.8', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '33.2', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '9.1', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '1937', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '63.1', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '80.7', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '70.1', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '23.7', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '45.7', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '17.2', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '1938', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '47.4', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '109.4', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '88.6', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '24.0', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '43.5', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '27.6', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '1939', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '52.8', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '140.8', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '88.9', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '24.3', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '43.6', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': '27.8', 'rowspan': 1, 'colspan': 1, 'label_key': None}], [{'text': '*German Imports in millions of Reichsmarks', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_000'}, {'text': '*German Imports in millions of Reichsmarks', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_001'}, {'text': '*German Imports in millions of Reichsmarks', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_002'}, {'text': '*German Imports in millions of Reichsmarks', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_003'}, {'text': '*German Imports in millions of Reichsmarks', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_004'}, {'text': '*German Imports in millions of Reichsmarks', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_005'}, {'text': '*German Imports in millions of Reichsmarks', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_006'}]], 'layout': {'width': 1960, 'padding': 45, 'cell_width': 267.14285714285717, 'font_size': 27, 'heights': [106, 67, 67, 67, 67, 67, 67, 223]}}
LABELS = {'cell_000_001': '소련', 'cell_000_002': '폴란드와 단치히', 'cell_000_003': '핀란드', 'cell_000_004': '에스토니아', 'cell_000_005': '라트비아', 'cell_000_006': '리투아니아', 'cell_007_000': '*독일 수입액 (단위: 백만 라이히스마르크)', 'cell_007_001': '*독일 수입액 (단위: 백만 라이히스마르크)', 'cell_007_002': '*독일 수입액 (단위: 백만 라이히스마르크)', 'cell_007_003': '*독일 수입액 (단위: 백만 라이히스마르크)', 'cell_007_004': '*독일 수입액 (단위: 백만 라이히스마르크)', 'cell_007_005': '*독일 수입액 (단위: 백만 라이히스마르크)', 'cell_007_006': '*독일 수입액 (단위: 백만 라이히스마르크)'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
