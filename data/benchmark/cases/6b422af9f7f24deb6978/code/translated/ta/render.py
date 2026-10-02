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
BASE_ID = 'qa_91b1981c99126252329915f5fd51ee5219fd3f36205a2305d97f2b4d2330c684'
LANGUAGE = 'ta'
DATA = {'rows': [[{'text': 'Photography Lighting Setup Analysis by Material Properties', 'label_key': 'table_title', 'colspan': 7, 'rowspan': 1}], [{'text': 'Material\nType', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Reflectance\n%', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Key Light\nSetup', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Fill Light\nRatio', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_003'}, {'text': 'Background', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_004'}, {'text': 'Special\nConsiderations', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_005'}, {'text': 'Recommended\nModifier', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_006'}], [{'text': 'Polished\nMetal', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_000'}, {'text': '85-95', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Softbox 45°', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_002'}, {'text': '1:2', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Dark/Black', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_004'}, {'text': 'Minimize reflections, use\nflags', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_005'}, {'text': 'Large octabox', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_006'}], [{'text': 'Brushed Steel', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_000'}, {'text': '60-75', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Strip light', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_002'}, {'text': '1:3', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Neutral gray', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_004'}, {'text': 'Cross-polarization\nhelpful', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_005'}, {'text': 'Linear softbox', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_006'}], [{'text': 'Chrome', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_000'}, {'text': '90-98', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Umbrella\ndiffused', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_002'}, {'text': '1:4', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Gradient', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_004'}, {'text': 'Control environment\nreflections', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_005'}, {'text': 'Beauty dish', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_006'}], [{'text': 'Cotton\nFabric', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_000'}, {'text': '40-60', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Direct key', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_002'}, {'text': '1:1.5', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'White/cream', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_004'}, {'text': 'Avoid harsh shadows', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_005'}, {'text': 'Standard softbox', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_006'}], [{'text': 'Silk', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_000'}, {'text': '45-65', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Feathered\nkey', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_002'}, {'text': '1:2', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Textured', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_004'}, {'text': 'Enhance fabric texture', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_005'}, {'text': 'Grid spot', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_006'}], [{'text': 'Velvet', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_000'}, {'text': '15-25', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Side lighting', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_002'}, {'text': '1:1', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Dark', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_004'}, {'text': 'Preserve deep blacks', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_005'}, {'text': 'Barn doors', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_006'}], [{'text': 'Clear Glass', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_000'}, {'text': '8-12', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Rim lighting', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_002'}, {'text': '1:8', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Pure white', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_004'}, {'text': 'Backlight for\ntransparency', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_005'}, {'text': 'Strip lights', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_006'}], [{'text': 'Frosted Glass', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_000'}, {'text': '20-35', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Transmitted\nlight', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_002'}, {'text': '1:3', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Gradient', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_004'}, {'text': 'Even illumination', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_005'}, {'text': 'Panel diffuser', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_006'}], [{'text': 'Colored\nGlass', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_000'}, {'text': '5-40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Color temp\nmatch', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_002'}, {'text': '1:4', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Complementary', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_004'}, {'text': 'Color accuracy critical', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_005'}, {'text': 'Color gels', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_006'}], [{'text': 'Wood (Oak)', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_000'}, {'text': '25-40', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Raking light', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_002'}, {'text': '1:2', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Natural tone', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_004'}, {'text': 'Emphasize grain', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_005'}, {'text': 'Fresnel spot', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_006'}], [{'text': 'Leather', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_000'}, {'text': '20-35', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Textural key', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_002'}, {'text': '1:1.5', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Brown/tan', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_004'}, {'text': 'Show surface detail', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_005'}, {'text': 'Honeycomb grid', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_006'}], [{'text': 'Stone/Marble', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_000'}, {'text': '30-50', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Multi-angle', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_002'}, {'text': '1:2.5', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Neutral', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_004'}, {'text': 'Avoid flat lighting', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_005'}, {'text': 'Multiple sources', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_012_006'}]], 'layout': {'width': 1960, 'padding': 45, 'cell_width': 267.14285714285717, 'font_size': 27, 'heights': [67, 184, 223, 145, 145, 145, 184, 145, 184, 145, 184, 184, 184, 184]}}
LABELS = {'cell_000_000': 'பொருள்\nவகை', 'cell_000_001': 'ஒளிப்பிரதிபலிப்பு\n%', 'cell_000_002': 'முதன்மை ஒளி\nஅமைப்பு', 'cell_000_003': 'நிரப்பு ஒளி\nவிகிதம்', 'cell_000_004': 'பின்னணி', 'cell_000_005': 'சிறப்புக்\nகவனக்குறிப்புகள்', 'cell_000_006': 'பரிந்துரைக்கப்படும்\nஒளி மாற்றி', 'cell_001_000': 'மெருகூட்டிய\nஉலோகம்', 'cell_001_002': 'மென்பெட்டி 45°', 'cell_001_004': 'அடர்நிறம்/கருப்பு', 'cell_001_005': 'பிரதிபலிப்புகளைக் குறைக்கவும்,\nஒளித்தடுப்புகளைப் பயன்படுத்தவும்', 'cell_001_006': 'பெரிய எண்கோண ஒளிப்பெட்டி', 'cell_002_000': 'தூரிகையால் தேய்த்த எஃகு', 'cell_002_002': 'பட்டை விளக்கு', 'cell_002_004': 'நடுநிலைச் சாம்பல்', 'cell_002_005': 'குறுக்குத் தளவிளைவு\nஉதவும்', 'cell_002_006': 'நீள்வடிவ மென்பெட்டி', 'cell_003_000': 'குரோம்', 'cell_003_002': 'குடை வழியே பரவலாக்கிய\nஒளி', 'cell_003_004': 'படிப்படியான நிறமாற்றம்', 'cell_003_005': 'சுற்றுப்புறப் பிரதிபலிப்புகளைக்\nகட்டுப்படுத்தவும்', 'cell_003_006': 'அழகொளித் தட்டு', 'cell_004_000': 'பருத்தித்\nதுணி', 'cell_004_002': 'நேரடி முதன்மை ஒளி', 'cell_004_004': 'வெள்ளை/பாலேடு நிறம்', 'cell_004_005': 'கடுமையான நிழல்களைத் தவிர்க்கவும்', 'cell_004_006': 'வழக்கமான மென்பெட்டி', 'cell_005_000': 'பட்டு', 'cell_005_002': 'ஒளிக்கற்றையின் விளிம்பால் வழங்கும்\nமுதன்மை ஒளி', 'cell_005_004': 'மேற்பரப்பு அமைப்புடையது', 'cell_005_005': 'துணியின் இழையமைப்பை மேம்படுத்தவும்', 'cell_005_006': 'வலைப்பின்னலுடன் கூடிய குவியொளி', 'cell_006_000': 'வெல்வெட்', 'cell_006_002': 'பக்கவாட்டு ஒளியமைப்பு', 'cell_006_004': 'அடர்நிறம்', 'cell_006_005': 'ஆழ்ந்த கருமையைப் பாதுகாக்கவும்', 'cell_006_006': 'ஒளித்தடுப்புக் கதவுகள்', 'cell_007_000': 'தெளிவான கண்ணாடி', 'cell_007_002': 'விளிம்பு ஒளியமைப்பு', 'cell_007_004': 'தூய வெள்ளை', 'cell_007_005': 'ஒளி ஊடுருவும் தன்மைக்குப்\nபின்புற ஒளி', 'cell_007_006': 'பட்டை விளக்குகள்', 'cell_008_000': 'மங்கலாக்கிய கண்ணாடி', 'cell_008_002': 'ஊடுருவும்\nஒளி', 'cell_008_004': 'படிப்படியான நிறமாற்றம்', 'cell_008_005': 'சீரான ஒளியூட்டம்', 'cell_008_006': 'ஒளி பரப்பும் பலகை', 'cell_009_000': 'வண்ணக்\nகண்ணாடி', 'cell_009_002': 'நிற வெப்பநிலைப்\nபொருத்தம்', 'cell_009_004': 'நிரப்பு நிறம்', 'cell_009_005': 'நிறத் துல்லியம் மிக முக்கியம்', 'cell_009_006': 'வண்ண ஜெல் வடிகட்டிகள்', 'cell_010_000': 'மரம் (ஓக்)', 'cell_010_002': 'மேற்பரப்பை ஒட்டிச் சாய்வாக விழும் ஒளி', 'cell_010_004': 'இயற்கை நிறச்சாயல்', 'cell_010_005': 'மர நாரமைப்பை முன்னிலைப்படுத்தவும்', 'cell_010_006': 'ஃபிரெனெல் குவியொளி', 'cell_011_000': 'தோல்', 'cell_011_002': 'மேற்பரப்பு அமைப்பை வெளிப்படுத்தும் முதன்மை ஒளி', 'cell_011_004': 'பழுப்பு/வெளிர் பழுப்பு', 'cell_011_005': 'மேற்பரப்பு நுண்விவரங்களைக் காட்டவும்', 'cell_011_006': 'தேன்கூடு வலைப்பின்னல்', 'cell_012_000': 'கல்/பளிங்கு', 'cell_012_002': 'பல கோணங்கள்', 'cell_012_004': 'நடுநிலை', 'cell_012_005': 'ஆழமற்ற தோற்றமளிக்கும் ஒளியமைப்பைத் தவிர்க்கவும்', 'cell_012_006': 'பல ஒளி மூலங்கள்', 'table_title': 'பொருட்களின் பண்புகளின்படி புகைப்பட ஒளியமைப்புப் பகுப்பாய்வு'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
