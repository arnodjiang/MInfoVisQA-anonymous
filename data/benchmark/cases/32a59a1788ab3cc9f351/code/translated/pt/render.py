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
    fig.patch.set_facecolor(data['colors']['background'])
    ax = fig.add_axes(data['plot_rect'])
    y = np.array(data['values'], dtype=float)
    x = np.linspace(data['x_range'][0], data['x_range'][1], len(y))
    ax.fill_between(x, 0, y, color=data['colors']['fill'], linewidth=0)
    ax.plot(x, y, color=data['colors']['edge'], linewidth=1.05)
    for p in data['separators']:
        height = np.interp(p, x, y)
        ax.plot([p, p], [0, height], color='white', alpha=0.3, linewidth=0.8)
    ax.set_xlim(*data['x_range'])
    ax.set_ylim(*data['y_range'])
    ax.axis('off')
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_xlim(0, W / 2)
    overlay.set_ylim(H / 2, 0)
    overlay.axis('off')
    for pts in data['leaders']:
        a = np.array(pts)
        overlay.plot(a[:, 0], a[:, 1], color=data['colors']['leader'], linewidth=1.35, solid_capstyle='butt')
    for yy in data['rule_y']:
        overlay.plot([10, 544], [yy, yy], color=data['colors']['rule'], linewidth=0.8)
    placements = []
    for (key, xx, yy, size, width, anchor) in data['text_placements']:
        placements.append({'key': key, 'x': xx / (W / 2), 'y': yy / (H / 2), 'size': size, 'max_width': width, 'rotation': 0, 'anchor': anchor})
    (left, bottom, width, height) = data['plot_rect']
    for (p, key) in zip(data['month_positions'], data['month_keys']):
        placements.append({'key': key, 'x': left + width * p / data['x_range'][1], 'y': 360 / (H / 2), 'size': 14, 'max_width': 0.065, 'rotation': 0, 'anchor': 'center'})
    return finish(fig, labels, placements)
BASE_ID = 'qa_7ddec4a68f6eb7ac174f52b4d5841624bcd6ba9bd763e88b6f4e08c5905ae236'
LANGUAGE = 'pt'
DATA = {'canvas': [1100, 798], 'plot_rect': [0.02, 0.1203, 0.9673, 0.4511], 'x_range': [0, 365], 'y_range': [0, 180], 'values': [34, 43, 42, 50, 48, 55, 52, 54, 58, 50, 53, 63, 61, 65, 63, 83, 88, 78, 86, 91, 74, 86, 90, 97, 97, 91, 111, 123, 119, 104, 136, 144, 139, 148, 176, 177, 153, 138, 141, 178, 151, 135, 72, 72, 83, 84, 74, 73, 93, 103, 71, 64, 69, 66, 52, 69, 62, 61, 70, 62, 67, 61, 60, 69, 64, 66, 75, 62, 69, 59, 63, 68, 62, 57, 47, 54, 56, 56, 54, 49, 57, 58, 51, 56, 59, 52, 59, 54, 68, 68, 75, 70, 75, 73, 80, 77, 71, 81, 68, 79, 83, 70, 61, 68, 45, 39, 44, 43, 39, 43, 36, 41, 40, 39, 34, 36, 38, 35, 32, 34, 32, 34, 36, 35, 31, 30, 33, 35, 37, 35, 37, 45, 44, 36, 44, 45, 41, 47, 46, 43, 43, 56, 49, 48, 45, 50, 54, 46, 48, 53, 47, 49, 57, 63, 66, 56, 51, 59, 56, 60, 54, 57, 49, 59, 54, 66, 73, 74, 77, 86, 91, 91, 91, 112, 122, 113, 125, 119, 142, 140, 145, 137, 168, 158, 170, 154, 174, 176, 156, 134, 97, 113, 115, 91, 55, 38, 37, 17, 37, 42, 40], 'month_positions': [15, 44, 74, 105, 136, 166, 197, 228, 258, 289, 319, 351], 'month_keys': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 'separators': [30, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], 'leaders': [[[65, 208], [72, 208], [72, 253]], [[81, 124], [109, 124], [109, 168]], [[185, 147], [143, 147], [143, 242]], [[187, 210], [173, 210], [173, 273]], [[155, 273], [192, 273]], [[284, 124], [278, 124], [278, 262], [264, 262]], [[278, 262], [296, 262]], [[465, 178], [501, 178]], [[463, 259], [537, 259], [537, 307]]], 'text_placements': [['title', 272, 21, 34, 0.8, 'center'], ['subtitle', 272, 34, 21, 0.83, 'center'], ['website', 272, 54, 22, 0.8, 'center'], ['valentine', 10, 208, 15, 0.12, 'left'], ['valentine_note', 10, 225, 14, 0.12, 'left'], ['spring', 29, 124, 15, 0.15, 'left'], ['spring_note', 29, 134, 14, 0.15, 'left'], ['april', 191, 148, 15, 0.19, 'left'], ['april_note', 191, 162, 14, 0.17, 'left'], ['monday', 191, 211, 15, 0.2, 'left'], ['monday_note', 191, 234, 14, 0.19, 'left'], ['summer', 287, 124, 15, 0.24, 'left'], ['summer_note', 287, 138, 14, 0.25, 'left'], ['christmas_before', 460, 183, 15, 0.25, 'right'], ['christmas_before_note', 460, 197, 14, 0.26, 'right'], ['christmas', 457, 259, 15, 0.23, 'right'], ['christmas_note', 457, 269, 14, 0.2, 'right'], ['credit', 10, 378, 17, 0.6, 'left'], ['credit_site', 10, 390, 17, 0.68, 'left'], ['source', 544, 378, 17, 0.4, 'right']], 'rule_y': [350, 366], 'colors': {'fill': '#e2e6f3', 'edge': '#b4bdcf', 'leader': '#626262', 'rule': '#b9b9b9', 'background': '#ffffff'}}
LABELS = {'title': 'Épocas com mais separações', 'subtitle': 'Segundo as atualizações de status no Facebook', 'website': 'InformationIsBeautiful.net', 'valentine': 'Dia de São Valentim', 'valentine_note': 'O namorado esqueceu\nde reservar o\nrestaurante?', 'spring': 'Férias de primavera', 'spring_note': 'Limpeza de primavera?', 'april': 'Dia da Mentira', 'april_note': 'Algum tipo de\nbrincadeira terrível', 'monday': 'Segundas-feiras', 'monday_note': 'Pessoas saindo de\nfins de semana terríveis,\npublicando suas más\nnotícias', 'summer': 'Férias de verão', 'summer_note': 'Quer ser jovem, livre\ne sem compromisso nestas férias?', 'christmas_before': 'Duas semanas antes das\nférias de Natal', 'christmas_before_note': 'Aliviar a consciência?', 'christmas': 'Dia de Natal', 'christmas_note': 'Cruel demais?', 'jan': 'JAN', 'feb': 'FEV', 'mar': 'MAR', 'apr': 'ABR', 'may': 'MAI', 'jun': 'JUN', 'jul': 'JUL', 'aug': 'AGO', 'sep': 'SET', 'oct': 'OUT', 'nov': 'NOV', 'dec': 'DEZ', 'credit': 'David McCandless e Lee Byron', 'credit_site': 'InformationIsBeautiful.net / LeeByron.com', 'source': 'fonte: Facebook Lexicon 2008'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
