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
    c = data['colors']
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(c['background'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    placements = []

    def text(key, x, y, size, width, anchor='left', color=None):
        p = {'key': key, 'x': x / W, 'y': y / H, 'size': size, 'max_width': width / W, 'anchor': anchor}
        if color is not None:
            p['color'] = color
        placements.append(p)

    def box(x, y, w, h):
        r = 18
        for (dx, dy, col) in [(0, 3, c['shadow']), (0, 0, c['edge'])]:
            ax.add_patch(Rectangle((x + r + dx, y + dy), w - 2 * r, h, facecolor=col, edgecolor='none'))
            ax.add_patch(Rectangle((x + dx, y + r + dy), w, h - 2 * r, facecolor=col, edgecolor='none'))
            for xx in [x + r, x + w - r]:
                for yy in [y + r, y + h - r]:
                    ax.add_patch(Circle((xx + dx, yy + dy), r, facecolor=col, edgecolor='none'))
        ax.add_patch(Rectangle((x + r, y + 1.5), w - 2 * r, h - 3, facecolor=c['panel'], edgecolor='none'))
        ax.add_patch(Rectangle((x + 1.5, y + r), w - 3, h - 2 * r, facecolor=c['panel'], edgecolor='none'))
        for xx in [x + r, x + w - r]:
            for yy in [y + r, y + h - r]:
                ax.add_patch(Circle((xx, yy), r - 1.5, facecolor=c['panel'], edgecolor='none'))
    text('title', 300, 124, 54, 950)
    text('credit', 1620, 131, 17, 400, 'right')
    for (i, card) in enumerate(data['cards']):
        x = data['card_x'][i % 3]
        y = data['card_y'][i // 3]
        (w, h) = data['card_size']
        box(x, y, w, h)
        (author, title, bg, fg) = card
        text(author, x + 16, y + 25, 15, 133, color=c['author'])
        text(title, x + 16, y + 69, 20, 137, color=c['ink'])
        (cw, ch) = data['cover_size']
        cx = x + w - cw - 11
        cy = y + 15
        ax.add_patch(Rectangle((cx, cy), cw, ch, facecolor=bg, edgecolor=c['edge'], linewidth=0.8))
        ax.add_patch(Rectangle((cx + 3, cy + 3), 3, ch - 6, facecolor=fg, alpha=0.25, edgecolor='none'))
        ax.plot([cx + 9, cx + cw - 7], [cy + 19, cy + 19], color=fg, lw=0.7)
        ax.plot([cx + 9, cx + cw - 7], [cy + ch - 13, cy + ch - 13], color=fg, lw=0.7)
        text(author, cx + cw / 2, cy + 9, 5.5, cw - 8, 'center', fg)
        text(title, cx + cw / 2, cy + ch / 2, 10, cw - 12, 'center', fg)
    for (x, y, w, h) in data['panel_boxes']:
        box(x, y, w, h)
    ax.add_patch(Circle((1130, 253), 19, facecolor='#c69458', edgecolor=c['edge']))
    ax.add_patch(Circle((1130, 245), 6, facecolor='#523727', edgecolor='none'))
    ax.add_patch(Rectangle((1123, 251), 15, 19, facecolor='#28231e', edgecolor='none'))
    text('summary', 1160, 253, 17, 440, color=c['ink'])
    ax.add_patch(Circle((1130, 546), 19, facecolor='#faf8e9', edgecolor=c['edge']))
    text('b_daring', 1130, 546, 7, 35, 'center', c['author'])
    text('favorite', 1160, 546, 17, 440, color=c['ink'])
    (x, y, w, h) = data['tree_area']
    ta = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ta.set_xlim(0, 1)
    ta.set_ylim(0, 1)
    ta.axis('off')
    for (i, (xx, yy, ww, hh, count, key)) in enumerate(data['tree_boxes']):
        ta.add_patch(Rectangle((xx, yy), ww, hh, facecolor=c['accent'] if i == 0 else c['panel'], edgecolor=c['edge'], linewidth=0.9))
        if key:
            text(key, x + xx * w + 12, y + (1 - yy - hh) * h + 21, 14, ww * w - 20, color='white' if i == 0 else '#111111')
    (x, y, w, h) = data['bars_area']
    n = len(data['pages'])
    ba = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ba.set_xlim(0, 324)
    ba.set_ylim(n - 0.55, -0.45)
    ba.axis('off')
    ba.barh(np.arange(n), data['pages'], height=0.84, color=[c['accent'] if i == data['highlight_index'] else c['panel'] for i in range(n)], edgecolor=c['edge'], linewidth=0.8)
    fig.canvas.draw()
    for (i, (key, v)) in enumerate(zip(data['bar_keys'], data['pages'])):
        (px, py) = ba.transData.transform((0, i))
        yy = H - py
        color = 'white' if i == data['highlight_index'] else '#111111'
        text(key, x + 5, yy, 14, min(v * w / 324 - 45, 360), color=color)
        if i == 0:
            text('pages_read', x + w - 8, yy, 14, 120, 'right', color)
        else:
            ba.text(v - 5, i, str(v), ha='right', va='center', fontsize=9, color=color)
    return finish(fig, labels, placements)
BASE_ID = 'qa_893d2ef71822f6af0ac69a1a9178df39322d4d7f0de3cdfc7c96bf45bfedd87c'
LANGUAGE = 'de'
DATA = {'canvas': [1920, 1080], 'card_x': [298, 560, 822], 'card_y': [228, 374, 519, 665, 810], 'card_size': [230, 122], 'cards': [['a_brown', 'b_atlas', '#ce0036', '#efcaa0'], ['a_friendship', 'b_friendship', '#f2a916', '#fff1b0'], ['a_brown', 'b_braving', '#78cfcc', '#164b4c'], ['a_creativity', 'b_creativity', '#f12110', '#181712'], ['a_brown', 'b_daring', '#f9f7ed', '#8db8b3'], ['a_brown', 'b_thought', '#314760', '#ef9c32'], ['a_ikigai', 'b_ikigai', '#a1d8e3', '#345154'], ['a_bach', 'b_jonathan', '#4136b5', '#ffffff'], ['a_frankl', 'b_meaning', '#a2deea', '#4a8194'], ['a_brown', 'b_rising', '#eff7f6', '#275782'], ['a_coelho', 'b_alchemist', '#fb6517', '#ffc322'], ['a_brown', 'b_gifts', '#279a97', '#e6c959'], ['a_ware', 'b_regrets', '#ffda09', '#e8ae21'], ['a_wooden', 'b_wooden', '#0487bd', '#ffdb19'], ['a_prentiss', 'b_zen', '#efa819', '#bd5735']], 'bar_keys': ['bar_creativity', 'bar_regrets', 'b_thought', 'b_rising', 'b_atlas', 'b_daring', 'b_friendship', 'b_wooden', 'b_ikigai', 'b_alchemist', 'b_gifts', 'b_meaning', 'b_braving', 'b_zen', 'b_jonathan'], 'pages': [324, 303, 285, 280, 276, 263, 212, 201, 185, 171, 170, 165, 163, 142, 127], 'highlight_index': 5, 'tree_boxes': [[0, 0, 0.4, 1, 6, 'tree_brown'], [0.4, 0, 0.2, 0.333333, 1, 'tree_prentiss'], [0.6, 0, 0.2, 0.333333, 1, 'tree_wooden'], [0.8, 0, 0.2, 0.333333, 1, 'tree_frankl'], [0.4, 0.333333, 0.2, 0.333334, 1, 'tree_ware'], [0.4, 0.666667, 0.2, 0.333333, 1, None], [0.6, 0.333333, 0.2, 0.333334, 1, None], [0.6, 0.666667, 0.2, 0.333333, 1, None], [0.8, 0.333333, 0.1, 0.666667, 1, None], [0.9, 0.333333, 0.1, 0.666667, 1, None]], 'panel_boxes': [[1090, 228, 530, 267], [1090, 520, 530, 414]], 'tree_area': [1110, 290, 495, 194], 'bars_area': [1109, 577, 489, 348], 'cover_size': [67, 91], 'colors': {'background': '#fff8f5', 'panel': '#f5eae4', 'edge': '#cdbfb4', 'shadow': '#ddd5cf', 'accent': '#ad650c', 'author': '#906423', 'ink': '#24211c'}}
LABELS = {'title': 'Bücher, die ich 2023 gelesen habe …', 'credit': 'gestaltet von @Ali15Tehrani', 'summary': 'Dieses Jahr habe ich 6 Bücher von Brené Brown gelesen – die pure Freude!', 'favorite': 'Müsste ich meinen Favoriten wählen, wäre es Großen Mut wagen!', 'pages_read': '324 Seiten gelesen', 'a_brown': 'Brené Brown', 'a_friendship': 'Sow & Friedman', 'a_creativity': 'Catmull & Wallace', 'a_ikigai': 'García & Miralles', 'a_bach': 'Richard Bach', 'a_frankl': 'Viktor Frankl', 'a_coelho': 'Paulo Coelho', 'a_ware': 'Bronnie Ware', 'a_wooden': 'John Wooden', 'a_prentiss': 'Chris Prentiss', 'b_atlas': 'Atlas des Herzens', 'b_friendship': 'Große Freundschaft', 'b_braving': 'Der Wildnis trotzen', 'b_creativity': 'Kreativität GmbH', 'b_daring': 'Großen Mut wagen', 'b_thought': 'Ich dachte, es geht nur mir so', 'b_ikigai': 'Der Sinn des Lebens', 'b_jonathan': 'Die Möwe Jonathan L.', 'b_meaning': 'Der Mensch auf der Suche nach Sinn', 'b_rising': 'Stark wieder aufstehen', 'b_alchemist': 'Der Alchemist', 'b_gifts': 'Die Gaben der Unvollkommenheit', 'b_regrets': 'Die 5 größten Bedauern Sterbender', 'b_wooden': 'Wooden', 'b_zen': 'Zen und die Kunst des Glücklichseins', 'bar_creativity': 'Kreativität GmbH', 'bar_regrets': 'Die 5 größten Bedauern Sterbender', 'tree_brown': 'Brown', 'tree_ware': 'Ware', 'tree_prentiss': 'Prentiss', 'tree_wooden': 'Wooden', 'tree_frankl': 'Frankl'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
