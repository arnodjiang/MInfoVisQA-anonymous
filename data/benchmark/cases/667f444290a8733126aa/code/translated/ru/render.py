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
    fig.patch.set_facecolor('#f5f7f9')
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_axis_off()
    bg.add_patch(Rectangle((0.009, 0.012), 0.981, 0.976, facecolor='white', edgecolor='#e7ebef', linewidth=0.7))
    ax = fig.add_axes(data['axes_bounds'])
    x = np.array(data['years'])
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(0, 1000000)
    for i in data['years']:
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color='#f9f9f9', zorder=0)
    ax.set_yticks(data['y_ticks'])
    ax.set_yticklabels([format(v, ',').replace(',', ' ') for v in data['y_ticks']], fontsize=10, color='#666666')
    ax.grid(axis='y', linestyle=':', linewidth=0.7, color='#b8b8b8', zorder=1)
    bottom = np.zeros(len(x))
    for (key, color) in zip(data['series'], data['colors']):
        vals = np.array(data[key])
        ax.bar(x, vals, bottom=bottom, width=data['bar_width'], color=color, edgecolor='white', linewidth=0.65, zorder=3)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0, pad=14)
    for s in ['left', 'right', 'top']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#112b40')
    ax.spines['bottom'].set_linewidth(0.7)
    placements = [{'key': 'y_axis', 'x': 0.06, 'y': 0.354, 'size': 12, 'max_width': 0.28, 'rotation': 90, 'anchor': 'center'}]
    (left, bottom_a, width, height) = data['axes_bounds']
    for i in data['years']:
        placements.append({'key': 'year_' + str(i), 'x': left + (i + 0.5) * width / 16, 'y': 0.665, 'size': 14, 'max_width': 0.105, 'rotation': 45, 'anchor': 'right'})
    for (key, color, lx) in zip(data['series'], data['colors'], data['legend_positions']):
        bg.plot([lx], [1 - 0.783], marker='o', markersize=10, color=color, transform=bg.transAxes)
        placements.append({'key': key, 'x': lx + 0.017, 'y': 0.785, 'size': 16, 'max_width': 0.14, 'rotation': 0, 'anchor': 'left'})
    placements.extend([{'key': 'copyright', 'x': 0.943, 'y': 0.893, 'size': 14, 'max_width': 0.23, 'rotation': 0, 'anchor': 'right'}, {'key': 'additional', 'x': 0.049, 'y': 0.936, 'size': 14, 'max_width': 0.35, 'rotation': 0, 'anchor': 'left'}, {'key': 'source', 'x': 0.948, 'y': 0.936, 'size': 14, 'max_width': 0.25, 'rotation': 0, 'anchor': 'right'}])
    for cx in [0.039, 0.96]:
        bg.add_patch(Circle((cx, 1 - 0.935), 0.0065, transform=bg.transAxes, facecolor='#008be2', edgecolor='none'))
        bg.plot([cx, cx], [1 - 0.938, 1 - 0.932], transform=bg.transAxes, color='white', linewidth=1)
    for (j, yy) in enumerate(data['toolbar_y']):
        bg.add_patch(Rectangle((0.93, 1 - yy - 0.031), 0.044, 0.062, transform=bg.transAxes, facecolor='white', edgecolor='#edf0f3', linewidth=0.8))
        cx = 0.952
        cy = 1 - yy
        col = '#bdc7d3'
        if j == 0:
            angles = np.arange(10) * math.pi / 5 + math.pi / 2
            radii = np.array([0.009, 0.004] * 5)
            bg.add_patch(Polygon(np.column_stack([cx + np.cos(angles) * radii, cy + np.sin(angles) * radii * 1.4]), transform=bg.transAxes, color=col))
        elif j == 1:
            bg.add_patch(Polygon([[cx - 0.007, cy - 0.006], [cx + 0.007, cy - 0.006], [cx + 0.004, cy + 0.002], [cx + 0.004, cy + 0.008], [cx - 0.004, cy + 0.008], [cx - 0.004, cy + 0.002]], transform=bg.transAxes, color=col))
            bg.plot(cx, cy - 0.009, 'o', ms=2, color=col, transform=bg.transAxes)
        elif j == 2:
            bg.plot(cx, cy, marker=(8, 1, 0), ms=11, color=col, transform=bg.transAxes)
            bg.plot(cx, cy, 'o', ms=3, color='white', transform=bg.transAxes)
        elif j == 3:
            bg.plot([cx + 0.004, cx - 0.005, cx + 0.004], [cy + 0.008, cy, cy - 0.008], '-o', ms=4, lw=1.6, color=col, transform=bg.transAxes)
        elif j == 4:
            for dx in [-0.004, 0.004]:
                bg.add_patch(Rectangle((cx + dx - 0.003, cy), 0.005, 0.008, transform=bg.transAxes, color=col))
                bg.plot([cx + dx + 0.002, cx + dx + 0.001, cx + dx - 0.002], [cy, cy - 0.006, cy - 0.008], color=col, lw=1.8, transform=bg.transAxes)
        else:
            bg.add_patch(Rectangle((cx - 0.007, cy - 0.005), 0.014, 0.01, transform=bg.transAxes, fill=False, edgecolor=col))
            bg.add_patch(Rectangle((cx - 0.004, cy + 0.003), 0.008, 0.008, transform=bg.transAxes, facecolor='white', edgecolor=col))
            bg.add_patch(Rectangle((cx - 0.004, cy - 0.009), 0.008, 0.008, transform=bg.transAxes, facecolor='white', edgecolor=col))
    bg.plot([0.953, 0.953], [0.095, 0.114], color='#4c4c4c', lw=1, transform=bg.transAxes)
    bg.add_patch(Polygon([[0.953, 0.113], [0.959, 0.111], [0.965, 0.114], [0.965, 0.104], [0.959, 0.102], [0.953, 0.104]], transform=bg.transAxes, color='#4c4c4c'))
    return finish(fig, labels, placements)
BASE_ID = 'qa_5a45c0fbd94873cf430f6535c46b1bfed66372994c5d173dde721ee53f9f53b0'
LANGUAGE = 'ru'
DATA = {'canvas': [1000, 696], 'years': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'undergraduate': [239000, 235000, 236000, 242000, 269000, 272000, 289000, 308000, 339000, 369000, 399000, 426000, 441000, 442000, 432000, 419000], 'graduate': [263000, 260000, 266000, 277000, 283000, 293000, 296000, 300000, 309000, 329000, 359000, 382000, 386000, 382000, 377000, 371000], 'non_degree': [30000, 30000, 36000, 46000, 53000, 56000, 60000, 70000, 73000, 78000, 93000, 86000, 73000, 63000, 63000, 60000], 'series': ['undergraduate', 'graduate', 'non_degree'], 'colors': ['#2b76dc', '#112b40', '#b9b9b9'], 'y_ticks': [0, 250000, 500000, 750000, 1000000], 'axes_bounds': [0.166, 0.377, 0.742, 0.54], 'bar_width': 0.77, 'legend_positions': [0.298, 0.45, 0.559], 'toolbar_y': [0.067, 0.143, 0.218, 0.294, 0.37, 0.445]}
LABELS = {'y_axis': 'Число студентов', 'undergraduate': 'Бакалавриат', 'graduate': 'Магистратура и аспирантура', 'non_degree': 'Обучение без получения степени', 'year_0': '2004/05', 'year_1': '2005/06', 'year_2': '2006/07', 'year_3': '2007/08', 'year_4': '2008/09', 'year_5': '2009/10', 'year_6': '2010/11', 'year_7': '2011/12', 'year_8': '2012/13', 'year_9': '2013/14', 'year_10': '2014/15', 'year_11': '2015/16', 'year_12': '2016/17', 'year_13': '2017/18', 'year_14': '2018/19', 'year_15': '2019/20', 'copyright': '© Statista 2021', 'additional': 'Дополнительная информация', 'source': 'Показать источник'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
