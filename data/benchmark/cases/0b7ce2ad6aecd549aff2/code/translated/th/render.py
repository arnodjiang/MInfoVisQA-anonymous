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

    def panel_00(data, labels):
        (W, H) = (data['width'], data['height'])
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis('off')
        placements = []
        h = data['header_height']
        n = len(data['row_keys'])
        rh = (0.995 - h) / n
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=data['border_color'], edgecolor='none'))
        ax.add_patch(Rectangle((0.007, 0.004), 0.982, h - 0.004, facecolor=data['header_color'], edgecolor='none'))
        placements.append({'key': 'title', 'x': 0.014, 'y': 0.018, 'size': 39, 'max_width': 0.6, 'anchor': 'left', 'color': 'white'})
        ax.plot([0.018, 0.171], [0.052, 0.052], color=data['accent_color'], linewidth=3)
        for (key, x) in zip(['households', 'hh_percent', 'us_percent', 'index'], data['column_x'][1:]):
            placements.append({'key': key, 'x': x, 'y': 0.04, 'size': 22, 'max_width': 0.15, 'anchor': 'center', 'color': 'white'})
        for (i, key) in enumerate(data['row_keys']):
            y = h + i * rh
            yc = y + rh / 2
            ax.add_patch(Rectangle((0.007, y), 0.982, rh, facecolor=data['row_colors'][i], edgecolor=data['grid_color'], linewidth=1.1))
            placements.append({'key': key, 'x': data['column_x'][0], 'y': yc, 'size': 23, 'max_width': 0.405, 'anchor': 'left'})
            vals = [format(data['households'][i], ',d'), format(data['hh_percent'][i], '.2f') + '%', format(data['us_percent'][i], '.1f') + '%', str(data['index'][i])]
            for (x, v) in zip(data['column_x'][1:], vals):
                ax.text(x, yc, v, ha='center', va='center', fontsize=17, color='#25332b')
        ax.add_patch(Rectangle((0.005, 0.003), 0.986, 0.992, fill=False, edgecolor=data['border_color'], linewidth=2))
        return finish(fig, labels, placements)

    def panel_01(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 165)
        ax.set_ylim(135, 0)
        ax.set_axis_off()
        ax.set_facecolor(data['background'])
        for pts in data['icon_polygons']:
            ax.add_patch(Polygon(pts, closed=True, facecolor=data['icon_color'], edgecolor='none'))
        for (x, y, w, h) in data['icon_rectangles']:
            ax.add_patch(Rectangle((x, y), w, h, facecolor=data['icon_color'], edgecolor='none'))
        for (x, y, w, h) in data['icon_cutouts']:
            ax.add_patch(Rectangle((x, y), w, h, facecolor=data['background'], edgecolor='none'))
        placements = [{'key': 'title', 'x': 0.5, 'y': 0.06, 'size': 53, 'max_width': 0.72, 'anchor': 'center', 'color': data['foreground']}]
        for m in data['metrics']:
            ax.text(m['x'] * 165, m['y'] * 135, m['format'].format(m['value']), ha='center', va='center', fontsize=m['size'] * 72 / 100, color=data['foreground'], fontfamily='DejaVu Sans', fontweight='normal')
            placements.append({'key': m['caption'], 'x': m['x'], 'y': m['caption_y'], 'size': m['caption_size'], 'max_width': 0.46 if m['caption'] != 'ratio' else 0.36, 'anchor': 'center', 'color': data['foreground']})
        return finish(fig, labels, placements)

    def panel_02(data, labels):
        (W, H) = (900, 1050)
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis('off')
        placements = [{'key': 'title', 'x': 0.48, 'y': 0.062, 'size': 72, 'max_width': 0.88, 'anchor': 'center'}]
        c = data['cyan']
        iw = data['icon_width']
        ih = data['icon_height']
        for (i, cx) in enumerate(data['centers']):
            cy = data['icon_y'][i]

            def pts(p):
                return [(cx + (x - 0.5) * iw, cy + (y - 0.5) * ih) for (x, y) in p]

            def line(p, lw=8):
                q = pts(p)
                ax.plot([v[0] for v in q], [v[1] for v in q], color=c, lw=lw * 0.72, solid_capstyle='round', solid_joinstyle='round')

            def box(x, y, w, h, fill=False):
                q = pts([[x, y]])[0]
                ax.add_patch(Rectangle(q, w * iw, h * ih, edgecolor=c, facecolor=c if fill else 'none', linewidth=4.5))
            if i < 2:
                if i == 1:
                    ax.add_patch(Polygon(pts(data['cap_diamond']), closed=True, facecolor=c, edgecolor=c, linewidth=3))
                else:
                    line(data['cap_diamond'])
                line(data['cap_base'], 6)
                line(data['tassel'], 6)
                line([[0.1, 0.87], [0.15, 1.03]], 4)
            elif i == 2:
                for (x, y, w, h) in data['book_blocks']:
                    box(x, y, w, h)
                line([[-0.03, 1.04], [1, 1.04]], 7)
                line([[0.31, 0.24], [0.45, 0.24]], 5)
                line([[0.31, 0.34], [0.45, 0.34]], 5)
                line([[0.31, 0.75], [0.45, 0.75]], 4)
                line([[0.07, 0.57], [0.17, 0.57]], 4)
                line([[0.61, 0.64], [0.88, 0.64]], 4)
                line([[0.61, 0.74], [0.88, 0.74]], 4)
            else:
                line(data['school_outline'], 6)
                line([[-0.06, 1.06], [1.05, 1.06]], 7)
                line([[0.07, 0.25], [0.2, 0.1], [0.33, 0.25]], 5)
                line([[0.69, 0.25], [0.81, 0.1], [0.94, 0.25]], 5)
                for (x, y) in data['school_windows']:
                    box(x, y, 0.055, 0.095, True)
                box(0.43, 0.68, 0.14, 0.34)
                q = pts([[0.5, 0.34]])[0]
                ax.add_patch(Circle(q, 0.013, facecolor='none', edgecolor=c, linewidth=4))
            ax.text(cx, data['value_y'][i], str(data['percentages'][i]) + '%', ha='center', va='center', fontsize=81, color=data['text'], fontweight='normal')
            placements.append({'key': data['category_keys'][i], 'x': cx, 'y': data['caption_y'][i], 'size': 39, 'max_width': 0.48, 'anchor': 'center'})
        return finish(fig, labels, placements)

    def panel_03(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 239)
        ax.set_ylim(264, 0)
        ax.set_axis_off()
        placements = [{'key': 'title', 'x': 0.012, 'y': 0.031, 'size': 36, 'max_width': 0.65, 'anchor': 'left'}]
        land = Polygon(data['outline'], closed=True, facecolor=data['map_color'], edgecolor='#454746', lw=1)
        ax.add_patch(land)
        for road in data['minor_roads']:
            p = np.array(road)
            (line,) = ax.plot(p[:, 0], p[:, 1], color='#646664', lw=1.7, alpha=0.35)
            line.set_clip_path(land)
        for road in data['major_roads']:
            p = np.array(road)
            (line,) = ax.plot(p[:, 0], p[:, 1], color='#747573', lw=2.4, alpha=0.4)
            line.set_clip_path(land)
        area = Polygon(data['highlight'], closed=True, facecolor='#ad3547', edgecolor='#8c3540', alpha=0.46, lw=2)
        ax.add_patch(area)
        area.set_clip_path(land)
        for road in data['red_roads']:
            p = np.array(road)
            (line,) = ax.plot(p[:, 0], p[:, 1], color='#ac3748', alpha=0.7, lw=3)
            line.set_clip_path(land)
        p = np.array(data['boundary'])
        (line,) = ax.plot(p[:, 0], p[:, 1], color='#272f30', lw=4)
        line.set_clip_path(land)
        for (key, x, y, s) in data['place_labels']:
            placements.append({'key': key, 'x': x / 239, 'y': y / 264, 'size': s * 4, 'max_width': 0.2, 'anchor': 'center'})
        (x, y) = data['marker']
        ax.add_patch(Circle((x, y), 7, color='#087fc3', alpha=0.18))
        ax.add_patch(Rectangle((x - 3, y - 1), 6, 8, facecolor='#16a8e9', edgecolor='none'))
        ax.add_patch(Circle((x, y - 2), 3.3, facecolor='#6fe7ff', edgecolor='#12a1de', lw=2))
        ax.add_patch(Circle((x, y - 2), 1.7, facecolor='#dcffff', edgecolor='none'))
        for (x, y, w, h) in data['controls']:
            ax.add_patch(Rectangle((x, y), w, h, facecolor='#f0efeb', edgecolor='#b8bab7', lw=1))
        ax.plot([223, 229], [54, 54], color='#5f625f', lw=3)
        ax.plot([226, 226], [51, 57], color='#5f625f', lw=3)
        ax.plot([223, 229], [69, 69], color='#5f625f', lw=3)
        ax.add_patch(Polygon([[222.5, 90], [226, 87], [229.5, 90], [228.5, 90], [228.5, 94], [226.5, 94], [226.5, 91.5], [225, 91.5], [225, 94], [223.5, 94], [223.5, 90]], facecolor='#777b77', edgecolor='none'))
        ax.add_patch(Circle((227, 38), 2.6, fill=False, edgecolor='#999b96', lw=2))
        ax.plot([229, 231], [40, 42], color='#999b96', lw=2)
        return finish(fig, labels, placements)

    def panel_04(data, labels):
        (W, H) = data['canvas']
        c = data['colors']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(c['background'])
        ax = fig.add_axes(data['axes_rect'])
        ax.set_facecolor(c['background'])
        ax.bar(data['x'], data['bars_percent'], width=data['bar_width'], color=c['bar'], edgecolor=c['bar'], linewidth=0, zorder=2)
        ax.plot(data['x'], data['comparison_percent'], color=c['line'], linewidth=1.4, marker='o', markersize=3.8, markerfacecolor=c['marker'], markeredgecolor=c['line'], markeredgewidth=0.5, zorder=3)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_yticks(data['yticks'])
        ax.set_xticks(data['x'])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', colors=c['text'], labelsize=10, length=4, width=0.6, pad=5)
        ax.tick_params(axis='x', colors=c['text'], length=3, width=0.6)
        ax.grid(axis='y', color=c['grid'], linewidth=0.65, alpha=0.8, zorder=0)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_color(c['text'])
            ax.spines[s].set_linewidth(0.8)
        placements = [{'key': 'title', 'x': 0.5, 'y': 0.039, 'size': 29, 'max_width': 0.7, 'anchor': 'center'}, {'key': 'y_axis', 'x': 0.016, 'y': 0.449, 'size': 15, 'max_width': 0.26, 'rotation': 90, 'anchor': 'center'}]
        fig.canvas.draw()
        for (x, key) in zip(data['x'], data['age_keys']):
            (px, py) = ax.transData.transform((x, 0))
            placements.append({'key': key, 'x': px / W, 'y': 1 - py / H + 0.036, 'size': 13, 'max_width': 0.09, 'rotation': 45, 'anchor': 'right'})
        overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
        overlay.set_xlim(0, 1)
        overlay.set_ylim(0, 1)
        overlay.set_axis_off()
        (sx, sy, sw, sh) = data['selector_rect']
        overlay.add_patch(Rectangle((sx, sy), sw, sh, facecolor=c['selector'], edgecolor=c['selector_border'], linewidth=0.7))
        overlay.add_patch(Polygon(data['selector_arrow'], closed=True, facecolor=c['selector_border'], edgecolor='none'))
        placements.extend([{'key': 'comparison_caption', 'x': 0.773, 'y': 0.951, 'size': 13, 'max_width': 0.33, 'anchor': 'right'}, {'key': 'comparison_selection', 'x': sx + 0.009, 'y': 1 - sy - sh / 2, 'size': 11, 'max_width': sw - 0.033, 'anchor': 'left', 'color': '#77766b'}])
        return finish(fig, labels, placements)

    def panel_05(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis('off')
        placements = []

        def put(key, x, y, size, wide, anchor='left'):
            placements.append({'key': key, 'x': x / W, 'y': y / H, 'size': size, 'max_width': wide / W, 'rotation': 0, 'anchor': anchor})
        put('title', 13, 20, 19, 950)
        put('largest', 13, 48, 16, 980)
        put('smallest', 13, 74, 16, 980)
        for (key, x, wide) in [('variable', 15, 330), ('value', 362, 140), ('difference', 511, 170)]:
            put(key, x, 101, 14, wide)
        (x, y, w, h) = data['table_bounds']
        n = len(data['row_keys'])
        rh = h / n
        for (i, key) in enumerate(data['row_keys']):
            cy = y + (i + 0.5) * rh
            ax.add_patch(Rectangle((x, y + i * rh), w, rh, facecolor=data['row_fills'][i % 2], edgecolor='none'))
            a = data['bar_left_pixels'][i]
            b = data['bar_right_pixels'][i]
            ax.add_patch(Rectangle((a, cy - data['bar_height'] / 2), b - a, data['bar_height'], facecolor=data['bar_colors'][i], edgecolor='#9fdee7', linewidth=0.35))
            put(key, x + 6, cy, data['table_text_size'], 330)
            ax.text(368, cy, format(data['values_percent'][i], '.1f') + '%', ha='left', va='center', fontsize=10, color='#eeeae0')
            d = data['differences_percentage_points'][i]
            ax.text(514, cy, format(d, '+.1f') + '%', ha='left', va='center', fontsize=10, color='#24afc0' if d > 0 else '#b7dce4')
        for xx in data['column_edges']:
            ax.plot([xx, xx], [y, y + h], color=data['grid_color'], linewidth=0.7)
        for i in range(n + 1):
            yy = y + i * rh
            ax.plot([x, x + w], [yy, yy], color=data['grid_color'], linewidth=0.65)
        ax.plot([data['baseline_pixel']] * 2, [y, y + h], color='#a9a69d', linewidth=0.7)
        put('comparison_caption', 1005, 493, 14, 365, 'right')
        (sx, sy, sw, sh) = data['selector_bounds']
        ax.add_patch(Rectangle((sx, sy), sw, sh, facecolor='#fcfcfb', edgecolor='#888577', linewidth=0.8))
        put('comparison_source', sx + 9, sy + sh / 2, 12, sw - 33)
        ax.add_patch(Polygon([[sx + sw - 20, sy + 14], [sx + sw - 10, sy + 14], [sx + sw - 15, sy + 20]], closed=True, facecolor='#666666', edgecolor='none'))
        return finish(fig, labels, placements)

    def panel_06(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis('off')
        placements = [{'key': 'title', 'x': 24 / W, 'y': 25 / H, 'size': 25, 'max_width': 0.8, 'anchor': 'left'}]
        for (i, c) in enumerate(data['cards']):
            top = c['top']
            cy = top + data['portrait_offset_y']
            cx = data['portrait_x']
            r = data['portrait_radius']
            ax.add_patch(Rectangle((data['card_left'], top), data['card_width'], c['height'], facecolor=data['card_color'], edgecolor=data['card_edge'], linewidth=1.5))
            ax.plot([data['column_x']] * 2, [top, top + c['height']], color=data['separator'], lw=1)
            clip = Circle((cx, cy), r - 5, facecolor=data['portrait_backgrounds'][i], edgecolor='none')
            ax.add_patch(clip)
            for p in data['people'][i]:
                x = cx + p['x']
                y = cy + p['y']
                hr = p['head_r']
                bw = p['body_w']
                bh = p['body_h']
                shapes = [Polygon([[x - bw / 2, y + bh], [x - bw / 2, y + hr + 10], [x - 9, y + hr - 1], [x + 9, y + hr - 1], [x + bw / 2, y + hr + 10], [x + bw / 2, y + bh]], facecolor=p['shirt'], edgecolor='none'), Circle((x, y - 3), hr + 4, facecolor=p['hair'], edgecolor='none'), Circle((x, y + 3), hr, facecolor=p['skin'], edgecolor='none'), Polygon([[x - hr, y - 1], [x - hr, y - hr + 1], [x, y - hr - 5], [x + hr, y - hr + 3], [x + hr, y + 1], [x + 5, y - hr + 7], [x - 6, y - hr + 5]], facecolor=p['hair'], edgecolor='none')]
                for s in shapes:
                    s.set_clip_path(clip)
                    ax.add_patch(s)
                for dx in [-6, 6]:
                    eye = Circle((x + dx, y + 2), 1.4, facecolor='#584839', edgecolor='none')
                    eye.set_clip_path(clip)
                    ax.add_patch(eye)
                ax.plot([x - 4, x + 4], [y + 12, y + 12], color='#976f5c', lw=0.8, clip_path=clip)
            ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor=c['accent'], linewidth=7))
            bx = data['badge_x']
            by = top + data['badge_offset_y']
            ax.add_patch(Circle((bx, by), data['badge_radius'], facecolor='#f0f3eb', edgecolor=c['accent'], linewidth=5))
            placements.append({'key': c['badge'], 'x': bx / W, 'y': by / H, 'size': 20, 'max_width': 0.055, 'anchor': 'center', 'color': '#666b65'})
            placements.append({'key': c['name'], 'x': data['name_x'] / W, 'y': (top + data['name_offset_y']) / H, 'size': 24, 'max_width': 0.32, 'anchor': 'left', 'color': '#ffffff'})
            gy = top + data['group_offset_y'] + (17 if i == 2 else 0)
            placements.append({'key': c['group'], 'x': data['name_x'] / W, 'y': gy / H, 'size': 16, 'max_width': 0.31, 'anchor': 'left', 'color': '#aeb0aa'})
            ax.text(data['percent_x'], top + data['percent_offset_y'], str(c['percent']) + '%', fontsize=20, color='#f4f4ef', fontweight='bold', ha='left', va='center')
            placements.append({'key': 'share_caption', 'x': data['percent_x'] / W, 'y': (top + data['caption_offset_y']) / H, 'size': 15, 'max_width': 0.25, 'anchor': 'left', 'color': '#aeb0aa'})
            ax.plot(data['chevron_x'], [top + v for v in data['chevron_offset_y']], color='#e8e9e3', lw=3, solid_capstyle='round')
        return finish(fig, labels, placements)
    functions = [panel_00, panel_01, panel_02, panel_03, panel_04, panel_05, panel_06]
    (width, height) = data['canvas']
    image = Image.new('RGB', (width, height), 'white')
    boxes = []
    for (i, panel) in enumerate(data['panels']):
        local = {k: labels[v] for (k, v) in panel['label_map'].items()}
        (part, part_boxes) = functions[i](panel['data'], local)
        (left, top, right, bottom) = panel['bbox']
        (x, y) = (round(left * width), round(top * height))
        (w, h) = (round((right - left) * width), round((bottom - top) * height))
        (sx, sy) = (w / part.width, h / part.height)
        image.paste(part.resize((w, h), Image.Resampling.LANCZOS), (x, y))
        for box in part_boxes:
            b = dict(box)
            (a, bb, c, d) = b['box']
            b['box'] = [round(x + a * sx), round(y + bb * sy), round(x + c * sx), round(y + d * sy)]
            b['label_key'] = panel['label_map'].get(b.get('label_key'), b.get('label_key'))
            b['effective_font_size'] = b.get('font_size', 0) * min(sx, sy)
            b['inside_canvas'] = b['box'][0] >= 0 and b['box'][1] >= 0 and (b['box'][2] <= width) and (b['box'][3] <= height)
            boxes.append(b)
    for p in data['global_placements']:
        b = put(image, labels[p['key']], p['x'] * width, p['y'] * height, p.get('size', 30), max_width=p.get('max_width', 0.8) * width)
        b['label_key'] = p['key']
        boxes.append(b)
    return (image, boxes)
BASE_ID = 'qa_9793d5f4d5bed0969050c8eebcbd190f034023151404820d4bed4a7156c86dbd'
LANGUAGE = 'th'
DATA = {'canvas': [2800, 1770], 'panels': [{'bbox': [0.007, 0.158, 0.275, 0.994], 'data': {'width': 900, 'height': 1780, 'header_height': 0.071, 'row_keys': ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14'], 'households': [3763, 1421, 0, 2606, 3386, 0, 3376, 5544, 1289, 20, 5802, 0, 247, 0], 'hh_percent': [13.49, 5.17, 0, 9.49, 12.33, 0, 12.01, 20.25, 4.69, 0.07, 20.41, 0, 0.9, 0], 'us_percent': [9.9, 5.7, 3.3, 7.5, 11.5, 12, 7.1, 10.9, 5.8, 6.2, 4.2, 6.1, 4.9, 1.4], 'index': [138, 92, 0, 127, 107, 0, 184, 186, 81, 1, 329, 0, 23, 0], 'row_colors': ['#f5822b', '#64b546', '#83cde3', '#ef8cb2', '#f4e829', '#079148', '#ac8bc1', '#0aabd5', '#ed607b', '#e9d2a4', '#00aaa1', '#ffbf08', '#a58ac0', '#a7c576'], 'column_x': [0.017, 0.51, 0.648, 0.786, 0.937], 'border_color': '#787c76', 'header_color': '#626463', 'grid_color': '#b5bab1', 'accent_color': '#008fac'}, 'label_map': {'title': 'panel_00_title', 'households': 'panel_00_households', 'hh_percent': 'panel_00_hh_percent', 'us_percent': 'panel_00_us_percent', 'index': 'panel_00_index', 'r1': 'panel_00_r1', 'r2': 'panel_00_r2', 'r3': 'panel_00_r3', 'r4': 'panel_00_r4', 'r5': 'panel_00_r5', 'r6': 'panel_00_r6', 'r7': 'panel_00_r7', 'r8': 'panel_00_r8', 'r9': 'panel_00_r9', 'r10': 'panel_00_r10', 'r11': 'panel_00_r11', 'r12': 'panel_00_r12', 'r13': 'panel_00_r13', 'r14': 'panel_00_r14'}, 'crop_pixel_bbox': [6, 82, 227, 518], 'crop_sha256': '3d330277d0449fb50410a093b85b8b9e20e1f3c8176105315f9cec92a27d93bb', 'api_request_sha256': '338b2f946bb8775442b3f0db9305b81e7f76885cdc13a0dc490023db6117269d'}, {'bbox': [0.285, 0.015, 0.486, 0.274], 'data': {'canvas': [990, 810], 'background': '#302511', 'foreground': '#ffffff', 'icon_color': '#009dbc', 'metrics': [{'value': 375958, 'format': '${:,.0f}', 'x': 0.221, 'y': 0.381, 'size': 80, 'caption': 'home_value', 'caption_y': 0.485, 'caption_size': 29}, {'value': 65779, 'format': '${:,.0f}', 'x': 0.764, 'y': 0.381, 'size': 80, 'caption': 'income', 'caption_y': 0.485, 'caption_size': 29}, {'value': 5.7, 'format': '{:.1f}', 'x': 0.5, 'y': 0.578, 'size': 78, 'caption': 'ratio', 'caption_y': 0.695, 'caption_size': 29}, {'value': 35.7, 'format': '{:.1f}', 'x': 0.206, 'y': 0.847, 'size': 84, 'caption': 'age', 'caption_y': 0.948, 'caption_size': 29}, {'value': 27809, 'format': '{:,.0f}', 'x': 0.785, 'y': 0.847, 'size': 84, 'caption': 'population', 'caption_y': 0.948, 'caption_size': 29}], 'icon_polygons': [[[25, 29], [33, 22], [41, 29]], [[30, 25], [40, 17], [51, 25]], [[33, 23], [33, 21], [36, 21], [36, 23]], [[117, 22], [125, 18], [134, 23]]], 'icon_rectangles': [[27, 29, 12, 7], [32, 25, 17, 10], [119, 23, 14, 12], [117, 24, 3, 11], [130, 28, 5, 7]], 'icon_cutouts': [[35, 27, 3, 2], [40, 27, 3, 2], [45, 27, 2, 2], [41, 31, 5, 4], [29, 31, 2, 2], [33, 31, 3, 2], [121, 25, 3, 2], [126, 25, 3, 2], [121, 29, 3, 2], [126, 29, 3, 2], [121, 33, 3, 2], [126, 33, 3, 2], [131, 30, 2, 2], [131, 33, 2, 2]]}, 'label_map': {'title': 'panel_01_title', 'home_value': 'panel_01_home_value', 'income': 'panel_01_income', 'ratio': 'panel_01_ratio', 'age': 'panel_01_age', 'population': 'panel_01_population'}, 'crop_pixel_bbox': [235, 8, 400, 143], 'crop_sha256': '5573ea1f31b99fa53e84d48e87c22e2bf7eea0fbc0789734f397db3d0625f166', 'api_request_sha256': 'ec376106889126e39f0101feafdfce583a81e243307cbe7ba4911d2240c305ae'}, {'bbox': [0.528, 0.015, 0.667, 0.274], 'data': {'percentages': [11, 21, 32, 36], 'category_keys': ['no_hs', 'hs', 'college', 'degree'], 'centers': [0.21, 0.75, 0.21, 0.75], 'icon_y': [0.215, 0.215, 0.685, 0.685], 'value_y': [0.402, 0.402, 0.865, 0.865], 'caption_y': [0.489, 0.489, 0.958, 0.958], 'background': '#30291b', 'cyan': '#02bddd', 'text': '#f6f2e9', 'icon_width': 0.24, 'icon_height': 0.155, 'cap_diamond': [[0.08, 0.38], [0.43, 0.06], [0.87, 0.33], [0.48, 0.65], [0.08, 0.38]], 'cap_base': [[0.22, 0.5], [0.22, 0.79], [0.46, 0.95], [0.71, 0.73], [0.71, 0.47]], 'tassel': [[0.1, 0.4], [0.1, 0.92], [0.04, 1.04]], 'book_blocks': [[0.03, 0.49, 0.19, 0.49], [0.25, 0.15, 0.27, 0.83], [0.56, 0.55, 0.37, 0.43]], 'school_outline': [[0.01, 1.02], [0.01, 0.42], [0.14, 0.42], [0.14, 0.25], [0.27, 0.25], [0.27, 0.43], [0.38, 0.43], [0.38, 0.17], [0.5, 0.04], [0.63, 0.17], [0.63, 0.43], [0.74, 0.43], [0.74, 0.25], [0.87, 0.25], [0.87, 0.43], [0.99, 0.43], [0.99, 1.02], [0.01, 1.02]], 'school_windows': [[0.1, 0.55], [0.25, 0.55], [0.69, 0.55], [0.85, 0.55], [0.1, 0.77], [0.25, 0.77], [0.69, 0.77], [0.85, 0.77]]}, 'label_map': {'title': 'panel_02_title', 'no_hs': 'panel_02_no_hs', 'hs': 'panel_02_hs', 'college': 'panel_02_college', 'degree': 'panel_02_degree'}, 'crop_pixel_bbox': [435, 8, 550, 143], 'crop_sha256': '0ad54bf9920a0406f88f3d66ec3bba6113ff9d21019f06069e5b8e7e377d59a4', 'api_request_sha256': '13ec66ce0b6acb111ac74e3a63a55c5b86c5b269dd232f504afee955b04d4f86'}, {'bbox': [0.698, 0.015, 0.988, 0.523], 'data': {'canvas': [956, 1056], 'bounds': [0, 239, 264, 0], 'background': '#2b220f', 'map_color': '#505252', 'outline': [[232, 35], [112, 35], [89, 39], [65, 48], [46, 62], [29, 83], [17, 108], [9, 135], [8, 155], [13, 183], [25, 207], [44, 229], [67, 245], [94, 255], [120, 259], [148, 255], [174, 245], [197, 229], [213, 209], [224, 184], [232, 157]], 'highlight': [[54, 157], [68, 148], [84, 140], [101, 136], [111, 127], [113, 117], [125, 112], [133, 119], [133, 135], [146, 139], [153, 147], [165, 151], [170, 164], [161, 174], [169, 182], [183, 187], [194, 192], [173, 190], [158, 183], [148, 188], [144, 195], [136, 185], [125, 184], [121, 174], [109, 174], [100, 168], [82, 165], [76, 158], [63, 166], [58, 175], [53, 171]], 'boundary': [[11, 176], [20, 167], [30, 169], [40, 160], [52, 154], [64, 156], [73, 146], [82, 141], [91, 139], [96, 133], [108, 134], [112, 139], [120, 136], [134, 136], [143, 140], [151, 137], [158, 133], [171, 131], [180, 127], [184, 122], [190, 119], [193, 110], [201, 109], [207, 105], [206, 95], [211, 95], [214, 87], [222, 84]], 'major_roads': [[[11, 88], [46, 87], [78, 87], [95, 94], [112, 108], [116, 132], [120, 158], [127, 184], [134, 211], [143, 253]], [[14, 193], [42, 192], [68, 191], [93, 189], [118, 187], [145, 187], [174, 191], [202, 196], [217, 208]], [[37, 72], [38, 96], [44, 121], [49, 143], [54, 159], [59, 187], [60, 224]], [[20, 127], [49, 127], [74, 129], [94, 129], [117, 129]], [[87, 48], [87, 71], [99, 83], [117, 87], [145, 88], [177, 87], [205, 87], [229, 83]]], 'minor_roads': [[[24, 103], [42, 100], [66, 102], [83, 106], [100, 105]], [[26, 113], [67, 113], [67, 136], [94, 137]], [[20, 145], [42, 147], [64, 146], [66, 156], [94, 155], [102, 159], [132, 158], [166, 158], [196, 160], [220, 163]], [[22, 180], [44, 180], [44, 172], [73, 173], [95, 174], [99, 179], [119, 179]], [[73, 96], [74, 118], [73, 148], [75, 175], [76, 202], [80, 235]], [[96, 100], [95, 124], [101, 145], [100, 179], [103, 203], [128, 204], [162, 205], [200, 208]], [[139, 37], [137, 53], [151, 58], [158, 53], [176, 57], [189, 64], [216, 66]], [[134, 49], [132, 75], [127, 96], [125, 120], [125, 147]], [[161, 68], [157, 98], [154, 124], [157, 147]], [[182, 99], [178, 124], [181, 146], [187, 172], [191, 192]], [[39, 211], [64, 213], [89, 211], [110, 214], [129, 213]], [[52, 61], [57, 78], [58, 104], [61, 125]], [[13, 157], [34, 154], [44, 157], [49, 174]], [[113, 220], [112, 248]], [[153, 213], [151, 245]], [[92, 54], [117, 55], [116, 71], [93, 71]], [[102, 155], [103, 169], [114, 170], [114, 186]]], 'red_roads': [[[52, 156], [72, 155], [93, 160], [111, 162], [132, 165], [146, 179], [170, 186], [188, 190]], [[114, 99], [123, 114], [124, 139], [124, 169], [135, 190]], [[84, 140], [103, 139], [124, 144], [150, 150], [165, 155]]], 'place_labels': [['san_bernardino', 58, 121, 7.2], ['highland', 110, 107, 7.3], ['colton', 38, 148, 6.4], ['loma_linda', 92, 169, 5.7], ['redlands', 135, 162, 6.7], ['mentone', 169, 147, 5.4], ['yucaipa', 217, 180, 7], ['grand_terrace', 50, 179, 5.1], ['running_springs', 204, 42, 4.8]], 'marker': [125, 152], 'controls': [[221, 48, 10, 13], [221, 61, 10, 16], [221, 85, 10, 11]]}, 'label_map': {'title': 'panel_03_title', 'san_bernardino': 'panel_03_san_bernardino', 'highland': 'panel_03_highland', 'colton': 'panel_03_colton', 'loma_linda': 'panel_03_loma_linda', 'redlands': 'panel_03_redlands', 'mentone': 'panel_03_mentone', 'yucaipa': 'panel_03_yucaipa', 'grand_terrace': 'panel_03_grand_terrace', 'running_springs': 'panel_03_running_springs'}, 'crop_pixel_bbox': [575, 8, 814, 272], 'crop_sha256': '1823af4b05a6b4cc3adac9b73a52e5fb7ceb3fcafe89153ab449a1a62609323e', 'api_request_sha256': '4f231b839832509d134fa0863f32754cfce96710940395909fea3a6b5242c2ba'}, {'bbox': [0.28, 0.276, 0.696, 0.632], 'data': {'canvas': [1029, 555], 'axes_rect': [0.063, 0.205, 0.917, 0.686], 'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 'age_keys': ['age_0', 'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'age_9', 'age_10', 'age_11', 'age_12', 'age_13', 'age_14', 'age_15', 'age_16', 'age_17'], 'bars_percent': [6.21, 6.07, 5.93, 6.76, 8, 7.93, 7.45, 6.76, 5.86, 5.93, 5.79, 6.07, 5.59, 4.76, 3.45, 2.34, 1.66, 2.07], 'comparison_percent': [7.45, 7.24, 7.1, 7.17, 7.72, 8.55, 7.59, 6.83, 5.93, 5.93, 6.07, 5.86, 4.97, 4.21, 2.83, 1.72, 1.03, 1.1], 'yticks': [0, 2, 4, 6, 8], 'xlim': [-0.5, 17.5], 'ylim': [0, 8.8], 'bar_width': 0.75, 'selector_rect': [0.797, 0.018, 0.19, 0.063], 'selector_arrow': [[0.965, 0.057], [0.978, 0.057], [0.9715, 0.04]], 'colors': {'background': '#2b2412', 'bar': '#00b4d5', 'line': '#8c7851', 'marker': '#eee9d8', 'grid': '#928b75', 'text': '#eeeade', 'selector': '#f8f8f5', 'selector_border': '#a39f91'}}, 'label_map': {'title': 'panel_04_title', 'y_axis': 'panel_04_y_axis', 'age_0': 'panel_04_age_0', 'age_1': 'panel_04_age_1', 'age_2': 'panel_04_age_2', 'age_3': 'panel_04_age_3', 'age_4': 'panel_04_age_4', 'age_5': 'panel_04_age_5', 'age_6': 'panel_04_age_6', 'age_7': 'panel_04_age_7', 'age_8': 'panel_04_age_8', 'age_9': 'panel_04_age_9', 'age_10': 'panel_04_age_10', 'age_11': 'panel_04_age_11', 'age_12': 'panel_04_age_12', 'age_13': 'panel_04_age_13', 'age_14': 'panel_04_age_14', 'age_15': 'panel_04_age_15', 'age_16': 'panel_04_age_16', 'age_17': 'panel_04_age_17', 'comparison_caption': 'panel_04_comparison_caption', 'comparison_selection': 'panel_04_comparison_selection'}, 'crop_pixel_bbox': [231, 144, 574, 329], 'crop_sha256': '9aace7d673b5d2b0d0d4f88913d55606f23d52bf8c334be0e54da3b15f4adf49', 'api_request_sha256': 'd78e5ef0fb4015f03dfb1a026424220b2af90e9901e4a8b8982ec16eac80a98f'}, {'bbox': [0.28, 0.635, 0.693, 0.988], 'data': {'canvas': [1020, 552], 'background': '#2b230f', 'table_bounds': [15, 114, 987, 228], 'column_edges': [15, 361, 509, 658, 1002], 'row_keys': ['income_0', 'income_1', 'income_2', 'income_3', 'income_4', 'income_5', 'income_6', 'income_7', 'income_8'], 'values_percent': [8.9, 8.8, 8.0, 13.2, 17.9, 13.6, 14.7, 7.5, 8.1], 'differences_percentage_points': [-1.6, -1.1, -0.5, -0.6, -0.3, 0.1, 0.6, 0.9, 2.4], 'bar_left_pixels': [717, 750, 792, 786, 811, 830, 830, 830, 830], 'bar_right_pixels': [830, 830, 830, 830, 830, 836, 873, 893, 998], 'bar_colors': ['#a4d7e8', '#8dcde0', '#7fc5d8', '#6abed4', '#52b8d0', '#39b3ce', '#21adcc', '#08a9cd', '#00abd1'], 'baseline_pixel': 830, 'bar_height': 18, 'table_text_size': 14, 'row_fills': ['#342f23', '#676461'], 'grid_color': '#a49e8d', 'selector_bounds': [801, 510, 204, 33]}, 'label_map': {'title': 'panel_05_title', 'largest': 'panel_05_largest', 'smallest': 'panel_05_smallest', 'variable': 'panel_05_variable', 'value': 'panel_05_value', 'difference': 'panel_05_difference', 'income_0': 'panel_05_income_0', 'income_1': 'panel_05_income_1', 'income_2': 'panel_05_income_2', 'income_3': 'panel_05_income_3', 'income_4': 'panel_05_income_4', 'income_5': 'panel_05_income_5', 'income_6': 'panel_05_income_6', 'income_7': 'panel_05_income_7', 'income_8': 'panel_05_income_8', 'comparison_caption': 'panel_05_comparison_caption', 'comparison_source': 'panel_05_comparison_source'}, 'crop_pixel_bbox': [231, 331, 571, 515], 'crop_sha256': '8bdf6423774f6b387da6d9aa2f738887f1c5db133a0c50226f83398009dc68e8', 'api_request_sha256': 'db05abae8e544e867f78ca9b9171ee739a1dd16398dccb7458119c89cdda753c'}, {'bbox': [0.702, 0.527, 0.989, 0.8], 'data': {'canvas': [948, 568], 'background': '#292315', 'card_color': '#50514e', 'card_edge': '#665e4b', 'separator': '#696a65', 'cards': [{'top': 72, 'height': 158, 'name': 'segment_1', 'group': 'group_1', 'badge': 'badge_1', 'percent': 12.3, 'accent': '#d9a56e'}, {'top': 232, 'height': 158, 'name': 'segment_2', 'group': 'group_2', 'badge': 'badge_2', 'percent': 10.3, 'accent': '#52c6bc'}, {'top': 392, 'height': 158, 'name': 'segment_3', 'group': 'group_3', 'badge': 'badge_3', 'percent': 8.6, 'accent': '#48bace'}], 'card_left': 24, 'card_width': 910, 'column_x': 572, 'portrait_x': 105, 'portrait_offset_y': 79, 'portrait_radius': 59, 'badge_x': 184, 'badge_offset_y': 62, 'badge_radius': 25, 'name_x': 248, 'name_offset_y': 65, 'group_offset_y': 103, 'percent_x': 601, 'percent_offset_y': 64, 'caption_offset_y': 103, 'chevron_x': [881, 896, 910], 'chevron_offset_y': [64, 78, 64], 'people': [[{'x': -21, 'y': -7, 'head_r': 17, 'skin': '#e4c1a3', 'hair': '#86694c', 'shirt': '#e9dfc8', 'body_w': 39, 'body_h': 46}, {'x': 22, 'y': -9, 'head_r': 18, 'skin': '#dab494', 'hair': '#ccc4b6', 'shirt': '#809b97', 'body_w': 42, 'body_h': 48}], [{'x': -22, 'y': -7, 'head_r': 18, 'skin': '#aa7860', 'hair': '#302c26', 'shirt': '#253d39', 'body_w': 41, 'body_h': 47}, {'x': 21, 'y': -8, 'head_r': 17, 'skin': '#d4a88c', 'hair': '#493a2c', 'shirt': '#e3d9c5', 'body_w': 38, 'body_h': 47}], [{'x': -22, 'y': -7, 'head_r': 17, 'skin': '#c09378', 'hair': '#514334', 'shirt': '#d8e5e4', 'body_w': 39, 'body_h': 48}, {'x': 20, 'y': -6, 'head_r': 17, 'skin': '#e2bca2', 'hair': '#81694e', 'shirt': '#f0eae1', 'body_w': 39, 'body_h': 47}]], 'portrait_backgrounds': ['#b4c9bf', '#7fae9d', '#b4d4d8']}, 'label_map': {'title': 'panel_06_title', 'segment_1': 'panel_06_segment_1', 'group_1': 'panel_06_group_1', 'segment_2': 'panel_06_segment_2', 'group_2': 'panel_06_group_2', 'segment_3': 'panel_06_segment_3', 'group_3': 'panel_06_group_3', 'badge_1': 'panel_06_badge_1', 'badge_2': 'panel_06_badge_2', 'badge_3': 'panel_06_badge_3', 'share_caption': 'panel_06_share_caption'}, 'crop_pixel_bbox': [578, 275, 815, 417], 'crop_sha256': '823601edfe38c1d1a8365e4f7085d919602b508f2282cfc122252f9fdfa2bedb', 'api_request_sha256': '45bf5a7c1e1ea95c5b68f9da9719051b1f58d2014fad65ab737ded8c9057a8f9'}], 'global_placements': [{'key': 'title', 'x': 0.181, 'y': 0.066, 'size': 30, 'max_width': 0.19}, {'key': 'subtitle', 'x': 0.18, 'y': 0.132, 'size': 8, 'max_width': 0.19}, {'key': 'branding', 'x': 0.956, 'y': 0.947, 'size': 24, 'max_width': 0.077}, {'key': 'branding_tagline', 'x': 0.954, 'y': 0.976, 'size': 5, 'max_width': 0.078}]}
LABELS = {'title': 'การแบ่งกลุ่ม Tapestry', 'subtitle': 'สายใยของย่านชุมชนในอเมริกา', 'branding': 'esri', 'branding_tagline': 'ศาสตร์แห่งสถานที่', 'panel_00_title': 'รูปแบบชีวิต Tapestry', 'panel_00_households': 'ครัวเรือน', 'panel_00_hh_percent': '% ครัวเรือน', 'panel_00_us_percent': '% ครัวเรือนในสหรัฐฯ', 'panel_00_index': 'ดัชนี', 'panel_00_r1': 'ย่านที่พักอาศัยมั่งคั่ง (1)', 'panel_00_r2': 'ย่านหรู (2)', 'panel_00_r3': 'คนเมืองย่านหรู (3)', 'panel_00_r4': 'ย่านครอบครัว (4)', 'panel_00_r5': 'คนเมืองเจเนอเรชันเอ็กซ์ (5)', 'panel_00_r6': 'ชีวิตชนบทแสนอบอุ่น (6)', 'panel_00_r7': 'ชุมชนชาติพันธุ์ (7)', 'panel_00_r8': 'กลุ่มระดับกลาง (8)', 'panel_00_r9': 'วิถีผู้สูงวัย (9)', 'panel_00_r10': 'ชุมชนชนบทห่างไกล (10)', 'panel_00_r11': 'คนโสดใจกลางเมือง (11)', 'panel_00_r12': 'ถิ่นบ้านเกิด (12)', 'panel_00_r13': 'คลื่นลูกใหม่ (13)', 'panel_00_r14': 'นักศึกษาและผู้รักชาติ (14)', 'panel_01_title': 'ข้อมูลสำคัญ', 'panel_01_home_value': 'ค่ามัธยฐานมูลค่าบ้าน', 'panel_01_income': 'ค่ามัธยฐานรายได้ครัวเรือน', 'panel_01_ratio': 'อัตราส่วนมูลค่าบ้าน /\nรายได้', 'panel_01_age': 'อายุมัธยฐาน', 'panel_01_population': 'ประชากร', 'panel_02_title': 'การศึกษา', 'panel_02_no_hs': 'ไม่จบมัธยมปลาย', 'panel_02_hs': 'จบมัธยมปลาย', 'panel_02_college': 'เคยเรียนระดับอุดมศึกษา', 'panel_02_degree': 'ปริญญาตรีขึ้นไป', 'panel_03_title': 'สำนักงานใหญ่ Esri', 'panel_03_san_bernardino': 'แซน\nเบอร์นาร์ดิโน', 'panel_03_highland': 'ไฮแลนด์', 'panel_03_colton': 'โคลตัน', 'panel_03_loma_linda': 'โลมาลินดา', 'panel_03_redlands': 'เรดแลนด์ส', 'panel_03_mentone': 'เมนโทน', 'panel_03_yucaipa': 'ยูไคปา', 'panel_03_grand_terrace': 'แกรนด์เทอร์เรซ', 'panel_03_running_springs': 'รันนิงสปริงส์', 'panel_04_title': 'โครงสร้างอายุ', 'panel_04_y_axis': 'ร้อยละ', 'panel_04_age_0': '0-4', 'panel_04_age_1': '5-9', 'panel_04_age_2': '10-14', 'panel_04_age_3': '15-19', 'panel_04_age_4': '20-24', 'panel_04_age_5': '25-29', 'panel_04_age_6': '30-34', 'panel_04_age_7': '35-39', 'panel_04_age_8': '40-44', 'panel_04_age_9': '45-49', 'panel_04_age_10': '50-54', 'panel_04_age_11': '55-59', 'panel_04_age_12': '60-64', 'panel_04_age_13': '65-69', 'panel_04_age_14': '70-74', 'panel_04_age_15': '75-79', 'panel_04_age_16': '80-84', 'panel_04_age_17': '85+', 'panel_04_comparison_caption': 'ข้อมูลแสดงการเปรียบเทียบกับ', 'panel_04_comparison_selection': 'เทศมณฑลอีสต์แฮมป์เชียร์', 'panel_05_title': 'ครัวเรือนจำแนกตามรายได้', 'panel_05_largest': 'กลุ่มใหญ่ที่สุด: $50,000 - $74,999 (17.9%)', 'panel_05_smallest': 'กลุ่มเล็กที่สุด: $150,000 - $199,999 (7.5%)', 'panel_05_variable': 'ตัวแปร', 'panel_05_value': 'ค่า', 'panel_05_difference': 'ผลต่าง', 'panel_05_income_0': '< $10,000', 'panel_05_income_1': '$10,000 - $24,999', 'panel_05_income_2': '$25,000 - $34,999', 'panel_05_income_3': '$35,000 - $49,999', 'panel_05_income_4': '$50,000 - $74,999', 'panel_05_income_5': '$75,000 - $99,999', 'panel_05_income_6': '$100,000 - $149,999', 'panel_05_income_7': '$150,000 - $199,999', 'panel_05_income_8': '$200,000 +', 'panel_05_comparison_caption': 'แท่งกราฟแสดงความแตกต่างจาก', 'panel_05_comparison_source': 'เทศมณฑลแซนเบอร์นาร์ดิโน', 'panel_06_title': 'กลุ่ม Tapestry', 'panel_06_segment_1': 'ผู้อาศัยนอกเขตชานเมือง', 'panel_06_group_1': 'ย่านที่พักอาศัยมั่งคั่ง', 'panel_06_segment_2': 'คนหนุ่มสาวผู้ไม่หยุดนิ่ง', 'panel_06_group_2': 'คนโสดใจกลางเมือง', 'panel_06_segment_3': 'คนทำงานรุ่นใหม่\nอนาคตไกล', 'panel_06_group_3': 'กลุ่มระดับกลาง', 'panel_06_badge_1': '1E', 'panel_06_badge_2': '11B', 'panel_06_badge_3': '8C', 'panel_06_share_caption': 'ของครัวเรือน'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
