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
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        (sw, sh) = data['source_extent']
        ax.set_xlim(0, sw)
        ax.set_ylim(sh, 0)
        ax.axis('off')
        c = data['colors']
        (x, y, w, h) = data['border']
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=c['border'], linewidth=1.7, linestyle=(0, (2, 2))))
        ax.plot(data['line_x'], data['line_y_image_estimated'], color=c['line'], linewidth=5.6, solid_capstyle='round', solid_joinstyle='round')
        (x0, x1, y) = data['average_line']
        ax.plot([x0, x1], [y, y], color=c['average'], linewidth=3.4, linestyle=(0, (4, 2)), dash_capstyle='butt')
        (x, y, w, h) = data['bullet_rectangle']
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c['bar'], edgecolor='none'))
        (x, y0, y1) = data['target_line']
        ax.plot([x, x], [y0, y1], color=c['target'], linewidth=3.5, solid_capstyle='butt')
        ax.add_patch(Polygon(data['down_triangle'], closed=True, facecolor=c['change'], edgecolor='none'))
        placements = []
        for (key, x, y, size, mw, anchor) in data['text_layout']:
            placements.append({'key': key, 'x': x, 'y': y, 'size': size, 'max_width': mw, 'rotation': 0, 'anchor': anchor, 'color': c['change'] if key == 'change' else '#595b57' if key == 'average' else c['line']})
        return finish(fig, labels, placements)

    def panel_01(data, labels):
        (W, H) = data['canvas']
        (w, h) = data['extent']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis('off')
        placements = []

        def text(key, pos, size, color='#2b2b2b', anchor='center', width=0.3):
            placements.append({'key': key, 'x': pos[0] / w, 'y': pos[1] / h, 'size': size * W / w, 'color': color, 'anchor': anchor, 'max_width': width, 'rotation': 0})
        ax.add_patch(Rectangle(data['border'][:2], data['border'][2], data['border'][3], fill=False, edgecolor='#d4d4d4', linewidth=1.8, linestyle=(0, (2, 2))))
        for t in data['title_parts']:
            text(t['key'], t['position'], 40, t['color'], width=0.25)
        m = data['plot_mapping']
        for s in data['series']:
            y = m['baseline'] - np.array(s['values']) * m['pixels_per_unit']
            ax.plot(data['x'], y, color=s['color'], linewidth=5 * W / w, solid_capstyle='round', solid_joinstyle='round')
            ax.scatter([data['x'][-1]], [y[-1]], s=(10 * W / w) ** 2, color=s['color'], zorder=4)
            text(s['endpoint'], [data['endpoint_x'], float(y[-1]) + data['endpoint_y_offset']], 29, s['color'], 'left', 0.1)
        d = data['divider']
        ax.plot([d[0], d[0]], [d[1], d[2]], color='#eeeeee', linewidth=0.7)
        for b in data['bars']:
            r = b['rectangle']
            ax.add_patch(Rectangle(r[:2], r[2], r[3], facecolor=b['color'], edgecolor='none'))
            (x, y0, y1) = b['target']
            ax.plot([x, x], [y0, y1], color='black', linewidth=3.6 * W / w, solid_capstyle='butt')
            text(b['category'], b['category_position'], 28, width=0.26)
            text(b['total'], b['total_position'], 59, width=0.2)
            placements[-1]['weight'] = 'bold'
            text(b['change'], b['change_position'], 29, b['change_color'], width=0.14)
            ax.add_patch(Polygon(b['triangle'], closed=True, facecolor=b['change_color'], edgecolor='none'))
        return finish(fig, labels, placements)

    def panel_02(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        (sw, sh) = data['source_extent']
        ax.set_xlim(0, sw)
        ax.set_ylim(sh, 0)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        c = data['colors']
        scale = W / sw
        (x, y, w, h) = data['border_rect']
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=c['border'], linewidth=1.5 * scale, linestyle=(0, (3, 2))))
        v = np.array(data['ticket_values'])
        (lo, hi) = data['line_value_limits']
        (y0, y1) = data['line_pixel_y_limits']
        yy = y0 + (v - lo) * (y1 - y0) / (hi - lo)
        ax.plot(data['line_x'], yy, color=c['line'], linewidth=4.8 * scale, solid_capstyle='round', solid_joinstyle='round')
        ax.scatter([data['line_x'][0], data['line_x'][-1]], [yy[0], yy[-1]], s=70 * scale * scale, color=c['line'], zorder=3)
        (x, y, w, h) = data['bullet_rect']
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c['fill'], edgecolor=c['outline'], linewidth=3.8 * scale))
        (x, y0, y1) = data['target_segment']
        ax.plot([x, x], [y0, y1], color=c['outline'], linewidth=3.6 * scale, solid_capstyle='butt')
        ax.add_patch(Polygon(data['up_triangle'], closed=True, facecolor=c['accent'], edgecolor=c['accent'], linewidth=0.5))
        placements = [dict(p) for p in data['text_positions']]
        return finish(fig, labels, placements)

    def panel_03(data, labels):
        (W, H) = data['canvas']
        sx = W / data['coordinate_extent'][0]
        sy = H / data['coordinate_extent'][1]
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 717)
        ax.set_ylim(760, 0)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        (x, y, w, h) = data['border']
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='#d5d5d5', linewidth=1.5, linestyle=(0, (3, 2))))
        placements = [{'key': 'title', 'x': 0.5, 'y': 46 / 760, 'size': 40 * sy, 'max_width': 0.92, 'anchor': 'center'}]
        for (i, key) in enumerate(data['category_keys']):
            cy = data['bar_centers'][i]
            left = data['bar_left']
            ax.add_patch(Rectangle((left, cy - data['bar_height'] / 2), data['bar_ends'][i] - left, data['bar_height'], facecolor=data['bar_colors'][i], edgecolor='none'))
            tx = data['target_x'][i]
            ax.plot([tx, tx], [data['target_top'][i], data['target_bottom'][i]], color='black', linewidth=3.6 * sx, solid_capstyle='butt')
            placements.append({'key': key, 'x': data['category_x'] / 717, 'y': data['category_y'][i] / 760, 'size': 28 * sy, 'max_width': 0.25, 'anchor': 'center'})
            ax.text(data['value_right'], data['value_y'][i], format(data['values'][i], ','), ha='right', va='center', fontsize=58 * sy * 72 / 100, fontweight='bold', color='#292b28', fontfamily='DejaVu Sans')
            yy = data['change_y'][i]
            color = data['change_colors'][i]
            ax.text(data['change_right'], yy, format(data['changes'][i], '.1f') + '%', ha='right', va='center', fontsize=29 * sy * 72 / 100, color=color, fontfamily='DejaVu Sans')
            xx = data['triangle_x']
            hw = data['triangle_half_width']
            hh = data['triangle_half_height']
            direction = data['directions'][i]
            ax.add_patch(Polygon([[xx, yy - direction * hh], [xx - hw, yy + direction * hh], [xx + hw, yy + direction * hh]], closed=True, facecolor=color, edgecolor='none'))
        return finish(fig, labels, placements)

    def panel_04(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0.019, 0, 0.429, 1])
        ax.set_xlim(0, data['scale_max'])
        ax.set_ylim(0, 1)
        ax.axis('off')
        overlay = fig.add_axes([0, 0, 1, 1])
        overlay.set_xlim(0, 1)
        overlay.set_ylim(0, 1)
        overlay.axis('off')
        overlay.patch.set_alpha(0)
        overlay.add_patch(Rectangle((0.003, 0.003), 0.994, 0.994, fill=False, edgecolor='#d4d4d4', linewidth=2, linestyle=(0, (3, 2))))
        placements = [{'key': 'title', 'x': 0.5, 'y': 0.059, 'size': 56, 'max_width': 0.86, 'anchor': 'center'}]
        for (i, y) in enumerate(data['row_y']):
            ax.barh(y, data['values'][i], height=data['bar_height'], color=data['bar_colors'][i], edgecolor='none')
            t = data['targets_estimated'][i]
            ax.plot([t, t], [y - data['target_height'] / 2, y + data['target_height'] / 2], color='black', linewidth=5, solid_capstyle='butt')
            overlay.plot([0.448, 0.448], [y - 0.04, y + 0.04], color='#f7f7f7', linewidth=1)
            placements.append({'key': data['category_keys'][i], 'x': 0.599, 'y': 1 - y + 0.006, 'size': 38, 'max_width': 0.225, 'anchor': 'center'})
            placements.append({'key': data['value_keys'][i], 'x': 0.968, 'y': 1 - y - 0.024, 'size': 82, 'max_width': 0.24, 'anchor': 'right'})
            placements.append({'key': data['change_keys'][i], 'x': 0.922, 'y': 1 - y + 0.059, 'size': 41, 'max_width': 0.15, 'anchor': 'right', 'color': data['change_colors'][i]})
            cx = 0.95
            cy = y - 0.059
            dx = 0.016
            dy = 0.018
            if data['changes_percent'][i] > 0:
                pts = [[cx - dx, cy - dy], [cx + dx, cy - dy], [cx, cy + dy]]
            else:
                pts = [[cx - dx, cy + dy], [cx + dx, cy + dy], [cx, cy - dy]]
            overlay.add_patch(Polygon(pts, closed=True, facecolor=data['change_colors'][i], edgecolor='none'))
        return finish(fig, labels, placements)

    def panel_05(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor('white')
        placements = []
        fig.add_artist(Rectangle((0.003, 0.003), 0.993, 0.994, transform=fig.transFigure, fill=False, edgecolor=data['colors']['border'], linewidth=1.8, linestyle=(0, (3, 1))))
        placements.append({'key': 'title', 'x': 0.5, 'y': 0.08, 'size': 52, 'max_width': 0.88, 'anchor': 'center'})
        ax = fig.add_axes([0.123, 0.455, 0.858, 0.412])
        x = np.arange(len(data['categories']))
        ax.bar(x, data['counts'], width=data['bar_width'], color=data['colors']['bars'])
        ax.set_xlim(-0.52, 6.52)
        ax.set_ylim(*data['y_range'])
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_yticks(data['y_ticks'])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', length=10, width=2, color=data['colors']['ticks'])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (key, yy) in zip(data['y_label_keys'], data['y_label_canvas_positions']):
            placements.append({'key': key, 'x': 0.099, 'y': yy, 'size': 37, 'max_width': 0.092, 'anchor': 'right'})
        for (i, key) in enumerate(data['category_label_keys']):
            xx = 0.123 + 0.858 * (i + 0.52) / 7.04
            placements.append({'key': key, 'x': xx, 'y': 0.591, 'size': 40, 'max_width': 0.11, 'anchor': 'center'})
        placements.append({'key': 'summary_title', 'x': 0.5, 'y': 0.676, 'size': 40, 'max_width': 0.92, 'anchor': 'center'})
        bx = fig.add_axes([0.02, 0.016, 0.602, 0.255])
        bx.set_xlim(*data['summary_range'])
        bx.set_ylim(0, 1)
        bx.barh([0.5], [data['summary_actual']], height=0.5, color=data['colors']['bullet'])
        bx.plot([data['summary_target'], data['summary_target']], [0, 1], color='black', linewidth=4.5)
        bx.axis('off')
        placements.append({'key': 'summary_value', 'x': 0.8, 'y': 0.832, 'size': 98, 'max_width': 0.31, 'anchor': 'center'})
        placements.append({'key': 'change', 'x': 0.8, 'y': 0.922, 'size': 40, 'max_width': 0.28, 'anchor': 'center', 'color': data['colors']['change']})
        return finish(fig, labels, placements)

    def panel_06(data, labels):
        (W, H) = data['canvas']
        (cw, ch) = data['coordinate_extent']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        placements = []

        def text(key, x, y, size, color='#303030', anchor='center', weight='normal', width=0.3):
            placements.append({'key': key, 'x': x / cw, 'y': y / ch, 'size': size * W / cw, 'color': color, 'anchor': anchor, 'weight': weight, 'max_width': width, 'rotation': 0})
        b = data['border']
        ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, edgecolor='#d2d2d2', linewidth=1.3, linestyle='--'))
        for (key, x, y, size, color, weight) in data['title_parts']:
            text(key, x, y, size, color, 'left', weight, 0.34 if key == 'title_prefix' else 0.26)
        for s in data['series']:
            ax.plot(s['x'], s['y'], color=s['color'], linewidth=5.2 * W / cw, solid_capstyle='round', solid_joinstyle='round')
            ax.scatter([s['x'][-1]], [s['y'][-1]], s=65 * (W / cw) ** 2, color=s['color'], zorder=3)
        for (key, x, y, color) in data['endpoint_labels']:
            text(key, x, y, 30, color, width=0.12)
        sep = data['separator']
        ax.plot([sep[0], sep[0]], [sep[1], sep[2]], color='#f3f0f0', linewidth=0.7)
        for row in data['bullets']:
            (x, y, w, h) = row['bar']
            ax.add_patch(Rectangle((x, y), w, h, color=data['bar_color'], linewidth=0))
            (x, y0, y1) = row['target']
            ax.plot([x, x], [y0, y1], color='black', linewidth=3.6 * W / cw, solid_capstyle='butt')
            text(row['label'], *row['label_position'], 29, width=0.25)
            text(row['total_label'], *row['total_position'], 61, anchor='right', weight='bold', width=0.26)
            text(row['change_label'], *row['change_position'], 30, color=data['change_color'], anchor='right', width=0.16)
            ax.add_patch(Polygon(row['triangle'], closed=True, facecolor=data['change_color'], edgecolor='none'))
        return finish(fig, labels, placements)

    def panel_07(data, labels):
        (W, H) = data['canvas']
        (rw, rh) = data['reference_size']
        s = W / rw
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, rw)
        ax.set_ylim(rh, 0)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        (bx, by, bw, bh) = data['border']
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor='#cfcfcf', linewidth=1.2, linestyle='--'))
        (tx, ty) = data['title_position']
        placements = [{'key': 'title', 'x': tx / rw, 'y': ty / rh, 'size': 40 * s, 'max_width': 0.92, 'anchor': 'center'}]
        for (i, y) in enumerate(data['row_centers']):
            h = data['bar_heights'][i]
            ax.add_patch(Rectangle((data['bar_start'], y - h / 2), data['bar_widths'][i], h, facecolor=data['bar_colors'][i], edgecolor='none'))
            t = data['target_x'][i]
            th = data['target_heights'][i]
            ax.plot([t, t], [y - th / 2, y + th / 2], color='black', linewidth=3.6 * s, solid_capstyle='butt')
            placements.append({'key': data['category_keys'][i], 'x': data['category_x'] / rw, 'y': data['category_positions_y'][i] / rh, 'size': 28 * s, 'max_width': 0.245, 'anchor': 'center'})
            ax.text(data['value_x'], data['value_positions_y'][i], format(data['values'][i], ','), ha='right', va='center', fontsize=43 * s, fontweight='bold', color='#292929', fontfamily='DejaVu Sans')
            cy = data['change_positions_y'][i]
            col = data['change_colors'][i]
            placements.append({'key': data['change_keys'][i], 'x': data['change_x'] / rw, 'y': cy / rh, 'size': 30 * s, 'max_width': 0.14, 'anchor': 'right', 'color': col})
            cx = data['triangle_x']
            tw = data['triangle_width']
            th = data['triangle_height']
            d = data['directions'][i]
            tip = cy - d * th / 2
            base = cy + d * th / 2
            ax.add_patch(Polygon([[cx - tw / 2, base], [cx + tw / 2, base], [cx, tip]], closed=True, facecolor=col, edgecolor='none'))
        return finish(fig, labels, placements)

    def panel_08(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0.123, 0.087, 0.859, 0.762])
        ax.bar(data['hours'], data['tickets'], width=data['bar_width'], color='#323331', linewidth=0)
        ax.set_xlim(data['x_limits'])
        ax.set_ylim(data['y_limits'])
        ax.set_xticks(data['x_ticks'])
        ax.set_yticks(data['y_ticks'])
        ax.tick_params(axis='x', labelsize=20, colors='#666666', length=5, color='#f4f4f4', pad=12)
        ax.tick_params(axis='y', labelsize=20, colors='#666666', length=12, width=2, color='#f5f5f5', pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('white')
        fig.add_artist(Rectangle((0.003, 0.003), 0.993, 0.993, transform=fig.transFigure, fill=False, edgecolor='#d0d0d0', linewidth=1.5, linestyle='--'))
        placements = [{'key': 'title', 'x': 0.5, 'y': 0.077, 'size': 49, 'max_width': 0.9, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)
    functions = [panel_00, panel_01, panel_02, panel_03, panel_04, panel_05, panel_06, panel_07, panel_08]
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
BASE_ID = 'qa_49e9af6ac6eb8a5ecb354373e9de700da9ff021b149f7de468bac87bf0b0788f'
LANGUAGE = 'zh'
DATA = {'canvas': [2800, 3111], 'panels': [{'bbox': [0.013, 0.058, 0.332, 0.363], 'data': {'canvas': [940, 1000], 'source_extent': [717, 762], 'line_x': [49, 101, 153, 205, 256, 308, 360, 412, 464, 516, 567, 619, 670], 'line_y_image_estimated': [247, 254, 442, 310, 247, 312, 386, 406, 363, 360, 129, 175, 431], 'average_value': 647, 'summary_value': 8415, 'change_percent': 0.3, 'average_line': [14, 696, 310], 'bullet_rectangle': [14, 596, 397, 102], 'target_line': [411, 543, 748], 'down_triangle': [[603, 685], [627, 685], [615, 711]], 'border': [2, 2, 713, 758], 'colors': {'line': '#30312f', 'average': '#91938f', 'bar': '#df5458', 'target': '#050505', 'change': '#639fba', 'border': '#d6d7d4'}, 'text_layout': [['title', 0.5, 0.078, 51, 0.92, 'center'], ['average', 0.031, 0.444, 39, 0.35, 'left'], ['summary', 0.805, 0.824, 94, 0.34, 'center'], ['change', 0.782, 0.916, 39, 0.2, 'center']]}, 'label_map': {'title': 'panel_00_title', 'average': 'panel_00_average', 'summary': 'panel_00_summary', 'change': 'panel_00_change'}, 'crop_pixel_bbox': [29, 145, 746, 907], 'crop_sha256': '2c0ee19c8f023ea1dfc5a414226ac8f10a1938a942170b6faf2d96cc6f16bbbf', 'api_request_sha256': 'a65d417548e5ec4ccc9779985e3900928057442c3b5333064dcdcff2e9665d49'}, {'bbox': [0.34, 0.058, 0.659, 0.363], 'data': {'canvas': [1000, 1064], 'extent': [716, 762], 'x': [50, 98, 146, 194, 242, 290, 338, 386, 434, 482, 530, 578, 626], 'series': [{'values': [33, 37, 35, 26, 34, 24, 32, 36, 34, 27, 26, 33, 44], 'color': '#191c24', 'endpoint': 'major_endpoint'}, {'values': [2, 10, 6, 11, 12, 16, 6, 11, 13, 5, 17, 7, 5], 'color': '#599f4a', 'endpoint': 'critical_endpoint'}], 'plot_mapping': {'baseline': 462.5, 'pixels_per_unit': 7.5}, 'bars': [{'rectangle': [14, 521, 287, 65], 'color': '#589f48', 'target': [301, 488, 618], 'category': 'major', 'category_position': [428, 556], 'total': 'major_total', 'total_position': [638, 536], 'change': 'major_change', 'change_position': [624, 599], 'change_color': '#e68e32', 'triangle': [[669, 610], [692, 610], [680, 584]]}, {'rectangle': [14, 651, 83, 65], 'color': '#e35257', 'target': [105, 618, 748], 'category': 'critical', 'category_position': [428, 686], 'total': 'critical_total', 'total_position': [638, 666], 'change': 'critical_change', 'change_position': [624, 729], 'change_color': '#60a4c4', 'triangle': [[669, 714], [692, 714], [680, 741]]}], 'title_parts': [{'key': 'title_major', 'position': [214, 60], 'color': '#292929'}, {'key': 'title_critical', 'position': [365, 60], 'color': '#084c09'}, {'key': 'title_tickets', 'position': [508, 60], 'color': '#222222'}], 'border': [2, 2, 712, 758], 'divider': [321, 610, 751], 'endpoint_x': 634, 'endpoint_y_offset': 5}, 'label_map': {'title_major': 'panel_01_title_major', 'title_critical': 'panel_01_title_critical', 'title_tickets': 'panel_01_title_tickets', 'major': 'panel_01_major', 'critical': 'panel_01_critical', 'major_total': 'panel_01_major_total', 'critical_total': 'panel_01_critical_total', 'major_change': 'panel_01_major_change', 'critical_change': 'panel_01_critical_change', 'major_endpoint': 'panel_01_major_endpoint', 'critical_endpoint': 'panel_01_critical_endpoint'}, 'crop_pixel_bbox': [764, 145, 1481, 907], 'crop_sha256': 'd7e68132285e6cc7ec1a8f957a396fb6c850c63aa0bb7ca3dfbee8fcfdb888bb', 'api_request_sha256': 'a51382c731f31868e6f28dcae7e87a590dd73a66619db05b9a8850b3226b4c2f'}, {'bbox': [0.667, 0.058, 0.986, 0.363], 'data': {'canvas': [950, 1008], 'source_extent': [718, 762], 'line_x': [125, 165, 204, 244, 283, 322, 361, 400, 438, 477, 517, 556, 595], 'ticket_values': [1642, 1650.3, 1659.2, 1663.2, 1671.5, 1679.5, 1688.6, 1696.6, 1704.6, 1712.7, 1723.6, 1729, 1731], 'line_value_limits': [1642, 1731], 'line_pixel_y_limits': [441, 130], 'bullet_rect': [17, 594, 397, 107], 'target_segment': [391, 543, 748], 'up_triangle': [[604, 709], [625, 709], [614.5, 686]], 'border_rect': [2, 2, 714, 758], 'colors': {'line': '#30302e', 'fill': '#589f49', 'outline': '#080a07', 'accent': '#ef9136', 'border': '#d5d5d5'}, 'text_positions': [{'key': 'title', 'x': 0.5, 'y': 0.079, 'size': 53, 'max_width': 0.85, 'anchor': 'center'}, {'key': 'start_value', 'x': 0.159, 'y': 0.575, 'size': 40, 'max_width': 0.155, 'anchor': 'right'}, {'key': 'end_value', 'x': 0.839, 'y': 0.18, 'size': 40, 'max_width': 0.153, 'anchor': 'left'}, {'key': 'change', 'x': 0.799, 'y': 0.825, 'size': 94, 'max_width': 0.3, 'anchor': 'center'}, {'key': 'percent_change', 'x': 0.829, 'y': 0.917, 'size': 40, 'max_width': 0.22, 'anchor': 'right', 'color': '#ef9136'}]}, 'label_map': {'title': 'panel_02_title', 'start_value': 'panel_02_start_value', 'end_value': 'panel_02_end_value', 'change': 'panel_02_change', 'percent_change': 'panel_02_percent_change'}, 'crop_pixel_bbox': [1499, 145, 2217, 907], 'crop_sha256': '921a6f88b54e83be5740a689eba26f108c37cae7848638cc7a34bb547f937bb4', 'api_request_sha256': '37b1dc5d12a1ce30b854286c17ff5d9e4a4513b0e71e3bdaf663a5fe24064c66'}, {'bbox': [0.013, 0.37, 0.332, 0.674], 'data': {'canvas': [944, 1000], 'coordinate_extent': [717, 760], 'bar_left': 14, 'bar_height': 60, 'bar_centers': [174, 336, 501, 664], 'bar_ends': [303, 228, 157, 88], 'target_x': [304, 229, 170, 89], 'target_top': [124, 289, 451, 616], 'target_bottom': [221, 386, 549, 714], 'category_keys': ['systems', 'access_login', 'software', 'hardware'], 'category_y': [177, 340, 506, 668], 'values': [3382, 2502, 1668, 863], 'value_y': [156, 321, 484, 649], 'changes': [0.9, 1.0, 5.9, 2.7], 'change_y': [216, 381, 545, 710], 'directions': [1, 1, -1, 1], 'bar_colors': ['#599f4b', '#599f4b', '#e25459', '#599f4b'], 'change_colors': ['#ee902c', '#ee902c', '#60a2c2', '#ee902c'], 'category_x': 419, 'value_right': 693, 'change_right': 660, 'triangle_x': 680, 'triangle_half_width': 12, 'triangle_half_height': 13, 'border': [2, 2, 712, 756]}, 'label_map': {'title': 'panel_03_title', 'systems': 'panel_03_systems', 'access_login': 'panel_03_access_login', 'software': 'panel_03_software', 'hardware': 'panel_03_hardware'}, 'crop_pixel_bbox': [29, 924, 746, 1684], 'crop_sha256': '31ca808bbec2b6c4e91ca577cb0717fc7a8b5a618400061e3eac900e47defc66', 'api_request_sha256': 'e80a1b8c415e5bbfb6e870fa2525280fe20e3a1248126f2842e8bb6ed290ca3e'}, {'bbox': [0.34, 0.37, 0.659, 0.674], 'data': {'canvas': [1000, 1060], 'values': [1428, 1314, 3074, 2599], 'targets_estimated': [1415, 1445, 3150, 2535], 'row_y': [0.774, 0.558, 0.343, 0.126], 'category_keys': ['low', 'medium', 'high', 'unassigned'], 'value_keys': ['value_low', 'value_medium', 'value_high', 'value_unassigned'], 'change_keys': ['change_low', 'change_medium', 'change_high', 'change_unassigned'], 'changes_percent': [2.8, -7.2, -1.7, 3.5], 'bar_colors': ['#589e4c', '#df5357', '#df5357', '#589e4c'], 'change_colors': ['#e78d31', '#64a1bf', '#64a1bf', '#e78d31'], 'bar_height': 0.089, 'target_height': 0.128, 'scale_max': 3300}, 'label_map': {'title': 'panel_04_title', 'low': 'panel_04_low', 'medium': 'panel_04_medium', 'high': 'panel_04_high', 'unassigned': 'panel_04_unassigned', 'value_low': 'panel_04_value_low', 'value_medium': 'panel_04_value_medium', 'value_high': 'panel_04_value_high', 'value_unassigned': 'panel_04_value_unassigned', 'change_low': 'panel_04_change_low', 'change_medium': 'panel_04_change_medium', 'change_high': 'panel_04_change_high', 'change_unassigned': 'panel_04_change_unassigned'}, 'crop_pixel_bbox': [764, 924, 1481, 1684], 'crop_sha256': '3a85c0ae61d18e2f774184bab640fad632f8b160d6a1c85a03f762e960d766ad', 'api_request_sha256': '3f5859da092e054cb65ba161d9575e94b0af26fc7730d59c150dc100e6b384cc'}, {'bbox': [0.667, 0.37, 0.986, 0.674], 'data': {'canvas': [950, 1006], 'categories': [0, 5, 10, 15, 20, 25, 30], 'counts': [3703, 2300, 1090, 665, 250, 150, 150], 'category_label_keys': ['x0', 'x5', 'x10', 'x15', 'x20', 'x25', 'x30'], 'y_ticks': [0, 1000, 2000, 3000], 'y_label_keys': ['y0', 'y1', 'y2', 'y3'], 'y_label_canvas_positions': [0.521, 0.448, 0.343, 0.241], 'y_range': [0, 4000], 'bar_width': 0.74, 'summary_actual': 3703, 'summary_target': 3775, 'summary_range': [0, 4100], 'colors': {'bars': '#333332', 'bullet': '#df5558', 'change': '#63a2c0', 'border': '#d4d4d4', 'ticks': '#f5f5f5'}}, 'label_map': {'title': 'panel_05_title', 'summary_title': 'panel_05_summary_title', 'y0': 'panel_05_y0', 'y1': 'panel_05_y1', 'y2': 'panel_05_y2', 'y3': 'panel_05_y3', 'x0': 'panel_05_x0', 'x5': 'panel_05_x5', 'x10': 'panel_05_x10', 'x15': 'panel_05_x15', 'x20': 'panel_05_x20', 'x25': 'panel_05_x25', 'x30': 'panel_05_x30', 'summary_value': 'panel_05_summary_value', 'change': 'panel_05_change'}, 'crop_pixel_bbox': [1499, 924, 2217, 1684], 'crop_sha256': 'bb70e2a54878a4ae532b478a05cedda85abe52d8af0cd1dc0172c91b690b3c91', 'api_request_sha256': '737e3ad9d2dc56ec947f9613d8f5c65e92dfd06629359d8c8b58ff5a71df7ef4'}, {'bbox': [0.013, 0.681, 0.332, 0.989], 'data': {'canvas': [1000, 1074], 'coordinate_extent': [717, 770], 'series': [{'key': 'title_requests', 'color': '#171b23', 'x': [50, 98, 146, 194, 242, 290, 338, 386, 434, 482, 530, 578, 626], 'y': [177, 159, 208, 164, 160, 185, 177, 208, 184, 168, 131, 145, 177]}, {'key': 'title_issues', 'color': '#f39231', 'x': [50, 98, 146, 194, 242, 290, 338, 386, 434, 482, 530, 578, 626], 'y': [406, 426, 422, 434, 424, 414, 440, 413, 426, 442, 424, 422, 450]}], 'bullets': [{'label': 'issue', 'total_label': 'issue_total', 'change_label': 'issue_change', 'total': 2110, 'change_percent': 0.5, 'bar': [14, 522, 97, 67], 'target': [111, 489, 625], 'label_position': [408, 560], 'total_position': [695, 537], 'change_position': [660, 600], 'triangle': [[668, 585], [692, 585], [680, 613]]}, {'label': 'request', 'total_label': 'request_total', 'change_label': 'request_change', 'total': 6305, 'change_percent': 0.3, 'bar': [14, 657, 290, 67], 'target': [304, 624, 757], 'label_position': [408, 694], 'total_position': [695, 672], 'change_position': [660, 734], 'triangle': [[668, 720], [692, 720], [680, 748]]}], 'title_parts': [['title_prefix', 57, 61, 40, '#303030', 'normal'], ['title_requests', 292, 61, 40, '#303030', 'bold'], ['title_vs', 477, 61, 40, '#303030', 'normal'], ['title_issues', 540, 61, 40, '#ef9030', 'normal']], 'endpoint_labels': [['request_endpoint', 651, 213, '#171b23'], ['issue_endpoint', 661, 447, '#ee902d']], 'border': [2, 3, 713, 765], 'separator': [321, 515, 709], 'bar_color': '#df5558', 'change_color': '#65a5c3'}, 'label_map': {'title_prefix': 'panel_06_title_prefix', 'title_requests': 'panel_06_title_requests', 'title_vs': 'panel_06_title_vs', 'title_issues': 'panel_06_title_issues', 'request_endpoint': 'panel_06_request_endpoint', 'issue_endpoint': 'panel_06_issue_endpoint', 'issue': 'panel_06_issue', 'request': 'panel_06_request', 'issue_total': 'panel_06_issue_total', 'request_total': 'panel_06_request_total', 'issue_change': 'panel_06_issue_change', 'request_change': 'panel_06_request_change'}, 'crop_pixel_bbox': [29, 1701, 746, 2471], 'crop_sha256': '7a99609d17025881e3d7e4fe56d772e0440618bf90208f11effad749fd2068e6', 'api_request_sha256': '054560887917160635676873cc08e556508e1857b685bc4fa4b867beaa3955de'}, {'bbox': [0.34, 0.681, 0.659, 0.989], 'data': {'canvas': [932, 1001], 'reference_size': [717, 770], 'row_centers': [176, 341, 508.5, 673.5], 'bar_start': 14, 'bar_widths': [203, 186, 278, 290], 'bar_heights': [68, 68, 67, 67], 'target_x': [229, 201, 304, 306], 'target_heights': [98, 98, 97, 97], 'values': [1757, 1635, 2413, 2515], 'value_positions_y': [158, 323, 491, 655], 'category_positions_y': [179, 346, 512, 679], 'change_positions_y': [220, 385, 552, 718], 'category_keys': ['unsatisfied', 'satisfied', 'highly_satisfied', 'unknown'], 'change_keys': ['change_down_1', 'change_up_1', 'change_down_2', 'change_up_2'], 'directions': [-1, 1, -1, 1], 'bar_colors': ['#df5356', '#589f4e', '#df5356', '#589f4e'], 'change_colors': ['#65a3c2', '#ec8e32', '#65a3c2', '#ec8e32'], 'title_position': [358.5, 45], 'category_x': 428, 'value_x': 694, 'change_x': 660, 'triangle_x': 680, 'triangle_width': 24, 'triangle_height': 27, 'border': [2, 3, 712, 765]}, 'label_map': {'title': 'panel_07_title', 'unsatisfied': 'panel_07_unsatisfied', 'satisfied': 'panel_07_satisfied', 'highly_satisfied': 'panel_07_highly_satisfied', 'unknown': 'panel_07_unknown', 'change_down_1': 'panel_07_change_down_1', 'change_up_1': 'panel_07_change_up_1', 'change_down_2': 'panel_07_change_down_2', 'change_up_2': 'panel_07_change_up_2'}, 'crop_pixel_bbox': [764, 1701, 1481, 2471], 'crop_sha256': '46851bd3b051942a3dd0a003e75ffe2b68792aebac7a91739c9bff5ec1ba97e1', 'api_request_sha256': 'd6526cd56e3609e0308592abfa14b311ba93557b8db7168900c12a9bfc0562bc'}, {'bbox': [0.667, 0.681, 0.986, 0.989], 'data': {'hours': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'tickets': [356, 363, 342, 345, 396, 357, 335, 365, 351, 342, 343, 345, 382, 315, 360, 350, 360, 338, 353, 331, 338, 331, 363, 357], 'x_ticks': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], 'y_ticks': [0, 100, 200, 300, 400], 'x_limits': [-0.5, 23.5], 'y_limits': [0, 410], 'bar_width': 0.88, 'canvas': [940, 1008]}, 'label_map': {'title': 'panel_08_title'}, 'crop_pixel_bbox': [1499, 1701, 2217, 2471], 'crop_sha256': 'e0272d9c1e520a4c24e1fd656cd71f2ea318ca2010e4c86cbe0214ab50872fd8', 'api_request_sha256': '38d2f98d0d053f703f9a46244b889c4c98fff2463297c24563043973207b4edd'}], 'global_placements': [{'key': 'title', 'x': 0.504, 'y': 0.032, 'size': 68, 'max_width': 0.9}]}
LABELS = {'title': '服务台绩效', 'panel_00_title': '每周工单量', 'panel_00_average': '平均：647', 'panel_00_summary': '8,415', 'panel_00_change': '0.3%', 'panel_01_title_major': '严重与', 'panel_01_title_critical': '危急', 'panel_01_title_tickets': '工单', 'panel_01_major': '严重', 'panel_01_critical': '危急', 'panel_01_major_total': '421', 'panel_01_critical_total': '121', 'panel_01_major_change': '0.5%', 'panel_01_critical_change': '6.2%', 'panel_01_major_endpoint': '44', 'panel_01_critical_endpoint': '5', 'panel_02_title': '未关闭工单', 'panel_02_start_value': '1,642', 'panel_02_end_value': '1,731', 'panel_02_change': '+89', 'panel_02_percent_change': '5.4%', 'panel_03_title': '工单涉及类别', 'panel_03_systems': '系统', 'panel_03_access_login': '访问 /\n登录', 'panel_03_software': '软件', 'panel_03_hardware': '硬件', 'panel_04_title': '工单优先级', 'panel_04_low': '低', 'panel_04_medium': '中', 'panel_04_high': '高', 'panel_04_unassigned': '未指定', 'panel_04_value_low': '1,428', 'panel_04_value_medium': '1,314', 'panel_04_value_high': '3,074', 'panel_04_value_unassigned': '2,599', 'panel_04_change_low': '2.8%', 'panel_04_change_medium': '7.2%', 'panel_04_change_high': '1.7%', 'panel_04_change_unassigned': '3.5%', 'panel_05_title': '工单关闭所需天数', 'panel_05_summary_title': '不到5天内关闭的工单', 'panel_05_y0': '0千', 'panel_05_y1': '1千', 'panel_05_y2': '2千', 'panel_05_y3': '3千', 'panel_05_x0': '0', 'panel_05_x5': '5', 'panel_05_x10': '10', 'panel_05_x15': '15', 'panel_05_x20': '20', 'panel_05_x25': '25', 'panel_05_x30': '30+', 'panel_05_summary_value': '3,703', 'panel_05_change': '2.1% ▼', 'panel_06_title_prefix': '工单类型：', 'panel_06_title_requests': '请求', 'panel_06_title_vs': '对比', 'panel_06_title_issues': '问题', 'panel_06_request_endpoint': '479', 'panel_06_issue_endpoint': '132', 'panel_06_issue': '问题', 'panel_06_request': '请求', 'panel_06_issue_total': '2,110', 'panel_06_request_total': '6,305', 'panel_06_issue_change': '0.5%', 'panel_06_request_change': '0.3%', 'panel_07_title': '客户满意度', 'panel_07_unsatisfied': '不满意', 'panel_07_satisfied': '满意', 'panel_07_highly_satisfied': '非常\n满意', 'panel_07_unknown': '未知', 'panel_07_change_down_1': '4.2%', 'panel_07_change_up_1': '1.6%', 'panel_07_change_down_2': '3.3%', 'panel_07_change_up_2': '0.4%', 'panel_08_title': '工单创建时段'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
