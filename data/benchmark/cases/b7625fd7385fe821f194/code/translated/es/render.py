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
        (cw, ch) = data['coordinates']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.axis('off')
        placements = []

        def text(key, x, y, size, width, anchor='center'):
            placements.append({'key': key, 'x': x / cw, 'y': y / ch, 'size': size * W / cw, 'max_width': width / cw, 'rotation': 0, 'anchor': anchor})
        ax.add_patch(Rectangle((0, 0), cw, ch, color=data['background']))
        for (x, y, w, h) in data['silhouettes']:
            ax.add_patch(Rectangle((x, y), w, h, facecolor='#082c3d', alpha=0.42, edgecolor='none'))
        ax.add_patch(Rectangle((0, 317), cw, 66, facecolor='#0b374a', edgecolor='none'))
        for (i, p) in enumerate(data['foreground_polygons']):
            ax.add_patch(Polygon(p, facecolor=['#9bdb72', '#cee4e9'][i], edgecolor='none'))
        base = data['base']
        for (bi, (x, y, w, front, kind)) in enumerate(data['buildings']):
            (face, side, roof, dark, light) = data['palette'][kind]
            rise = 10
            ax.add_patch(Polygon([[x, y + 5], [x + front, y + rise], [x + front, base], [x, base - 2]], facecolor=face, edgecolor='none'))
            ax.add_patch(Polygon([[x + front, y + rise], [x + w, y + 7], [x + w, base - 1], [x + front, base]], facecolor=side, edgecolor='none'))
            ax.add_patch(Polygon([[x, y + 5], [x + w - front, y - 2], [x + w, y + 7], [x + front, y + rise]], facecolor=roof, edgecolor='none'))
            (dx, dy, ww, hh, ncol, margin, step, side_step, rowstep, sw, sh, mod, rem) = data['window_geometry']
            for c in range(int(ncol)):
                xx = x + dx + c * step
                for r in range(int((base - y - 14) / dy)):
                    yy = y + 11 + r * dy + c * 0.12
                    ax.add_patch(Polygon([[xx, yy], [xx + ww, yy + 0.35], [xx + ww, yy + hh + 0.35], [xx, yy + hh]], facecolor=dark, alpha=0.86, edgecolor='none'))
            for c in range(int((w - front - 6) / side_step)):
                xx = x + front + margin + c * side_step
                for r in range(int((base - y - 18) / rowstep)):
                    yy = y + 17 + r * rowstep - c * 0.48
                    lit = (r * 7 + c * 11 + bi * 3) % mod < rem
                    ax.add_patch(Rectangle((xx, yy), sw, sh / 2, facecolor=light if lit else dark, alpha=0.95 if lit else 0.38, edgecolor='none'))
            for c in range(1, 9):
                xx = x + front + (w - front) * c / 9
                ax.plot([xx, xx], [y + 11, base - 2], color=roof, alpha=0.09, linewidth=0.55)
        for (x, y, w, h, px, py, end, key, kind) in data['callouts']:
            ax.plot([px, px], [end, py], color='#e4f1ee', linewidth=1)
            ax.scatter([px], [py], s=39, color='#f5faf3', edgecolors='#dcebe5', linewidths=0.6, zorder=8)
            color = '#b9d849' if kind else '#08354a'
            ax.text((x + w / 2) / cw, 1 - (y + h / 2) / ch, labels[key], transform=ax.transAxes, ha='center', va='center', fontsize=1, color=color, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': color, 'edgecolor': color})
            ax.add_patch(Rectangle((x + 7, y), w - 14, h, facecolor=color, edgecolor='none'))
            ax.add_patch(Rectangle((x, y + 7), w, h - 14, facecolor=color, edgecolor='none'))
            for (cx, cy) in [(x + 7, y + 7), (x + w - 7, y + 7), (x + 7, y + h - 7), (x + w - 7, y + h - 7)]:
                ax.add_patch(Circle((cx, cy), 7, facecolor=color, edgecolor='none'))
            ax.plot([x + 7, x + w - 7], [y, y], color='#e8f2e8', lw=0.9)
            ax.plot([x + 7, x + w - 7], [y + h, y + h], color='#e8f2e8', lw=0.9)
            ax.plot([x, x], [y + 7, y + h - 7], color='#e8f2e8', lw=0.9)
            ax.plot([x + w, x + w], [y + 7, y + h - 7], color='#e8f2e8', lw=0.9)
            text(key, x + w / 2, y + h / 2, 27, w - 12)
        for (x, y, key) in data['categories']:
            text(key, x, y, 22, 205)
        text('legend_heading', 116, 225, 22, 220)
        for (x, y, w, h, key, kind) in data['legend']:
            color = '#b9d849' if kind else '#08354a'
            ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='#dce9e5', linewidth=0.85))
            text(key, x + w / 2, y + h / 2, 19, w - 8)
        for (x, y, key) in data['header_positions']:
            text(key, x, y, 27, 300, 'left')
        return finish(fig, labels, placements)

    def panel_01(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 488)
        ax.set_ylim(408, 0)
        ax.axis('off')
        c = data['colors']
        placements = []
        ax.add_patch(Polygon(data['banner'], closed=True, facecolor=data['banner_color'], edgecolor='none'))
        ax.add_patch(Polygon(data['corner'], closed=True, facecolor=c['ink'], edgecolor='none'))
        for vertices in data['hexagons']:
            ax.add_patch(Polygon(vertices, closed=True, facecolor=c['hex_fill'], edgecolor=c['hex_edge'], linewidth=data['stroke_widths'][0]))

        def line(points, lw=None):
            p = np.array(points)
            ax.plot(p[:, 0], p[:, 1], color=c['icon'], linewidth=lw or data['stroke_widths'][1], solid_capstyle='round', solid_joinstyle='round')
        v = data['virus']
        (cx, cy) = v['center']
        r = v['radius']
        outline = []
        for angle in v['angles']:
            for (da, rr) in [(-13, r), (-5, r), (-4, r + 8), (-9, r + 10), (-8, r + 14), (0, r + 15), (5, r + 12), (3, r + 8), (5, r), (13, r)]:
                a = math.radians(angle + da)
                outline.append([cx + rr * math.cos(a), cy + rr * math.sin(a)])
        outline.append(outline[0])
        line(outline)
        for (x, y, rr) in v['holes']:
            ax.add_patch(Circle((x, y), rr, facecolor='none', edgecolor=c['icon'], linewidth=data['stroke_widths'][1]))
        for path in data['puzzle_paths']:
            line(path)
        for arrow in data['arrows']:
            ax.add_patch(Polygon(arrow, closed=True, facecolor='none', edgecolor=c['icon'], linewidth=data['stroke_widths'][1], joinstyle='round'))
        m = data['magnifier']
        ax.add_patch(Circle(m['center'], m['radius'], facecolor=c['hex_fill'], edgecolor=c['icon'], linewidth=data['stroke_widths'][2]))
        ax.text(*data['question_center'], '?', ha='center', va='center', fontsize=31, color=c['icon'], fontfamily='DejaVu Sans')
        for (x, y, w, h) in data['cards']:
            rr = 10
            pts = []
            for (xx, yy, start) in [(x + w - rr, y + rr, -90), (x + w - rr, y + h - rr, 0), (x + rr, y + h - rr, 90), (x + rr, y + rr, 180)]:
                for a in np.linspace(start, start + 90, 10):
                    pts.append([xx + rr * math.cos(math.radians(a)), yy + rr * math.sin(math.radians(a))])
            ax.add_patch(Polygon(pts, closed=True, facecolor=c['card'], edgecolor=c['hex_edge'], linewidth=data['stroke_widths'][0]))

        def place(key, xy, size, width):
            placements.append({'key': key, 'x': xy[0] / 488, 'y': xy[1] / 408, 'size': size, 'max_width': width, 'anchor': 'center', 'rotation': 0})
        place('heading', data['heading_center'], 48, 0.64)
        for (key, xy) in zip(data['value_keys'], data['value_centers']):
            place(key, xy, 70, 0.22)
        for (key, xy) in zip(data['category_keys'], data['category_centers']):
            place(key, xy, 44, 0.31)
        return finish(fig, labels, placements)

    def panel_02(data, labels):
        (W, H) = (1068, 706)
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis('off')
        placements = []
        ax.add_patch(Polygon(data['tower'], closed=True, facecolor=data['tower_color'], edgecolor='none'))
        for key in ['document_hex', 'lock_hex']:
            ax.add_patch(Polygon(data[key], closed=True, facecolor='#daeff6', edgecolor='#e5f5fb', linewidth=2))

        def line(points, color='#8399a2', lw=2):
            p = np.array(points)
            ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, solid_capstyle='round')

        def ellipse(cx, cy, rx, ry, color, lw=2, fill=None):
            t = np.linspace(0, 2 * np.pi, 100)
            p = np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])
            if fill:
                ax.add_patch(Polygon(p, facecolor=fill, edgecolor=color, lw=lw))
            else:
                line(p, color, lw)

        def rounded(x, y, w, h, r):
            pts = []
            for (cx, cy, start) in [(x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)]:
                for a in np.linspace(start, start + 90, 16):
                    pts.append([cx + r * np.cos(a * np.pi / 180), cy + r * W / H * np.sin(a * np.pi / 180)])
            ax.add_patch(Polygon(pts, closed=True, facecolor=data['callout_color'], edgecolor='#f5fbfc', lw=2.4))
        for (k, v, y) in zip(data['row_keys'], data['percentages'], data['row_y']):
            rounded(data['box_x'], y - data['box_height'] / 2, data['box_width'], data['box_height'], data['box_radius'])
            ax.text(0.459, y + 0.003, format(v, '.1f') + '%', ha='center', va='center', fontsize=31, color='#102e3b', fontfamily='DejaVu Sans', fontstretch='condensed')
            placements.append({'key': k, 'x': 0.317, 'y': y, 'size': 44, 'max_width': 0.306, 'anchor': 'right'})
        for k in ['document_outline', 'document_left_bottom', 'fold', 'stamp_handle', 'stamp_base', 'stamp_line']:
            line(data[k], lw=2.8)
        for seg in data['document_lines']:
            line([[seg[0], seg[1]], [seg[2], seg[3]]], lw=2.5)
        ellipse(0.797, 0.342, 0.02, 0.031, '#8399a2', 2.5)
        line([[0.788, 0.342], [0.795, 0.352], [0.807, 0.33]], lw=2.7)
        for (rx, ry) in [(0.024, 0.057), (0.015, 0.043)]:
            t = np.linspace(np.pi, 2 * np.pi, 60)
            line(np.column_stack([0.919 + rx * np.cos(t), 0.108 + ry * np.sin(t)]), lw=2)
        line([[0.895, 0.108], [0.895, 0.088]], lw=2)
        line([[0.943, 0.108], [0.943, 0.088]], lw=2)
        line(data['lock_body'], lw=2)
        t = np.linspace(0, 2 * np.pi, 60)
        ellipse(0.919, 0.143, 0.007, 0.01, '#8399a2', 2)
        line([[0.916, 0.152], [0.916, 0.17], [0.922, 0.17], [0.922, 0.152]], lw=2)
        placements.extend([{'key': 'title', 'x': 0.015, 'y': 0.015, 'size': 43, 'max_width': 0.731, 'anchor': 'left'}, {'key': 'note', 'x': 0.681, 'y': 0.601, 'size': 25, 'max_width': 0.293, 'anchor': 'left'}])
        return finish(fig, labels, placements)

    def panel_03(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor(data['background'])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis('off')
        placements = []
        for (x, y, w, h) in data['background_blocks']:
            ax.add_patch(Rectangle((x, y), w, h, color='#7bc6e9', alpha=0.35, lw=0))
        ax.add_patch(Polygon(data['ribbon_vertices'], closed=True, facecolor=data['ribbon'], edgecolor='none'))
        for d in data['donuts']:
            (x, y) = d['center']
            r = d['radius']
            a = fig.add_axes([x - r * H / W, 1 - y - r, 2 * r * H / W, 2 * r])
            a.pie(d['values'], colors=d['colors'], startangle=d['start'], counterclock=True, wedgeprops={'width': d['width'] / r, 'edgecolor': '#effbfc', 'linewidth': 1.5})
            a.set_xlim(-1.02, 1.02)
            a.set_ylim(-1.02, 1.02)
            a.axis('off')
        for d in data['callouts']:
            ax.text(d['x'], d['y'], str(d['value']) + '%', ha='center', va='center', fontsize=24, color=data['ink'], bbox={'boxstyle': 'round,pad=.27', 'facecolor': d['color'], 'edgecolor': '#eef9ee', 'linewidth': 1.6}, zorder=10)
        for d in [data['donuts'][0], data['donuts'][2]]:
            (x, y) = d['center']
            ax.text(x, y + 0.006, format(d['values'][0], '.1f') + '%', ha='center', va='center', fontsize=20, color=data['ink'], zorder=10)
        for (x, y, y2) in data['leaders']:
            ax.plot([x, x], [y, y2], color=data['ink'], lw=0.7)
            ax.plot(x, y, 'o', color='#071d23', markersize=4)
        for points in data['building_lines']:
            ax.plot([p[0] / 1000 for p in points], [p[1] / 400 for p in points], color=data['building_color'], lw=2, solid_capstyle='round')
        for (x, y, r) in data['building_circles']:
            ax.add_patch(Circle((x / 1000, y / 400), r / 1000, transform=ax.transAxes, visible=False))
        b = fig.add_axes([0.006, 0.7, 0.15, 0.29])
        b.set_xlim(6, 156)
        b.set_ylim(120, 4)
        b.set_aspect('equal')
        b.axis('off')
        for (x, y, r) in data['building_circles']:
            b.add_patch(Circle((x, y), r, fill=False, ec=data['building_color'], lw=2))
        b.text(132, 31, '$', ha='center', va='center', fontsize=26, color=data['building_color'])
        for (key, x, y, size, width, anchor) in [('headline', 0.174, 0.182, 29, 0.8, 'left'), ('all', 0.503, 0.565, 28, 0.17, 'center'), ('recipients', 0.362, 0.574, 26, 0.27, 'right'), ('nonrecipients', 0.644, 0.574, 26, 0.225, 'left'), ('left_caption', 0.008, 0.89, 23, 0.36, 'left'), ('right_caption', 0.986, 0.89, 23, 0.355, 'right')]:
            placements.append({'key': key, 'x': x, 'y': y, 'size': size, 'max_width': width, 'anchor': anchor, 'rotation': 0})
        return finish(fig, labels, placements)
    functions = [panel_00, panel_01, panel_02, panel_03]
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
BASE_ID = 'qa_280ef6241beac1bc01a3500003071ef2d3b74d71cad237a1a6e52d6a86992bf1'
LANGUAGE = 'es'
DATA = {'canvas': [2800, 4613], 'panels': [{'bbox': [0.178, 0.229, 0.978, 0.433], 'data': {'canvas': [1200, 504], 'coordinates': [912, 383], 'background': '#103b50', 'base': 318, 'values': [52.7, 46.4, 72.7, 63.0, 79.8, 71.9], 'buildings': [[264, 151, 87, 38, 0], [359, 167, 89, 38, 1], [493, 90, 89, 38, 0], [587, 119, 90, 38, 1], [718, 65, 88, 38, 0], [813, 91, 88, 38, 1]], 'palette': [['#319fcb', '#126486', '#65c5e0', '#08445e', '#9cdfed'], ['#b5d749', '#357968', '#d2e45b', '#255956', '#c7e84e']], 'window_geometry': [4, 6, 2.3, 3.1, 7, 5, 5, 10, 11, 4, 7, 17, 3], 'callouts': [[269, 92, 87, 39, 311, 151, 131, 'value_1', 0], [362, 108, 87, 39, 405, 167, 147, 'value_2', 1], [498, 29, 87, 40, 541, 90, 69, 'value_3', 0], [594, 59, 87, 39, 637, 119, 98, 'value_4', 1], [720, 6, 87, 39, 763, 65, 45, 'value_5', 0], [815, 31, 88, 39, 859, 91, 70, 'value_6', 1]], 'categories': [[365, 349, 'product'], [583, 347, 'process'], [809, 347, 'any']], 'category_lines': [['Product', 'innovation¹'], ['Business process', 'innovation²'], ['Any type of', 'innovation']], 'silhouettes': [[0, 190, 36, 129], [33, 183, 33, 136], [71, 146, 34, 173], [110, 164, 54, 154], [171, 147, 27, 171], [198, 115, 37, 203], [235, 155, 29, 163], [290, 0, 35, 318], [327, 41, 33, 277], [366, 164, 31, 154], [393, 71, 23, 247], [424, 0, 30, 318], [456, 0, 24, 318], [610, 16, 32, 302], [642, 41, 40, 277], [683, 158, 31, 160], [879, 0, 33, 318]], 'foreground_polygons': [[[0, 318], [283, 383], [0, 383]], [[224, 373], [264, 383], [224, 383]]], 'legend': [[13, 268, 89, 29, 'period_1', 0], [128, 268, 101, 29, 'period_2', 1]], 'header_positions': [[-8, 18, 'fragment_line_1'], [0, 57, 'fragment_line_2']]}, 'label_map': {'fragment_line_1': 'panel_00_fragment_line_1', 'fragment_line_2': 'panel_00_fragment_line_2', 'product': 'panel_00_product', 'process': 'panel_00_process', 'any': 'panel_00_any', 'legend_heading': 'panel_00_legend_heading', 'period_1': 'panel_00_period_1', 'period_2': 'panel_00_period_2', 'value_1': 'panel_00_value_1', 'value_2': 'panel_00_value_2', 'value_3': 'panel_00_value_3', 'value_4': 'panel_00_value_4', 'value_5': 'panel_00_value_5', 'value_6': 'panel_00_value_6'}, 'crop_pixel_bbox': [203, 430, 1115, 813], 'crop_sha256': 'ecc1d9f2f97bcbfeeeb28896fffc46912f5e39c58fd444795e523f0af3a6ac7b', 'api_request_sha256': 'f620f44a2131d7623db4d08d2e142099b0f78965ad78e0cfbfff0c28c8bd0038'}, {'bbox': [0.02, 0.433, 0.448, 0.65], 'data': {'canvas': [976, 816], 'background': '#bfe4f6', 'banner_color': '#8acb65', 'banner': [[73, 0], [403, 0], [403, 71], [238, 153], [73, 61]], 'corner': [[444, 0], [488, 0], [488, 10]], 'hexagons': [[[4, 173], [40, 110], [112, 110], [148, 173], [112, 235], [40, 235]], [[170, 242], [205, 179], [277, 179], [313, 242], [277, 304], [205, 304]], [[335, 173], [371, 110], [443, 110], [479, 173], [443, 235], [371, 235]]], 'cards': [[19, 211, 116, 51], [186, 292, 115, 51], [348, 211, 115, 51]], 'values': [29.4, 28.5, 28.1], 'value_keys': ['covid_value', 'skills_value', 'risk_value'], 'value_centers': [[77, 237], [244, 319], [406, 237]], 'category_keys': ['covid', 'skills', 'risk'], 'category_centers': [[77, 312], [239, 374], [406, 306]], 'heading_center': [240, 43], 'virus': {'center': [74, 165], 'radius': 26, 'angles': [-95, -58, -24, 17, 60, 108, 158, 200, 244], 'holes': [[62, 161, 5], [78, 151, 4], [71, 172, 5], [89, 170, 4], [62, 181, 3], [79, 185, 3]]}, 'puzzle_paths': [[[210, 233], [210, 203], [239, 203], [239, 213], [244, 212], [248, 216], [248, 221], [245, 224], [239, 223], [239, 233], [230, 233], [231, 226], [228, 222], [223, 222], [220, 226], [220, 233], [210, 233]], [[239, 233], [268, 233], [268, 203], [239, 203]], [[210, 233], [210, 261], [239, 261], [239, 252], [233, 253], [229, 250], [229, 245], [233, 242], [239, 244], [239, 233]], [[239, 233], [249, 233], [248, 239], [251, 243], [256, 243], [259, 239], [258, 233], [268, 233]], [[239, 261], [249, 261], [246, 257], [247, 253], [251, 251], [255, 252], [258, 257], [268, 257], [268, 247], [282, 247], [282, 276], [253, 276], [253, 267], [247, 268], [243, 265], [244, 261], [249, 261]], [[268, 247], [261, 247], [262, 252], [259, 256], [255, 256]]], 'arrows': [[[410, 146], [425, 134], [419, 129], [435, 125], [434, 142], [429, 138], [416, 151]], [[418, 158], [438, 158], [438, 152], [454, 163], [439, 173], [439, 167], [418, 167]], [[410, 180], [424, 190], [419, 194], [435, 200], [434, 181], [430, 186], [417, 175]]], 'magnifier': {'center': [391, 163], 'radius': 28}, 'question_center': [391, 164], 'colors': {'icon': '#899ea6', 'hex_fill': '#d5edf7', 'hex_edge': '#eef8fa', 'card': '#afe18f', 'ink': '#003244'}, 'stroke_widths': [1.5, 1.7, 2.4]}, 'label_map': {'heading': 'panel_01_heading', 'covid': 'panel_01_covid', 'skills': 'panel_01_skills', 'risk': 'panel_01_risk', 'covid_value': 'panel_01_covid_value', 'skills_value': 'panel_01_skills_value', 'risk_value': 'panel_01_risk_value'}, 'crop_pixel_bbox': [23, 813, 511, 1221], 'crop_sha256': '3398fcddb97bc948656f5ffbc78a1d2882b244818d19f0e80d69ea48c51ef4c5', 'api_request_sha256': '3da8d11c9d1a045fe58031466b87b370e333a097864f1b67c24fda2db955016b'}, {'bbox': [0.507, 0.473, 0.975, 0.661], 'data': {'percentages': [45.0, 43.7, 33.0], 'row_keys': ['nda', 'trademarks', 'patents'], 'row_y': [0.531, 0.707, 0.893], 'background': '#c2e5f8', 'tower_color': '#42aec8', 'callout_color': '#6bcde5', 'tower': [[0.649, 1.0], [0.649, 0.258], [0.815, 0.092], [0.982, 0.26], [0.982, 0.926], [1.0, 0.896], [1.0, 1.0]], 'document_hex': [[0.699, 0.287], [0.805, 0.199], [0.909, 0.288], [0.909, 0.476], [0.805, 0.565], [0.699, 0.474]], 'lock_hex': [[0.852, 0.074], [0.919, 0.016], [0.984, 0.073], [0.984, 0.19], [0.919, 0.248], [0.852, 0.19]], 'box_x': 0.35, 'box_width': 0.218, 'box_height': 0.146, 'box_radius': 0.022, 'document_outline': [[0.748, 0.309], [0.767, 0.285], [0.845, 0.285], [0.845, 0.352]], 'document_left_bottom': [[0.748, 0.309], [0.748, 0.477], [0.824, 0.477]], 'fold': [[0.748, 0.309], [0.767, 0.309], [0.767, 0.285]], 'document_lines': [[0.759, 0.365, 0.805, 0.365], [0.759, 0.389, 0.814, 0.389], [0.759, 0.41, 0.811, 0.41], [0.759, 0.431, 0.797, 0.431], [0.759, 0.451, 0.796, 0.451]], 'stamp_handle': [[0.837, 0.416], [0.849, 0.387], [0.85, 0.367], [0.859, 0.353], [0.873, 0.355], [0.875, 0.369], [0.867, 0.383], [0.859, 0.394], [0.853, 0.428]], 'stamp_base': [[0.811, 0.418], [0.827, 0.414], [0.847, 0.426], [0.862, 0.441], [0.858, 0.459], [0.851, 0.477], [0.833, 0.475], [0.807, 0.449], [0.811, 0.418]], 'stamp_line': [[0.808, 0.434], [0.83, 0.443], [0.856, 0.465]], 'lock_body': [[0.884, 0.119], [0.897, 0.109], [0.929, 0.11], [0.951, 0.122], [0.951, 0.188], [0.93, 0.196], [0.899, 0.194], [0.884, 0.187], [0.884, 0.119]]}, 'label_map': {'title': 'panel_02_title', 'nda': 'panel_02_nda', 'trademarks': 'panel_02_trademarks', 'patents': 'panel_02_patents', 'note': 'panel_02_note'}, 'crop_pixel_bbox': [578, 888, 1112, 1241], 'crop_sha256': 'ddc9ef1fe4256b48b1592330b50c74d078c5e1364dfe044f02616dcf1465d6a6', 'api_request_sha256': 'e2738405e250bc0179b3950a5e0f5ec7d7c07e5c06a3ebc02dd84cf0b521abb5'}, {'bbox': [0.063, 0.67, 0.94, 0.879], 'data': {'canvas': [1200, 480], 'background': '#94d3f1', 'ribbon': '#bee6f8', 'ink': '#082a39', 'green': '#85c66b', 'teal': '#389bac', 'pale': '#b1e1f5', 'donuts': [{'center': [0.071, 0.565], 'radius': 0.155, 'width': 0.047, 'values': [83.3, 16.7], 'start': 210, 'colors': ['#85c66b', '#b1e1f5']}, {'center': [0.503, 0.584], 'radius': 0.282, 'width': 0.076, 'values': [34.1, 65.9], 'start': 118.62, 'colors': ['#85c66b', '#389bac']}, {'center': [0.927, 0.565], 'radius': 0.155, 'width': 0.047, 'values': [66, 34], 'start': 61.2, 'colors': ['#389bac', '#b1e1f5']}], 'callouts': [{'x': 0.371, 'y': 0.404, 'value': 34.1, 'color': '#b4e69a'}, {'x': 0.637, 'y': 0.404, 'value': 65.9, 'color': '#6ac9e2'}], 'ribbon_vertices': [[0.102, 0.562], [0.145, 0.407], [0.853, 0.407], [0.898, 0.562], [0.853, 0.712], [0.145, 0.712]], 'leaders': [[0.068, 0.723, 0.8], [0.926, 0.723, 0.8]], 'building_lines': [[[14, 32], [57, 8], [100, 32], [100, 38], [14, 38], [14, 32]], [[21, 39], [23, 46], [35, 46], [38, 39]], [[77, 39], [79, 46], [91, 46], [94, 39]], [[24, 46], [24, 76], [34, 76], [34, 46]], [[79, 46], [79, 63]], [[89, 46], [89, 63]], [[40, 54], [74, 54], [74, 64], [40, 64], [40, 54]], [[45, 68], [45, 86]], [[57, 68], [57, 86]], [[69, 68], [69, 86]], [[23, 76], [21, 87], [75, 87]], [[17, 87], [16, 96], [76, 96]], [[9, 108], [9, 100], [76, 100]], [[89, 72], [89, 65], [96, 63], [115, 63], [121, 66], [121, 72]], [[79, 95], [79, 111], [82, 115], [132, 115], [135, 112], [135, 96]], [[78, 73], [135, 73], [136, 83], [132, 91], [115, 94]], [[97, 94], [82, 92], [77, 87], [77, 77], [78, 73]], [[103, 87], [111, 87], [112, 98], [103, 98], [103, 87]]], 'building_circles': [[57, 24, 6], [132, 31, 19]], 'building_color': '#667779', 'background_blocks': [[0.101, 0, 0.018, 0.05], [0.149, 0, 0.026, 0.023], [0.181, 0, 0.026, 0.068], [0.233, 0, 0.034, 0.048], [0.352, 0, 0.026, 0.025], [0.376, 0, 0.034, 0.033], [0.464, 0, 0.021, 0.032], [0.489, 0.03, 0.028, 0.052], [0.521, 0, 0.045, 0.055], [0.579, 0.014, 0.024, 0.053], [0.609, 0, 0.032, 0.055], [0.667, 0, 0.027, 0.062], [0.741, 0, 0.033, 0.031], [0.787, 0, 0.026, 0.101], [0.815, 0.03, 0.033, 0.054], [0.872, 0, 0.026, 0.086], [0.905, 0, 0.039, 0.088], [0.934, 0.08, 0.059, 0.059]]}, 'label_map': {'headline': 'panel_03_headline', 'all': 'panel_03_all', 'recipients': 'panel_03_recipients', 'nonrecipients': 'panel_03_nonrecipients', 'left_caption': 'panel_03_left_caption', 'right_caption': 'panel_03_right_caption'}, 'crop_pixel_bbox': [72, 1258, 1072, 1651], 'crop_sha256': 'be5d8e4a559a95c66f351b30769d73544701bca88bded0b74ebe9c29a0a35164', 'api_request_sha256': '7fc850f9a607f8bcad158488ee059d1f1c119e3a939cc629a7a6150a73067c53'}], 'global_placements': [{'key': 'title', 'x': 0.317, 'y': 0.036, 'size': 58, 'max_width': 0.53}, {'key': 'introduction', 'x': 0.268, 'y': 0.172, 'size': 27, 'max_width': 0.42}, {'key': 'comparison', 'x': 0.774, 'y': 0.125, 'size': 23, 'max_width': 0.29}, {'key': 'footnote_1', 'x': 0.352, 'y': 0.892, 'size': 11, 'max_width': 0.654}, {'key': 'footnote_2', 'x': 0.352, 'y': 0.91, 'size': 11, 'max_width': 0.654}, {'key': 'footnote_3', 'x': 0.352, 'y': 0.928, 'size': 11, 'max_width': 0.654}, {'key': 'source', 'x': 0.192, 'y': 0.946, 'size': 11, 'max_width': 0.334}, {'key': 'copyright', 'x': 0.869, 'y': 0.92, 'size': 11, 'max_width': 0.226}, {'key': 'agency', 'x': 0.161, 'y': 0.971, 'size': 14, 'max_width': 0.153}, {'key': 'website', 'x': 0.5, 'y': 0.971, 'size': 20, 'max_width': 0.2}, {'key': 'wordmark', 'x': 0.908, 'y': 0.965, 'size': 46, 'max_width': 0.15}]}
LABELS = {'title': 'INNOVACIÓN\nen las empresas canadienses\n2020 a 2022', 'introduction': 'Alrededor de 7 de cada 10 empresas canadienses (71.9%) introdujeron innovaciones durante el período 2020–2022, frente a 8 de cada 10 empresas (79.8%) durante el período 2017–2019.', 'comparison': 'La innovación de productos y la innovación de procesos empresariales disminuyeron en el período 2020-2022, en comparación con el período 2017-2019.', 'footnote_1': '1. Una innovación de producto es un bien o servicio nuevo o mejorado que difiere significativamente de los bienes o servicios anteriores de la empresa en cuanto a sus características, funciones o especificaciones de rendimiento y que se ha introducido en el mercado.', 'footnote_2': '2. Una innovación de proceso empresarial es un proceso nuevo o mejorado para una o más actividades empresariales que difiere significativamente de los procesos empresariales anteriores de la empresa y que esta ha puesto en uso en sus operaciones internas o externas.', 'footnote_3': '3. El 83.3% representa a las empresas que recibieron financiación pública para apoyar sus otras actividades de innovación (p. ej., investigación y desarrollo), que dieron lugar a innovaciones de productos o de procesos empresariales entre 2020 y 2022.', 'source': 'Fuente: Oficina de Estadística de Canadá, Encuesta sobre Innovación y Estrategia Empresarial, 2022.', 'copyright': '© Su Majestad el Rey en su calidad de soberano de Canadá,\nrepresentado por el ministro de Industria, 2024', 'agency': 'Oficina de Estadística de Canadá', 'website': 'CONFIGURE_LOCALLY', 'wordmark': 'Canadá', 'panel_00_fragment_line_1': 'empresas (79.8%) durante', 'panel_00_fragment_line_2': 'el período 2019.', 'panel_00_product': 'Innovación de productos¹', 'panel_00_process': 'Innovación de procesos empresariales²', 'panel_00_any': 'Cualquier tipo de innovación', 'panel_00_legend_heading': 'Porcentaje de\nempresas innovadoras', 'panel_00_period_1': '2017–2019', 'panel_00_period_2': '2020–2022', 'panel_00_value_1': '52.7%', 'panel_00_value_2': '46.4%', 'panel_00_value_3': '72.7%', 'panel_00_value_4': '63.0%', 'panel_00_value_5': '79.8%', 'panel_00_value_6': '71.9%', 'panel_01_heading': 'Los 3 principales obstáculos\na la innovación señalados\npor las empresas', 'panel_01_covid': 'Impactos\ncausados por\nla COVID-19', 'panel_01_skills': 'Falta de\ncompetencias', 'panel_01_risk': 'Incertidumbre\ny riesgo', 'panel_01_covid_value': '29.4%', 'panel_01_skills_value': '28.5%', 'panel_01_risk_value': '28.1%', 'panel_02_title': 'Tipos más comunes de protección de la propiedad intelectual (PI) solicitados por empresas innovadoras que desarrollaron nuevos productos', 'panel_02_nda': 'Acuerdos de\nconfidencialidad', 'panel_02_trademarks': 'Marcas comerciales', 'panel_02_patents': 'Patentes', 'panel_02_note': 'Nota: Los porcentajes\nrepresentan la proporción\nde empresas que desarrollaron\nnuevos productos y solicitaron\nsu protección mediante\nestos distintos\nmecanismos.', 'panel_03_headline': 'Entre todas las empresas, las beneficiarias de apoyo público\npara sus actividades relacionadas con la innovación fueron más innovadoras', 'panel_03_all': 'Todas las\nempresas', 'panel_03_recipients': 'Beneficiarias de\napoyo público\npara la innovación', 'panel_03_nonrecipients': 'No beneficiarias de\napoyo público\npara la innovación', 'panel_03_left_caption': 'Proporción de empresas innovadoras\nentre las beneficiarias de apoyo\npúblico para la innovación³', 'panel_03_right_caption': 'Proporción de empresas innovadoras\nentre las no beneficiarias de apoyo\npúblico para la innovación'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
