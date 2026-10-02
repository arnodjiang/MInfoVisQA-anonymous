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
        ax = fig.add_axes([0.29, 0.16, 0.695, 0.655])
        placements = []
        hm = data['heatmap']
        (xmin, xmax, ymin, ymax) = data['bounds']
        xs = np.linspace(xmin, xmax, hm['nx'])
        ys = np.linspace(ymin, ymax, hm['ny'])
        (X, Y) = np.meshgrid(xs, ys)
        (cx, cy) = hm['center']
        (rx, ry) = hm['radii']
        r = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        Z = 0.75 - hm['edge_strength'] * r ** 2
        for (a, b, c) in hm['waves']:
            Z += c * np.sin(a * X + b * Y)
        for (x, y, sx, sy, v) in hm['hotspots']:
            Z += v * np.exp(-((X - x) / sx) ** 2 - ((Y - y) / sy) ** 2)
        rng = np.random.default_rng(hm['seed'])
        noise = rng.normal(0, hm['noise'], Z.shape)
        for i in range(hm['blur_passes']):
            noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, 1, 1)) / 3
        Z += noise
        patch = Polygon(data['outline'], closed=True, facecolor='none', edgecolor='#608891', linewidth=1.6, zorder=4)
        ax.add_patch(patch)
        im = ax.imshow(Z, extent=data['bounds'], origin='lower', cmap='RdYlBu_r', vmin=hm['range'][0], vmax=hm['range'][1], interpolation='nearest', aspect='auto', zorder=1)
        im.set_clip_path(patch)
        for line in data['guide_lines']:
            p = np.array(line)
            ax.plot(p[:, 0], p[:, 1], color='#4e5350', lw=1.6, clip_on=False, zorder=5)
        s = data['square_size']
        for (x, y) in data['squares']:
            ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fill=False, edgecolor='#292d22', linewidth=1.5, zorder=6))
        (x0, x1, y) = data['scale_bar']
        ax.plot([x0, x1], [y, y], color='#202522', lw=3, zorder=7)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.tick_params(axis='both', labelsize=16, length=3, pad=6, color='#999999')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color('#bcbcbc')
            ax.spines[side].set_linewidth(0.8)
        placements.extend([{'key': 'panel', 'x': 0.012, 'y': 0.085, 'size': 115, 'max_width': 0.12, 'anchor': 'left'}, {'key': 'condition', 'x': 0.637, 'y': 0.085, 'size': 65, 'max_width': 0.56, 'anchor': 'center'}, {'key': 'subpanel', 'x': 0.012, 'y': 0.52, 'size': 80, 'max_width': 0.16, 'anchor': 'left'}, {'key': 'x_axis', 'x': 0.645, 'y': 0.959, 'size': 50, 'max_width': 0.1, 'anchor': 'center'}, {'key': 'y_axis', 'x': 0.255, 'y': 0.522, 'size': 50, 'max_width': 0.06, 'anchor': 'center'}])
        ax.annotate('', xy=(0.285, 0.48), xytext=(0.14, 0.48), xycoords=fig.transFigure, textcoords=fig.transFigure, arrowprops={'arrowstyle': '->', 'lw': 1.4, 'color': '#777777'}, annotation_clip=False)
        return finish(fig, labels, placements)

    def panel_01(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.14, 0.17, 0.57, 0.66])
        placements = []
        boundary = np.array(data['boundary'])
        rng = np.random.RandomState(data['seed'])
        (gx, gy) = data['grid']
        (xx, yy) = np.meshgrid(np.arange(0, 2290, gx), np.arange(0, 2080, gy))
        x = xx.ravel()
        y = yy.ravel()
        inside = np.zeros(x.shape, dtype=bool)
        for i in range(len(boundary)):
            a = boundary[i]
            b = boundary[(i + 1) % len(boundary)]
            cross = ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
            inside ^= cross
        x = x[inside]
        y = y[inside]
        val = np.full(x.shape, data['base_count'])
        for (cx, cy, sx, sy, amp) in data['hotspots']:
            val += amp * np.exp(-((x - cx) / sx) ** 2 - ((y - cy) / sy) ** 2)
        texture = (np.sin(x / 29 + np.sin(y / 43) * 2) + np.sin(y / 27 + x / 61) + np.sin(x / 71 - y / 39)) / 3
        val += data['texture_amplitude'] * (0.43 * texture + 0.57 * (rng.rand(len(x)) - 0.5))
        val = np.clip(val, *data['vlim'])
        keep = rng.rand(len(x)) > 0.1
        ax.scatter(x[keep] + rng.uniform(-4, 4, keep.sum()), y[keep] + rng.uniform(-4, 4, keep.sum()), c=val[keep], cmap='RdYlBu_r', vmin=data['vlim'][0], vmax=data['vlim'][1], marker='s', s=8, linewidths=0, alpha=0.8)
        ax.add_patch(Polygon(boundary, closed=True, facecolor='none', edgecolor='#73969e', linewidth=1.25))
        for path in data['line_paths']:
            p = np.array(path)
            ax.plot(p[:, 0], p[:, 1], color='#4d4d3b', lw=1.4)
        for ((cx, cy), w) in zip(data['patch_centers'], data['patch_widths']):
            ax.add_patch(Rectangle((cx - w / 2, cy - w / 2), w, w, fill=False, edgecolor='#383c22', linewidth=1.45))
        (a, b, ybar) = data['scale_bar']
        ax.plot([a, b], [ybar, ybar], color='#363636', lw=3)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_xticks(data['xticks'])
        ax.set_yticks(data['yticks'])
        ax.tick_params(labelsize=15, length=4, pad=3, color='#555555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_color('#888888')
            ax.spines[s].set_linewidth(0.8)
        cax = fig.add_axes([0.76, 0.235, 0.025, 0.535])
        grad = np.linspace(*data['vlim'], 256).reshape(-1, 1)
        cax.imshow(grad, origin='lower', aspect='auto', extent=[0, 1, 0, 2], cmap='RdYlBu_r', vmin=0, vmax=2)
        cax.set_xticks([])
        cax.set_yticks(data['color_ticks'])
        cax.set_yticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'])
        cax.yaxis.tick_right()
        cax.tick_params(labelsize=14, length=0, pad=6)
        for s in cax.spines.values():
            s.set_visible(False)
        placements.extend([{'key': 'treated', 'x': 0.425, 'y': 0.065, 'size': 28, 'max_width': 0.59, 'anchor': 'center'}, {'key': 'x', 'x': 0.425, 'y': 0.95, 'size': 26, 'max_width': 0.15, 'anchor': 'center'}, {'key': 'y', 'x': 0.04, 'y': 0.5, 'size': 26, 'max_width': 0.15, 'rotation': 90, 'anchor': 'center'}, {'key': 'gene_count', 'x': 0.955, 'y': 0.5, 'size': 25, 'max_width': 0.51, 'rotation': 90, 'anchor': 'center'}])
        return finish(fig, labels, placements)

    def panel_02(data, labels):
        fig = plt.figure(figsize=(10.8, 6.75), dpi=100)
        ax = fig.add_axes([0.375, 0.119, 0.537, 0.859])
        ax.set_xlim(data['limits'])
        ax.set_ylim(data['limits'])
        ax.set_xticks(data['ticks'])
        ax.set_yticks(data['ticks'])
        ax.tick_params(axis='both', labelsize=38, length=12, width=1.8, direction='in', pad=7)
        for s in ax.spines.values():
            s.set_linewidth(3)
        for c in data['clusters']:
            ax.add_patch(Polygon(c['polygon'], closed=True, facecolor=data['color'], edgecolor=data['color'], linewidth=0.6))
            p = np.array(c['points'])
            ax.scatter(p[:, 0], p[:, 1], s=np.array(c['sizes']) * 10, c=data['color'], linewidths=0)
        p = np.array(data['specks'])
        ax.scatter(p[:, 0], p[:, 1], s=np.array(data['speck_sizes']) * 9, c=data['color'], linewidths=0)
        p = np.array(data['scale_line'])
        ax.plot(p[:, 0], p[:, 1], color='black', lw=5, solid_capstyle='butt')
        for line in data['zoom_lines']:
            p = np.array(line)
            ax.plot(p[:, 0], p[:, 1], color='#555555', lw=2, clip_on=False)
        placements = [{'key': 'panel', 'x': 0.054, 'y': 0.421, 'size': 76, 'max_width': 0.19, 'anchor': 'center'}, {'key': 'scale', 'x': 0.477, 'y': 0.826, 'size': 52, 'max_width': 0.22, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_03(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        placements = []
        ax.set_xlim(data['limits'])
        ax.set_ylim(data['limits'])
        ax.set_xticks(data['ticks'])
        ax.set_yticks(data['ticks'])
        ax.tick_params(axis='both', labelsize=36, length=4, width=1, pad=6)
        for sp in ax.spines.values():
            sp.set_linewidth(1.7)
        rng = np.random.default_rng(data['seed'])
        color = data['color']
        for (x, y, sx, sy, n) in data['clusters']:
            a = rng.uniform(0, 2 * np.pi, n)
            r = np.sqrt(rng.uniform(0, 1, n))
            xx = x + sx * r * np.cos(a)
            yy = y + sy * r * np.sin(a)
            ax.scatter(xx, yy, s=rng.uniform(7, 34, n), c=color, marker='s', linewidths=0, zorder=3)
            a = np.linspace(0, 2 * np.pi, 13)[:-1]
            rad = rng.uniform(0.65, 1.05, len(a))
            verts = np.column_stack([x + sx * 0.85 * rad * np.cos(a), y + sy * 0.85 * rad * np.sin(a)])
            ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor=color, linewidth=0.2, zorder=4))
        p = np.array(data['specks'])
        ax.scatter(p[:, 0], p[:, 1], s=rng.uniform(2, 12, len(p)), c=color, linewidths=0, marker='.', alpha=0.85)
        for (x1, y1, x2, y2) in data['zoom_lines']:
            ax.plot([x1, x2], [y1, y2], color='#404040', lw=1.5, clip_on=False)
        (x1, x2, y) = data['scale_bar']
        ax.plot([x1, x2], [y, y], color='black', lw=3, solid_capstyle='butt')
        (left, bottom, width, height) = data['axes']
        placements.append({'key': 'scale', 'x': left + width * 20 / 150, 'y': 1 - bottom - height * 9 / 150, 'size': 48, 'max_width': 0.24, 'rotation': 0, 'anchor': 'center'})
        return finish(fig, labels, placements)

    def panel_04(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0.338, 0.123, 0.504, 0.848])
        (lo, hi) = data['color_range']
        cmap = plt.get_cmap('inferno')
        ax.imshow(data['heatmap'], origin='lower', extent=data['extent'], interpolation='nearest', aspect='auto', cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#aaaaaa')
            s.set_linewidth(0.65)
        cb = fig.add_axes([0.873, 0.123, 0.027, 0.848])
        grad = np.linspace(lo, hi, data['color_gradient_samples']).reshape(-1, 1)
        cb.imshow(grad, origin='lower', extent=[0, 1, lo, hi], aspect='auto', cmap=cmap, vmin=lo, vmax=hi)
        cb.set_xticks([])
        cb.set_yticks(data['color_ticks'])
        cb.yaxis.tick_right()
        cb.tick_params(axis='y', length=0, pad=10, labelsize=35)
        for s in cb.spines.values():
            s.set_color('#999999')
            s.set_linewidth(0.65)
        placements = [{'key': 'panel', 'x': 0.055, 'y': 0.444, 'size': 58, 'max_width': 0.15, 'rotation': 0, 'anchor': 'center'}, {'key': 'radius', 'x': 0.302, 'y': 0.464, 'size': 51, 'max_width': 0.43, 'rotation': 90, 'anchor': 'center'}, {'key': 'codensity', 'x': 0.596, 'y': 0.946, 'size': 51, 'max_width': 0.57, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_05(data, labels):
        fig = plt.figure(figsize=(data['width'] / 100, data['height'] / 100), dpi=100)
        ax = fig.add_axes(data['axes_bounds'])
        im = ax.imshow(np.array(data['heatmap']), origin='upper', extent=data['extent'], aspect='auto', interpolation='bilinear', cmap='inferno', vmin=data['color_limits'][0], vmax=data['color_limits'][1])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#777777')
            s.set_linewidth(0.7)
        cax = fig.add_axes(data['colorbar_bounds'])
        cb = fig.colorbar(im, cax=cax, ticks=data['color_ticks'])
        cb.ax.tick_params(labelsize=44, length=10, width=0.7, pad=7)
        cb.outline.set_edgecolor('#777777')
        cb.outline.set_linewidth(0.7)
        placements = []
        for (i, key) in enumerate(['radius', 'codensity']):
            placements.append({'key': key, 'x': data['label_positions'][i][0], 'y': data['label_positions'][i][1], 'size': data['label_sizes'][i], 'rotation': data['label_rotations'][i], 'max_width': 0.7, 'anchor': 'center'})
        return finish(fig, labels, placements)

    def panel_06(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes(data['heatmap_box'])
        cb = fig.add_axes(data['colorbar_box'])
        (ny, nx) = data['grid_shape']
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        (X, Y) = np.meshgrid(x, y)
        z = np.full((ny, nx), data['base_value'])
        for (cx, cy, sx, sy, a) in data['gaussian_features']:
            z += a * np.exp(-0.5 * ((X - cx) / sx) ** 2 - 0.5 * ((Y - cy) / sy) ** 2)
        (cy, sy, a) = data['edge_feature']
        z += a * np.exp(-0.5 * ((Y - cy) / sy) ** 2)
        palette = np.array(data['palette_rgb']) / 255.0
        stops = data['palette_values']
        rgb = np.stack([np.interp(z, stops, palette[:, i]) for i in range(3)], axis=-1)
        ax.imshow(rgb, origin='lower', extent=data['extent'], interpolation='bilinear', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#55505a')
            spine.set_linewidth(0.7)
        (lo, hi) = data['color_limits']
        v = np.linspace(lo, hi, 512)
        grad = np.stack([np.interp(v, stops, palette[:, i]) for i in range(3)], axis=-1)[:, None, :]
        cb.imshow(grad, origin='lower', extent=[0, 1, lo, hi], aspect='auto', interpolation='bilinear')
        cb.set_xticks([])
        cb.set_yticks(data['color_ticks'])
        cb.set_yticklabels(['%.1f' % v for v in data['color_ticks']])
        cb.yaxis.tick_right()
        cb.tick_params(axis='y', labelsize=36 * 72 / 100, length=5, width=0.8, pad=6)
        for spine in cb.spines.values():
            spine.set_color('#615a45')
            spine.set_linewidth(0.8)
        placements = [{'key': 'panel', 'x': 0.06, 'y': 0.425, 'size': 69, 'max_width': 0.19, 'anchor': 'center'}, {'key': 'radius', 'x': 0.294, 'y': 0.44, 'size': 57, 'max_width': 0.46, 'rotation': 90, 'anchor': 'center'}, {'key': 'codensity', 'x': 0.578, 'y': 0.921, 'size': 56, 'max_width': 0.67, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_07(data, labels):
        fig = plt.figure(figsize=(data['canvas'][0] / 100, data['canvas'][1] / 100), dpi=100)
        ax = fig.add_axes(data['axes'])
        im = ax.imshow(np.array(data['heatmap']), cmap='inferno', vmin=data['limits'][0], vmax=data['limits'][1], origin='upper', aspect='auto', interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#444444')
            s.set_linewidth(0.7)
        cax = fig.add_axes(data['colorbar_axes'])
        cb = fig.colorbar(im, cax=cax, ticks=data['color_ticks'])
        cb.ax.set_yticklabels(['0', '0.1', '0.2', '0.3'])
        cb.ax.tick_params(labelsize=37, length=7, width=1, pad=8)
        cb.outline.set_linewidth(0.7)
        placements = [{'key': 'radius', 'x': 0.042, 'y': 0.574, 'size': 70, 'max_width': 0.65, 'rotation': 90, 'anchor': 'center'}, {'key': 'codensity', 'x': 0.4315, 'y': 0.904, 'size': 70, 'max_width': 0.72, 'rotation': 0, 'anchor': 'center'}]
        return finish(fig, labels, placements)

    def panel_08(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor('white')
        ax = fig.add_axes(data['image_box'])
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_facecolor(data['background'])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        rng = np.random.default_rng(data['seed'])
        t = data['texture']
        n = t['point_count']
        x = rng.uniform(0, 1, n)
        y = rng.uniform(0, 1, n)
        weights = (np.sin(x * 29 + y * 13) + np.cos(y * 39 - x * 16) + 2) / 4
        c = np.minimum((weights * len(t['colors'])).astype(int), len(t['colors']) - 1)
        ax.scatter(x, y, s=rng.uniform(*t['point_size'], n), c=np.array(t['colors'])[c], alpha=t['alpha'], linewidths=0)
        cells = data['cells']
        theta = np.linspace(0, 2 * np.pi, cells['rim_points'])
        for i in range(cells['count']):
            (cx, cy) = rng.uniform(0, 1, 2)
            r = rng.uniform(*cells['radius'])
            asp = rng.uniform(*cells['aspect'])
            phase = rng.uniform(0, 2 * np.pi)
            rad = r * (1 + 0.12 * np.sin(3 * theta + phase) + 0.08 * np.cos(5 * theta - phase))
            xx = cx + rad * np.cos(theta)
            yy = cy + rad * asp * np.sin(theta)
            ax.fill(xx, yy, color=cells['core_colors'][i % len(cells['core_colors'])], alpha=0.45, linewidth=0)
            ax.plot(xx, yy, color=cells['rim_colors'][i % len(cells['rim_colors'])], alpha=0.55, lw=0.65)
        for (cx, cy, rx, ry) in data['dark_regions']:
            ax.fill(cx + rx * np.cos(theta), cy + ry * np.sin(theta), color=data['background'], alpha=0.85, linewidth=0)
        b = data['blue_spots']
        ax.scatter(rng.uniform(0, 1, b['count']), rng.uniform(0, 1, b['count']), s=rng.uniform(*b['size'], b['count']), color=b['color'], alpha=b['alpha'], linewidths=0)
        m = data['magenta_spots']
        for (center, n, spread, color) in zip(m['centers'], m['counts'], m['spread'], m['colors']):
            xy = rng.normal(center, spread, (n, 2))
            ax.scatter(xy[:, 0], xy[:, 1], s=rng.uniform(0.8, 5, n), color=color, alpha=0.35, linewidths=0)
        for points in data['outlines']:
            p = np.array(points)
            q = []
            for i in range(len(p) - 1):
                p0 = p[(i - 1) % (len(p) - 1)]
                p1 = p[i]
                p2 = p[i + 1]
                p3 = p[(i + 2) % (len(p) - 1)]
                for u in np.linspace(0, 1, 8, endpoint=False):
                    q.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u))
            q = np.array(q + [q[0]])
            ax.plot(q[:, 0], q[:, 1], color='white', lw=data['outline_width'] * W / 284, solid_capstyle='round')
        placements = [{'key': 'title', 'x': 0.496, 'y': 0.026, 'size': 42, 'max_width': 0.65, 'anchor': 'center'}, {'key': 'panel', 'x': 0.008, 'y': 0.001, 'size': 45, 'max_width': 0.07, 'anchor': 'left'}, {'key': 'cropped_channel', 'x': 0.935, 'y': 0.034, 'size': 39, 'max_width': 0.065, 'anchor': 'left'}]
        return finish(fig, labels, placements)

    def panel_09(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        fig.patch.set_facecolor('white')
        ax = fig.add_axes(data['image_bounds'])
        ax.set(xlim=(0, 1), ylim=(1, 0))
        ax.set_facecolor(data['background'])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        rng = np.random.default_rng(data['seed'])
        for i in range(data['cell_count']):
            (x, y) = rng.uniform(0, 1, 2)
            r = rng.uniform(*data['cell_radius_range'])
            t = np.linspace(0, 2 * np.pi, 19)
            a = rng.uniform(0, np.pi)
            rx = r * (1 + 0.14 * rng.normal(size=len(t)))
            ry = r * rng.uniform(0.7, 1.6)
            xx = rx * np.cos(t)
            yy = ry * np.sin(t)
            p = np.column_stack((x + xx * np.cos(a) - yy * np.sin(a), y + xx * np.sin(a) + yy * np.cos(a)))
            c = data['cell_colors'][i % len(data['cell_colors'])]
            ax.add_patch(Polygon(p, closed=True, facecolor='#130729', edgecolor=c, linewidth=rng.uniform(0.5, 1.8), alpha=0.64))
        n = data['puncta_count']
        x = rng.uniform(0, 1, n)
        y = rng.uniform(0, 1, n)
        colors = np.asarray(data['puncta_colors'])
        idx = rng.choice(len(colors), n, p=[0.29, 0.27, 0.21, 0.08, 0.14, 0.01])
        sizes = rng.uniform(*data['puncta_size_range'], n)
        ax.scatter(x, y, s=sizes, c=colors[idx], alpha=0.22, linewidths=0)
        for center in data['bright_centers']:
            k = data['bright_points_per_center']
            p = rng.normal(center, data['bright_spread'], size=(k, 2))
            ax.scatter(p[:, 0], p[:, 1], s=rng.uniform(1, 13, k), color='#b409cc', alpha=0.25, linewidths=0)
        ax.add_patch(Polygon(data['void_outline'], closed=True, facecolor='#06010d', edgecolor='#10031c', linewidth=9, alpha=0.94))
        for key in ['upper_outline', 'lower_outline']:
            p = np.asarray(data[key])
            ax.plot(p[:, 0], p[:, 1], color='white', linewidth=1.9, solid_joinstyle='round', solid_capstyle='round')
        placements = [{'key': 'title', 'x': 0.504, 'y': 0.033, 'size': 43, 'max_width': 0.66, 'anchor': 'center'}, {'key': 'cropped_label', 'x': 0.001, 'y': 0.033, 'size': 40, 'max_width': 0.11, 'anchor': 'left'}]
        return finish(fig, labels, placements)

    def panel_10(data, labels):
        (W, H) = data['canvas']
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0.175, 0.113, 0.789, 0.841])
        placements = []
        for (i, series) in enumerate(data['series']):
            ax.bar(np.array(data['x']) + data['offsets'][i], data['values'][i], width=data['bar_width'], color=data['colors'][i], edgecolor='black', linewidth=1.6, yerr=data['errors'][i], error_kw={'ecolor': '#454545', 'elinewidth': 1.6, 'capsize': 13, 'capthick': 1.6})
        ax.set_xlim(*data['x_limits'])
        ax.set_ylim(*data['y_limits'])
        ax.set_yticks(data['y_ticks'])
        ax.set_xticks(data['x'])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', length=0, labelsize=34, pad=18)
        ax.tick_params(axis='x', length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.6)
        ax.spines['bottom'].set_linewidth(1.6)
        for b in data['brackets']:
            ax.plot(b['x'], b['y'], color='#303030', linewidth=1.6)
        fig.canvas.draw()
        for (x, key) in zip(data['x'], data['groups']):
            p = ax.transData.transform((x, 0))
            placements.append({'key': key, 'x': float(p[0] / W), 'y': 0.941, 'size': 47, 'max_width': 0.39, 'anchor': 'center'})
        for b in data['brackets']:
            p = ax.transData.transform((b['text_x'], b['text_y']))
            placements.append({'key': b['label'], 'x': float(p[0] / W), 'y': float(1 - p[1] / H), 'size': 46, 'max_width': 0.2, 'anchor': 'center'})
        placements.append({'key': 'y_axis', 'x': 0.03, 'y': 0.47, 'size': 47, 'max_width': 0.6, 'rotation': 90, 'anchor': 'center'})
        return finish(fig, labels, placements)

    def panel_11(data, labels):
        fig = plt.figure(figsize=(9.5, 10.3), dpi=100)
        ax = fig.add_axes([0.28, 0.12, 0.67, 0.83])
        for g in data['groups']:
            ax.scatter(g['x'], g['y'], s=250, c=g['color'], edgecolors='none', alpha=0.9)
        ax.set_xlim(data['xlim'])
        ax.set_ylim(data['ylim'])
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(['%.1f' % v for v in data['yticks']], fontsize=38)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([])
        ax.tick_params(axis='both', length=0, pad=20)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_linewidth(3)
        ax.plot(data['bracket_x'], data['bracket_y'], color='#393939', lw=4)
        placements = [{'key': 'ylabel', 'x': 0.087, 'y': 0.54, 'size': 49, 'max_width': 0.74, 'rotation': 90, 'anchor': 'center'}, {'key': 'significance', 'x': 0.615, 'y': 0.043, 'size': 45, 'max_width': 0.25, 'rotation': 0, 'anchor': 'center'}]
        for (i, g) in enumerate(data['groups']):
            placements.append({'key': g['label'], 'x': 0.28 + 0.67 * (i + 0.5) / 2, 'y': 0.94, 'size': 48, 'max_width': 0.32, 'rotation': 0, 'anchor': 'center'})
        return finish(fig, labels, placements)
    functions = [panel_00, panel_01, panel_02, panel_03, panel_04, panel_05, panel_06, panel_07, panel_08, panel_09, panel_10, panel_11]
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
BASE_ID = 'qa_42f90ff127442166df7d386c8b2ec23a5aa86e92a9b3f55b10dd6ad8f643c992'
LANGUAGE = 'ur'
DATA = {'canvas': [2800, 1829], 'panels': [{'bbox': [0.0, 0.0, 0.211, 0.273], 'data': {'canvas': [1080, 915], 'bounds': [0, 3100, 0, 2450], 'xticks': [0, 1000, 2000, 3000], 'yticks': [0, 500, 1000, 1500, 2000], 'outline': [[0, 1370], [310, 1775], [740, 2100], [1200, 2350], [1710, 2390], [2120, 2370], [2490, 2160], [2780, 1815], [2990, 1450], [3050, 920], [2990, 490], [2775, 265], [2380, 60], [2050, 225], [1710, 470], [1160, 570], [630, 715], [225, 900], [0, 1100]], 'squares': [[760, 1050], [835, 945], [780, 1230], [945, 1430], [775, 1360], [1250, 1210], [1370, 1570], [1490, 1740], [2200, 1790], [2540, 1430], [2310, 1120], [2330, 1040], [1980, 1155], [2110, 960], [2100, 725], [1920, 490], [1770, 410], [2240, 450], [2440, 510], [2440, 730]], 'square_size': 145, 'guide_lines': [[[1330, 2380], [1330, 550], [2660, -600]], [[370, -650], [2580, 810]]], 'scale_bar': [60, 350, 350], 'heatmap': {'nx': 75, 'ny': 62, 'seed': 19, 'range': [-1.05, 1.45], 'center': [1480, 1240], 'radii': [1580, 1250], 'edge_strength': 1.5, 'noise': 0.34, 'waves': [[0.0045, 0.0021, 0.22], [0.012, -0.007, 0.2], [0.019, 0.014, 0.12]], 'hotspots': [[750, 1100, 330, 490, 0.5], [1610, 1710, 280, 370, 0.45], [2330, 950, 310, 470, 0.45], [1550, 430, 400, 240, 0.25]], 'blur_passes': 1}}, 'label_map': {'panel': 'panel_00_panel', 'condition': 'panel_00_condition', 'subpanel': 'panel_00_subpanel', 'x_axis': 'panel_00_x_axis', 'y_axis': 'panel_00_y_axis'}, 'crop_pixel_bbox': [0, 0, 216, 183], 'crop_sha256': '2709741c32586943c26362312820cdbd9cb007e0f276fc9ac3a22ad4d31fb0d4', 'api_request_sha256': 'e7239a1fa8aebafe56a6d35689021804e4f8903beb011b5a3ad98a1a2a398892'}, {'bbox': [0.211, 0.0, 0.437, 0.273], 'data': {'canvas': [1000, 780], 'xlim': [0, 2250], 'ylim': [-50, 2075], 'xticks': [0, 1000, 2000], 'yticks': [0, 500, 1000, 1500, 2000], 'boundary': [[0, 1570], [200, 1770], [470, 1950], [790, 2050], [1050, 2065], [1370, 2020], [1680, 1940], [1950, 1840], [2140, 1710], [2240, 1440], [2280, 1110], [2230, 830], [2100, 590], [1880, 410], [1610, 280], [1300, 180], [950, 120], [650, 70], [330, 0], [120, 85], [0, 270]], 'patch_centers': [[290, 1300], [400, 1430], [490, 1530], [760, 1500], [920, 1490], [770, 1090], [920, 1050], [735, 925], [685, 640], [515, 800], [1170, 820], [1320, 1190], [1440, 1360], [1425, 1490], [1480, 1290], [1640, 1170], [1550, 860], [1710, 615]], 'patch_widths': [130, 125, 125, 140, 135, 145, 125, 155, 145, 155, 150, 160, 150, 135, 150, 145, 155, 145], 'line_paths': [[[650, 0], [650, 1970]], [[360, 0], [1450, 1520], [1850, 0]], [[0, 555], [1980, 530]]], 'scale_bar': [30, 450, 205], 'grid': [13, 14], 'seed': 31, 'vlim': [0, 2], 'color_ticks': [0, 0.5, 1, 1.5, 2], 'hotspots': [[360, 1300, 430, 550, 0.58], [900, 1670, 410, 270, 0.72], [1370, 1280, 290, 580, 0.66], [820, 650, 400, 330, 0.61], [1560, 390, 440, 190, 0.53], [1900, 1130, 230, 470, 0.25]], 'base_count': 0.31, 'texture_amplitude': 0.75}, 'label_map': {'treated': 'panel_01_treated', 'x': 'panel_01_x', 'y': 'panel_01_y', 'gene_count': 'panel_01_gene_count'}, 'crop_pixel_bbox': [216, 0, 447, 183], 'crop_sha256': '9896844499b698cfbb3de95276e1e5996aa00fd03c9d32cfeb1517fc53940f4d', 'api_request_sha256': 'b52e683cd1c746f685ce029ad0960784db896bb8af2877b33d5d6183c3ca2dbd'}, {'bbox': [0.0, 0.273, 0.211, 0.475], 'data': {'limits': [0, 150], 'ticks': [0, 50, 100, 150], 'color': '#410075', 'clusters': [{'polygon': [[109, 35], [107, 37], [111, 39], [108, 42], [112, 42], [111, 46], [114, 43], [116, 41], [120, 41], [121, 39], [124, 38], [121, 36], [118, 36], [116, 34], [113, 36]], 'points': [[108, 39], [111, 38], [113, 40], [115, 39], [117, 38], [120, 38], [114, 42], [112, 44], [118, 37], [116, 35], [122, 37]], 'sizes': [3, 6, 8, 8, 6, 3, 5, 3, 5, 3, 2]}, {'polygon': [[113, 113], [114, 116], [114, 119], [116, 121], [119, 121], [120, 118], [119, 116], [122, 114], [119, 113], [117, 114], [115, 113]], 'points': [[115, 118], [117, 120], [118, 117], [116, 115], [120, 114], [117, 123]], 'sizes': [6, 7, 6, 6, 3, 2]}], 'specks': [[105, 44], [112, 47], [109, 34], [127, 134], [132, 142]], 'speck_sizes': [1, 1, 1, 0.7, 0.6], 'scale_line': [[8, 22], [38, 22]], 'zoom_lines': [[[0, 150], [10, 157]], [[150, 150], [144, 157]]]}, 'label_map': {'panel': 'panel_02_panel', 'scale': 'panel_02_scale'}, 'crop_pixel_bbox': [0, 183, 216, 318], 'crop_sha256': 'eca6051b830d6d998052c5fe0b75a3e4e19da42540619ccb266d7f4a8818a77b', 'api_request_sha256': '026eb7beb2ed01eaed02442fadfac2b8dd7b6ffc597f3b139018d0508a1e9b18'}, {'bbox': [0.211, 0.273, 0.405, 0.475], 'data': {'canvas': [1000, 900], 'axes': [0.2, 0.32, 0.58, 0.64], 'limits': [0, 150], 'ticks': [0, 50, 100, 150], 'color': '#49006c', 'clusters': [[32, 144, 2.5, 2.0, 25], [82, 130, 4.8, 4.1, 85], [87, 137, 2.9, 3.2, 35], [90, 124, 5.0, 3.3, 65], [96, 120, 2.9, 2.2, 20], [15, 79, 4.1, 4.6, 65], [25, 83, 2.0, 2.3, 18], [24, 71, 2.4, 2.1, 20], [16, 60, 4.0, 3.0, 55], [14, 53, 4.9, 3.8, 85], [51, 61, 1.9, 2.0, 13], [74, 57, 3.3, 4.0, 55], [80, 52, 5.0, 4.7, 110], [86, 47, 2.6, 3.1, 28], [50, 35, 3.5, 3.5, 52], [55, 31, 3.4, 2.5, 38], [62, 13, 3.1, 2.5, 30], [67, 8, 2.5, 2.4, 23], [71, 4, 2.8, 1.5, 20], [94, 86, 2.8, 2.0, 23], [134, 18, 3.8, 2.0, 28], [147, 59, 2.3, 5.1, 40], [150, 66, 2.1, 3.6, 20], [127, 65, 0.7, 2.2, 6]], 'specks': [[40, 143], [41, 140], [38, 123], [35, 114], [34, 112], [72, 137], [70, 143], [105, 134], [110, 131], [109, 119], [116, 149], [121, 126], [127, 111], [120, 91], [130, 88], [98, 106], [42, 50], [50, 66], [55, 41], [60, 45], [61, 30], [47, 25], [45, 14], [57, 18], [77, 23], [91, 30], [93, 32], [97, 29], [103, 34], [107, 28], [109, 23], [120, 20], [121, 9], [136, 38], [134, 23], [136, 22], [90, 11], [86, 14], [104, 61], [104, 53], [42, 94], [31, 87], [31, 80], [8, 71], [7, 40], [34, 35], [84, 38], [98, 23], [108, 6], [71, 72], [125, 47]], 'scale_bar': [7, 32, 20], 'zoom_lines': [[0, 150, 10, 162], [150, 150, 145, 165]], 'seed': 17}, 'label_map': {'scale': 'panel_03_scale'}, 'crop_pixel_bbox': [216, 183, 415, 318], 'crop_sha256': '298e4ecaae97077d2d8811a378e2fb8f3d2e417b0ce175f633a1f46cb80124dd', 'api_request_sha256': '5b3e6f8a44d961a224c5cc941f9e4381eb12d5da6753872c77d2b81096bf9e35'}, {'bbox': [0.0, 0.524, 0.229, 0.733], 'data': {'heatmap': [[1.35, 1.35, 1.35, 1.35], [1.35, 1.35, 1.35, 1.35], [1.35, 1.35, 1.35, 1.35], [1.35, 1.35, 1.35, 1.35]], 'extent': [0, 1, 0, 1], 'color_range': [0, 15], 'color_ticks': [0, 5, 10, 15], 'color_gradient_samples': 256, 'canvas': [936, 556]}, 'label_map': {'panel': 'panel_04_panel', 'radius': 'panel_04_radius', 'codensity': 'panel_04_codensity'}, 'crop_pixel_bbox': [0, 351, 234, 490], 'crop_sha256': 'e5a9c83b97a7fdfaf0457fb1f6b22979f7ec95770ace2e845c8ab3bcfc015291', 'api_request_sha256': '42f2e8978d66b9fcbec4ff2cd5d436d99b88f92752a163a6f5bd82c69824c0b2'}, {'bbox': [0.229, 0.524, 0.405, 0.733], 'data': {'width': 905, 'height': 695, 'extent': [0, 1, 0, 1], 'color_limits': [0, 15], 'color_ticks': [0, 5, 10, 15], 'heatmap': [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1, 0.9, 0.6, 0.5], [0.5, 1.8, 3.2, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.5, 3.6, 3.9, 2.4, 0.8, 0.5], [0.5, 3.8, 7.4, 8.4, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.3, 8.5, 7.8, 3, 0.8, 0.5], [0.5, 3.7, 7.2, 11.5, 12.8, 12.8, 12.8, 12.8, 12.8, 12.8, 12.9, 12.4, 8.5, 3.1, 1, 0.5], [0.6, 2.7, 6.4, 9.3, 10.9, 10.9, 10.9, 10.9, 10.9, 10.9, 10.7, 9.7, 5, 3, 2.1, 0.8], [0.7, 2.9, 5.6, 5.1, 4.1, 4, 4, 4, 4, 4, 4, 4, 3.6, 3, 2.5, 1], [0.6, 1.4, 1.8, 1.4, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1, 0.7], [0.6, 1.5, 2.2, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.3, 2, 1], [0.5, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.6], [0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5], [0.6, 1, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.2, 0.8], [0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7], [0.6, 1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 0.8]], 'axes_bounds': [0.116, 0.129, 0.652, 0.835], 'colorbar_bounds': [0.807, 0.129, 0.033, 0.835], 'label_positions': [[0.06, 0.46], [0.441, 0.945]], 'label_sizes': [63, 63], 'label_rotations': [90, 0]}, 'label_map': {'radius': 'panel_05_radius', 'codensity': 'panel_05_codensity'}, 'crop_pixel_bbox': [234, 351, 415, 490], 'crop_sha256': 'bae2dda7fa9b03b6e49e7fe2558b0eedaa47c09d3078efb507f93b9169ba53c9', 'api_request_sha256': '5f836f60d5ca78e24421784f4e31fa2680a9c2a8dc1b66f20e143aea29545f69'}, {'bbox': [0.0, 0.783, 0.233, 1.0], 'data': {'canvas': [1000, 607], 'heatmap_box': [0.333, 0.165, 0.491, 0.809], 'colorbar_box': [0.855, 0.165, 0.025, 0.809], 'extent': [0, 1, 0, 1], 'grid_shape': [100, 100], 'base_value': 0.006, 'color_limits': [0, 0.32], 'color_ticks': [0, 0.1, 0.2, 0.3], 'palette_values': [0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32], 'palette_rgb': [[22, 12, 45], [44, 12, 69], [81, 15, 87], [119, 25, 86], [160, 42, 74], [200, 66, 53], [230, 106, 34], [244, 156, 25], [249, 207, 62]], 'gaussian_features': [[0.956, 0.711, 0.022, 0.029, 0.018], [0.935, 0.681, 0.025, 0.02, 0.009], [0.971, 0.204, 0.018, 0.022, 0.007], [0.839, 0.025, 0.15, 0.012, 0.006], [0.41, 0.019, 0.29, 0.008, 0.002]], 'edge_feature': [0.018, 0.007, 0.004]}, 'label_map': {'panel': 'panel_06_panel', 'radius': 'panel_06_radius', 'codensity': 'panel_06_codensity'}, 'crop_pixel_bbox': [0, 524, 239, 669], 'crop_sha256': '55a8d3ecf0cb57eb39cef66beeecc3d0a636031b5e0d1a17ced95c80b8534db5', 'api_request_sha256': '480f9abd7cc3083f201a6e1da6306db825d3ef35d0edfb181c8780baf5413555'}, {'bbox': [0.233, 0.783, 0.407, 1.0], 'data': {'heatmap': [[0.022, 0.037, 0.075, 0.139, 0.214, 0.262, 0.288, 0.269, 0.219, 0.165, 0.139, 0.128, 0.118, 0.112, 0.109, 0.108], [0.027, 0.049, 0.108, 0.19, 0.248, 0.293, 0.305, 0.25, 0.181, 0.15, 0.139, 0.132, 0.125, 0.119, 0.115, 0.113], [0.029, 0.061, 0.135, 0.206, 0.226, 0.26, 0.268, 0.211, 0.166, 0.157, 0.155, 0.151, 0.145, 0.137, 0.128, 0.119], [0.03, 0.066, 0.139, 0.174, 0.186, 0.206, 0.215, 0.2, 0.19, 0.192, 0.196, 0.199, 0.193, 0.176, 0.151, 0.134], [0.035, 0.065, 0.102, 0.126, 0.155, 0.186, 0.219, 0.234, 0.229, 0.222, 0.223, 0.229, 0.227, 0.209, 0.18, 0.158], [0.034, 0.058, 0.085, 0.103, 0.115, 0.132, 0.157, 0.179, 0.183, 0.178, 0.174, 0.178, 0.193, 0.222, 0.24, 0.214], [0.035, 0.064, 0.094, 0.109, 0.115, 0.124, 0.143, 0.158, 0.16, 0.159, 0.153, 0.162, 0.196, 0.244, 0.279, 0.248], [0.036, 0.065, 0.094, 0.113, 0.127, 0.139, 0.137, 0.13, 0.124, 0.123, 0.125, 0.144, 0.194, 0.247, 0.288, 0.254], [0.035, 0.051, 0.077, 0.096, 0.112, 0.123, 0.119, 0.11, 0.105, 0.104, 0.104, 0.121, 0.153, 0.183, 0.209, 0.19], [0.031, 0.043, 0.059, 0.079, 0.095, 0.108, 0.113, 0.115, 0.116, 0.11, 0.102, 0.103, 0.112, 0.125, 0.14, 0.133], [0.032, 0.044, 0.061, 0.077, 0.081, 0.084, 0.091, 0.097, 0.095, 0.088, 0.084, 0.078, 0.073, 0.072, 0.076, 0.082], [0.035, 0.055, 0.096, 0.139, 0.15, 0.145, 0.15, 0.157, 0.165, 0.167, 0.16, 0.152, 0.14, 0.119, 0.107, 0.099], [0.045, 0.077, 0.132, 0.176, 0.171, 0.16, 0.174, 0.188, 0.197, 0.186, 0.173, 0.165, 0.162, 0.139, 0.123, 0.104], [0.039, 0.061, 0.092, 0.114, 0.118, 0.12, 0.127, 0.125, 0.126, 0.131, 0.125, 0.116, 0.111, 0.1, 0.092, 0.076], [0.027, 0.035, 0.042, 0.048, 0.052, 0.05, 0.047, 0.045, 0.045, 0.046, 0.046, 0.045, 0.044, 0.041, 0.037, 0.03], [0.021, 0.028, 0.034, 0.037, 0.038, 0.036, 0.034, 0.032, 0.033, 0.034, 0.035, 0.035, 0.034, 0.03, 0.025, 0.021], [0.018, 0.025, 0.031, 0.034, 0.035, 0.032, 0.028, 0.027, 0.027, 0.027, 0.028, 0.03, 0.03, 0.027, 0.022, 0.019], [0.02, 0.028, 0.036, 0.04, 0.04, 0.037, 0.034, 0.032, 0.03, 0.03, 0.031, 0.032, 0.032, 0.029, 0.024, 0.019], [0.017, 0.023, 0.028, 0.031, 0.031, 0.029, 0.027, 0.025, 0.024, 0.024, 0.025, 0.027, 0.027, 0.024, 0.02, 0.016], [0.025, 0.039, 0.049, 0.053, 0.052, 0.049, 0.045, 0.042, 0.04, 0.039, 0.041, 0.044, 0.045, 0.04, 0.033, 0.024], [0.027, 0.043, 0.057, 0.064, 0.064, 0.063, 0.06, 0.058, 0.057, 0.057, 0.059, 0.061, 0.06, 0.054, 0.044, 0.03], [0.019, 0.027, 0.033, 0.036, 0.036, 0.035, 0.033, 0.031, 0.03, 0.03, 0.03, 0.031, 0.03, 0.027, 0.023, 0.018], [0.024, 0.037, 0.049, 0.055, 0.057, 0.056, 0.054, 0.051, 0.049, 0.049, 0.05, 0.052, 0.051, 0.046, 0.037, 0.025], [0.045, 0.061, 0.068, 0.071, 0.073, 0.072, 0.068, 0.065, 0.064, 0.064, 0.064, 0.065, 0.064, 0.058, 0.048, 0.035]], 'limits': [0, 0.32], 'color_ticks': [0, 0.1, 0.2, 0.3], 'canvas': [1000, 815], 'axes': [0.105, 0.172, 0.653, 0.803], 'colorbar_axes': [0.797, 0.172, 0.035, 0.803]}, 'label_map': {'radius': 'panel_07_radius', 'codensity': 'panel_07_codensity'}, 'crop_pixel_bbox': [239, 524, 417, 669], 'crop_sha256': 'cf8a691083801974776ba90b0ac5cf2c57c980e4b7429b285300aa2a98d45bd5', 'api_request_sha256': 'e4d367e8d83d0f0fb67c6cd16f5d39946a819f818973f3c396a6ebc93c2702ef'}, {'bbox': [0.441, 0.034, 0.719, 0.521], 'data': {'canvas': [900, 1033], 'image_box': [0.007, 0.009, 0.986, 0.919], 'background': '#070214', 'seed': 37, 'texture': {'point_count': 47000, 'point_size': [0.3, 3.5], 'colors': ['#110628', '#190839', '#200b55', '#250d70', '#371060', '#501052', '#681165'], 'alpha': 0.35}, 'cells': {'count': 440, 'radius': [0.01, 0.032], 'aspect': [0.55, 1.3], 'rim_points': 32, 'rim_colors': ['#170d45', '#21105d', '#23116a', '#311067'], 'core_colors': ['#100826', '#150931', '#160b40']}, 'blue_spots': {'count': 1600, 'size': [0.5, 8], 'color': '#211ba0', 'alpha': 0.3}, 'magenta_spots': {'centers': [[0.169, 0.238], [0.252, 0.246], [0.391, 0.372], [0.502, 0.413], [0.534, 0.546], [0.532, 0.604], [0.716, 0.884]], 'counts': [130, 180, 250, 140, 160, 80, 65], 'spread': [0.03, 0.034, 0.034, 0.013, 0.025, 0.024, 0.017], 'colors': ['#5e0c75', '#71107f', '#761486', '#a019a0', '#771187', '#611179', '#701479']}, 'dark_regions': [[0.112, 0.08, 0.048, 0.044], [0.606, 0.049, 0.059, 0.066], [0.178, 0.311, 0.06, 0.05], [0.643, 0.626, 0.054, 0.061], [0.657, 0.829, 0.044, 0.047], [0.891, 0.057, 0.055, 0.066]], 'outlines': [[[0.364, 0.472], [0.369, 0.505], [0.363, 0.546], [0.346, 0.582], [0.323, 0.608], [0.287, 0.63], [0.241, 0.641], [0.196, 0.642], [0.146, 0.634], [0.104, 0.617], [0.069, 0.589], [0.043, 0.553], [0.027, 0.515], [0.027, 0.48], [0.041, 0.447], [0.063, 0.42], [0.101, 0.398], [0.148, 0.384], [0.194, 0.38], [0.244, 0.386], [0.293, 0.402], [0.335, 0.427], [0.357, 0.45], [0.364, 0.472]], [[0.97, 0.424], [0.969, 0.459], [0.957, 0.491], [0.933, 0.518], [0.902, 0.537], [0.864, 0.551], [0.825, 0.554], [0.786, 0.547], [0.752, 0.533], [0.725, 0.51], [0.708, 0.479], [0.699, 0.446], [0.699, 0.409], [0.708, 0.374], [0.724, 0.345], [0.751, 0.328], [0.789, 0.315], [0.832, 0.31], [0.871, 0.312], [0.907, 0.326], [0.935, 0.349], [0.954, 0.378], [0.967, 0.404], [0.97, 0.424]]], 'outline_width': 1.6}, 'label_map': {'panel': 'panel_08_panel', 'title': 'panel_08_title', 'cropped_channel': 'panel_08_cropped_channel'}, 'crop_pixel_bbox': [452, 23, 736, 349], 'crop_sha256': 'c385bfe4891c3a01b785220751b01b2bbf6fee70dacc2b87f4a51ce04ac50b5d', 'api_request_sha256': '903067d1c1197e8dc4d41a4869755353a59e953f4407851dd9ea2f4bf433b4e7'}, {'bbox': [0.721, 0.034, 1.0, 0.521], 'data': {'canvas': [900, 1026], 'seed': 26, 'image_bounds': [0.012, 0.009, 0.98, 0.919], 'background': '#10031e', 'cell_count': 640, 'cell_radius_range': [0.011, 0.026], 'cell_colors': ['#24105d', '#321074', '#451066', '#191059', '#521078'], 'puncta_count': 14500, 'puncta_colors': ['#341075', '#51118a', '#760793', '#9c0bac', '#291063', '#cb18d0'], 'puncta_size_range': [0.5, 7], 'bright_centers': [[0.49, 0.28], [0.56, 0.31], [0.59, 0.25], [0.51, 0.22], [0.44, 0.4], [0.48, 0.47], [0.58, 0.48], [0.66, 0.52], [0.64, 0.75], [0.58, 0.72], [0.53, 0.79], [0.7, 0.83], [0.77, 0.8]], 'bright_spread': [0.015, 0.012], 'bright_points_per_center': 100, 'void_outline': [[0.66, 0.255], [0.69, 0.23], [0.77, 0.205], [0.82, 0.2], [0.87, 0.179], [0.94, 0.195], [1, 0.178], [1, 0.292], [0.94, 0.28], [0.88, 0.298], [0.81, 0.277], [0.76, 0.299], [0.69, 0.3], [0.65, 0.288]], 'upper_outline': [[0.691, 0.505], [0.681, 0.463], [0.654, 0.426], [0.623, 0.398], [0.584, 0.379], [0.548, 0.372], [0.51, 0.379], [0.477, 0.398], [0.447, 0.429], [0.426, 0.465], [0.416, 0.503], [0.42, 0.541], [0.439, 0.573], [0.472, 0.598], [0.515, 0.612], [0.563, 0.617], [0.611, 0.605], [0.651, 0.583], [0.68, 0.549], [0.691, 0.505]], 'lower_outline': [[0.813, 0.764], [0.797, 0.721], [0.763, 0.685], [0.72, 0.657], [0.674, 0.642], [0.633, 0.64], [0.586, 0.654], [0.545, 0.679], [0.509, 0.709], [0.482, 0.744], [0.47, 0.777], [0.47, 0.807], [0.485, 0.835], [0.515, 0.859], [0.56, 0.876], [0.614, 0.886], [0.668, 0.886], [0.72, 0.876], [0.767, 0.855], [0.8, 0.828], [0.813, 0.797], [0.813, 0.764]]}, 'label_map': {'title': 'panel_09_title', 'cropped_label': 'panel_09_cropped_label'}, 'crop_pixel_bbox': [738, 23, 1024, 349], 'crop_sha256': '78f656db9b438e72560012b09a114c641d0439bed61f77b97095e1bdd32288fe', 'api_request_sha256': '733860bac59f3ee440a33721374e18664118381df449d4da5f2f2f265d4beeb4'}, {'bbox': [0.439, 0.605, 0.713, 0.989], 'data': {'canvas': [1008, 925], 'groups': ['control', 'treated'], 'series': ['glom', 'non_glom'], 'x': [0, 1], 'values': [[146, 174], [151, 160]], 'errors': [[1, 6], [6, 7]], 'colors': ['#5B7DB8', '#D88B60'], 'offsets': [-0.2, 0.2], 'bar_width': 0.4, 'x_limits': [-0.49, 1.49], 'y_limits': [0, 252], 'y_ticks': [0, 50, 100, 150, 200, 250], 'brackets': [{'x': [-0.2, -0.2, 0.2, 0.2], 'y': [169, 207, 207, 169], 'label': 'not_significant', 'text_x': 0, 'text_y': 216}, {'x': [0.8, 0.8, 1.2, 1.2], 'y': [191, 229, 229, 191], 'label': 'significant', 'text_x': 1, 'text_y': 238}]}, 'label_map': {'y_axis': 'panel_10_y_axis', 'control': 'panel_10_control', 'treated': 'panel_10_treated', 'not_significant': 'panel_10_not_significant', 'significant': 'panel_10_significant', 'glom': 'panel_10_glom', 'non_glom': 'panel_10_non_glom'}, 'crop_pixel_bbox': [450, 405, 730, 662], 'crop_sha256': '694497f731900d344e32b38760ca5586e597ffe2c778c3f761d6422d573225f3', 'api_request_sha256': 'b7f0ecac2bd1a9e217ed01ca98ec525ba4be5707519def9966832f1a1e76d610'}, {'bbox': [0.737, 0.605, 0.97, 0.989], 'data': {'groups': [{'label': 'control', 'color': '#202321', 'x': [0.91, 0.95, 0.97, 1.02, 1.06, 0.94, 1.04, 0.99, 1.07, 0.9, 0.95, 1.02, 1.05, 0.98, 1.01, 0.96, 1.08, 0.92, 1.03, 1.0, 1.07, 0.94, 0.99, 1.04, 0.97, 1.02, 0.9, 1.08, 0.95, 1.04], 'y': [1.047, 1.039, 1.024, 1.02, 1.028, 1.014, 1.01, 1.006, 1.002, 1.003, 0.998, 0.996, 0.998, 1.006, 1.011, 0.992, 0.987, 0.982, 0.983, 0.975, 0.963, 0.966, 0.977, 0.994, 1.017, 0.995, 0.938, 0.941, 0.972, 0.848]}, {'label': 'treated', 'color': '#ef6da2', 'x': [1.94, 2.0, 2.08, 1.96, 2.04, 1.92, 2.03, 1.98, 2.07, 1.94, 2.02, 1.99, 1.9, 2.05, 1.96, 2.07, 1.93, 2.01, 1.97, 2.06, 1.92, 2.0, 2.04, 1.96, 2.07, 1.91, 2.01, 1.95, 2.05, 1.98, 2.08, 2.02, 2.1, 2.06], 'y': [1.109, 1.087, 1.09, 1.078, 1.068, 1.061, 1.06, 1.053, 1.043, 1.046, 1.037, 1.03, 1.026, 1.025, 1.024, 1.017, 1.015, 1.013, 1.004, 1.006, 1.0, 0.999, 0.995, 0.991, 0.989, 0.981, 0.978, 0.982, 0.982, 0.962, 0.947, 0.946, 0.951, 0.896]}], 'xlim': [0.5, 2.5], 'ylim': [0.8, 1.2], 'yticks': [0.8, 0.9, 1.0, 1.1, 1.2], 'bracket_x': [1, 1, 2, 2], 'bracket_y': [1.125, 1.183, 1.183, 1.125]}, 'label_map': {'ylabel': 'panel_11_ylabel', 'control': 'panel_11_control', 'treated': 'panel_11_treated', 'significance': 'panel_11_significance'}, 'crop_pixel_bbox': [755, 405, 993, 662], 'crop_sha256': '8b606237f51512d6a62e2f5a2e020b82233bbb3e3e988ab02179602d3cf056a1', 'api_request_sha256': 'fcca5d0656b4ebe15b3faccb19d6709b967fbcf6f4f5c632f25a3b3da35bfc61'}], 'global_placements': [{'key': 'section_A', 'x': 0.011, 'y': 0.024, 'size': 26, 'max_width': 0.025}, {'key': 'section_B', 'x': 0.451, 'y': 0.025, 'size': 26, 'max_width': 0.025}, {'key': 'section_C', 'x': 0.451, 'y': 0.569, 'size': 26, 'max_width': 0.025}, {'key': 'patch_transition', 'x': 0.222, 'y': 0.499, 'size': 13, 'max_width': 0.095}, {'key': 'average_transition', 'x': 0.223, 'y': 0.75, 'size': 12, 'max_width': 0.09}, {'key': 'fluorescence_magenta', 'x': 0.721, 'y': 0.024, 'size': 13, 'max_width': 0.05}, {'key': 'fluorescence_blue', 'x': 0.721, 'y': 0.051, 'size': 13, 'max_width': 0.05}, {'key': 'bar_legend_blue', 'x': 0.552, 'y': 0.576, 'size': 13, 'max_width': 0.045}, {'key': 'bar_legend_orange', 'x': 0.65, 'y': 0.576, 'size': 13, 'max_width': 0.075}]}
LABELS = {'title': '', 'section_A': 'A', 'section_B': 'B', 'section_C': 'C', 'patch_transition': 'ہر قطعے کے لیے مخصوص\nMPH منظرنامہ', 'average_transition': 'تمام قطعوں کی\nاوسط', 'fluorescence_magenta': 'CD45', 'fluorescence_blue': 'مرکزے', 'bar_legend_blue': 'گلومیرولس', 'bar_legend_orange': 'غیر گلومیرولس', 'panel_00_panel': 'A', 'panel_00_condition': 'کنٹرول', 'panel_00_subpanel': '(i)', 'panel_00_x_axis': 'x', 'panel_00_y_axis': 'y', 'panel_01_treated': 'علاج شدہ', 'panel_01_x': 'x', 'panel_01_y': 'y', 'panel_01_gene_count': 'جینوں کی تعداد\n(ہموار کردہ)', 'panel_02_panel': '(ii)', 'panel_02_scale': '20μm', 'panel_03_scale': '20μm', 'panel_04_panel': '(iii)', 'panel_04_radius': 'نصف قطر', 'panel_04_codensity': 'ہم کثافت', 'panel_05_radius': 'نصف قطر', 'panel_05_codensity': 'ہم کثافت', 'panel_06_panel': '(iv)', 'panel_06_radius': 'نصف قطر', 'panel_06_codensity': 'ہم کثافت', 'panel_07_radius': 'نصف قطر', 'panel_07_codensity': 'ہم کثافت', 'panel_08_panel': 'B', 'panel_08_title': 'کنٹرول', 'panel_08_cropped_channel': 'مرکز', 'panel_09_title': 'علاج شدہ', 'panel_09_cropped_label': 'ے', 'panel_10_y_axis': 'اوسط شدت', 'panel_10_control': 'کنٹرول', 'panel_10_treated': 'علاج شدہ', 'panel_10_not_significant': 'شماریاتی طور پر غیر معنی خیز', 'panel_10_significant': '**', 'panel_10_glom': 'گلومیرولس', 'panel_10_non_glom': 'غیر گلومیرولس', 'panel_11_ylabel': 'مجموعی/مرکزی شدت کے تناسب\nکی اوسط', 'panel_11_control': 'کنٹرول', 'panel_11_treated': 'علاج شدہ', 'panel_11_significance': '***'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
