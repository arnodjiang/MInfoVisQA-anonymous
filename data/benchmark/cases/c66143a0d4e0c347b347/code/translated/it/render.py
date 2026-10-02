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
BASE_ID = 'qa_819b5c102555a31ab2ef156a4e1539bc1a2dca8e55b69cb45da8db2286f6aa2a'
LANGUAGE = 'it'
DATA = {'rows': [[{'text': 'Table 1: Historical Timeline of Data Storage', 'label_key': 'table_title', 'colspan': 4, 'rowspan': 1}], [{'text': 'Year', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Technology', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Capacity (Typical)', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Notes', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_000_003'}], [{'text': '1956', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'IBM 350 RAMAC', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_001'}, {'text': '5 MB', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_002'}, {'text': 'First commercial hard disk drive', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_001_003'}], [{'text': '1973', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Winchester HDD', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_001'}, {'text': '30 MB', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_002'}, {'text': 'Significant reduction in size and cost', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_002_003'}], [{'text': '1980', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Seagate ST-506', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_001'}, {'text': '5 MB', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_002'}, {'text': 'Early personal computer HDD', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_003_003'}], [{'text': '1982', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Compact Disc (CD)', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_001'}, {'text': '650 MB', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_002'}, {'text': 'Revolutionized audio storage', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_004_003'}], [{'text': '1991', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'ZIP Drive', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_001'}, {'text': '100 MB', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_002'}, {'text': 'Popular removable storage', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_005_003'}], [{'text': '1995', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'DVD', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_001'}, {'text': '4.7 GB', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_002'}, {'text': 'Enabled storage of video content', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_006_003'}], [{'text': '1999', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'SD Card', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_001'}, {'text': '8 MB - 2 GB', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_002'}, {'text': 'Used in digital cameras and portable devices', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_007_003'}], [{'text': '2000', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'USB Flash Drive', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_001'}, {'text': '8 MB - 1 TB+', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_002'}, {'text': 'Highly portable and versatile', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_008_003'}], [{'text': '2007', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Solid State Drive (SSD)', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_001'}, {'text': '64 GB - 8 TB+', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_002'}, {'text': 'Faster and more durable than HDDs', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_009_003'}], [{'text': '2010', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'Cloud Storage', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_001'}, {'text': 'Variable', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_002'}, {'text': 'Accessible over the internet', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_010_003'}], [{'text': '2020', 'rowspan': 1, 'colspan': 1, 'label_key': None}, {'text': 'NVMe SSD', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_001'}, {'text': '256 GB - 8 TB+', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_002'}, {'text': 'Extremely fast storage for high-performance computing', 'rowspan': 1, 'colspan': 1, 'label_key': 'cell_011_003'}]], 'layout': {'width': 1500, 'padding': 45, 'cell_width': 352.5, 'font_size': 27, 'heights': [67, 106, 145, 145, 184, 145, 145, 145, 184, 184, 145, 106, 184]}}
LABELS = {'cell_000_000': 'Anno', 'cell_000_001': 'Tecnologia', 'cell_000_002': 'Capacità (tipica)', 'cell_000_003': 'Note', 'cell_001_001': 'IBM 350 RAMAC', 'cell_001_002': '5 MB', 'cell_001_003': 'Primo disco rigido commerciale', 'cell_002_001': 'Disco rigido Winchester', 'cell_002_002': '30 MB', 'cell_002_003': 'Riduzione significativa delle dimensioni e dei costi', 'cell_003_001': 'Seagate ST-506', 'cell_003_002': '5 MB', 'cell_003_003': 'Uno dei primi dischi rigidi per personal computer', 'cell_004_001': 'Disco compatto (CD)', 'cell_004_002': '650 MB', 'cell_004_003': "Ha rivoluzionato l'archiviazione audio", 'cell_005_001': 'Unità ZIP', 'cell_005_002': '100 MB', 'cell_005_003': 'Supporto di archiviazione rimovibile molto diffuso', 'cell_006_001': 'DVD', 'cell_006_002': '4.7 GB', 'cell_006_003': "Ha consentito l'archiviazione di contenuti video", 'cell_007_001': 'Scheda SD', 'cell_007_002': '8 MB - 2 GB', 'cell_007_003': 'Usata nelle fotocamere digitali e nei dispositivi portatili', 'cell_008_001': 'Chiavetta USB', 'cell_008_002': '8 MB - 1 TB+', 'cell_008_003': 'Estremamente portatile e versatile', 'cell_009_001': 'Unità a stato solido (SSD)', 'cell_009_002': '64 GB - 8 TB+', 'cell_009_003': 'Più veloce e resistente dei dischi rigidi', 'cell_010_001': 'Archiviazione nel cloud', 'cell_010_002': 'Variabile', 'cell_010_003': 'Accessibile tramite Internet', 'cell_011_001': 'SSD NVMe', 'cell_011_002': '256 GB - 8 TB+', 'cell_011_003': 'Archiviazione estremamente veloce per il calcolo ad alte prestazioni', 'table_title': "Tabella 1: Cronologia storica dell'archiviazione dei dati"}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
