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
BASE_ID = 'qa_8f1cbc7436566fdb226e50112a1972b787ec78fa9956627f7d8e86cc839b44d5'
LANGUAGE = 'fa'
DATA = {'rows': [[{'text': 'Series #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Season #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Title', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Notes', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_003'}, {'text': 'Original air date', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_004'}], [{'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Charity"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_002'}, {'text': "Alfie, Dee Dee, and Melanie are supposed to be helping their parents at a carnival by working the dunking booth. When Goo arrives and announces their favorite basketball player, Kendall Gill, is at the Comic Book Store signing autographs, the boys decide to ditch the carnival. This leaves Melanie and Jennifer to work the booth and both end up soaked. But the Comic Book Store is packed and much to Alfie and Dee Dee's surprise their father has to interview Kendall Gill. Goo comes up with a plan to get Alfie and Dee Dee, Gill's signature before getting them back at the local carnival, but are caught by Roger. All ends well for everyone except Alfie and Goo, who must endure being soaked at the dunking booth.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_003'}, {'text': 'October 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_004'}], [{'text': '2', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Practical Joke War"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_002'}, {'text': "Alfie and Goo unleash harsh practical jokes on Dee Dee and his friends. Dee Dee, Harry and Donnel retaliate by pulling a practical joke on Alfie with the trick gum. After Alfie and Goo get even with Dee Dee and his friends, Melanie and Deonne help them get even. Soon, Alfie and Goo declare a practical joke war on Melanie, Dee Dee and their friends. This eventually stops when Roger and Jennifer end up on the wrong end of the practical joke war after being announced as the winner of a magazine contest for Best Family Of The Year. They set their children straight for their behavior and will have a talk with their friends' parents as well.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_003'}, {'text': 'October 22, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_004'}], [{'text': '3', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Weekend Aunt Helen Came"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_002'}, {'text': "The boy's mother, Jennifer, leaves for the weekend and she leaves the father, Roger, in charge. However, he lets the kids run wild. Alfie and Dee Dee's Aunt Helen then comes to oversee the house until Jennifer gets back. Meanwhile, Alfie throws a basketball at Goo, which hits him in the head, giving him temporary amnesia. In this case of memory loss, Goo acts like a nerd, does homework on a weekend, wants to be called Milton instead of Goo, and he even calls Alfie Alfred. He is much nicer to Deonne and Dee Dee, but is somewhat rude to Melanie. The only thing that will reverse this is another hit in the head.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_003'}, {'text': 'November 1, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_004'}], [{'text': '4', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Robin Hood Play"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_002'}, {'text': "Alfie's school is performing the play Robin Hood and Alfie is chosen to play the part of Robin Hood. Alfie is excited at this prospect, but he does not want to wear tights because he feels that tights are for girls. However, he reconsiders his stance on tights when Dee Dee wisely tells him not to let that affect his performance as Robin Hood.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_003'}, {'text': 'November 9, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_004'}], [{'text': '5', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Basketball Tryouts"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_002'}, {'text': "Alfie tries out for the basketball team and doesn't make it even after showing off his basketball skills. However, Harry, Dee Dee and Donnell make the team. Alfie is depressed and doesn't want to attend the celebration party. However, Goo sets him straight by telling him it was his own fault for not being a team player and kept the ball to himself.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_003'}, {'text': 'November 30, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_004'}], [{'text': '6', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Where\'s the Snake?"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_002'}, {'text': "Dee Dee gets a snake, but he doesn't want his parents to know about it. However, things get complicated when he loses the snake in the house. Meanwhile, Melanie and Deonne are assigned by their teacher to take care of her beloved pet rabbit, Duchess for the weekend. This causes both Alfie and Dee Dee to be concerned for Duchess when they learn from Goo that snakes eat rabbits.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_003'}, {'text': 'December 6, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_004'}], [{'text': '7', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Girlfriend"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_002'}, {'text': 'A girl kisses Dee Dee in front of Harry and Donnell. They promise not to tell, but it slips and everyone laughs at Dee Dee. Dee Dee ends his friendship with Harry and Donnell and hangs out with Alfie and Goo. Soon, Alfie and Goo finally get the three to talk to each other.', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_003'}, {'text': 'December 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_004'}], [{'text': '8', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Haircut"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_002'}, {'text': "Dee Dee wants to get a hair cut by Cool Doctor Money and have his name shaved in his head. His parents will not let him do this, but Goo offers to do it for five dollars. However, when Goo messes up Dee Dee's hair and spells his name wrong, his parents find out the truth and Dee Dee is forced to have his hair shaved off. In addition to that, his friends tease him about his bald head, causing a fight between the boys along with Goo and Alfie. In a b-story, Alfie and Goo try to play a practical joke on Dee Dee involving a jalapeño lollipop. It backfires when Roger is the unwitting victim and it leads to him chasing the boys around.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_003'}, {'text': 'December 20, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_004'}], [{'text': '9', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee Runs Away"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_002'}, {'text': "Dee Dee has been waiting to go to a monster truck show all week. But Alfie and Goo's baseball team makes it to the tournament and everyone forgets about the monster truck show. Dee Dee feels ignored and runs away from home with Harry and Donnell. It's up to Alfie and Goo to try and convince him to come home.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_003'}, {'text': 'December 28, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_004'}], [{'text': '10', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '\'"Donnell\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_002'}, {'text': "Donnell is having a birthday party and brags about all the dancing and cool people who will be there. Harry says that he knows how to dance so Dee Dee feels left out because he doesn't know how to dance. Later on, Harry admits to Dee Dee alone that he can't dance either and only lied so he doesn't get teased by Donnell. So, they ask Alfie to help them learn how to dance. He refuses to help because Dee Dee previously told on him to Roger about his and Goo's plans to cheat on their math quiz. Alfie eventually agrees, after Melanie threatens to refuse to help him with his math homework. Soon Dee Dee and Harry learn Donnell's secret and were forced to teach him how to dance. After the party, Dee Dee tells Alfie about it and finds out that he knew Donnell was a liar.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_003'}, {'text': 'January 5, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_004'}], [{'text': '11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Alfie\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_002'}, {'text': "Goo and Melanie pretend they are dating and they leave Alfie out of everything. He ends up bored and starts hanging out with Dee Dee and his friends. However, it just isn't the same without Goo. Later on, Alfie learns about the surprise birthday party that Goo and Melanie had been planning with everyone else (except for Dee Dee, who couldn't know since he would've told).", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_003'}, {'text': 'January 19, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_004'}], [{'text': '12', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Candy Sale"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_002'}, {'text': "Alfie and Goo are selling candy to make money for some expensive jackets, but they are not having any luck. However, when Dee Dee start helping them sell candy, they start to make money and asks him to help them out. Soon Goo and Alfie finds themselves confronted by Melanie, Deonne, Harry and Donnell for Dee Dee's share of the money. They soon learn the boys have used the money to buy three expensive jackets for themselves and Dee Dee as a token of their gratitude. They quickly apologize to Alfie and Goo for their quick judgment.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_003'}, {'text': 'January 26, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_004'}], [{'text': '13', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Big Bully"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_002'}, {'text': "Dee Dee gets beat up at school and his friends try to teach him how to fight back. Goo, however, tells him to bluff, but the plan backfires and Dee Dee gets hit because of it. When Alfie confronts the bully, he learns that Dee Dee was picked on by a girl. Alfie and Goo decide to confront her. However, when some of their classmates, who happen to be the girls' siblings, learn they are bullying their sister, they intervene.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_003'}, {'text': 'February 2, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_004'}]], 'layout': {'width': 1500, 'padding': 45, 'cell_width': 282.0, 'font_size': 27, 'heights': [106, 2875, 2212, 2563, 1393, 1198, 1471, 1159, 2602, 1198, 2953, 1354, 2017, 1549]}}
LABELS = {'cell_000_000': 'شماره در مجموعه', 'cell_000_001': 'شماره در فصل', 'cell_000_002': 'عنوان', 'cell_000_003': 'یادداشت\u200cها', 'cell_000_004': 'تاریخ پخش اولیه', 'cell_001_002': '«خیریه»', 'cell_001_003': 'قرار است آلفی، دی\u200cدی و ملانی در جشن کارناوال، با ادارهٔ غرفهٔ انداختن در آب به والدینشان کمک کنند. وقتی گو می\u200cرسد و خبر می\u200cدهد که بسکتبالیست محبوبشان، کندال گیل، در فروشگاه کتاب\u200cهای مصور مشغول امضا دادن است، پسرها تصمیم می\u200cگیرند جشن را رها کنند. در نتیجه ملانی و جنیفر می\u200cمانند تا غرفه را اداره کنند و هر دو حسابی خیس می\u200cشوند. اما فروشگاه کتاب\u200cهای مصور شلوغ است و آلفی و دی\u200cدی با تعجب می\u200cفهمند که پدرشان قرار است با کندال گیل مصاحبه کند. گو نقشه\u200cای می\u200cکشد تا پیش از برگرداندن آلفی و دی\u200cدی به جشن محلی، امضای گیل را برایشان بگیرد، اما راجر مچشان را می\u200cگیرد. ماجرا برای همه به\u200cخوبی تمام می\u200cشود، جز آلفی و گو که باید خیس شدن در غرفهٔ انداختن در آب را تحمل کنند.', 'cell_001_004': 'اکتبر 15, 1994', 'cell_002_002': '«جنگ شوخی\u200cهای عملی»', 'cell_002_003': 'آلفی و گو شوخی\u200cهای عملی آزاردهنده\u200cای با دی\u200cدی و دوستانش می\u200cکنند. دی\u200cدی، هری و دانل با آدامس حقه\u200cدار از آلفی انتقام می\u200cگیرند. پس از آنکه آلفی و گو تلافی می\u200cکنند، ملانی و دیون به دی\u200cدی و دوستانش کمک می\u200cکنند تا جوابشان را بدهند. طولی نمی\u200cکشد که آلفی و گو علیه ملانی، دی\u200cدی و دوستانشان جنگ شوخی\u200cهای عملی راه می\u200cاندازند. این ماجرا سرانجام زمانی متوقف می\u200cشود که راجر و جنیفر، پس از اعلام نامشان به\u200cعنوان برندگان مسابقهٔ «بهترین خانوادهٔ سال» یک مجله، قربانی این جنگ شوخی\u200cها می\u200cشوند. آن\u200cها فرزندانشان را بابت رفتارشان سرزنش می\u200cکنند و قصد دارند با والدین دوستانشان هم صحبت کنند.', 'cell_002_004': 'اکتبر 22, 1994', 'cell_003_002': '«آخر هفته\u200cای که عمه هلن آمد»', 'cell_003_003': 'جنیفر، مادر پسرها، آخر هفته به سفر می\u200cرود و مسئولیت خانه را به راجر، پدرشان، می\u200cسپارد. اما او بچه\u200cها را به حال خود می\u200cگذارد تا هر کاری می\u200cخواهند بکنند. سپس عمه هلن، عمهٔ آلفی و دی\u200cدی، می\u200cآید تا زمان بازگشت جنیفر بر خانه نظارت کند. در همین حال، آلفی توپ بسکتبالی به طرف گو پرتاب می\u200cکند که به سرش می\u200cخورد و باعث فراموشی موقت او می\u200cشود. گو در این وضعیت مثل یک بچه\u200cدرس\u200cخوان رفتار می\u200cکند، آخر هفته تکالیفش را انجام می\u200cدهد، می\u200cخواهد به\u200cجای گو او را میلتون صدا کنند و حتی آلفی را آلفرد خطاب می\u200cکند. او با دیون و دی\u200cدی خیلی مهربان\u200cتر شده، اما با ملانی کمی بی\u200cادبانه رفتار می\u200cکند. تنها چیزی که می\u200cتواند او را به حالت قبل برگرداند، ضربهٔ دیگری به سرش است.', 'cell_003_004': 'نوامبر 1, 1994', 'cell_004_002': '«نمایش رابین هود»', 'cell_004_003': 'مدرسهٔ آلفی قرار است نمایش رابین هود را اجرا کند و آلفی برای نقش رابین هود انتخاب می\u200cشود. او از این فرصت هیجان\u200cزده است، اما نمی\u200cخواهد جوراب\u200cشلواری بپوشد، چون فکر می\u200cکند جوراب\u200cشلواری مخصوص دخترهاست. بااین\u200cحال، وقتی دی\u200cدی عاقلانه به او می\u200cگوید نگذارد این موضوع بر اجرای نقش رابین هود تأثیر بگذارد، در نظرش تجدیدنظر می\u200cکند.', 'cell_004_004': 'نوامبر 9, 1994', 'cell_005_002': '«آزمون انتخاب تیم بسکتبال»', 'cell_005_003': 'آلفی در آزمون انتخاب تیم بسکتبال شرکت می\u200cکند، اما با وجود به نمایش گذاشتن مهارت\u200cهایش انتخاب نمی\u200cشود. در مقابل، هری، دی\u200cدی و دانل به تیم راه پیدا می\u200cکنند. آلفی دلگیر است و نمی\u200cخواهد به جشن برود. اما گو به او گوشزد می\u200cکند که تقصیر خودش بوده، چون تیمی بازی نکرده و توپ را فقط پیش خودش نگه داشته است.', 'cell_005_004': 'نوامبر 30, 1994', 'cell_006_002': '«مار کجاست؟»', 'cell_006_003': 'دی\u200cدی یک مار می\u200cگیرد، اما نمی\u200cخواهد والدینش از آن باخبر شوند. وقتی مار را در خانه گم می\u200cکند، اوضاع پیچیده می\u200cشود. در همین حال، معلم ملانی و دیون از آن\u200cها می\u200cخواهد آخر هفته از خرگوش خانگی محبوبش، داچس، مراقبت کنند. وقتی آلفی و دی\u200cدی از گو می\u200cشنوند که مارها خرگوش می\u200cخورند، هر دو نگران داچس می\u200cشوند.', 'cell_006_004': 'دسامبر 6, 1994', 'cell_007_002': '«دوست\u200cدختر دی\u200cدی»', 'cell_007_003': 'دختری جلوی هری و دانل دی\u200cدی را می\u200cبوسد. آن\u200cها قول می\u200cدهند به کسی نگویند، اما موضوع از دهانشان می\u200cپرد و همه به دی\u200cدی می\u200cخندند. دی\u200cدی دوستی\u200cاش را با هری و دانل به هم می\u200cزند و با آلفی و گو وقت می\u200cگذراند. سرانجام آلفی و گو موفق می\u200cشوند آن سه نفر را وادار کنند با هم حرف بزنند.', 'cell_007_004': 'دسامبر 15, 1994', 'cell_008_002': '«مدل موی دی\u200cدی»', 'cell_008_003': 'دی\u200cدی می\u200cخواهد کول دکتر مانی موهایش را کوتاه کند و نامش را با تراشیدن مو روی سرش بنویسد. والدینش اجازه نمی\u200cدهند، اما گو پیشنهاد می\u200cکند این کار را در ازای پنج دلار انجام دهد. وقتی گو موهای دی\u200cدی را خراب می\u200cکند و نامش را هم اشتباه می\u200cنویسد، والدینش حقیقت را می\u200cفهمند و دی\u200cدی مجبور می\u200cشود همهٔ موهایش را بتراشد. علاوه بر این، دوستانش سرِ کچل او را مسخره می\u200cکنند و همین باعث دعوا میان پسرها، گو و آلفی می\u200cشود. در داستان فرعی، آلفی و گو می\u200cخواهند با یک آب\u200cنبات چوبی هالوپینو با دی\u200cدی شوخی کنند. نقشه نتیجهٔ معکوس می\u200cدهد، چون راجر بی\u200cخبر قربانی آن می\u200cشود و دنبال پسرها می\u200cافتد.', 'cell_008_004': 'دسامبر 20, 1994', 'cell_009_002': '«دی\u200cدی فرار می\u200cکند»', 'cell_009_003': 'دی\u200cدی تمام هفته منتظر بوده به نمایش مانستر تراک\u200cها برود. اما تیم بیسبال آلفی و گو به مسابقات راه پیدا می\u200cکند و همه نمایش مانستر تراک\u200cها را فراموش می\u200cکنند. دی\u200cدی احساس می\u200cکند نادیده گرفته شده و همراه هری و دانل از خانه فرار می\u200cکند. حالا آلفی و گو باید تلاش کنند او را به بازگشت به خانه راضی کنند.', 'cell_009_004': 'دسامبر 28, 1994', 'cell_010_002': '«جشن تولد دانل»', 'cell_010_003': 'دانل قرار است جشن تولد بگیرد و دربارهٔ رقص و آدم\u200cهای باحالی که به مهمانی می\u200cآیند پز می\u200cدهد. هری می\u200cگوید رقص بلد است و دی\u200cدی، که بلد نیست برقصد، احساس می\u200cکند از جمع کنار مانده است. بعدتر هری در خلوت به دی\u200cدی اعتراف می\u200cکند که او هم رقص بلد نیست و فقط برای اینکه دانل مسخره\u200cاش نکند دروغ گفته است. پس از آلفی می\u200cخواهند به آن\u200cها رقص یاد بدهد. او قبول نمی\u200cکند، چون دی\u200cدی قبلاً نقشهٔ او و گو برای تقلب در امتحان کوتاه ریاضی را به راجر لو داده است. سرانجام، پس از آنکه ملانی تهدید می\u200cکند دیگر در تکالیف ریاضی کمکش نخواهد کرد، آلفی می\u200cپذیرد. کمی بعد دی\u200cدی و هری به راز دانل پی می\u200cبرند و مجبور می\u200cشوند به او رقص یاد بدهند. بعد از مهمانی، دی\u200cدی ماجرا را برای آلفی تعریف می\u200cکند و می\u200cفهمد که او از قبل می\u200cدانسته دانل دروغ می\u200cگوید.', 'cell_010_004': 'ژانویه 5, 1995', 'cell_011_002': '«جشن تولد آلفی»', 'cell_011_003': 'گو و ملانی وانمود می\u200cکنند با هم رابطهٔ عاشقانه دارند و آلفی را از همهٔ برنامه\u200cها کنار می\u200cگذارند. او حوصله\u200cاش سر می\u200cرود و شروع می\u200cکند به وقت گذراندن با دی\u200cدی و دوستانش. اما بدون گو، هیچ\u200cچیز مثل قبل نیست. بعدتر آلفی از جشن تولد غافلگیرانه\u200cای باخبر می\u200cشود که گو و ملانی همراه بقیه برایش تدارک دیده بودند؛ به\u200cجز دی\u200cدی که نباید چیزی می\u200cدانست، چون حتماً آن را لو می\u200cداد.', 'cell_011_004': 'ژانویه 19, 1995', 'cell_012_002': '«فروش آب\u200cنبات»', 'cell_012_003': 'آلفی و گو برای خرید چند کاپشن گران\u200cقیمت آب\u200cنبات می\u200cفروشند، اما موفقیتی ندارند. وقتی دی\u200cدی در فروش کمکشان می\u200cکند، درآمدشان بیشتر می\u200cشود و از او می\u200cخواهند به کمکش ادامه دهد. کمی بعد، ملانی، دیون، هری و دانل برای گرفتن سهم دی\u200cدی از پول، به آلفی و گو اعتراض می\u200cکنند. اما خیلی زود می\u200cفهمند که آن دو با آن پول سه کاپشن گران\u200cقیمت برای خودشان و دی\u200cدی خریده\u200cاند تا از او قدردانی کنند. آن\u200cها فوراً بابت قضاوت عجولانه\u200cشان از آلفی و گو عذرخواهی می\u200cکنند.', 'cell_012_004': 'ژانویه 26, 1995', 'cell_013_002': '«قلدر بزرگ»', 'cell_013_003': 'دی\u200cدی در مدرسه کتک می\u200cخورد و دوستانش سعی می\u200cکنند به او یاد بدهند چطور از خودش دفاع کند. اما گو به او می\u200cگوید بلوف بزند؛ نقشه نتیجهٔ معکوس می\u200cدهد و دی\u200cدی به\u200cخاطر آن کتک می\u200cخورد. وقتی آلفی با فرد قلدر روبه\u200cرو می\u200cشود، می\u200cفهمد کسی که دی\u200cدی را آزار داده یک دختر بوده است. آلفی و گو تصمیم می\u200cگیرند با او برخورد کنند. اما چند نفر از هم\u200cکلاسی\u200cهایشان که خواهر و برادر آن دختر هستند، وقتی می\u200cفهمند آن دو خواهرشان را اذیت می\u200cکنند، مداخله می\u200cکنند.', 'cell_013_004': 'فوریه 2, 1995'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
