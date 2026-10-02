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
LANGUAGE = 'hi'
DATA = {'rows': [[{'text': 'Series #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Season #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Title', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Notes', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_003'}, {'text': 'Original air date', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_004'}], [{'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Charity"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_002'}, {'text': "Alfie, Dee Dee, and Melanie are supposed to be helping their parents at a carnival by working the dunking booth. When Goo arrives and announces their favorite basketball player, Kendall Gill, is at the Comic Book Store signing autographs, the boys decide to ditch the carnival. This leaves Melanie and Jennifer to work the booth and both end up soaked. But the Comic Book Store is packed and much to Alfie and Dee Dee's surprise their father has to interview Kendall Gill. Goo comes up with a plan to get Alfie and Dee Dee, Gill's signature before getting them back at the local carnival, but are caught by Roger. All ends well for everyone except Alfie and Goo, who must endure being soaked at the dunking booth.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_003'}, {'text': 'October 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_004'}], [{'text': '2', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Practical Joke War"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_002'}, {'text': "Alfie and Goo unleash harsh practical jokes on Dee Dee and his friends. Dee Dee, Harry and Donnel retaliate by pulling a practical joke on Alfie with the trick gum. After Alfie and Goo get even with Dee Dee and his friends, Melanie and Deonne help them get even. Soon, Alfie and Goo declare a practical joke war on Melanie, Dee Dee and their friends. This eventually stops when Roger and Jennifer end up on the wrong end of the practical joke war after being announced as the winner of a magazine contest for Best Family Of The Year. They set their children straight for their behavior and will have a talk with their friends' parents as well.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_003'}, {'text': 'October 22, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_004'}], [{'text': '3', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Weekend Aunt Helen Came"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_002'}, {'text': "The boy's mother, Jennifer, leaves for the weekend and she leaves the father, Roger, in charge. However, he lets the kids run wild. Alfie and Dee Dee's Aunt Helen then comes to oversee the house until Jennifer gets back. Meanwhile, Alfie throws a basketball at Goo, which hits him in the head, giving him temporary amnesia. In this case of memory loss, Goo acts like a nerd, does homework on a weekend, wants to be called Milton instead of Goo, and he even calls Alfie Alfred. He is much nicer to Deonne and Dee Dee, but is somewhat rude to Melanie. The only thing that will reverse this is another hit in the head.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_003'}, {'text': 'November 1, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_004'}], [{'text': '4', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Robin Hood Play"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_002'}, {'text': "Alfie's school is performing the play Robin Hood and Alfie is chosen to play the part of Robin Hood. Alfie is excited at this prospect, but he does not want to wear tights because he feels that tights are for girls. However, he reconsiders his stance on tights when Dee Dee wisely tells him not to let that affect his performance as Robin Hood.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_003'}, {'text': 'November 9, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_004'}], [{'text': '5', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Basketball Tryouts"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_002'}, {'text': "Alfie tries out for the basketball team and doesn't make it even after showing off his basketball skills. However, Harry, Dee Dee and Donnell make the team. Alfie is depressed and doesn't want to attend the celebration party. However, Goo sets him straight by telling him it was his own fault for not being a team player and kept the ball to himself.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_003'}, {'text': 'November 30, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_004'}], [{'text': '6', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Where\'s the Snake?"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_002'}, {'text': "Dee Dee gets a snake, but he doesn't want his parents to know about it. However, things get complicated when he loses the snake in the house. Meanwhile, Melanie and Deonne are assigned by their teacher to take care of her beloved pet rabbit, Duchess for the weekend. This causes both Alfie and Dee Dee to be concerned for Duchess when they learn from Goo that snakes eat rabbits.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_003'}, {'text': 'December 6, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_004'}], [{'text': '7', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Girlfriend"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_002'}, {'text': 'A girl kisses Dee Dee in front of Harry and Donnell. They promise not to tell, but it slips and everyone laughs at Dee Dee. Dee Dee ends his friendship with Harry and Donnell and hangs out with Alfie and Goo. Soon, Alfie and Goo finally get the three to talk to each other.', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_003'}, {'text': 'December 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_004'}], [{'text': '8', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Haircut"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_002'}, {'text': "Dee Dee wants to get a hair cut by Cool Doctor Money and have his name shaved in his head. His parents will not let him do this, but Goo offers to do it for five dollars. However, when Goo messes up Dee Dee's hair and spells his name wrong, his parents find out the truth and Dee Dee is forced to have his hair shaved off. In addition to that, his friends tease him about his bald head, causing a fight between the boys along with Goo and Alfie. In a b-story, Alfie and Goo try to play a practical joke on Dee Dee involving a jalapeño lollipop. It backfires when Roger is the unwitting victim and it leads to him chasing the boys around.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_003'}, {'text': 'December 20, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_004'}], [{'text': '9', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee Runs Away"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_002'}, {'text': "Dee Dee has been waiting to go to a monster truck show all week. But Alfie and Goo's baseball team makes it to the tournament and everyone forgets about the monster truck show. Dee Dee feels ignored and runs away from home with Harry and Donnell. It's up to Alfie and Goo to try and convince him to come home.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_003'}, {'text': 'December 28, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_004'}], [{'text': '10', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '\'"Donnell\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_002'}, {'text': "Donnell is having a birthday party and brags about all the dancing and cool people who will be there. Harry says that he knows how to dance so Dee Dee feels left out because he doesn't know how to dance. Later on, Harry admits to Dee Dee alone that he can't dance either and only lied so he doesn't get teased by Donnell. So, they ask Alfie to help them learn how to dance. He refuses to help because Dee Dee previously told on him to Roger about his and Goo's plans to cheat on their math quiz. Alfie eventually agrees, after Melanie threatens to refuse to help him with his math homework. Soon Dee Dee and Harry learn Donnell's secret and were forced to teach him how to dance. After the party, Dee Dee tells Alfie about it and finds out that he knew Donnell was a liar.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_003'}, {'text': 'January 5, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_004'}], [{'text': '11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Alfie\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_002'}, {'text': "Goo and Melanie pretend they are dating and they leave Alfie out of everything. He ends up bored and starts hanging out with Dee Dee and his friends. However, it just isn't the same without Goo. Later on, Alfie learns about the surprise birthday party that Goo and Melanie had been planning with everyone else (except for Dee Dee, who couldn't know since he would've told).", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_003'}, {'text': 'January 19, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_004'}], [{'text': '12', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Candy Sale"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_002'}, {'text': "Alfie and Goo are selling candy to make money for some expensive jackets, but they are not having any luck. However, when Dee Dee start helping them sell candy, they start to make money and asks him to help them out. Soon Goo and Alfie finds themselves confronted by Melanie, Deonne, Harry and Donnell for Dee Dee's share of the money. They soon learn the boys have used the money to buy three expensive jackets for themselves and Dee Dee as a token of their gratitude. They quickly apologize to Alfie and Goo for their quick judgment.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_003'}, {'text': 'January 26, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_004'}], [{'text': '13', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Big Bully"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_002'}, {'text': "Dee Dee gets beat up at school and his friends try to teach him how to fight back. Goo, however, tells him to bluff, but the plan backfires and Dee Dee gets hit because of it. When Alfie confronts the bully, he learns that Dee Dee was picked on by a girl. Alfie and Goo decide to confront her. However, when some of their classmates, who happen to be the girls' siblings, learn they are bullying their sister, they intervene.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_003'}, {'text': 'February 2, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_004'}]], 'layout': {'width': 1500, 'padding': 45, 'cell_width': 282.0, 'font_size': 27, 'heights': [106, 2875, 2212, 2563, 1393, 1198, 1471, 1159, 2602, 1198, 2953, 1354, 2017, 1549]}}
LABELS = {'cell_000_000': 'श्रृंखला क्रमांक', 'cell_000_001': 'सीज़न क्रमांक', 'cell_000_002': 'शीर्षक', 'cell_000_003': 'टिप्पणियाँ', 'cell_000_004': 'मूल प्रसारण तिथि', 'cell_001_002': '"परोपकार"', 'cell_001_003': 'एल्फ़ी, डी डी और मेलनी को मेले में लोगों को पानी में गिराने वाले बूथ पर काम करके अपने माता-पिता की मदद करनी है। जब गू आकर बताता है कि उनका पसंदीदा बास्केटबॉल खिलाड़ी केंडल गिल कॉमिक बुक की दुकान में ऑटोग्राफ़ दे रहा है, तो लड़के मेला छोड़कर जाने का फ़ैसला करते हैं। इससे मेलनी और जेनिफ़र को बूथ संभालना पड़ता है और दोनों भीग जाती हैं। लेकिन कॉमिक बुक की दुकान खचाखच भरी है और एल्फ़ी तथा डी डी यह जानकर बहुत हैरान होते हैं कि उनके पिता को केंडल गिल का साक्षात्कार लेना है। गू एक योजना बनाता है जिससे एल्फ़ी और डी डी को गिल का ऑटोग्राफ़ दिलाकर स्थानीय मेले में वापस पहुँचाया जा सके, लेकिन रॉजर उन्हें पकड़ लेता है। अंत में सबके लिए सब ठीक हो जाता है, सिवाय एल्फ़ी और गू के, जिन्हें पानी में गिराने वाले बूथ पर भीगना पड़ता है।', 'cell_001_004': 'अक्टूबर 15, 1994', 'cell_002_002': '"शरारतों की जंग"', 'cell_002_003': 'एल्फ़ी और गू, डी डी और उसके दोस्तों के साथ तीखी शरारतें करते हैं। डी डी, हैरी और डोनेल जवाब में शरारती च्यूइंग गम से एल्फ़ी को छकाते हैं। एल्फ़ी और गू जब डी डी और उसके दोस्तों से बदला लेते हैं, तो मेलनी और डिओन बदला लेने में उनकी मदद करती हैं। जल्द ही एल्फ़ी और गू, मेलनी, डी डी और उनके दोस्तों के विरुद्ध शरारतों की जंग छेड़ देते हैं। यह जंग तब जाकर रुकती है जब एक पत्रिका की वर्ष के सर्वश्रेष्ठ परिवार की प्रतियोगिता के विजेता घोषित होने के बाद रॉजर और जेनिफ़र भी इन शरारतों का शिकार हो जाते हैं। वे अपने बच्चों को उनके व्यवहार के लिए फटकारते हैं और उनके दोस्तों के माता-पिता से भी बात करने का फ़ैसला करते हैं।', 'cell_002_004': 'अक्टूबर 22, 1994', 'cell_003_002': '"वह सप्ताहांत जब आंटी हेलन आईं"', 'cell_003_003': 'लड़कों की माँ जेनिफ़र सप्ताहांत के लिए बाहर जाती है और पिता रॉजर को ज़िम्मेदारी सौंप जाती है। लेकिन वह बच्चों को मनमानी करने देता है। फिर एल्फ़ी और डी डी की आंटी हेलन, जेनिफ़र के लौटने तक घर की देखरेख करने आती हैं। इस बीच, एल्फ़ी गू की ओर बास्केटबॉल फेंकता है, जो उसके सिर पर लगती है और उसकी याददाश्त कुछ समय के लिए चली जाती है। याददाश्त खोने पर गू किताबी कीड़े जैसा व्यवहार करने लगता है, सप्ताहांत में होमवर्क करता है, चाहता है कि उसे गू के बजाय मिल्टन कहा जाए और एल्फ़ी को भी अल्फ़्रेड कहकर बुलाता है। वह डिओन और डी डी से कहीं ज़्यादा अच्छे ढंग से पेश आता है, लेकिन मेलनी से कुछ रूखा व्यवहार करता है। सिर पर एक और चोट ही उसे पहले जैसा कर सकती है।', 'cell_003_004': 'नवंबर 1, 1994', 'cell_004_002': '"रॉबिन हुड का नाटक"', 'cell_004_003': 'एल्फ़ी के स्कूल में रॉबिन हुड नाटक का मंचन हो रहा है और एल्फ़ी को रॉबिन हुड की भूमिका के लिए चुना जाता है। एल्फ़ी इस अवसर से उत्साहित है, लेकिन वह टाइट्स नहीं पहनना चाहता क्योंकि उसे लगता है कि टाइट्स लड़कियों के लिए होती हैं। हालाँकि, जब डी डी समझदारी से उसे कहता है कि इस बात का असर रॉबिन हुड के रूप में उसके अभिनय पर नहीं पड़ना चाहिए, तो वह टाइट्स के बारे में अपने रुख पर फिर से विचार करता है।', 'cell_004_004': 'नवंबर 9, 1994', 'cell_005_002': '"बास्केटबॉल टीम की चयन परीक्षा"', 'cell_005_003': 'एल्फ़ी बास्केटबॉल टीम की चयन परीक्षा में भाग लेता है, लेकिन अपना बास्केटबॉल कौशल दिखाने के बावजूद चुना नहीं जाता। हालाँकि, हैरी, डी डी और डोनेल टीम में चुने जाते हैं। एल्फ़ी उदास है और जश्न की पार्टी में नहीं जाना चाहता। लेकिन गू उसे समझाता है कि गलती उसी की थी, क्योंकि वह टीम के साथ मिलकर खेलने के बजाय गेंद अपने ही पास रखता रहा।', 'cell_005_004': 'नवंबर 30, 1994', 'cell_006_002': '"साँप कहाँ है?"', 'cell_006_003': 'डी डी एक साँप लाता है, लेकिन वह नहीं चाहता कि उसके माता-पिता को इसकी खबर हो। हालाँकि, जब साँप घर में कहीं गुम हो जाता है, तो मामला उलझ जाता है। इस बीच, मेलनी और डिओन की शिक्षिका उन्हें सप्ताहांत में अपनी प्यारी पालतू खरगोश डचेस की देखभाल का ज़िम्मा सौंपती हैं। जब एल्फ़ी और डी डी को गू से पता चलता है कि साँप खरगोश खाते हैं, तो दोनों डचेस के लिए चिंतित हो जाते हैं।', 'cell_006_004': 'दिसंबर 6, 1994', 'cell_007_002': '"डी डी की प्रेमिका"', 'cell_007_003': 'एक लड़की हैरी और डोनेल के सामने डी डी को चूमती है। वे किसी को न बताने का वादा करते हैं, लेकिन उनके मुँह से बात निकल जाती है और सभी डी डी पर हँसते हैं। डी डी, हैरी और डोनेल से दोस्ती तोड़ देता है और एल्फ़ी तथा गू के साथ समय बिताने लगता है। जल्द ही एल्फ़ी और गू आखिरकार तीनों को एक-दूसरे से बात करने के लिए राज़ी कर लेते हैं।', 'cell_007_004': 'दिसंबर 15, 1994', 'cell_008_002': '"डी डी की बाल कटाई"', 'cell_008_003': 'डी डी कूल डॉक्टर मनी से बाल कटवाना चाहता है और सिर के बालों में अपना नाम मुँडवाना चाहता है। उसके माता-पिता इसकी अनुमति नहीं देते, लेकिन गू पाँच डॉलर में ऐसा करने की पेशकश करता है। हालाँकि, जब गू डी डी के बाल बिगाड़ देता है और उसके नाम की वर्तनी भी गलत लिख देता है, तो उसके माता-पिता को सच पता चल जाता है और डी डी को सिर मुँडवाना पड़ता है। इसके अलावा, उसके दोस्त उसके गंजे सिर का मज़ाक उड़ाते हैं, जिससे लड़कों के बीच झगड़ा हो जाता है, जिसमें गू और एल्फ़ी भी शामिल होते हैं। एक उपकथा में, एल्फ़ी और गू हलापेन्यो मिर्च वाली लॉलीपॉप से डी डी के साथ शरारत करने की कोशिश करते हैं। यह शरारत उलटी पड़ जाती है जब रॉजर अनजाने में उसका शिकार हो जाता है और लड़कों के पीछे दौड़ने लगता है।', 'cell_008_004': 'दिसंबर 20, 1994', 'cell_009_002': '"डी डी घर से भाग जाता है"', 'cell_009_003': 'डी डी पूरे हफ़्ते से मॉन्स्टर ट्रक शो में जाने का इंतज़ार कर रहा है। लेकिन एल्फ़ी और गू की बेसबॉल टीम टूर्नामेंट में पहुँच जाती है और सभी मॉन्स्टर ट्रक शो के बारे में भूल जाते हैं। डी डी को लगता है कि उसकी अनदेखी हो रही है और वह हैरी तथा डोनेल के साथ घर से भाग जाता है। अब एल्फ़ी और गू को उसे घर लौटने के लिए मनाने की कोशिश करनी है।', 'cell_009_004': 'दिसंबर 28, 1994', 'cell_010_002': '\'"डोनेल की जन्मदिन की पार्टी"', 'cell_010_003': 'डोनेल जन्मदिन की पार्टी दे रहा है और वहाँ होने वाले नाच-गाने और आने वाले मज़ेदार लोगों के बारे में शेखी बघारता है। हैरी कहता है कि उसे नाचना आता है, इसलिए डी डी खुद को अलग-थलग महसूस करता है क्योंकि उसे नाचना नहीं आता। बाद में हैरी अकेले में डी डी से मान लेता है कि उसे भी नाचना नहीं आता और उसने सिर्फ़ इसलिए झूठ कहा था ताकि डोनेल उसे चिढ़ाए नहीं। इसलिए दोनों एल्फ़ी से नाचना सीखने में मदद माँगते हैं। वह मदद करने से इनकार कर देता है क्योंकि डी डी ने पहले रॉजर को गणित की प्रश्नोत्तरी में नकल करने की उसकी और गू की योजना बता दी थी। आखिरकार एल्फ़ी मान जाता है, जब मेलनी धमकी देती है कि वह गणित के होमवर्क में उसकी मदद नहीं करेगी। जल्द ही डी डी और हैरी को डोनेल का राज़ पता चलता है और उन्हें उसे नाचना सिखाना पड़ता है। पार्टी के बाद डी डी एल्फ़ी को इस बारे में बताता है और उसे पता चलता है कि एल्फ़ी पहले से जानता था कि डोनेल झूठ बोल रहा है।', 'cell_010_004': 'जनवरी 5, 1995', 'cell_011_002': '"एल्फ़ी की जन्मदिन की पार्टी"', 'cell_011_003': 'गू और मेलनी प्रेम-संबंध में होने का नाटक करते हैं और एल्फ़ी को हर चीज़ से दूर रखते हैं। वह ऊब जाता है और डी डी तथा उसके दोस्तों के साथ समय बिताने लगता है। लेकिन गू के बिना पहले जैसा मज़ा नहीं आता। बाद में एल्फ़ी को जन्मदिन की उस सरप्राइज़ पार्टी के बारे में पता चलता है जिसकी योजना गू और मेलनी बाकी सभी के साथ मिलकर बना रहे थे। इसमें डी डी शामिल नहीं था, क्योंकि उसे पता होता तो वह बता देता।', 'cell_011_004': 'जनवरी 19, 1995', 'cell_012_002': '"टॉफ़ियों की बिक्री"', 'cell_012_003': 'एल्फ़ी और गू कुछ महँगी जैकेटों के लिए पैसे जुटाने को टॉफ़ियाँ बेच रहे हैं, लेकिन उन्हें कोई सफलता नहीं मिल रही। हालाँकि, जब डी डी टॉफ़ियाँ बेचने में उनकी मदद करने लगता है, तो उनकी कमाई होने लगती है और वे उससे मदद जारी रखने को कहते हैं। जल्द ही मेलनी, डिओन, हैरी और डोनेल, डी डी के हिस्से के पैसों को लेकर गू और एल्फ़ी से जवाब माँगते हैं। उन्हें जल्द ही पता चलता है कि लड़कों ने उन पैसों से अपने लिए और आभार जताने के लिए डी डी के लिए कुल तीन महँगी जैकेटें खरीदी हैं। वे जल्दबाज़ी में राय बनाने के लिए तुरंत एल्फ़ी और गू से माफ़ी माँगते हैं।', 'cell_012_004': 'जनवरी 26, 1995', 'cell_013_002': '"बड़ा धौंसबाज़"', 'cell_013_003': 'स्कूल में डी डी की पिटाई होती है और उसके दोस्त उसे पलटकर लड़ना सिखाने की कोशिश करते हैं। लेकिन गू उसे झूठी धौंस दिखाने की सलाह देता है। यह योजना उलटी पड़ जाती है और इसके कारण डी डी को मार पड़ती है। जब एल्फ़ी धौंस जमाने वाले का सामना करता है, तो उसे पता चलता है कि डी डी को एक लड़की ने सताया था। एल्फ़ी और गू उससे भिड़ने का फ़ैसला करते हैं। हालाँकि, जब उनके कुछ सहपाठियों को, जो उस लड़की के भाई-बहन हैं, पता चलता है कि वे उनकी बहन पर धौंस जमा रहे हैं, तो वे बीच-बचाव करते हैं।', 'cell_013_004': 'फ़रवरी 2, 1995'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
