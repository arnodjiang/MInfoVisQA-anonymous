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
LANGUAGE = 'ta'
DATA = {'rows': [[{'text': 'Series #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Season #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Title', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Notes', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_003'}, {'text': 'Original air date', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_004'}], [{'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Charity"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_002'}, {'text': "Alfie, Dee Dee, and Melanie are supposed to be helping their parents at a carnival by working the dunking booth. When Goo arrives and announces their favorite basketball player, Kendall Gill, is at the Comic Book Store signing autographs, the boys decide to ditch the carnival. This leaves Melanie and Jennifer to work the booth and both end up soaked. But the Comic Book Store is packed and much to Alfie and Dee Dee's surprise their father has to interview Kendall Gill. Goo comes up with a plan to get Alfie and Dee Dee, Gill's signature before getting them back at the local carnival, but are caught by Roger. All ends well for everyone except Alfie and Goo, who must endure being soaked at the dunking booth.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_003'}, {'text': 'October 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_004'}], [{'text': '2', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Practical Joke War"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_002'}, {'text': "Alfie and Goo unleash harsh practical jokes on Dee Dee and his friends. Dee Dee, Harry and Donnel retaliate by pulling a practical joke on Alfie with the trick gum. After Alfie and Goo get even with Dee Dee and his friends, Melanie and Deonne help them get even. Soon, Alfie and Goo declare a practical joke war on Melanie, Dee Dee and their friends. This eventually stops when Roger and Jennifer end up on the wrong end of the practical joke war after being announced as the winner of a magazine contest for Best Family Of The Year. They set their children straight for their behavior and will have a talk with their friends' parents as well.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_003'}, {'text': 'October 22, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_004'}], [{'text': '3', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Weekend Aunt Helen Came"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_002'}, {'text': "The boy's mother, Jennifer, leaves for the weekend and she leaves the father, Roger, in charge. However, he lets the kids run wild. Alfie and Dee Dee's Aunt Helen then comes to oversee the house until Jennifer gets back. Meanwhile, Alfie throws a basketball at Goo, which hits him in the head, giving him temporary amnesia. In this case of memory loss, Goo acts like a nerd, does homework on a weekend, wants to be called Milton instead of Goo, and he even calls Alfie Alfred. He is much nicer to Deonne and Dee Dee, but is somewhat rude to Melanie. The only thing that will reverse this is another hit in the head.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_003'}, {'text': 'November 1, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_004'}], [{'text': '4', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Robin Hood Play"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_002'}, {'text': "Alfie's school is performing the play Robin Hood and Alfie is chosen to play the part of Robin Hood. Alfie is excited at this prospect, but he does not want to wear tights because he feels that tights are for girls. However, he reconsiders his stance on tights when Dee Dee wisely tells him not to let that affect his performance as Robin Hood.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_003'}, {'text': 'November 9, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_004'}], [{'text': '5', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Basketball Tryouts"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_002'}, {'text': "Alfie tries out for the basketball team and doesn't make it even after showing off his basketball skills. However, Harry, Dee Dee and Donnell make the team. Alfie is depressed and doesn't want to attend the celebration party. However, Goo sets him straight by telling him it was his own fault for not being a team player and kept the ball to himself.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_003'}, {'text': 'November 30, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_004'}], [{'text': '6', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Where\'s the Snake?"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_002'}, {'text': "Dee Dee gets a snake, but he doesn't want his parents to know about it. However, things get complicated when he loses the snake in the house. Meanwhile, Melanie and Deonne are assigned by their teacher to take care of her beloved pet rabbit, Duchess for the weekend. This causes both Alfie and Dee Dee to be concerned for Duchess when they learn from Goo that snakes eat rabbits.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_003'}, {'text': 'December 6, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_004'}], [{'text': '7', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Girlfriend"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_002'}, {'text': 'A girl kisses Dee Dee in front of Harry and Donnell. They promise not to tell, but it slips and everyone laughs at Dee Dee. Dee Dee ends his friendship with Harry and Donnell and hangs out with Alfie and Goo. Soon, Alfie and Goo finally get the three to talk to each other.', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_003'}, {'text': 'December 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_004'}], [{'text': '8', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Haircut"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_002'}, {'text': "Dee Dee wants to get a hair cut by Cool Doctor Money and have his name shaved in his head. His parents will not let him do this, but Goo offers to do it for five dollars. However, when Goo messes up Dee Dee's hair and spells his name wrong, his parents find out the truth and Dee Dee is forced to have his hair shaved off. In addition to that, his friends tease him about his bald head, causing a fight between the boys along with Goo and Alfie. In a b-story, Alfie and Goo try to play a practical joke on Dee Dee involving a jalapeño lollipop. It backfires when Roger is the unwitting victim and it leads to him chasing the boys around.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_003'}, {'text': 'December 20, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_004'}], [{'text': '9', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee Runs Away"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_002'}, {'text': "Dee Dee has been waiting to go to a monster truck show all week. But Alfie and Goo's baseball team makes it to the tournament and everyone forgets about the monster truck show. Dee Dee feels ignored and runs away from home with Harry and Donnell. It's up to Alfie and Goo to try and convince him to come home.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_003'}, {'text': 'December 28, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_004'}], [{'text': '10', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '\'"Donnell\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_002'}, {'text': "Donnell is having a birthday party and brags about all the dancing and cool people who will be there. Harry says that he knows how to dance so Dee Dee feels left out because he doesn't know how to dance. Later on, Harry admits to Dee Dee alone that he can't dance either and only lied so he doesn't get teased by Donnell. So, they ask Alfie to help them learn how to dance. He refuses to help because Dee Dee previously told on him to Roger about his and Goo's plans to cheat on their math quiz. Alfie eventually agrees, after Melanie threatens to refuse to help him with his math homework. Soon Dee Dee and Harry learn Donnell's secret and were forced to teach him how to dance. After the party, Dee Dee tells Alfie about it and finds out that he knew Donnell was a liar.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_003'}, {'text': 'January 5, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_004'}], [{'text': '11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Alfie\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_002'}, {'text': "Goo and Melanie pretend they are dating and they leave Alfie out of everything. He ends up bored and starts hanging out with Dee Dee and his friends. However, it just isn't the same without Goo. Later on, Alfie learns about the surprise birthday party that Goo and Melanie had been planning with everyone else (except for Dee Dee, who couldn't know since he would've told).", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_003'}, {'text': 'January 19, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_004'}], [{'text': '12', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Candy Sale"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_002'}, {'text': "Alfie and Goo are selling candy to make money for some expensive jackets, but they are not having any luck. However, when Dee Dee start helping them sell candy, they start to make money and asks him to help them out. Soon Goo and Alfie finds themselves confronted by Melanie, Deonne, Harry and Donnell for Dee Dee's share of the money. They soon learn the boys have used the money to buy three expensive jackets for themselves and Dee Dee as a token of their gratitude. They quickly apologize to Alfie and Goo for their quick judgment.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_003'}, {'text': 'January 26, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_004'}], [{'text': '13', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Big Bully"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_002'}, {'text': "Dee Dee gets beat up at school and his friends try to teach him how to fight back. Goo, however, tells him to bluff, but the plan backfires and Dee Dee gets hit because of it. When Alfie confronts the bully, he learns that Dee Dee was picked on by a girl. Alfie and Goo decide to confront her. However, when some of their classmates, who happen to be the girls' siblings, learn they are bullying their sister, they intervene.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_003'}, {'text': 'February 2, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_004'}]], 'layout': {'width': 1500, 'padding': 45, 'cell_width': 282.0, 'font_size': 27, 'heights': [106, 2875, 2212, 2563, 1393, 1198, 1471, 1159, 2602, 1198, 2953, 1354, 2017, 1549]}}
LABELS = {'cell_000_000': 'தொடர் #', 'cell_000_001': 'பருவம் #', 'cell_000_002': 'தலைப்பு', 'cell_000_003': 'குறிப்புகள்', 'cell_000_004': 'முதலில் ஒளிபரப்பான தேதி', 'cell_001_002': '“தொண்டு”', 'cell_001_003': 'Alfie, Dee Dee மற்றும் Melanie ஆகியோர் ஒரு திருவிழாவில் தண்ணீரில் விழவைக்கும் விளையாட்டுச் சாவடியில் வேலை செய்து தங்கள் பெற்றோருக்கு உதவ வேண்டியிருக்கிறது. அவர்களுக்குப் பிடித்த கூடைப்பந்து வீரரான Kendall Gill, Comic Book Store-இல் ரசிகர்களுக்குக் கையொப்பமிட்டுக் கொண்டிருப்பதாக Goo வந்து தெரிவிக்கும்போது, சிறுவர்கள் திருவிழாவிலிருந்து நழுவிச் செல்ல முடிவு செய்கிறார்கள். இதனால் Melanie மற்றும் Jennifer ஆகியோர் சாவடியைக் கவனித்துக்கொள்ள வேண்டியதாகிறது; இருவரும் முற்றிலும் நனைந்து போகிறார்கள். ஆனால் Comic Book Store-இல் கூட்டம் நிரம்பி வழிகிறது. தங்கள் தந்தை Kendall Gill-ஐப் பேட்டி காண வேண்டியிருப்பதை அறிந்து Alfie மற்றும் Dee Dee மிகவும் ஆச்சரியப்படுகிறார்கள். Alfie மற்றும் Dee Dee ஆகியோருக்கு Gill-இன் கையொப்பத்தை வாங்கித் தந்துவிட்டு, அவர்களை உள்ளூர்த் திருவிழாவுக்குத் திரும்ப அழைத்துச் செல்ல Goo ஒரு திட்டம் தீட்டுகிறான். ஆனால் அவர்கள் Roger-இடம் மாட்டிக்கொள்கிறார்கள். தண்ணீரில் விழவைக்கும் விளையாட்டுச் சாவடியில் நனைவதைச் சகித்துக்கொள்ள வேண்டிய Alfie மற்றும் Goo ஆகியோரைத் தவிர மற்ற அனைவருக்கும் எல்லாம் நன்றாகவே முடிகிறது.', 'cell_001_004': 'அக்டோபர் 15, 1994', 'cell_002_002': '“குறும்பு விளையாட்டுப் போர்”', 'cell_002_003': 'Alfie மற்றும் Goo, Dee Dee மற்றும் அவனது நண்பர்களிடம் கடுமையான குறும்புச் சேட்டைகளைச் செய்கிறார்கள். Dee Dee, Harry மற்றும் Donnel, தந்திரமான சூயிங்கத்தைப் பயன்படுத்தி Alfie-யிடம் ஒரு குறும்புச் சேட்டை செய்து பதிலடி கொடுக்கிறார்கள். Alfie மற்றும் Goo, Dee Dee மற்றும் அவனது நண்பர்களைப் பழிவாங்கிய பிறகு, அவர்களுக்குப் பதிலடி கொடுக்க Melanie மற்றும் Deonne உதவுகிறார்கள். விரைவில், Melanie, Dee Dee மற்றும் அவர்களது நண்பர்களுக்கு எதிராக Alfie மற்றும் Goo குறும்புச் சேட்டைப் போரை அறிவிக்கிறார்கள். ஆண்டின் சிறந்த குடும்பத்திற்கான பத்திரிகைப் போட்டியில் வெற்றியாளர்களாக அறிவிக்கப்பட்ட Roger மற்றும் Jennifer, இந்தக் குறும்புச் சேட்டைப் போரில் பாதிக்கப்படும்போது இது இறுதியாக முடிவுக்கு வருகிறது. அவர்கள் தங்கள் பிள்ளைகளின் நடத்தையைக் கண்டித்துத் திருத்துகிறார்கள்; பிள்ளைகளின் நண்பர்களுடைய பெற்றோரிடமும் பேசவிருக்கிறார்கள்.', 'cell_002_004': 'அக்டோபர் 22, 1994', 'cell_003_002': '“அத்தை Helen வந்த வார இறுதி”', 'cell_003_003': 'சிறுவனின் தாயான Jennifer வார இறுதியைக் கழிக்க வெளியே செல்கிறார்; தந்தையான Roger பொறுப்பில் இருக்குமாறு விட்டுச் செல்கிறார். ஆனால், அவர் குழந்தைகளைத் தங்கள் இஷ்டம்போல் கட்டுக்கடங்காமல் நடந்துகொள்ள அனுமதிக்கிறார். பின்னர், Jennifer திரும்பி வரும்வரை வீட்டைக் கவனித்துக்கொள்ள Alfie மற்றும் Dee Dee ஆகியோரின் அத்தையான Helen வருகிறார். இதற்கிடையில், Alfie ஒரு கூடைப்பந்தை Goo மீது வீசுகிறான்; அது அவனது தலையில் பட்டு, அவனுக்குத் தற்காலிக நினைவிழப்பை ஏற்படுத்துகிறது. இந்த நினைவிழப்பால், Goo ஒரு புத்தகப்புழுவைப் போல நடந்துகொள்கிறான், வார இறுதியிலும் வீட்டுப்பாடம் செய்கிறான், தன்னை Goo என்பதற்குப் பதிலாக Milton என்று அழைக்க வேண்டும் என விரும்புகிறான்; Alfie-ஐக்கூட Alfred என்று அழைக்கிறான். அவன் Deonne மற்றும் Dee Dee ஆகியோரிடம் முன்பைவிட மிகவும் கனிவாக நடந்துகொள்கிறான், ஆனால் Melanie-யிடம் சற்றே முரட்டுத்தனமாக நடந்துகொள்கிறான். தலையில் மீண்டும் ஓர் அடி படுவதால் மட்டுமே இந்த நிலையை மாற்ற முடியும்.', 'cell_003_004': 'நவம்பர் 1, 1994', 'cell_004_002': '“Robin Hood நாடகம்”', 'cell_004_003': 'Alfie படிக்கும் பள்ளியில் Robin Hood நாடகம் அரங்கேற்றப்படுகிறது; Robin Hood பாத்திரத்தில் நடிக்க Alfie தேர்ந்தெடுக்கப்படுகிறான். இந்த வாய்ப்பால் Alfie உற்சாகமடைகிறான், ஆனால் உடலோடு ஒட்டிய காலுறைகள் பெண்களுக்கானவை என்று நினைப்பதால் அவற்றை அணிய அவன் விரும்பவில்லை. இருப்பினும், அது Robin Hood பாத்திரத்தில் அவனது நடிப்பைப் பாதிக்க விடக்கூடாது என்று Dee Dee புத்திசாலித்தனமாக அறிவுறுத்தும்போது, உடலோடு ஒட்டிய காலுறைகள் குறித்த தனது நிலைப்பாட்டை அவன் மறுபரிசீலனை செய்கிறான்.', 'cell_004_004': 'நவம்பர் 9, 1994', 'cell_005_002': '“கூடைப்பந்து அணிக்கான தேர்வுகள்”', 'cell_005_003': 'Alfie கூடைப்பந்து அணிக்கான தேர்வில் பங்கேற்கிறான்; தனது கூடைப்பந்துத் திறமைகளை வெளிப்படுத்திய பிறகும் அவன் தேர்வாகவில்லை. ஆனால் Harry, Dee Dee மற்றும் Donnell ஆகியோர் அணிக்குத் தேர்வாகிறார்கள். Alfie மனமுடைந்து போய், கொண்டாட்ட விருந்தில் கலந்துகொள்ள விரும்பவில்லை. ஆனால், அணியுடன் இணைந்து விளையாடாமல் பந்தைத் தன்னிடமே வைத்திருந்தது அவனுடைய தவறுதான் என்று சொல்லி Goo அவனுக்குப் புரியவைக்கிறான்.', 'cell_005_004': 'நவம்பர் 30, 1994', 'cell_006_002': '“பாம்பு எங்கே?”', 'cell_006_003': 'Dee Dee ஒரு பாம்பைப் பெறுகிறான், ஆனால் அதைப் பற்றித் தனது பெற்றோருக்குத் தெரியக்கூடாது என்று விரும்புகிறான். இருப்பினும், வீட்டிற்குள் பாம்பைத் தொலைத்துவிடும்போது நிலைமை சிக்கலாகிறது. இதற்கிடையில், Melanie மற்றும் Deonne ஆகியோரிடம் அவர்களது ஆசிரியை, தனது அன்புச் செல்ல முயலான Duchess ஐ வார இறுதியில் பார்த்துக்கொள்ளும் பொறுப்பை ஒப்படைக்கிறார். பாம்புகள் முயல்களைச் சாப்பிடும் என்று Goo விடமிருந்து அறிந்ததும், Alfie மற்றும் Dee Dee இருவரும் Duchess குறித்து கவலைப்படுகிறார்கள்.', 'cell_006_004': 'டிசம்பர் 6, 1994', 'cell_007_002': '“Dee Dee இன் காதலி”', 'cell_007_003': 'Harry மற்றும் Donnell முன்னிலையில் ஒரு பெண் Dee Dee-க்கு முத்தமிடுகிறாள். இதை யாரிடமும் சொல்ல மாட்டோம் என்று அவர்கள் வாக்குறுதி அளிக்கிறார்கள். ஆனால், தவறுதலாக விஷயம் வெளியே தெரியவந்து, எல்லோரும் Dee Dee-யைப் பார்த்துச் சிரிக்கிறார்கள். Dee Dee, Harry மற்றும் Donnell உடனான நட்பை முறித்துக்கொண்டு, Alfie மற்றும் Goo உடன் நேரம் செலவிடுகிறான். விரைவிலேயே, Alfie மற்றும் Goo இறுதியாக அந்த மூவரையும் ஒருவருடன் ஒருவர் பேச வைக்கிறார்கள்.', 'cell_007_004': 'டிசம்பர் 15, 1994', 'cell_008_002': '“Dee Dee-யின் முடிவெட்டு”', 'cell_008_003': 'Dee Dee, Cool Doctor Money-யிடம் முடிவெட்டிக்கொண்டு, தனது தலையில் தன் பெயர் தெரியும்படி முடியைச் சிரைத்துக்கொள்ள விரும்புகிறான். அவனது பெற்றோர் இதற்கு அனுமதிக்கவில்லை. ஆனால், ஐந்து டாலர்களுக்கு அதைச் செய்து தருவதாக Goo கூறுகிறான். எனினும், Goo, Dee Dee-யின் தலைமுடியை அலங்கோலமாக்கி, அவனது பெயரையும் தவறாக எழுதிவிடுவதால், அவனது பெற்றோருக்கு உண்மை தெரிந்துவிடுகிறது; Dee Dee தனது தலைமுடியை முழுவதுமாகச் சிரைத்துக்கொள்ள வேண்டிய கட்டாயம் ஏற்படுகிறது. அதோடு, அவனது மொட்டைத் தலையைப் பற்றி நண்பர்கள் கிண்டல் செய்வதால், Goo மற்றும் Alfie உட்படச் சிறுவர்களிடையே சண்டை ஏற்படுகிறது. ஒரு துணைக்கதையில், Alfie மற்றும் Goo, ஜலப்பீன்யோ மிளகாய் லாலிபாப்பைப் பயன்படுத்தி Dee Dee-யிடம் குறும்பு செய்ய முயல்கிறார்கள். ஆனால், எதுவும் அறியாத Roger அதற்கு இரையாகிவிடுவதால், அவர்களது திட்டம் அவர்களுக்கே எதிராகத் திரும்புகிறது; இதனால் அவன் அந்தச் சிறுவர்களைத் துரத்துகிறான்.', 'cell_008_004': 'டிசம்பர் 20, 1994', 'cell_009_002': '“Dee Dee வீட்டை விட்டு ஓடிப்போகிறான்”', 'cell_009_003': 'Dee Dee இந்த வாரம் முழுவதும் மான்ஸ்டர் டிரக் நிகழ்ச்சிக்குச் செல்லக் காத்திருக்கிறான். ஆனால் Alfie மற்றும் Goo-வின் பேஸ்பால் அணி போட்டித் தொடருக்குத் தகுதி பெறுவதால், எல்லோரும் மான்ஸ்டர் டிரக் நிகழ்ச்சியை மறந்துவிடுகிறார்கள். தன்னை யாரும் கண்டுகொள்ளவில்லை என்று உணரும் Dee Dee, Harry மற்றும் Donnell உடன் வீட்டை விட்டு ஓடிப்போகிறான். அவனை வீட்டுக்குத் திரும்பி வரச் சம்மதிக்க வைக்க முயற்சிப்பது Alfie மற்றும் Goo-வின் பொறுப்பாகிறது.', 'cell_009_004': 'டிசம்பர் 28, 1994', 'cell_010_002': '“Donnell-இன் பிறந்தநாள் விருந்து”', 'cell_010_003': 'Donnell பிறந்தநாள் விருந்து நடத்தவிருக்கிறான்; அங்கு நடக்கவிருக்கும் நடனம் பற்றியும் வரவிருக்கும் அசத்தலான நபர்கள் பற்றியும் பெருமையடித்துக்கொள்கிறான். தனக்கு நடனமாடத் தெரியும் என்று Harry சொல்கிறான்; அதனால், தனக்கு நடனமாடத் தெரியாததால் Dee Dee தனித்து விடப்பட்டதாக உணர்கிறான். பின்னர், Harry தனியாக Dee Dee-யிடம் தனக்கும் நடனமாடத் தெரியாது என்றும், Donnell தன்னைக் கேலி செய்யாமல் இருப்பதற்காகவே பொய் சொன்னதாகவும் ஒப்புக்கொள்கிறான். எனவே, நடனமாடக் கற்றுக்கொள்ள உதவுமாறு அவர்கள் Alfie-யிடம் கேட்கிறார்கள். கணிதக் குறுந்தேர்வில் தானும் Goo-வும் ஏமாற்றத் திட்டமிட்டிருந்ததை Dee Dee முன்பு Roger-யிடம் சொல்லிக்கொடுத்ததால், Alfie உதவ மறுக்கிறான். அவனுடைய கணித வீட்டுப்பாடத்திற்கு உதவ மாட்டேன் என்று Melanie மிரட்டிய பிறகு, Alfie இறுதியில் ஒப்புக்கொள்கிறான். விரைவில் Dee Dee-யும் Harry-யும் Donnell-இன் ரகசியத்தை அறிந்துகொண்டு, அவனுக்கு நடனமாடக் கற்றுக்கொடுக்க வேண்டிய கட்டாயத்திற்கு உள்ளாகிறார்கள். விருந்துக்குப் பிறகு, Dee Dee அதைப் பற்றி Alfie-யிடம் சொல்கிறான்; Donnell ஒரு பொய்யன் என்பது Alfie-க்கு ஏற்கெனவே தெரிந்திருந்ததை அறிந்துகொள்கிறான்.', 'cell_010_004': 'ஜனவரி 5, 1995', 'cell_011_002': '“Alfie-யின் பிறந்தநாள் விருந்து”', 'cell_011_003': 'Goo மற்றும் Melanie தாங்கள் காதலிப்பதாக நடித்து, எல்லாவற்றிலிருந்தும் Alfie-ஐ ஒதுக்கிவைக்கிறார்கள். இதனால் சலிப்படைந்த அவன், Dee Dee மற்றும் அவனது நண்பர்களுடன் நேரம் செலவிடத் தொடங்குகிறான். ஆனாலும், Goo இல்லாமல் முன்புபோல் இருப்பதில்லை. பின்னர், Goo மற்றும் Melanie மற்ற அனைவருடனும் சேர்ந்து திட்டமிட்டுக்கொண்டிருந்த ஆச்சரியப் பிறந்தநாள் விருந்தைப் பற்றி Alfie அறிகிறான் (Dee Dee-ஐத் தவிர; அவனுக்குத் தெரிந்திருந்தால் சொல்லியிருப்பான் என்பதால் அவனுக்குத் தெரியக்கூடாது).', 'cell_011_004': 'ஜனவரி 19, 1995', 'cell_012_002': '“மிட்டாய் விற்பனை”', 'cell_012_003': 'விலையுயர்ந்த சில ஜாக்கெட்டுகளை வாங்கப் பணம் சம்பாதிப்பதற்காக Alfie மற்றும் Goo மிட்டாய் விற்கிறார்கள், ஆனால் அவர்களுக்கு அதிர்ஷ்டம் கைகூடவில்லை. எனினும், மிட்டாய் விற்க Dee Dee அவர்களுக்கு உதவத் தொடங்கியதும், அவர்கள் பணம் சம்பாதிக்கத் தொடங்குகிறார்கள்; தொடர்ந்து உதவும்படி அவனைக் கேட்கிறார்கள். விரைவில், பணத்தில் Dee Dee-க்குரிய பங்கு குறித்து Melanie, Deonne, Harry மற்றும் Donnell ஆகியோர் Goo மற்றும் Alfie-யிடம் கேள்வி எழுப்புகிறார்கள். அந்தச் சிறுவர்கள் தங்களுக்கும், தங்கள் நன்றியின் அடையாளமாக Dee Dee-க்கும் மூன்று விலையுயர்ந்த ஜாக்கெட்டுகளை வாங்க அந்தப் பணத்தைப் பயன்படுத்தியிருப்பதை அவர்கள் விரைவில் அறிகிறார்கள். அவசரப்பட்டு முடிவுசெய்ததற்காக அவர்கள் உடனே Alfie மற்றும் Goo-விடம் மன்னிப்புக் கேட்கிறார்கள்.', 'cell_012_004': 'ஜனவரி 26, 1995', 'cell_013_002': '“பெரிய அடாவடிக்காரன்”', 'cell_013_003': 'Dee Dee பள்ளியில் அடிவாங்குகிறான்; அவனது நண்பர்கள் அவனுக்குத் திருப்பிச் சண்டையிடக் கற்றுக்கொடுக்க முயல்கிறார்கள். ஆனால் Goo அவனைப் பொய்யாக மிரட்டச் சொல்கிறான். அந்தத் திட்டம் எதிர்விளைவை ஏற்படுத்தி, அதனால் Dee Dee அடிவாங்குகிறான். Alfie அவனைத் துன்புறுத்தியவரை எதிர்கொள்ளும்போது, Dee Dee ஒரு பெண்ணால் துன்புறுத்தப்பட்டதை அறிகிறான். Alfie மற்றும் Goo அவளை எதிர்கொள்ள முடிவு செய்கிறார்கள். ஆனால், அவர்களது வகுப்புத் தோழர்களில் சிலர், அந்தப் பெண்ணின் உடன்பிறந்தவர்கள் என்பதால், இவர்கள் தங்கள் சகோதரியை மிரட்டித் துன்புறுத்துவதை அறிந்து தலையிடுகிறார்கள்.', 'cell_013_004': 'பிப்ரவரி 2, 1995'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
