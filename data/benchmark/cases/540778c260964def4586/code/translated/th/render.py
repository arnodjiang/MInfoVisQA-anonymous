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
LANGUAGE = 'th'
DATA = {'rows': [[{'text': 'Series #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Season #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Title', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Notes', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_003'}, {'text': 'Original air date', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_004'}], [{'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Charity"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_002'}, {'text': "Alfie, Dee Dee, and Melanie are supposed to be helping their parents at a carnival by working the dunking booth. When Goo arrives and announces their favorite basketball player, Kendall Gill, is at the Comic Book Store signing autographs, the boys decide to ditch the carnival. This leaves Melanie and Jennifer to work the booth and both end up soaked. But the Comic Book Store is packed and much to Alfie and Dee Dee's surprise their father has to interview Kendall Gill. Goo comes up with a plan to get Alfie and Dee Dee, Gill's signature before getting them back at the local carnival, but are caught by Roger. All ends well for everyone except Alfie and Goo, who must endure being soaked at the dunking booth.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_003'}, {'text': 'October 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_004'}], [{'text': '2', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Practical Joke War"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_002'}, {'text': "Alfie and Goo unleash harsh practical jokes on Dee Dee and his friends. Dee Dee, Harry and Donnel retaliate by pulling a practical joke on Alfie with the trick gum. After Alfie and Goo get even with Dee Dee and his friends, Melanie and Deonne help them get even. Soon, Alfie and Goo declare a practical joke war on Melanie, Dee Dee and their friends. This eventually stops when Roger and Jennifer end up on the wrong end of the practical joke war after being announced as the winner of a magazine contest for Best Family Of The Year. They set their children straight for their behavior and will have a talk with their friends' parents as well.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_003'}, {'text': 'October 22, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_004'}], [{'text': '3', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Weekend Aunt Helen Came"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_002'}, {'text': "The boy's mother, Jennifer, leaves for the weekend and she leaves the father, Roger, in charge. However, he lets the kids run wild. Alfie and Dee Dee's Aunt Helen then comes to oversee the house until Jennifer gets back. Meanwhile, Alfie throws a basketball at Goo, which hits him in the head, giving him temporary amnesia. In this case of memory loss, Goo acts like a nerd, does homework on a weekend, wants to be called Milton instead of Goo, and he even calls Alfie Alfred. He is much nicer to Deonne and Dee Dee, but is somewhat rude to Melanie. The only thing that will reverse this is another hit in the head.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_003'}, {'text': 'November 1, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_004'}], [{'text': '4', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Robin Hood Play"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_002'}, {'text': "Alfie's school is performing the play Robin Hood and Alfie is chosen to play the part of Robin Hood. Alfie is excited at this prospect, but he does not want to wear tights because he feels that tights are for girls. However, he reconsiders his stance on tights when Dee Dee wisely tells him not to let that affect his performance as Robin Hood.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_003'}, {'text': 'November 9, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_004'}], [{'text': '5', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Basketball Tryouts"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_002'}, {'text': "Alfie tries out for the basketball team and doesn't make it even after showing off his basketball skills. However, Harry, Dee Dee and Donnell make the team. Alfie is depressed and doesn't want to attend the celebration party. However, Goo sets him straight by telling him it was his own fault for not being a team player and kept the ball to himself.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_003'}, {'text': 'November 30, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_004'}], [{'text': '6', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Where\'s the Snake?"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_002'}, {'text': "Dee Dee gets a snake, but he doesn't want his parents to know about it. However, things get complicated when he loses the snake in the house. Meanwhile, Melanie and Deonne are assigned by their teacher to take care of her beloved pet rabbit, Duchess for the weekend. This causes both Alfie and Dee Dee to be concerned for Duchess when they learn from Goo that snakes eat rabbits.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_003'}, {'text': 'December 6, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_004'}], [{'text': '7', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Girlfriend"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_002'}, {'text': 'A girl kisses Dee Dee in front of Harry and Donnell. They promise not to tell, but it slips and everyone laughs at Dee Dee. Dee Dee ends his friendship with Harry and Donnell and hangs out with Alfie and Goo. Soon, Alfie and Goo finally get the three to talk to each other.', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_003'}, {'text': 'December 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_004'}], [{'text': '8', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Haircut"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_002'}, {'text': "Dee Dee wants to get a hair cut by Cool Doctor Money and have his name shaved in his head. His parents will not let him do this, but Goo offers to do it for five dollars. However, when Goo messes up Dee Dee's hair and spells his name wrong, his parents find out the truth and Dee Dee is forced to have his hair shaved off. In addition to that, his friends tease him about his bald head, causing a fight between the boys along with Goo and Alfie. In a b-story, Alfie and Goo try to play a practical joke on Dee Dee involving a jalapeño lollipop. It backfires when Roger is the unwitting victim and it leads to him chasing the boys around.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_003'}, {'text': 'December 20, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_004'}], [{'text': '9', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee Runs Away"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_002'}, {'text': "Dee Dee has been waiting to go to a monster truck show all week. But Alfie and Goo's baseball team makes it to the tournament and everyone forgets about the monster truck show. Dee Dee feels ignored and runs away from home with Harry and Donnell. It's up to Alfie and Goo to try and convince him to come home.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_003'}, {'text': 'December 28, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_004'}], [{'text': '10', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '\'"Donnell\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_002'}, {'text': "Donnell is having a birthday party and brags about all the dancing and cool people who will be there. Harry says that he knows how to dance so Dee Dee feels left out because he doesn't know how to dance. Later on, Harry admits to Dee Dee alone that he can't dance either and only lied so he doesn't get teased by Donnell. So, they ask Alfie to help them learn how to dance. He refuses to help because Dee Dee previously told on him to Roger about his and Goo's plans to cheat on their math quiz. Alfie eventually agrees, after Melanie threatens to refuse to help him with his math homework. Soon Dee Dee and Harry learn Donnell's secret and were forced to teach him how to dance. After the party, Dee Dee tells Alfie about it and finds out that he knew Donnell was a liar.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_003'}, {'text': 'January 5, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_004'}], [{'text': '11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Alfie\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_002'}, {'text': "Goo and Melanie pretend they are dating and they leave Alfie out of everything. He ends up bored and starts hanging out with Dee Dee and his friends. However, it just isn't the same without Goo. Later on, Alfie learns about the surprise birthday party that Goo and Melanie had been planning with everyone else (except for Dee Dee, who couldn't know since he would've told).", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_003'}, {'text': 'January 19, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_004'}], [{'text': '12', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Candy Sale"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_002'}, {'text': "Alfie and Goo are selling candy to make money for some expensive jackets, but they are not having any luck. However, when Dee Dee start helping them sell candy, they start to make money and asks him to help them out. Soon Goo and Alfie finds themselves confronted by Melanie, Deonne, Harry and Donnell for Dee Dee's share of the money. They soon learn the boys have used the money to buy three expensive jackets for themselves and Dee Dee as a token of their gratitude. They quickly apologize to Alfie and Goo for their quick judgment.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_003'}, {'text': 'January 26, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_004'}], [{'text': '13', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Big Bully"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_002'}, {'text': "Dee Dee gets beat up at school and his friends try to teach him how to fight back. Goo, however, tells him to bluff, but the plan backfires and Dee Dee gets hit because of it. When Alfie confronts the bully, he learns that Dee Dee was picked on by a girl. Alfie and Goo decide to confront her. However, when some of their classmates, who happen to be the girls' siblings, learn they are bullying their sister, they intervene.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_003'}, {'text': 'February 2, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_004'}]], 'layout': {'width': 1500, 'padding': 45, 'cell_width': 282.0, 'font_size': 27, 'heights': [106, 2875, 2212, 2563, 1393, 1198, 1471, 1159, 2602, 1198, 2953, 1354, 2017, 1549]}}
LABELS = {'cell_000_000': 'ตอนที่ในซีรีส์', 'cell_000_001': 'ตอนที่ในซีซัน', 'cell_000_002': 'ชื่อ', 'cell_000_003': 'หมายเหตุ', 'cell_000_004': 'วันที่ออกอากาศครั้งแรก', 'cell_001_002': '"งานการกุศล"', 'cell_001_003': 'อัลฟี ดีดี และเมลานีต้องช่วยพ่อแม่ดูแลซุ้มปาเป้าตกน้ำในงานรื่นเริง แต่เมื่อกูมาบอกว่าเคนดัลล์ กิลล์ นักบาสเกตบอลคนโปรดของพวกเขากำลังแจกลายเซ็นที่ร้านหนังสือการ์ตูน เด็กหนุ่มทั้งสองจึงตัดสินใจหนีจากงาน ปล่อยให้เมลานีกับเจนนิเฟอร์ดูแลซุ้มจนเปียกปอนทั้งคู่ ทว่าร้านหนังสือการ์ตูนแน่นขนัด และอัลฟีกับดีดีก็ประหลาดใจที่พ่อของพวกเขาต้องมาสัมภาษณ์เคนดัลล์ กิลล์ กูคิดแผนให้อัลฟีกับดีดีได้ลายเซ็นของกิลล์ก่อนพากลับไปงานรื่นเริงในละแวกบ้าน แต่โรเจอร์จับได้ ทุกอย่างจบลงด้วยดีสำหรับทุกคน ยกเว้นอัลฟีกับกูที่ต้องยอมเปียกปอนอยู่ในซุ้มปาเป้าตกน้ำ', 'cell_001_004': '15 ตุลาคม 1994', 'cell_002_002': '"สงครามแกล้งกัน"', 'cell_002_003': 'อัลฟีกับกูแกล้งดีดีและเพื่อน ๆ อย่างรุนแรง ดีดี แฮร์รี และดอนเนลล์จึงเอาคืนด้วยการใช้หมากฝรั่งแกล้งอัลฟี หลังจากอัลฟีกับกูแก้แค้นดีดีกับเพื่อน ๆ เมลานีและดีออนน์ก็ช่วยพวกเขาเอาคืน ไม่นานอัลฟีกับกูจึงประกาศสงครามแกล้งเมลานี ดีดี และเพื่อน ๆ เรื่องทั้งหมดหยุดลงเมื่อโรเจอร์กับเจนนิเฟอร์กลายเป็นผู้เคราะห์ร้ายจากสงครามแกล้งกัน หลังได้รับการประกาศให้เป็นผู้ชนะการประกวดครอบครัวดีเด่นแห่งปีของนิตยสาร ทั้งคู่ตักเตือนลูก ๆ เรื่องพฤติกรรมที่ไม่เหมาะสม และจะไปคุยกับพ่อแม่ของเพื่อน ๆ ด้วย', 'cell_002_004': '22 ตุลาคม 1994', 'cell_003_002': '"สุดสัปดาห์ที่ป้าเฮเลนมาเยือน"', 'cell_003_003': 'เจนนิเฟอร์ แม่ของเด็ก ๆ ไม่อยู่บ้านช่วงสุดสัปดาห์ จึงให้โรเจอร์ผู้เป็นพ่อคอยดูแล แต่เขากลับปล่อยให้เด็ก ๆ เล่นซนกันเต็มที่ ป้าเฮเลนของอัลฟีกับดีดีจึงมาดูแลบ้านจนกว่าเจนนิเฟอร์จะกลับ ระหว่างนั้นอัลฟีขว้างลูกบาสเกตบอลใส่กูโดนศีรษะ ทำให้กูความจำเสื่อมชั่วคราว หลังสูญเสียความทรงจำ กูทำตัวเป็นเด็กเนิร์ด ทำการบ้านในวันหยุด อยากให้คนเรียกว่ามิลตันแทนกู และยังเรียกอัลฟีว่าอัลเฟรดด้วย เขาทำดีกับดีออนน์และดีดีมากขึ้น แต่กลับค่อนข้างหยาบคายกับเมลานี สิ่งเดียวที่จะทำให้เขากลับเป็นเหมือนเดิมได้คือต้องถูกกระแทกที่ศีรษะอีกครั้ง', 'cell_003_004': '1 พฤศจิกายน 1994', 'cell_004_002': '"ละครโรบินฮูด"', 'cell_004_003': 'โรงเรียนของอัลฟีกำลังจัดแสดงละครเรื่องโรบินฮูด และอัลฟีได้รับเลือกให้รับบทโรบินฮูด เขาตื่นเต้นกับโอกาสนี้ แต่ไม่อยากใส่กางเกงรัดรูปเพราะคิดว่าเป็นของผู้หญิง อย่างไรก็ตาม เขาทบทวนความคิดนี้ใหม่เมื่อดีดีเตือนอย่างมีเหตุผลว่าอย่าปล่อยให้เรื่องนี้กระทบการแสดงบทโรบินฮูดของเขา', 'cell_004_004': '9 พฤศจิกายน 1994', 'cell_005_002': '"คัดตัวนักบาสเกตบอล"', 'cell_005_003': 'อัลฟีเข้าคัดตัวทีมบาสเกตบอลแต่ไม่ผ่าน แม้จะแสดงทักษะการเล่นออกมาแล้วก็ตาม ส่วนแฮร์รี ดีดี และดอนเนลล์กลับติดทีม อัลฟีเสียใจจนไม่อยากไปร่วมงานฉลอง แต่กูเตือนสติเขาว่าเป็นความผิดของเขาเองที่ไม่ยอมเล่นเป็นทีมและเอาแต่ครองบอลไว้คนเดียว', 'cell_005_004': '30 พฤศจิกายน 1994', 'cell_006_002': '"งูอยู่ไหน?"', 'cell_006_003': 'ดีดีได้งูมาเลี้ยงแต่ไม่อยากให้พ่อแม่รู้ เรื่องกลับวุ่นวายเมื่องูหายไปในบ้าน ขณะเดียวกัน ครูฝากให้เมลานีกับดีออนน์ช่วยดูแลดัชเชส กระต่ายแสนรักของเธอในช่วงสุดสัปดาห์ อัลฟีกับดีดีจึงเป็นห่วงดัชเชสเมื่อรู้จากกูว่างูกินกระต่าย', 'cell_006_004': '6 ธันวาคม 1994', 'cell_007_002': '"แฟนของดีดี"', 'cell_007_003': 'เด็กผู้หญิงคนหนึ่งจูบดีดีต่อหน้าแฮร์รีกับดอนเนลล์ ทั้งสองสัญญาว่าจะไม่บอกใคร แต่กลับหลุดปากจนทุกคนหัวเราะเยาะดีดี ดีดีจึงเลิกคบแฮร์รีกับดอนเนลล์แล้วไปเที่ยวเล่นกับอัลฟีและกูแทน ในที่สุดอัลฟีกับกูก็ทำให้ทั้งสามคนยอมคุยกันได้', 'cell_007_004': '15 ธันวาคม 1994', 'cell_008_002': '"ทรงผมของดีดี"', 'cell_008_003': 'ดีดีอยากให้คูล ด็อกเตอร์ มันนีตัดผมให้และไถผมเป็นชื่อของตัวเอง พ่อแม่ไม่อนุญาต แต่กูเสนอจะทำให้ในราคาห้าดอลลาร์ ทว่ากูกลับตัดผมดีดีจนเสียทรงและสะกดชื่อผิด เมื่อพ่อแม่รู้ความจริง ดีดีจึงต้องโกนผมทิ้งทั้งหมด ซ้ำเพื่อน ๆ ยังล้อเรื่องหัวโล้นจนเกิดการทะเลาะกันในหมู่เด็ก ๆ รวมถึงกูกับอัลฟีด้วย ในเรื่องราวรอง อัลฟีกับกูพยายามแกล้งดีดีด้วยอมยิ้มพริกฮาลาเปญโญ แต่แผนกลับตาลปัตรเมื่อโรเจอร์ตกเป็นเหยื่อโดยไม่รู้ตัว จนเขาวิ่งไล่เด็กหนุ่มทั้งสองไปทั่ว', 'cell_008_004': '20 ธันวาคม 1994', 'cell_009_002': '"ดีดีหนีออกจากบ้าน"', 'cell_009_003': 'ดีดีรอจะไปดูการแสดงรถมอนสเตอร์ทรักมาตลอดสัปดาห์ แต่ทีมเบสบอลของอัลฟีกับกูได้เข้ารอบการแข่งขัน ทุกคนจึงลืมเรื่องการแสดงรถมอนสเตอร์ทรัก ดีดีรู้สึกว่าถูกละเลยจึงหนีออกจากบ้านไปกับแฮร์รีและดอนเนลล์ อัลฟีกับกูต้องพยายามเกลี้ยกล่อมให้เขากลับบ้าน', 'cell_009_004': '28 ธันวาคม 1994', 'cell_010_002': '\'"งานวันเกิดของดอนเนลล์"', 'cell_010_003': 'ดอนเนลล์กำลังจะจัดงานวันเกิดและคุยอวดว่าจะมีการเต้นรำและคนเจ๋ง ๆ มาร่วมงานมากมาย แฮร์รีบอกว่าตัวเองเต้นเป็น ดีดีจึงรู้สึกแปลกแยกเพราะเต้นไม่เป็น ต่อมาแฮร์รีสารภาพกับดีดีตามลำพังว่าเขาก็เต้นไม่เป็นเหมือนกัน ที่โกหกไปเพราะไม่อยากถูกดอนเนลล์ล้อ ทั้งสองจึงขอให้อัลฟีสอนเต้นให้ อัลฟีปฏิเสธเพราะก่อนหน้านี้ดีดีฟ้องโรเจอร์เรื่องที่เขากับกูวางแผนโกงแบบทดสอบคณิตศาสตร์ ในที่สุดอัลฟีก็ยอมช่วยหลังเมลานีขู่ว่าจะไม่ช่วยเขาทำการบ้านคณิตศาสตร์ ไม่นานดีดีกับแฮร์รีก็รู้ความลับของดอนเนลล์และจำต้องสอนเขาเต้น หลังงานเลี้ยง ดีดีเล่าเรื่องนี้ให้อัลฟีฟังและพบว่าอัลฟีรู้อยู่แล้วว่าดอนเนลล์โกหก', 'cell_010_004': '5 มกราคม 1995', 'cell_011_002': '"งานวันเกิดของอัลฟี"', 'cell_011_003': 'กูกับเมลานีแกล้งทำเป็นคบกันและกันอัลฟีออกจากทุกกิจกรรม เขาเบื่อจนเริ่มไปเที่ยวเล่นกับดีดีและเพื่อน ๆ แต่เมื่อไม่มีกู อะไร ๆ ก็ไม่เหมือนเดิม ต่อมาอัลฟีได้รู้เรื่องงานวันเกิดเซอร์ไพรส์ที่กูกับเมลานีวางแผนร่วมกับคนอื่น ๆ ทุกคน ยกเว้นดีดีที่รู้ไม่ได้เพราะคงเผลอบอกเขาแน่', 'cell_011_004': '19 มกราคม 1995', 'cell_012_002': '"ขายลูกกวาด"', 'cell_012_003': 'อัลฟีกับกูขายลูกกวาดเพื่อหาเงินซื้อเสื้อแจ็กเก็ตราคาแพง แต่ขายไม่ค่อยได้ เมื่อดีดีมาช่วยขาย พวกเขาจึงเริ่มได้เงินและขอให้ดีดีช่วยต่อ ไม่นานเมลานี ดีออนน์ แฮร์รี และดอนเนลล์ก็มาทวงเงินส่วนแบ่งของดีดีจากกูกับอัลฟี แต่แล้วก็ได้รู้ว่าทั้งสองนำเงินไปซื้อเสื้อแจ็กเก็ตราคาแพงสามตัวให้ตัวเองและดีดีเพื่อเป็นการขอบคุณ ทุกคนจึงรีบขอโทษอัลฟีกับกูที่ด่วนตัดสินพวกเขา', 'cell_012_004': '26 มกราคม 1995', 'cell_013_002': '"จอมเกเรตัวแสบ"', 'cell_013_003': 'ดีดีถูกทำร้ายที่โรงเรียน เพื่อน ๆ จึงพยายามสอนให้เขาสู้กลับ แต่กูแนะนำให้ขู่ ทว่าแผนกลับไม่ได้ผลจนดีดีถูกต่อย เมื่ออัลฟีไปเผชิญหน้ากับคนที่รังแกดีดี เขาก็รู้ว่าคนคนนั้นเป็นเด็กผู้หญิง อัลฟีกับกูจึงตัดสินใจไปเผชิญหน้ากับเธอ แต่เพื่อนร่วมชั้นบางคนซึ่งเป็นพี่น้องของเด็กผู้หญิงคนนั้นรู้ว่าทั้งสองกำลังรังแกพี่น้องของตน จึงเข้ามาห้าม', 'cell_013_004': '2 กุมภาพันธ์ 1995'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
