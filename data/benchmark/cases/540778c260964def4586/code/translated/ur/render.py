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
LANGUAGE = 'ur'
DATA = {'rows': [[{'text': 'Series #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_000'}, {'text': 'Season #', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_001'}, {'text': 'Title', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_002'}, {'text': 'Notes', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_003'}, {'text': 'Original air date', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_000_004'}], [{'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Charity"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_002'}, {'text': "Alfie, Dee Dee, and Melanie are supposed to be helping their parents at a carnival by working the dunking booth. When Goo arrives and announces their favorite basketball player, Kendall Gill, is at the Comic Book Store signing autographs, the boys decide to ditch the carnival. This leaves Melanie and Jennifer to work the booth and both end up soaked. But the Comic Book Store is packed and much to Alfie and Dee Dee's surprise their father has to interview Kendall Gill. Goo comes up with a plan to get Alfie and Dee Dee, Gill's signature before getting them back at the local carnival, but are caught by Roger. All ends well for everyone except Alfie and Goo, who must endure being soaked at the dunking booth.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_003'}, {'text': 'October 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_001_004'}], [{'text': '2', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Practical Joke War"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_002'}, {'text': "Alfie and Goo unleash harsh practical jokes on Dee Dee and his friends. Dee Dee, Harry and Donnel retaliate by pulling a practical joke on Alfie with the trick gum. After Alfie and Goo get even with Dee Dee and his friends, Melanie and Deonne help them get even. Soon, Alfie and Goo declare a practical joke war on Melanie, Dee Dee and their friends. This eventually stops when Roger and Jennifer end up on the wrong end of the practical joke war after being announced as the winner of a magazine contest for Best Family Of The Year. They set their children straight for their behavior and will have a talk with their friends' parents as well.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_003'}, {'text': 'October 22, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_002_004'}], [{'text': '3', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Weekend Aunt Helen Came"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_002'}, {'text': "The boy's mother, Jennifer, leaves for the weekend and she leaves the father, Roger, in charge. However, he lets the kids run wild. Alfie and Dee Dee's Aunt Helen then comes to oversee the house until Jennifer gets back. Meanwhile, Alfie throws a basketball at Goo, which hits him in the head, giving him temporary amnesia. In this case of memory loss, Goo acts like a nerd, does homework on a weekend, wants to be called Milton instead of Goo, and he even calls Alfie Alfred. He is much nicer to Deonne and Dee Dee, but is somewhat rude to Melanie. The only thing that will reverse this is another hit in the head.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_003'}, {'text': 'November 1, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_003_004'}], [{'text': '4', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Robin Hood Play"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_002'}, {'text': "Alfie's school is performing the play Robin Hood and Alfie is chosen to play the part of Robin Hood. Alfie is excited at this prospect, but he does not want to wear tights because he feels that tights are for girls. However, he reconsiders his stance on tights when Dee Dee wisely tells him not to let that affect his performance as Robin Hood.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_003'}, {'text': 'November 9, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_004_004'}], [{'text': '5', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Basketball Tryouts"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_002'}, {'text': "Alfie tries out for the basketball team and doesn't make it even after showing off his basketball skills. However, Harry, Dee Dee and Donnell make the team. Alfie is depressed and doesn't want to attend the celebration party. However, Goo sets him straight by telling him it was his own fault for not being a team player and kept the ball to himself.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_003'}, {'text': 'November 30, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_005_004'}], [{'text': '6', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Where\'s the Snake?"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_002'}, {'text': "Dee Dee gets a snake, but he doesn't want his parents to know about it. However, things get complicated when he loses the snake in the house. Meanwhile, Melanie and Deonne are assigned by their teacher to take care of her beloved pet rabbit, Duchess for the weekend. This causes both Alfie and Dee Dee to be concerned for Duchess when they learn from Goo that snakes eat rabbits.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_003'}, {'text': 'December 6, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_006_004'}], [{'text': '7', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Girlfriend"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_002'}, {'text': 'A girl kisses Dee Dee in front of Harry and Donnell. They promise not to tell, but it slips and everyone laughs at Dee Dee. Dee Dee ends his friendship with Harry and Donnell and hangs out with Alfie and Goo. Soon, Alfie and Goo finally get the three to talk to each other.', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_003'}, {'text': 'December 15, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_007_004'}], [{'text': '8', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee\'s Haircut"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_002'}, {'text': "Dee Dee wants to get a hair cut by Cool Doctor Money and have his name shaved in his head. His parents will not let him do this, but Goo offers to do it for five dollars. However, when Goo messes up Dee Dee's hair and spells his name wrong, his parents find out the truth and Dee Dee is forced to have his hair shaved off. In addition to that, his friends tease him about his bald head, causing a fight between the boys along with Goo and Alfie. In a b-story, Alfie and Goo try to play a practical joke on Dee Dee involving a jalapeño lollipop. It backfires when Roger is the unwitting victim and it leads to him chasing the boys around.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_003'}, {'text': 'December 20, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_008_004'}], [{'text': '9', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Dee Dee Runs Away"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_002'}, {'text': "Dee Dee has been waiting to go to a monster truck show all week. But Alfie and Goo's baseball team makes it to the tournament and everyone forgets about the monster truck show. Dee Dee feels ignored and runs away from home with Harry and Donnell. It's up to Alfie and Goo to try and convince him to come home.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_003'}, {'text': 'December 28, 1994', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_009_004'}], [{'text': '10', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '\'"Donnell\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_002'}, {'text': "Donnell is having a birthday party and brags about all the dancing and cool people who will be there. Harry says that he knows how to dance so Dee Dee feels left out because he doesn't know how to dance. Later on, Harry admits to Dee Dee alone that he can't dance either and only lied so he doesn't get teased by Donnell. So, they ask Alfie to help them learn how to dance. He refuses to help because Dee Dee previously told on him to Roger about his and Goo's plans to cheat on their math quiz. Alfie eventually agrees, after Melanie threatens to refuse to help him with his math homework. Soon Dee Dee and Harry learn Donnell's secret and were forced to teach him how to dance. After the party, Dee Dee tells Alfie about it and finds out that he knew Donnell was a liar.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_003'}, {'text': 'January 5, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_010_004'}], [{'text': '11', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Alfie\'s Birthday Party"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_002'}, {'text': "Goo and Melanie pretend they are dating and they leave Alfie out of everything. He ends up bored and starts hanging out with Dee Dee and his friends. However, it just isn't the same without Goo. Later on, Alfie learns about the surprise birthday party that Goo and Melanie had been planning with everyone else (except for Dee Dee, who couldn't know since he would've told).", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_003'}, {'text': 'January 19, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_011_004'}], [{'text': '12', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"Candy Sale"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_002'}, {'text': "Alfie and Goo are selling candy to make money for some expensive jackets, but they are not having any luck. However, when Dee Dee start helping them sell candy, they start to make money and asks him to help them out. Soon Goo and Alfie finds themselves confronted by Melanie, Deonne, Harry and Donnell for Dee Dee's share of the money. They soon learn the boys have used the money to buy three expensive jackets for themselves and Dee Dee as a token of their gratitude. They quickly apologize to Alfie and Goo for their quick judgment.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_003'}, {'text': 'January 26, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_012_004'}], [{'text': '13', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '1', 'colspan': 1, 'rowspan': 1, 'label_key': None}, {'text': '"The Big Bully"', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_002'}, {'text': "Dee Dee gets beat up at school and his friends try to teach him how to fight back. Goo, however, tells him to bluff, but the plan backfires and Dee Dee gets hit because of it. When Alfie confronts the bully, he learns that Dee Dee was picked on by a girl. Alfie and Goo decide to confront her. However, when some of their classmates, who happen to be the girls' siblings, learn they are bullying their sister, they intervene.", 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_003'}, {'text': 'February 2, 1995', 'colspan': 1, 'rowspan': 1, 'label_key': 'cell_013_004'}]], 'layout': {'width': 1500, 'padding': 45, 'cell_width': 282.0, 'font_size': 27, 'heights': [106, 2875, 2212, 2563, 1393, 1198, 1471, 1159, 2602, 1198, 2953, 1354, 2017, 1549]}}
LABELS = {'cell_000_000': 'سیریز نمبر', 'cell_000_001': 'سیزن نمبر', 'cell_000_002': 'عنوان', 'cell_000_003': 'نوٹس', 'cell_000_004': 'پہلی نشریات کی تاریخ', 'cell_001_002': '"خیرات"', 'cell_001_003': 'ایلفی، ڈی ڈی اور میلنی کو میلے میں لوگوں کو پانی میں گرانے والے اسٹال پر کام کرکے اپنے والدین کی مدد کرنی ہوتی ہے۔ جب گو آکر بتاتا ہے کہ ان کا پسندیدہ باسکٹ بال کھلاڑی کینڈل گل کامک بکس کی دکان پر آٹوگراف دے رہا ہے تو لڑکے میلے سے کھسکنے کا فیصلہ کرتے ہیں۔ یوں میلنی اور جینیفر کو اسٹال سنبھالنا پڑتا ہے اور دونوں بھیگ جاتی ہیں۔ مگر کامک بکس کی دکان کھچا کھچ بھری ہوتی ہے، اور ایلفی اور ڈی ڈی کو یہ جان کر بڑی حیرت ہوتی ہے کہ ان کے والد کو کینڈل گل کا انٹرویو لینا ہے۔ گو ایک منصوبہ بناتا ہے تاکہ ایلفی اور ڈی ڈی کو مقامی میلے میں واپس پہنچانے سے پہلے گل کا آٹوگراف دلوا سکے، مگر راجر انہیں پکڑ لیتا ہے۔ آخر میں ایلفی اور گو کے سوا سب کے لیے سب کچھ اچھا رہتا ہے؛ ان دونوں کو پانی میں گرانے والے اسٹال پر بھیگنا پڑتا ہے۔', 'cell_001_004': 'اکتوبر 15, 1994', 'cell_002_002': '"شرارتوں کی جنگ"', 'cell_002_003': 'ایلفی اور گو، ڈی ڈی اور اس کے دوستوں کے ساتھ سخت عملی شرارتیں کرتے ہیں۔ ڈی ڈی، ہیری اور ڈونل بدلہ لینے کے لیے ایلفی کے ساتھ شرارتی چیونگم والی چال چلتے ہیں۔ جب ایلفی اور گو، ڈی ڈی اور اس کے دوستوں سے بدلہ لیتے ہیں تو میلنی اور ڈیون ان کی مدد کرتی ہیں۔ جلد ہی ایلفی اور گو، میلنی، ڈی ڈی اور ان کے دوستوں کے خلاف شرارتوں کی جنگ چھیڑ دیتے ہیں۔ یہ سلسلہ تب رکتا ہے جب ایک رسالے کے سال کے بہترین خاندان کے مقابلے میں فاتح قرار دیے جانے کے بعد راجر اور جینیفر بھی اس جنگ کی شرارتوں کا نشانہ بن جاتے ہیں۔ وہ اپنے بچوں کو ان کے رویے پر ڈانٹتے ہیں اور ان کے دوستوں کے والدین سے بھی بات کرنے کا ارادہ کرتے ہیں۔', 'cell_002_004': 'اکتوبر 22, 1994', 'cell_003_002': '"وہ اختتامِ ہفتہ جب آنٹی ہیلن آئیں"', 'cell_003_003': 'لڑکوں کی ماں جینیفر اختتامِ ہفتہ پر باہر جاتی ہے اور ان کے والد راجر کو گھر کی ذمہ داری سونپتی ہے۔ مگر وہ بچوں کو کھلی چھوٹ دے دیتا ہے۔ پھر ایلفی اور ڈی ڈی کی آنٹی ہیلن، جینیفر کی واپسی تک گھر سنبھالنے آتی ہیں۔ اسی دوران ایلفی، گو کی طرف باسکٹ بال پھینکتا ہے جو اس کے سر پر لگتی ہے اور وہ عارضی طور پر یادداشت کھو بیٹھتا ہے۔ یادداشت کھونے پر گو کتابی کیڑے کی طرح برتاؤ کرتا ہے، اختتامِ ہفتہ پر ہوم ورک کرتا ہے، چاہتا ہے کہ اسے گو کے بجائے ملٹن کہا جائے، اور ایلفی کو بھی ایلفرڈ کہتا ہے۔ وہ ڈیون اور ڈی ڈی کے ساتھ بہت اچھا برتاؤ کرتا ہے، مگر میلنی سے قدرے بدتمیزی کرتا ہے۔ اسے پہلے جیسا کرنے کا واحد طریقہ سر پر ایک اور چوٹ لگنا ہے۔', 'cell_003_004': 'نومبر 1, 1994', 'cell_004_002': '"رابن ہڈ کا ڈراما"', 'cell_004_003': 'ایلفی کے اسکول میں رابن ہڈ کا ڈراما پیش کیا جا رہا ہے اور ایلفی کو رابن ہڈ کا کردار ادا کرنے کے لیے چنا جاتا ہے۔ ایلفی اس موقع پر بہت خوش ہے، مگر وہ چست پاجامہ نہیں پہننا چاہتا کیونکہ اس کے خیال میں یہ لڑکیوں کا لباس ہے۔ تاہم، جب ڈی ڈی اسے سمجھ داری سے کہتا ہے کہ اسے اس بات کو رابن ہڈ کی حیثیت سے اپنی اداکاری پر اثرانداز نہیں ہونے دینا چاہیے تو وہ اپنی رائے پر دوبارہ غور کرتا ہے۔', 'cell_004_004': 'نومبر 9, 1994', 'cell_005_002': '"باسکٹ بال ٹیم کے انتخابی آزمائشی مقابلے"', 'cell_005_003': 'ایلفی باسکٹ بال ٹیم میں انتخاب کے لیے آزمائشی مقابلے میں حصہ لیتا ہے، مگر اپنی مہارت دکھانے کے باوجود منتخب نہیں ہوتا۔ تاہم ہیری، ڈی ڈی اور ڈونل ٹیم میں جگہ بنا لیتے ہیں۔ ایلفی مایوس ہو جاتا ہے اور جشن کی تقریب میں نہیں جانا چاہتا۔ مگر گو اسے سمجھاتا ہے کہ یہ اسی کی غلطی تھی، کیونکہ وہ ٹیم کے ساتھ مل کر کھیلنے کے بجائے گیند اپنے پاس ہی رکھتا رہا۔', 'cell_005_004': 'نومبر 30, 1994', 'cell_006_002': '"سانپ کہاں ہے؟"', 'cell_006_003': 'ڈی ڈی ایک سانپ لاتا ہے، مگر وہ نہیں چاہتا کہ اس کے والدین کو اس کا پتا چلے۔ تاہم، جب سانپ گھر میں گم ہو جاتا ہے تو معاملہ پیچیدہ ہو جاتا ہے۔ اسی دوران میلنی اور ڈیون کی استانی انہیں اختتامِ ہفتہ پر اپنی پیاری پالتو خرگوش ڈچس کی دیکھ بھال کی ذمہ داری دیتی ہے۔ جب ایلفی اور ڈی ڈی کو گو سے پتا چلتا ہے کہ سانپ خرگوش کھاتے ہیں تو دونوں ڈچس کے لیے فکر مند ہو جاتے ہیں۔', 'cell_006_004': 'دسمبر 6, 1994', 'cell_007_002': '"ڈی ڈی کی گرل فرینڈ"', 'cell_007_003': 'ایک لڑکی ہیری اور ڈونل کے سامنے ڈی ڈی کو بوسہ دیتی ہے۔ وہ کسی کو نہ بتانے کا وعدہ کرتے ہیں، مگر ان کے منہ سے بات نکل جاتی ہے اور سب ڈی ڈی پر ہنستے ہیں۔ ڈی ڈی، ہیری اور ڈونل سے دوستی توڑ دیتا ہے اور ایلفی اور گو کے ساتھ وقت گزارنے لگتا ہے۔ آخرکار ایلفی اور گو ان تینوں کو ایک دوسرے سے بات کرنے پر آمادہ کر لیتے ہیں۔', 'cell_007_004': 'دسمبر 15, 1994', 'cell_008_002': '"ڈی ڈی کے بالوں کی کٹائی"', 'cell_008_003': 'ڈی ڈی کول ڈاکٹر منی سے بال کٹوانا چاہتا ہے اور سر کے بال اس طرح منڈوانا چاہتا ہے کہ اس کا نام بن جائے۔ اس کے والدین اجازت نہیں دیتے، مگر گو پانچ ڈالر میں یہ کام کرنے کی پیشکش کرتا ہے۔ تاہم، جب گو ڈی ڈی کے بال خراب کر دیتا ہے اور اس کا نام بھی غلط لکھ دیتا ہے تو والدین کو حقیقت معلوم ہو جاتی ہے اور ڈی ڈی کو سر منڈوانا پڑتا ہے۔ اس کے علاوہ، اس کے دوست گنجے سر پر اسے چھیڑتے ہیں، جس سے لڑکوں میں جھگڑا ہو جاتا ہے اور گو اور ایلفی بھی اس میں شامل ہو جاتے ہیں۔ ضمنی کہانی میں ایلفی اور گو، ڈی ڈی کے ساتھ ہلاپینو مرچ والی لالی پاپ کی شرارت کرنے کی کوشش کرتے ہیں۔ یہ الٹی پڑ جاتی ہے جب راجر انجانے میں اس کا شکار ہو جاتا ہے اور لڑکوں کو پکڑنے کے لیے ان کے پیچھے دوڑتا ہے۔', 'cell_008_004': 'دسمبر 20, 1994', 'cell_009_002': '"ڈی ڈی گھر سے بھاگ جاتا ہے"', 'cell_009_003': 'ڈی ڈی پورا ہفتہ مونسٹر ٹرک شو میں جانے کا منتظر رہا ہے۔ مگر ایلفی اور گو کی بیس بال ٹیم ٹورنامنٹ میں پہنچ جاتی ہے اور سب مونسٹر ٹرک شو بھول جاتے ہیں۔ ڈی ڈی خود کو نظرانداز محسوس کرتا ہے اور ہیری اور ڈونل کے ساتھ گھر سے بھاگ جاتا ہے۔ اب ایلفی اور گو کو اسے گھر واپس آنے پر آمادہ کرنے کی کوشش کرنی ہے۔', 'cell_009_004': 'دسمبر 28, 1994', 'cell_010_002': '\'"ڈونل کی سالگرہ کی تقریب"', 'cell_010_003': 'ڈونل اپنی سالگرہ کی تقریب دے رہا ہے اور شیخی بگھارتا ہے کہ وہاں خوب رقص ہوگا اور بڑے زبردست لوگ آئیں گے۔ ہیری کہتا ہے کہ اسے رقص کرنا آتا ہے، اس لیے ڈی ڈی خود کو الگ تھلگ محسوس کرتا ہے کیونکہ اسے رقص نہیں آتا۔ بعد میں ہیری تنہائی میں ڈی ڈی سے اعتراف کرتا ہے کہ اسے بھی رقص نہیں آتا اور اس نے صرف اس لیے جھوٹ بولا تھا کہ ڈونل اسے نہ چھیڑے۔ چنانچہ وہ ایلفی سے رقص سکھانے میں مدد مانگتے ہیں۔ وہ انکار کر دیتا ہے، کیونکہ ڈی ڈی نے پہلے راجر کو بتا دیا تھا کہ ایلفی اور گو ریاضی کے مختصر امتحان میں نقل کرنے کا منصوبہ بنا رہے ہیں۔ آخرکار ایلفی مان جاتا ہے، جب میلنی دھمکی دیتی ہے کہ وہ ریاضی کے ہوم ورک میں اس کی مدد نہیں کرے گی۔ جلد ہی ڈی ڈی اور ہیری کو ڈونل کا راز معلوم ہو جاتا ہے اور انہیں اسے بھی رقص سکھانا پڑتا ہے۔ تقریب کے بعد ڈی ڈی، ایلفی کو یہ بات بتاتا ہے اور اسے پتا چلتا ہے کہ ایلفی پہلے ہی جانتا تھا کہ ڈونل جھوٹ بول رہا ہے۔', 'cell_010_004': 'جنوری 5, 1995', 'cell_011_002': '"ایلفی کی سالگرہ کی تقریب"', 'cell_011_003': 'گو اور میلنی ایک دوسرے سے رومانوی تعلق کا ڈراما کرتے ہیں اور ایلفی کو ہر بات سے الگ رکھتے ہیں۔ وہ اکتا جاتا ہے اور ڈی ڈی اور اس کے دوستوں کے ساتھ وقت گزارنے لگتا ہے۔ مگر گو کے بغیر وہی مزہ نہیں آتا۔ بعد میں ایلفی کو سالگرہ کی اس اچانک تقریب کا پتا چلتا ہے جس کی تیاری گو اور میلنی باقی سب کے ساتھ مل کر کر رہے تھے، سوائے ڈی ڈی کے، جسے نہیں بتایا جا سکتا تھا کیونکہ وہ راز کھول دیتا۔', 'cell_011_004': 'جنوری 19, 1995', 'cell_012_002': '"ٹافیوں کی فروخت"', 'cell_012_003': 'ایلفی اور گو کچھ مہنگی جیکٹیں خریدنے کے لیے ٹافیاں بیچ کر پیسے جمع کر رہے ہیں، مگر انہیں کامیابی نہیں مل رہی۔ تاہم، جب ڈی ڈی ٹافیاں بیچنے میں ان کی مدد کرنے لگتا ہے تو ان کی کمائی شروع ہو جاتی ہے اور وہ اس سے مدد جاری رکھنے کو کہتے ہیں۔ جلد ہی میلنی، ڈیون، ہیری اور ڈونل، ڈی ڈی کے حصے کے پیسوں کے بارے میں گو اور ایلفی سے جواب طلب کرتے ہیں۔ انہیں جلد پتا چلتا ہے کہ لڑکوں نے ان پیسوں سے اپنے لیے اور شکریے کے طور پر ڈی ڈی کے لیے تین مہنگی جیکٹیں خریدی ہیں۔ وہ جلدبازی میں رائے قائم کرنے پر فوراً ایلفی اور گو سے معافی مانگتے ہیں۔', 'cell_012_004': 'جنوری 26, 1995', 'cell_013_002': '"بڑا غنڈا"', 'cell_013_003': 'اسکول میں ڈی ڈی کی پٹائی ہوتی ہے اور اس کے دوست اسے جواب میں لڑنا سکھانے کی کوشش کرتے ہیں۔ مگر گو اسے جھوٹی دھونس جمانے کا مشورہ دیتا ہے۔ یہ منصوبہ الٹا پڑتا ہے اور ڈی ڈی کو اسی وجہ سے مار کھانی پڑتی ہے۔ جب ایلفی تنگ کرنے والے کا سامنا کرتا ہے تو اسے معلوم ہوتا ہے کہ ڈی ڈی کو ایک لڑکی نے ستایا تھا۔ ایلفی اور گو اس کا سامنا کرنے کا فیصلہ کرتے ہیں۔ تاہم، جب ان کے کچھ ہم جماعتوں کو، جو اس لڑکی کے بہن بھائی ہیں، پتا چلتا ہے کہ یہ دونوں ان کی بہن کو دھمکا رہے ہیں تو وہ بیچ میں آ جاتے ہیں۔', 'cell_013_004': 'فروری 2, 1995'}
(image, boxes) = render(DATA, LABELS)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output, optimize=True)
output.with_suffix('.layout.json').write_text(json.dumps({'boxes': boxes, 'missing_glyphs': sorted(set(MISSING_GLYPHS))}, ensure_ascii=False, indent=2))
