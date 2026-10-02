pass
from functools import lru_cache
import freetype
import numpy as np
import uharfbuzz as hb
from PIL import Image
FONT = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
FONT_BYTES = open(FONT, 'rb').read()
HB_FACE = hb.Face(FONT_BYTES)
MISSING_GLYPHS = []

@lru_cache(maxsize=4096)
def mask(text, size):
    size = int(size)
    if not str(text):
        return Image.new('L', (1, max(1, size)), 0)
    font = hb.Font(HB_FACE)
    font.scale = (size * 64, size * 64)
    hb.ot_font_set_funcs(font)
    buf = hb.Buffer()
    buf.add_str(str(text))
    buf.guess_segment_properties()
    hb.shape(font, buf)
    ft = freetype.Face(FONT)
    ft.set_char_size(size * 64)
    parts = []
    penx = 0
    peny = 0
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

def wrap(text, max_width, size):
    text = str(text).strip()
    if not text:
        return ['']
    words = text.split(' ') if ' ' in text else list(text)
    join = ' ' if ' ' in text else ''
    lines = []
    current = ''
    for word in words:
        candidate = current + join + word if current else word
        if current and mask(candidate, size).width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
