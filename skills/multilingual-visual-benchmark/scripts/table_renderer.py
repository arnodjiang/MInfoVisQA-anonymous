pass
from PIL import Image, ImageDraw

def geometry(data):
    occupied = set()
    positioned = []
    ncols = 0
    for (row_i, row) in enumerate(data['rows']):
        col = 0
        for cell in row:
            while (row_i, col) in occupied:
                col += 1
            (rs, cs) = (cell.get('rowspan', 1), cell.get('colspan', 1))
            if not isinstance(rs, int) or not isinstance(cs, int) or min(rs, cs) < 1:
                raise ValueError('Spans must be positive integers')
            if row_i + rs > len(data['rows']):
                raise ValueError('Row span exceeds table')
            for r in range(row_i, row_i + rs):
                for c in range(col, col + cs):
                    if (r, c) in occupied:
                        raise ValueError('Overlapping table cells')
                    occupied.add((r, c))
            positioned.append((row_i, col, rs, cs, cell))
            col += cs
            ncols = max(ncols, col)
    if not ncols:
        raise ValueError('Empty table')
    if len(occupied) != len(data['rows']) * ncols:
        raise ValueError('Ragged table: supply explicit empty cells')
    return (positioned, ncols)

def layout(data, all_labels):
    (positioned, ncols) = geometry(data)
    width = max(1200, ncols * 260)
    padding = 40
    cw = (width - 2 * padding) / ncols
    size = 28
    heights = [65] * len(data['rows'])
    for labels in all_labels:
        for (row, col, rs, cs, cell) in positioned:
            text = labels[cell['label_key']] if cell.get('label_key') else cell['text']
            needed = len(wrap(text, cw * cs - 30, size)) * (size + 12) + 24
            deficit = max(0, needed - sum(heights[row:row + rs]))
            heights[row + rs - 1] += deficit
    return {'width': width, 'padding': padding, 'cell_width': cw, 'font_size': size, 'heights': heights}

def render(data, labels):
    (positioned, ncols) = geometry(data)
    cfg = data['layout']
    pad = cfg['padding']
    cw = cfg['cell_width']
    size = cfg['font_size']
    heights = cfg['heights']
    im = Image.new('RGB', (cfg['width'], round(sum(heights) + 2 * pad)), 'white')
    draw = ImageDraw.Draw(im)
    boxes = []
    for (row, col, rs, cs, cell) in positioned:
        x = pad + col * cw
        y = pad + sum(heights[:row])
        w = cs * cw
        h = sum(heights[row:row + rs])
        draw.rectangle((x, y, x + w, y + h), fill='#eaf0ec' if row == 0 else '#ffffff', outline='#9baea1', width=2)
        text = labels[cell['label_key']] if cell.get('label_key') else cell['text']
        lines = wrap(text, w - 30, size)
        start = y + h / 2 - (len(lines) - 1) * (size + 12) / 2
        for (j, line) in enumerate(lines):
            box = put(im, line, x + w / 2, start + j * (size + 12), size, max_width=w - 28)
            (a, b, c, d) = box['box']
            box['inside_cell'] = a >= x and b >= y and (c <= x + w) and (d <= y + h)
            boxes.append(box)
    return (im, boxes)
