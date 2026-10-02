pass
import argparse
import copy
import hashlib
import json
from pathlib import Path
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from .api import digest, read, save, now
from .pipeline import Builder, DEFAULT_OUT
from .prompts import CHART
DETECT = 'Describe the panel layout of this chart, not its numerical data. Return JSON only:\n{"regions":[{"id":"p00","bbox":[left,top,right,bottom],"description":"..."},...],\n "global_labels":{"title":"exact visible heading outside panels"},\n "global_placements":[{"key":"title","x":0.5,"y":0.025,"size":30,"max_width":0.9}],\n "notes":[]}.\nBounding boxes are normalized fractions of full image with origin TOP LEFT. Include\nevery chart panel, its full axis tick labels, panel title, and per-panel legend. Regions\nmust not overlap; shared outer heading/footnotes/legend text belongs in global_labels.\nGive enough margin around each panel so all its marks and words fit inside the crop.\nDo not omit small panels. Do not extract values or solve a question. Single charts\nwith many series can use one region. Do not duplicate header text in both global and\npanel regions. Numeric/scientific plot titles should also be retained as labels.\n'

def composite_code(panels):
    chunks = ['def render(data, labels):']
    for (i, panel) in enumerate(panels):
        code = panel['python_code'].replace('def render(', 'def panel_%02d(' % i, 1)
        chunks.append(textwrap.indent(code, '    '))
    chunks.append('    functions = [' + ', '.join(('panel_%02d' % i for i in range(len(panels)))) + ']')
    chunks.append("    width, height = data['canvas']\n    image = Image.new('RGB', (width, height), 'white')\n    boxes = []\n    for i, panel in enumerate(data['panels']):\n        local = {k: labels[v] for k, v in panel['label_map'].items()}\n        part, part_boxes = functions[i](panel['data'], local)\n        left, top, right, bottom = panel['bbox']\n        x, y = round(left * width), round(top * height)\n        w, h = round((right-left)*width), round((bottom-top)*height)\n        # If a local adapter places a label a few pixels beyond its panel, leave a\n        # measured right margin instead of silently clipping the label at composition.\n        max_right = max((box['box'][2] for box in part_boxes), default=part.width)\n        # Scale against the furthest reported glyph box, since an adapter can\n        # report coordinates beyond the local raster width.\n        effective_w = w\n        if max_right > 0:\n            effective_w = min(w, max(1, int(w * part.width / max_right) - 8))\n        sx, sy = effective_w/part.width, h/part.height\n        image.paste(part.resize((effective_w,h), Image.Resampling.LANCZOS), (x,y))\n        for box in part_boxes:\n            b = dict(box)\n            a,bb,c,d = b['box']\n            b['box'] = [round(x+a*sx),round(y+bb*sy),round(x+c*sx),round(y+d*sy)]\n            b['label_key'] = panel['label_map'].get(b.get('label_key'), b.get('label_key'))\n            b['effective_font_size'] = b.get('font_size',0)*min(sx,sy)\n            b['inside_canvas'] = b['box'][0]>=0 and b['box'][1]>=0 and b['box'][2]<=width and b['box'][3]<=height\n            boxes.append(b)\n    for p in data['global_placements']:\n        size = p.get('size',30)\n        max_width = min(p.get('max_width',0.8)*width, width-20)\n        text = labels[p['key']]\n        while mask(text,size).width > max_width and size > 10:size -= 1\n        if mask(text,size).width > width-20:text='\\n'.join(wrap(text,width-20,size))\n        half = mask(text,size).width/2\n        gx = min(max(p['x']*width,half+3),width-half-3)\n        b = put(image,text,gx,p['y']*height,size,max_width=max_width,anchor=p.get('anchor','center'))\n        b['label_key'] = p['key']\n        boxes.append(b)\n    return image, boxes")
    return '\n'.join(chunks) + '\n'

def recover(b, identifier):
    folder = b.folder(identifier)
    if (folder / 'recovery_complete.json').exists():
        return
    original = b.image(identifier)
    (layout, key) = b.api.call('panel_layout', identifier, DETECT, {'input': 'Locate all panels for a lossless structural crop plan.'}, image=original, max_tokens=4000)
    save(folder / 'panel_layout.json', layout)
    if not layout.get('regions'):
        raise ValueError('empty_layout')
    source = Image.open(original).convert('RGB')
    labels = dict(layout.get('global_labels', {}))
    parts = []
    numeric = []
    uncertainties = list(layout.get('notes', []))
    for (i, region) in enumerate(layout['regions']):
        bbox = region['bbox']
        (left, top, right, bottom) = map(float, bbox)
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError('invalid_crop_bbox')
        crop_path = folder / 'crops' / ('%02d.png' % i)
        crop_path.parent.mkdir(exist_ok=True)
        pixel_bbox = (round(left * source.width), round(top * source.height), round(right * source.width), round(bottom * source.height))
        if not crop_path.exists():
            source.crop(pixel_bbox).save(crop_path)
        prompt = CHART + '\nThis is one isolated panel crop. Return compact JSON and compact Python under 1800 output tokens. Use concise arrays and loops; preserve visible marks, axes, labels and uncertainty bands. Do not invent other panels.\n'
        (result, rkey) = b.api.call('panel_recovery_%02d' % i, identifier, prompt, {'panel_description': region.get('description', '')}, image=crop_path, max_tokens=9000)
        b.rt.validate_code(result['python_code'])
        mapping = {k: 'panel_%02d_' % i + k for k in result['labels']}
        labels.update({mapping[k]: v for (k, v) in result['labels'].items()})
        parts.append(result)
        numeric.append({'bbox': bbox, 'data': result['data'], 'label_map': mapping, 'crop_pixel_bbox': pixel_bbox, 'crop_sha256': hashlib.sha256(crop_path.read_bytes()).hexdigest(), 'api_request_sha256': rkey})
        uncertainties.extend(result['recovery'].get('uncertainties', []))
        print('panel recovered ' + identifier + ' ' + str(i + 1) + '/' + str(len(layout['regions'])), flush=True)
    width = 2800
    height = round(width * source.height / source.width)
    spec = {'kind': 'chart', 'id': identifier, 'base_id': b.identities[identifier]['base_id'], 'source': b.by_id[identifier]['source'], 'labels': labels, 'data': {'canvas': [width, height], 'panels': numeric, 'global_placements': layout.get('global_placements', [])}, 'python_code': composite_code(parts), 'recovery': {'method': 'API_panel_layout_and_per_panel_python_reconstruction', 'panel_count': len(parts), 'fidelity_confidence': 'pending_review', 'uncertainties': uncertainties}, 'recovery_request_sha256': key}
    save(folder / 'reconstruction_by_panels.json', spec)
    save(folder / 'spec.json', spec)
    save(folder / 'english_labels.json', {'labels': labels})
    b.render_one(identifier, folder / 'spec.json', folder / 'english_labels.json', folder / 'baseline.png')
    save(folder / 'recovery_complete.json', {'finished_at': now(), 'spec_sha256': digest(spec), 'mechanical_render_pass': True, 'semantic_fidelity': 'pending_api_review'})
    b.status()
    print('Split recovery complete ' + identifier, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ids', required=True)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    args.output = str(DEFAULT_OUT)
    b = Builder(args)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(recover, b, identifier): identifier for identifier in args.ids.split(',')}
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append({'id': identifier, 'error': str(exc)})
                print('Split recovery failed ' + identifier + ' ' + str(exc), flush=True)
    save(b.out / 'split_recovery_failures.json', failures)
if __name__ == '__main__':
    main()
