pass
import ast
import hashlib
import json
import pprint
from pathlib import Path
from recover_visual_specs import OUT, ROOT
from render_visual_benchmark import table_layout

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    specs = json.loads((OUT / 'specs.json').read_text())
    locales = {p.stem: json.loads(p.read_text()) for p in (OUT / 'locales').glob('*.json') if not p.name.endswith('.raw.json')}
    renderer = (ROOT / 'scripts/render_visual_benchmark.py').read_text()
    helper = (ROOT / 'scripts/multilingual_drawing.py').read_text()
    names = ['to_image', 'bounds', 'multipanel', 'heatmap', 'combo', 'table_image']
    funcs = '\n\n'.join((ast.get_source_segment(renderer, n) for n in ast.parse(renderer).body if isinstance(n, ast.FunctionDef) and n.name in names))
    funcs = funcs.replace('        from multilingual_drawing import mask\n', '')
    inputs = {'specs': specs, 'locales': locales, 'renderer': renderer, 'helper': helper}
    version = sha(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode())[:12]
    dest = OUT / 'code' / 'snapshots' / version
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    prefix = "# Standalone rendering snapshot: literal data and language labels below.\n# Usage: python THIS_FILE.py --output result.png [--font /path/to/Unicode.ttf]\nimport argparse, os\nfrom pathlib import Path\nparser=argparse.ArgumentParser()\nparser.add_argument('--output',default=str(Path(__file__).with_suffix('.png')))\nparser.add_argument('--font',default='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')\nargs=parser.parse_args()\nos.environ.setdefault('MPLCONFIGDIR',str(Path(args.output).resolve().parent/'.matplotlib_cache'))\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nfrom matplotlib.colors import TwoSlopeNorm\nfrom matplotlib.ticker import AutoMinorLocator\nfrom PIL import ImageDraw\n"
    helper = helper.replace("FONT='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'", 'FONT=args.font')
    for (lang, loc) in sorted(locales.items()):
        folder = dest / ('original_en' if lang == 'en' else 'translated/' + lang)
        folder.mkdir(parents=True, exist_ok=True)
        for (id, spec) in specs.items():
            layout = table_layout(spec, locales) if spec['kind'] == 'table' else None
            fn = {'multipanel_line': 'multipanel', 'heatmap': 'heatmap', 'dual_axis_combo': 'combo', 'table': 'table_image'}[spec['kind']]
            content = prefix + '\n' + helper + '\n' + funcs + '\n\n# Frozen source identity and data; shared across every language.\nSPEC = ' + pprint.pformat(spec, width=110, sort_dicts=False) + '\n\n# Literal figure text for this language.\nLABELS = ' + pprint.pformat(loc['labels'], width=110, sort_dicts=False) + '\nLAYOUT = ' + repr(layout) + '\n'
            content += f'\nimage,boxes={fn}(SPEC,LABELS' + (',LAYOUT' if layout else '') + ")\nassert all(b['inside_canvas'] for b in boxes)\nassert not MISSING_GLYPHS\npath=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)\nimage.save(path,optimize=True)\nprint(path)\n"
            path = folder / (id + '.py')
            data = content.encode()
            compile(content, str(path), 'exec')
            if path.exists():
                assert path.read_bytes() == data, 'Snapshot is immutable'
            else:
                path.write_bytes(data)
            manifest.append({'id': id, 'base_id': spec['base_id'], 'language': lang, 'kind': spec['kind'], 'code_path': str(path.relative_to(OUT)), 'code_sha256': sha(data), 'data_sha256': sha(json.dumps(spec, sort_keys=True).encode()), 'labels_sha256': sha(json.dumps(loc['labels'], ensure_ascii=False, sort_keys=True).encode()), 'image_path': f'images/{lang}/{id}.png'})
    (dest / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (OUT / 'code' / 'latest.json').write_text(json.dumps({'snapshot': version, 'directory': str(dest.relative_to(OUT)), 'files': len(manifest), 'manifest': str((dest / 'manifest.json').relative_to(OUT))}, indent=2))
    (OUT / 'code' / 'README.md').write_text(f'# 原始重绘与翻译代码存档\n\n当前快照：`snapshots/{version}/`，共55份可独立执行的Python代码（3张图和2张表 × 11语言）。\n\n- `original_en/`：5份英文重绘基线代码。\n- `translated/zh/` 等10个语言目录：50份翻译后代码。\n- 每份文件内嵌完整绘图实现、恢复数据、该语言的文字常量和固定表格布局，不依赖外部specs.json、locales或项目脚本。\n- 每个快照的manifest.json保存base_id、代码/数据/文字SHA256及对应图像路径。再次归档时输入变化会生成新快照，不覆盖旧代码。\n- “原始”指从图像恢复的英文重绘代码，并非上游作者的原始绘图源码。像素恢复误差仍适用。\n\n运行示例（项目根目录）：\n\n```sh\n.venv/bin/python data/visual_benchmark/pilot_5x11_v1/code/snapshots/{version}/translated/zh/e525de67b11a063bafc6.py --output /tmp/chart_zh.png\n```\n\n依赖：matplotlib、numpy、Pillow、uharfbuzz、freetype-py。默认使用macOS Arial Unicode字体；其他系统通过`--font`指定覆盖目标语言的字体。同环境同字体可复现当前PNG。\n\n归档命令：`.venv/bin/python scripts/archive_visual_code.py`，无需API调用。\n')
    print(dest)
if __name__ == '__main__':
    main()
