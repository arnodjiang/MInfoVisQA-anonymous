import json, re, hashlib
from pathlib import Path
from recover_visual_specs import OUT, ROOT
specs = json.loads((OUT / 'specs.json').read_text())
cases = json.loads((OUT / 'source_cases.json').read_text())
manifest = json.loads((OUT / 'image_manifest.json').read_text())
records = [json.loads(x) for x in (OUT / 'benchmark.jsonl').read_text().splitlines()]
locales = {p.stem: json.loads(p.read_text()) for p in (OUT / 'locales').glob('*.json') if not p.name.endswith('.raw.json')}
assert len({x['variant_id'] for x in records}) == 155
for r in records:
    assert (OUT / r['image']).exists()
    assert '[[' not in r['question']
    if r['visual_language'] == r['query_language']:
        for key in r['reference_label_keys']:
            assert locales[r['visual_language']]['labels'][key] in r['question']
for c in cases:
    if not c['table']['rows']:
        continue
    for (a, b) in zip(c['table']['rows'], specs[c['id']]['rows']):
        assert len(a) == len(b)
        for (x, y) in zip(a, b):
            assert x['text'] == y['text'] and x['colspan'] == y['colspan'] and (x['rowspan'] == y['rowspan'])
code = OUT / 'code'
code.mkdir(exist_ok=True)
for (id, s) in specs.items():
    if s['kind'] == 'table':
        continue
    fn = {'multipanel_line': 'multipanel', 'heatmap': 'heatmap', 'dual_axis_combo': 'combo'}[s['kind']]
    (code / (id + '.py')).write_text(f'''"""Recovered chart. Usage: .venv/bin/python {{__file__}} zh. Numeric data is in ../specs.json."""\nimport json,sys\nfrom pathlib import Path\nHERE=Path(__file__).resolve(); OUT=HERE.parent.parent\nsys.path.insert(0,str(OUT.parents[2]/'scripts'))\nfrom render_visual_benchmark import {fn}\nlang=sys.argv[1] if len(sys.argv)>1 else 'en'\nspec=json.loads((OUT/'specs.json').read_text())[\'{id}']\nlabels=json.loads((OUT/'locales'/f'{{lang}}.json').read_text())['labels']\nimage,boxes={fn}(spec,labels)\npath=OUT/'images'/lang/\'{id}.png';path.parent.mkdir(parents=True,exist_ok=True);image.save(path,optimize=True)\nprint(path)\n''')
v = json.loads((OUT / 'validation.json').read_text())
v.update(unique_variant_ids=True, same_language_reference_bindings_valid=True, original_table_cells_and_spans_preserved=True)
v['answer_checks']['e525de67b11a063bafc6']['official_acceptance'] = 'pending; pilot tolerance must not be used as official scoring rule'
(OUT / 'validation.json').write_text(json.dumps(v, ensure_ascii=False, indent=2))
(OUT / 'README.md').write_text('# 多语言图像构造试验（5 × 11）\n\n图像、QA 和溯源信息分别保存在下列文件中。\n\n- images/：55 张 PNG；英语重绘基线 + 中文、日语、韩语、法语、德语、西班牙语、葡萄牙语、俄语、阿拉伯语、印地语。\n- specs.json：三张图的 Python 绘图数据、像素校准与两张表的原始单元格。图形数值不随语言改变。\n- code/：每张 Chart 的可执行 Python 入口；共同绘图实现位于项目 scripts/render_visual_benchmark.py。\n- locales/：稳定标签键到目标语言文本的映射、带引用占位符的 QA、10 次非英语翻译的原始响应和请求指纹。英语直接保留，无额外调用。\n- source_cases.json：原始 QA、原始 HTML/Markdown 字符、上游版本 / 文件哈希 / 行号 / 槽位 / 稳定 base_id。\n- image_manifest.json：图片 SHA256、数据规格 SHA256、语言、文字边界与恢复方式。\n- benchmark.jsonl：155 个唯一配置（每个 base：11 个同语 + 10 个中文跨语 + 10 个英文跨语），不是155个独立问题。相同 base_id 的多语变体必须放在同一数据划分，防止泄漏。\n- validation.json：数值、字体、边界、引用绑定及身份检查。\n\n## 复现\n\n在项目根目录执行：\n\n```sh\n.venv/bin/python scripts/render_visual_benchmark.py\n.venv/bin/python scripts/finalize_visual_pilot.py\n```\n\n已有翻译缓存，以上命令不调用 API。首次恢复与翻译分别由 recover_visual_specs.py 和 localize_visual_specs.py 完成。当前恢复器为这三张图的具体校准实现，不是任意图像通用逆向工具。新图需新增校准规格并审核。\n\n文字渲染使用 HarfBuzz + FreeType，支持阿拉伯文连写与天城文字形；字体路径在 multilingual_drawing.py 中，当前使用 macOS Arial Unicode。跨机器复现需配置同等覆盖字体。数值使用原始阿拉伯数字，表格列序不因语言方向变化。\n\n## 精度与准入\n\n热图保留原图15个标注数值；两张表保留原始数值和合并单元格。折线图通过颜色与像素坐标提取，存在遮挡、近似颜色导致的曲线串线风险；双轴图数值由手工选取像素坐标估计，并非原始数据。英文基线与所有语言共享同一份恢复数值，保证语言间比较一致，但不代表恢复准确。\n\nCharXiv hopper:stand 的10小时重建读数约716.049，原答案750。±50仅用于本次试验诊断，不是上游评测规则，正式入库前须复核曲线并按正式准则判断答案有效性，不应修改数值以迎合答案。其余四题的数值关系检查通过。所有候选仍标记 candidate_requires_visual_semantic_review；未声称11种语言经过母语人工审核。\n\n文字键保持稳定，例如 task_hopper_stand 在中文图与中文问题中均绑定“单腿机器人：站立”，原始英文标识仍保留于源文件，便于扩充和追踪。跨语 QA 的标签引用是对应语义翻译，不要求和图中文字逐字相同。图像本身不包含 QA 答案或审核信息。\n')
print('Identity, source cells and label binding checks passed; reproduction entrypoints written.')
