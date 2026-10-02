pass
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from translate_multilingual_pilot import OUT as TEXT_OUT, ROOT, SOURCE
OUT = ROOT / 'data/visual_benchmark/pilot_5x11_v1'
LINE_ID = 'e525de67b11a063bafc6'
HEAT_ID = 'e9862f967583ff461a6b'
COMBO_ID = '2807106cc9f8feac0ee1'

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

def line_spec(case, labels):
    im = np.asarray(Image.open(ROOT / case['original_image']).convert('RGB')).astype(float)
    x0 = [66, 321, 575]
    xscale = [(211 - 66) / 200, (495 - 321) / 20, (773 - 575) / 80]
    yzero = [115, 260, 404, 549, 693, 838, 982]
    yone = [34, 179, 323, 468, 612, 757, 901]
    titles = [['ball_in_cup:catch', 'cartpole:swingup_sparse', 'cheetah:run'], ['cartpole:balance', 'finger:turn_easy', 'finger:spin'], ['cartpole:balance_sparse', 'hopper:stand', 'finger:turn_hard'], ['cartpole:swingup', 'pendulum:swingup', 'fish:swim'], ['reacher:easy', 'point_mass:easy', 'fish:upright'], ['walker:stand', 'reacher:hard', 'swimmer:swimmer15'], ['walker:walk', 'swimmer:swimmer6', 'walker:run']]
    labels.update({'learner_walltime': 'Learner walltime', 'episode_return': 'Episode return', 'unit_minutes': 'minutes', 'unit_hours': 'hours', 'difficulty_trivial': 'trivial', 'difficulty_easy': 'easy', 'difficulty_medium': 'medium', 'actors_title': 'Number of Actors', 'series_single': 'Single process', 'series_2': 'Distributed: 2 actors', 'series_8': 'Distributed: 8 actors', 'series_16': 'Distributed: 16 actors'})
    prototypes = np.array([[180, 195, 222], [60, 130, 181], [45, 75, 160], [22, 38, 68]], float)
    panels = []
    for row in range(7):
        for col in range(3):
            key = 'task_' + titles[row][col].replace(':', '_')
            labels[key] = titles[row][col]
            left = x0[col]
            right = [279, 532, 786][col]
            top = yone[row] - 2
            bottom = yzero[row] + 1
            crop = im[top:bottom, left:right + 1]
            (red, green, blue) = (crop[:, :, 0], crop[:, :, 1], crop[:, :, 2])
            colorful = (blue - red > 18) & (blue - green > 3) & (red < 235)
            distances = np.linalg.norm(crop[:, :, None, :] - prototypes[None, None, :, :], axis=3)
            assignment = np.argmin(distances, axis=2)
            traces = []
            for series in range(4):
                mask = colorful & (assignment == series)
                xs = []
                ys = []
                pixel_ys = []
                for x in range(mask.shape[1]):
                    hits = np.flatnonzero(mask[:, x])
                    if hits.size:
                        py = float(np.median(hits)) + top
                        xs.append(round(x / xscale[col], 4))
                        ys.append(round((yzero[row] - py) / (yzero[row] - yone[row]) * 1000, 3))
                        pixel_ys.append(round(py, 2))
                if len(ys) > 3:
                    ys = np.asarray(median_filter(np.asarray(ys), size=3)).tolist()
                traces.append({'label_key': ['series_single', 'series_2', 'series_8', 'series_16'][series], 'color': ['#b4c3de', '#3c82b5', '#2d4ba0', '#162644'][series], 'x': xs, 'y': ys, 'observed_pixel_y': pixel_ys, 'recovery': 'color-mask column medians; JPEG approximate'})
            panels.append({'row': row, 'col': col, 'title_key': key, 'difficulty_key': ['difficulty_trivial', 'difficulty_easy', 'difficulty_medium'][col], 'xlim': [0, [300, 25, 85][col]], 'ylim': [0, 1050], 'xticks': [[0, 100, 200], [0, 5, 10, 15, 20], [0, 20, 40, 60, 80]][col], 'yticks': [0, 250, 500, 750, 1000], 'series': traces, 'calibration': {'pixel_x_zero': left, 'pixels_per_x_unit': xscale[col], 'pixel_y_zero': yzero[row], 'pixel_y_1000': yone[row]}})
    return {'kind': 'multipanel_line', 'rows': 7, 'cols': 3, 'panels': panels, 'recovery_method': 'deterministic pixel digitization, independently of QA answer', 'uncertainty': 'Color overlap and JPEG loss can merge or split the four series; not original experimental data.'}

def heat_spec(labels):
    names = ['Negative', 'Positive', 'Neutral', 'Aggregate', 'Returns', 'Volatility']
    keys = ['heat_' + x.lower() for x in names]
    labels.update(dict(zip(keys, names)))
    entries = [[1, 0, -0.38], [2, 0, -0.68], [2, 1, -0.42], [3, 0, -0.86], [3, 1, 0.73], [3, 2, 0.26], [4, 0, -0.45], [4, 1, 0.19], [4, 2, 0.29], [4, 3, 0.37], [5, 0, 0.25], [5, 1, 0.058], [5, 2, -0.29], [5, 3, -0.16], [5, 4, -0.087]]
    return {'kind': 'heatmap', 'label_keys': keys, 'entries': entries, 'vmin': -0.9, 'vmax': 1.0, 'recovery_method': 'manual transcription of all visible numeric annotations', 'uncertainty': 'No raw matrix was downloaded; signs/decimals were read from the provided raster.'}

def combo_spec(case, labels):
    labels.update({'combo_population_axis': 'Pop × millions', 'combo_attacks_axis': 'Attack cases', 'combo_total': 'Total attacks per decade', 'combo_rate': 'Attacks per million people', 'combo_population': 'Population growth'})
    centers = [134, 170, 206, 242, 278, 315, 351, 387, 423, 459, 495]
    diamond_y = [204, 195, 186, 182, 170, 153, 134, 115, 96, 79, 54]
    bar_top = [225, 208, 159, 129, 185, 184, 189, 184, 175, 140, 59]
    triangle_y = [221, 207, 171, 153, 204, 211, 210, 221, 220, 211, 196]

    def convert(values, top):
        return [round((242 - y) * top / 214, 3) for y in values]
    return {'kind': 'dual_axis_combo', 'categories': [f'{1900 + 10 * i}-{1909 + 10 * i}' for i in range(11)], 'population': convert(diamond_y, 25), 'attacks': convert(bar_top, 140), 'rate': convert(triangle_y, 25), 'left_ylim': [0, 25], 'right_ylim': [0, 140], 'pixel_evidence': {'x_centers': centers, 'population_y': diamond_y, 'bar_top_y': bar_top, 'rate_y': triangle_y, 'y_zero': 242, 'y_top': 28}, 'axis_assignment': {'population': 'left', 'attacks': 'right', 'rate': 'left'}, 'recovery_method': 'manual mark coordinates transformed through visible axis calibration', 'uncertainty': 'Estimated values from pixels; attack-rate axis assignment follows per-population units and plotted height.'}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads((TEXT_OUT / 'source_cases.json').read_text())
    labels = {}
    specs = {}
    for case in source:
        if case['id'] == LINE_ID:
            spec = line_spec(case, labels)
        elif case['id'] == HEAT_ID:
            spec = heat_spec(labels)
        elif case['id'] == COMBO_ID:
            spec = combo_spec(case, labels)
        else:
            rows = case['table']['rows']
            for row in rows:
                for cell in row:
                    if cell['text_id']:
                        key = 'table_' + case['id'] + '_' + cell['text_id']
                        labels[key] = cell['text']
                        cell['label_key'] = key
                    else:
                        cell['label_key'] = None
            spec = {'kind': 'table', 'rows': rows, 'original_raw': case['table']['original_raw'], 'source_format': case['table']['source_format'], 'recovery_method': 'original structured source; no numeric OCR'}
        spec.update(id=case['id'], base_id=case['base_id'], source=case['source'], original_image=case['original_image'])
        specs[case['id']] = spec
    qas = {c['id']: {'question': c['question_en'], 'answer': c['answer_en']} for c in source}
    qas[LINE_ID]['question'] = 'For the subplot labeled [[task_hopper_stand]], what is the maximum episode return approximately achieved at 10 hours?'
    qas[HEAT_ID]['question'] = 'What is the greatest correlation value between [[heat_aggregate]] and another label?'
    tableid = '3cb6452a2a243405fae2'
    textkey = next((k for (k, v) in labels.items() if k.startswith('table_' + tableid) and v.strip() == 'Diapers'))
    qas[tableid]['question'] = 'How much did the [[%s]] category make in 2002?' % textkey
    mmtuid = '89a32f0298031a49cbe3'
    textkey = next((k for (k, v) in labels.items() if k.startswith('table_' + mmtuid) and v.strip() == 'Poland and Danzig'))
    qas[mmtuid]['question'] = 'What is the total increase in German imports from [[%s]] between 1934 and 1939?' % textkey
    save('specs.json', specs)
    save('labels_en.json', labels)
    save('qa_templates_en.json', qas)
    save('source_cases.json', source)
    hashes = {id: hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for (id, s) in specs.items()}
    save('data_hashes.json', hashes)
    print(f'Recovered {len(specs)} specs; {len(labels)} shared translation keys.')
if __name__ == '__main__':
    main()
