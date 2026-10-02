pass
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/minfovisqa-mpl')
import json, collections, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
p = ROOT / 'data/benchmark/validation_release/val.candidates.jsonl'
rows = [json.loads(l) for l in p.open()]
seeds = [r for r in rows if r['image_language'] == r['query_language'] == 'en']
counts = collections.Counter((r['visual_kind'] for r in seeds))
items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
assert len(seeds) == 128 and len(items) == 32
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
navy = '#154677'
sand = '#E5C687'
SIZE = (3.4, 3.55)
family_counts = {family: collections.Counter((r['visual_kind'] for r in rows if r['visual_family'] == family)) for family in ['chart', 'table']}
subgroups = {}
for (family, counter) in family_counts.items():
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    subgroups[family] = ordered[:4]
    if len(ordered) > 4:
        subgroups[family].append(('Other', sum((n for (_, n) in ordered[4:]))))
fig = plt.figure(figsize=SIZE)
fig.text(0.5, 0.975, '(a) QA configurations by visual type', ha='center', va='top', fontsize=10, fontweight='bold', color=navy)
plot_h = 0.58
plot_w = plot_h * SIZE[1] / SIZE[0]
ax = fig.add_axes([(1 - plot_w) / 2, 0.27, plot_w, plot_h])
ax.set_aspect('equal', adjustable='box', anchor='C')
(outer, _) = ax.pie([sum(family_counts[f].values()) for f in ['chart', 'table']], radius=1, startangle=90, counterclock=False, colors=[navy, sand], wedgeprops={'width': 0.25, 'edgecolor': 'white', 'linewidth': 1})
for (w, label, n, color) in zip(outer, ['Chart', 'Table'], [sum(family_counts[f].values()) for f in ['chart', 'table']], ['white', navy]):
    angle = np.deg2rad((w.theta1 + w.theta2) / 2)
    ax.text(0.875 * np.cos(angle), 0.875 * np.sin(angle), f'{label}\n{n:,}', ha='center', va='center', fontsize=8, fontweight='bold', color=color, rotation=(np.rad2deg(angle) + 90 + 90) % 180 - 90)
blue = ['#497BA5', '#7196B8', '#96B3CB', '#B6CBDB', '#D1DEE8', '#E4EBF0']
gold = ['#BDA574', '#D1B780', '#E0C998', '#EBDDAD', '#F4EBD4']
subitems = subgroups['chart'] + subgroups['table']
colors = blue[:len(subgroups['chart'])] + gold[:len(subgroups['table'])]
(inner, _) = ax.pie([n for (_, n) in subitems], radius=0.74, startangle=90, counterclock=False, colors=colors, wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 0.65})
for (w, (_, n)) in zip(inner, subitems):
    if n >= 350:
        angle = np.deg2rad((w.theta1 + w.theta2) / 2)
        ax.text(0.535 * np.cos(angle), 0.535 * np.sin(angle), f'{n:,}', ha='center', va='center', fontsize=9, color=navy)
ax.set_xlim(-1.03, 1.03)
ax.set_ylim(-1.03, 1.03)
for (col, family) in enumerate(['chart', 'table']):
    x = 0.015 + col * 0.51
    for (i, ((name, n), color)) in enumerate(zip(subgroups[family], blue if family == 'chart' else gold)):
        y = 0.215 - i * 0.04
        if name == 'Line Graph with Uncertainty Bands':
            name = 'Line + uncertainty bands'
        fig.patches.append(plt.Rectangle((x, y - 0.008), 0.018, 0.018, transform=fig.transFigure, facecolor=color, edgecolor='none'))
        fig.text(x + 0.026, y, f'{name} ({n:,})', fontsize=6.5, va='center', linespacing=0.92)
fig.savefig(OUT / 'visual_type_distribution_qa.pdf')
fig.savefig(OUT / 'visual_type_distribution_qa.png', dpi=220)
plt.close(fig)
langs = 'EN ZH JA KO FR DE ES PT RU AR HI IT NL PL TR VI ID TH SW FA UR BN TA TE'.split()
pairs = collections.Counter(((r['query_language'].upper(), r['image_language'].upper()) for r in rows))
assert len(pairs) == 70 and set(pairs.values()) == {128}
a = np.zeros((24, 24))
for (i, q) in enumerate(langs):
    for (j, v) in enumerate(langs):
        if (q, v) in pairs:
            a[i, j] = 1 if q == v else 2
fig = plt.figure(figsize=SIZE)
fig.text(0.5, 0.975, '(b) Language-pair coverage', ha='center', va='top', fontsize=10, fontweight='bold', color=navy)
plot_h = 0.58
plot_w = plot_h * SIZE[1] / SIZE[0]
ax = fig.add_axes([(1 - plot_w) / 2, 0.27, plot_w, plot_h])
ax.imshow(a, cmap=ListedColormap(['#F1F3F5', navy, sand]), vmin=0, vmax=2, interpolation='none')
ax.set_xticks(range(24), langs, rotation=90, fontsize=6)
ax.set_yticks(range(24), langs, fontsize=6)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('Visual language', labelpad=5, fontsize=7)
ax.set_ylabel('Question / answer language', labelpad=4, fontsize=7)
ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.45)
ax.tick_params(which='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.legend(handles=[Patch(facecolor=navy, label='LQA (24)'), Patch(facecolor=sand, label='XQA (46)'), Patch(facecolor='#F1F3F5', label='Not included')], loc='lower center', bbox_to_anchor=(0.5, 0.145), ncol=3, frameon=False, fontsize=6.5, handlelength=1, columnspacing=0.8)
fig.savefig(OUT / 'language_pair_coverage_v2.pdf')
fig.savefig(OUT / 'language_pair_coverage_v2.png', dpi=220)
plt.close(fig)
(OUT / 'figure_counts.json').write_text(json.dumps({'source': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'visual_types': dict(items), 'display_count_unit': 'QA configurations', 'display_subgroups': subgroups, 'family_totals': {f: sum(c.values()) for (f, c) in family_counts.items()}, 'top_four_tie_break': 'count descending, then category name ascending', 'seeds': 128, 'configurations_per_seed': 70}, indent=2))
print(OUT)
