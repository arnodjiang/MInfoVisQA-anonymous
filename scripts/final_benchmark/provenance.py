pass
import hashlib
import ast
import re
from pathlib import Path
from .api import digest, read

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_binding(folder, candidate=None, identity=None):
    folder = Path(folder)
    source = read(folder / 'source.json')
    (c, i) = (source['candidate'], source['identity'])
    if c['id'] != folder.name or i['id'] != folder.name:
        raise ValueError('source_case_id_mismatch:' + folder.name)
    if candidate is not None and c != candidate:
        raise ValueError('source_candidate_changed:' + folder.name)
    if identity is not None and {k: v for (k, v) in i.items() if k != 'membership'} != {k: v for (k, v) in identity.items() if k != 'membership'}:
        raise ValueError('source_identity_changed:' + folder.name)
    p = i['provenance']
    if any((c[k] != p[k] for k in ('file', 'row_index', 'slot'))) or c['source'] != i['source']:
        raise ValueError('source_locator_mismatch:' + folder.name)
    if c['question'] != i['question']:
        raise ValueError('source_query_mismatch:' + folder.name)
    originals = sorted((folder / 'original').glob('original.*'))
    expected = i['fingerprints'].get('image_sha256')
    if expected and (len(originals) != 1 or sha256(originals[0]) != expected):
        raise ValueError('original_image_mismatch:' + folder.name)
    if not expected and originals:
        raise ValueError('unexpected_original_image:' + folder.name)
    return {'case_id': c['id'], 'base_id': i['base_id'], 'source': c['source'], 'file': c['file'], 'row_index': c['row_index'], 'slot': c['slot'], 'original_image_sha256': expected, 'raw_record_sha256': digest(c['raw_record']), 'source_question_sha256': digest(c['question']), 'source_answer_sha256': digest(c['answer'])}

def verify_spec(spec, binding):
    for (key, value) in [('id', binding['case_id']), ('base_id', binding['base_id']), ('source', binding['source'])]:
        if spec.get(key) != value:
            raise ValueError('spec_identity_mismatch:' + key)
    if spec.get('source_binding', binding) != binding:
        raise ValueError('spec_source_binding_changed')

def locale_parent(spec, qa, binding):
    return {'source': binding, 'labels_sha256': digest(spec['labels']), 'qa_sha256': digest({k: qa[k] for k in ('question', 'answer', 'answer_template')})}

def verify_locale(locale, spec, qa, binding):
    if qa.get('source_binding', binding) != binding:
        raise ValueError('qa_source_binding_mismatch')
    if qa.get('label_dictionary_sha256', digest(spec['labels'])) != digest(spec['labels']):
        raise ValueError('qa_label_dictionary_changed')
    expected = locale_parent(spec, qa, binding)
    if 'input_binding' in locale and locale['input_binding'] != expected:
        raise ValueError('locale_input_binding_mismatch')
    if set(locale['labels']) != set(spec['labels']):
        raise ValueError('locale_label_keys_mismatch')
    for key in ('question', 'answer_template'):
        if sorted(re.findall('\\[\\[([^\\]]+)\\]\\]', locale[key])) != sorted(re.findall('\\[\\[([^\\]]+)\\]\\]', qa[key])):
            raise ValueError('locale_placeholder_mismatch:' + key)

def verify_polish(record, case_id, language, original_query, locale):
    expected = {k: locale['labels'][k] for k in re.findall('\\[\\[([^\\]]+)\\]\\]', locale['question'])}
    if record['id'] != case_id or record['language'] != language or record['original_source_query'] != original_query or (record['current_query'] != locale['question']) or (record['protected_bindings'] != expected):
        raise ValueError('query_polish_source_mismatch:' + case_id + ':' + language)

def render_binding(spec, labels):
    return {'spec_sha256': digest(spec), 'labels_sha256': digest(labels), 'runtime_sha256': sha256(Path(__file__).with_name('render_runtime.py')), 'worker_sha256': sha256(Path(__file__).with_name('worker.py')), 'drawing_sha256': sha256(Path(__file__).resolve().parents[2] / 'skills/multilingual-visual-benchmark/scripts/multilingual_drawing.py')}

def render_inputs(folder, languages):
    folder = Path(folder)
    spec = read(folder / 'spec.json')
    labels = {lang: read(folder / 'locales' / (lang + '.json'))['labels'] for lang in languages}
    return render_binding(spec, labels)

def render_is_current(image, spec, labels):
    image = Path(image)
    report_path = image.with_suffix('.layout.json')
    if not image.exists() or not report_path.exists():
        return False
    report = read(report_path)
    return report.get('render_binding') == render_binding(spec, labels) and report.get('image_sha256') == sha256(image) and report.get('all_text_inside_canvas') and report.get('all_text_inside_cells') and (not report.get('missing_glyphs'))

def verify_legacy_render(code, spec, language, labels):
    pass
    constants = {}
    for node in ast.parse(Path(code).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('BASE_ID', 'LANGUAGE', 'DATA', 'LABELS'):
                    constants[target.id] = ast.literal_eval(node.value)
    if constants != {'BASE_ID': spec['base_id'], 'LANGUAGE': language, 'DATA': spec['data'], 'LABELS': labels}:
        raise ValueError('legacy_render_inputs_changed: rerender before export')

def require_table_only(extracted):
    if extracted.get('non_tabular_visuals'):
        raise ValueError('mixed_visual_requires_full_reconstruction')
    notes = ' '.join(extracted.get('recovery', {}).get('uncertainties', []))
    if re.search('\\b(?:radial|circular|embedded|accompanying)\\s+(?:diagram|chart)|labels from .*diagram', notes, re.I):
        raise ValueError('mixed_visual_requires_full_reconstruction')
