pass
import ast
from pathlib import Path
from .api import read, save, digest, now
from .pipeline import DEFAULT_OUT, LANGUAGES

def main():
    results = []
    for folder in sorted((DEFAULT_OUT / 'cases').iterdir()):
        if not folder.is_dir():
            continue
        if not (folder / 'render_complete.json').exists():
            continue
        spec = read(folder / 'render_spec.json')
        for lang in LANGUAGES:
            path = folder / 'code' / ('original' if lang == 'en' else 'translated') / lang / 'render.py'
            errors = []
            try:
                tree = ast.parse(path.read_text())
                constants = {}
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in ('DATA', 'LABELS', 'LANGUAGE', 'BASE_ID'):
                                constants[target.id] = ast.literal_eval(node.value)
                if digest(constants['DATA']) != digest(spec['data']):
                    errors.append('embedded_data_mismatch')
                if constants['LABELS'] != read(folder / 'locales' / f'{lang}.json')['labels']:
                    errors.append('embedded_labels_mismatch')
                if constants['LANGUAGE'] != lang or constants['BASE_ID'] != spec['base_id']:
                    errors.append('embedded_identity_mismatch')
            except Exception as exc:
                errors.append(type(exc).__name__)
            results.append({'id': folder.name, 'language': lang, 'errors': errors})
    save(DEFAULT_OUT / 'validation_release/code_invariance.json', {'checked_at': now(), 'checked': len(results), 'passed': sum((not x['errors'] for x in results)), 'scope': 'All exported AST embedded data/labels/identity, not pixel execution or source truth', 'failures': [r for r in results if r['errors']]})
    print('code invariance', len(results), 'failed', sum((bool(x['errors']) for x in results)))
if __name__ == '__main__':
    main()
