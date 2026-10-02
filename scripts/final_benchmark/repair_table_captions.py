pass
import argparse
import html
import re
import shutil
from .api import read, save, digest, now
from .pipeline import Builder, DEFAULT_OUT, table_spec

def main():
    b = Builder(argparse.Namespace(output=str(DEFAULT_OUT), retry_failed=False))
    for identifier in ('5cabdebbe28b174ed423', '880670c8479fd04fc8af'):
        folder = b.folder(identifier)
        extracted = read(folder / 'table_extraction.json')
        raw = (folder / 'original/table.html').read_text()
        caption = re.search('<caption\\b[^>]*>(.*?)</caption>', raw, flags=re.I | re.S)
        if not caption:
            continue
        title = html.unescape(re.sub('<[^>]+>', '', caption[1])).strip()
        if extracted.get('title') == title:
            continue
        for path in (folder / 'locales').glob('*.json'):
            if path.stem != 'en':
                raise ValueError('Localized caption requires a new translation revision')
        old = read(folder / 'spec.json')
        save(folder / 'reconstruction_before_caption.json', old)
        extracted['title'] = title
        save(folder / 'table_extraction.json', extracted)
        spec = table_spec(extracted)
        for key in ('id', 'base_id', 'source', 'recovery_request_sha256'):
            spec[key] = old.get(key)
        save(folder / 'spec.json', spec)
        save(folder / 'english_labels.json', {'labels': spec['labels']})
        if (folder / 'locales/en.json').exists():
            loc = read(folder / 'locales/en.json')
            loc['labels'] = spec['labels']
            save(folder / 'locales/en.json', loc)
        b.render_one(identifier, folder / 'spec.json', folder / 'english_labels.json', folder / 'baseline.png')
        save(folder / 'recovery_complete.json', {'finished_at': now(), 'spec_sha256': digest(spec), 'mechanical_render_pass': True, 'semantic_fidelity': 'pending_api_review', 'caption_preserved': True})
        print('Caption retained ' + identifier, flush=True)
if __name__ == '__main__':
    main()
