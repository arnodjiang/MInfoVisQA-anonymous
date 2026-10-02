pass
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from scripts.final_benchmark import prompts, polish_queries, review, val_audit, audit24, pipeline
    from scripts.final_benchmark.languages24 import NEW_LANGUAGES, RULES
    dest = ROOT / 'prompts'
    dest.mkdir(exist_ok=True)
    entries = {name.lower() + '.txt': value for (name, value) in vars(prompts).items() if name.isupper() and isinstance(value, str)}
    entries.update({'query_copyedit.txt': polish_queries.COMMON, 'query_equivalence_review.txt': polish_queries.REVIEW, 'source_review.txt': review.PROMPT, 'baseline_multilingual_audit.txt': val_audit.PROMPT, 'expanded_language_audit.txt': audit24.PROMPT})
    languages = dict(pipeline.LANGUAGES, **NEW_LANGUAGES)
    for (lang, rule) in dict(polish_queries.RULES, **RULES).items():
        entries['query/' + lang + '.txt'] = polish_queries.COMMON + '\nTarget language: ' + languages[lang] + '\n' + rule + '\n'
    index = {}
    for (name, text) in entries.items():
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        index[name] = {'sha256': hashlib.sha256(text.encode()).hexdigest()}
    (dest / 'manifest.json').write_text(json.dumps(index, indent=2) + '\n')
    (ROOT / 'configs/languages.json').write_text(json.dumps({'languages': languages, 'cross_language_query_languages': ['zh', 'en'], 'answer_policy': 'same as query', 'configurations_per_base_qa': 3 * len(languages) - 2}, indent=2) + '\n')
    print('Exported', len(entries), 'prompt snapshots for', len(languages), 'languages.')
if __name__ == '__main__':
    main()
