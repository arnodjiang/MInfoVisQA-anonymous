pass
import hashlib
import json
from pathlib import Path
import zipfile
from .pipeline import ROOT

def main():
    files = []
    for directory in ('scripts/final_benchmark', 'skills/multilingual-visual-benchmark'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and (path.suffix in ('.py', '.md', '.txt', '.json', '.yaml') or path.name == 'LICENSE'):
                files.append(path)
    for name in ('profile_and_sample.py', 'build_expansion_registry.py', 'export_candidate_assets.py', 'download_benchmarks.py', 'load_local.py'):
        files.append(ROOT / 'scripts' / name)
    for name in ('BENCHMARK_DESIGN.md', 'EXPANSION_GUIDE.md'):
        files.append(ROOT / name)
    for name in ('CharXiv', 'ChartQA', 'ChartQAPro', 'MMTU', 'TableVQA-Bench', 'Visual-TableQA'):
        files.append(ROOT / 'data/metadata' / (name + '.json'))
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    destination = ROOT / 'data/releases/mvisqa-benchmark-source-v0.3.0.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        archive.writestr('README.md', '# MVisQA reproducible source package\n\nStart with scripts/final_benchmark/README.md. The reusable skill is in skills/multilingual-visual-benchmark/. Its synthetic examples run without upstream datasets. Supply your own API configuration and Unicode font.\n\nThis archive contains code, synthetic examples and public upstream revision metadata, not dataset assets, API responses, credentials, or system fonts. Dataset terms remain applicable. SOURCE_MANIFEST.json records every included source file.\n')
    print(destination)
    print('Source files:', len(files))
if __name__ == '__main__':
    main()
