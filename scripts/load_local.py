pass
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT / 'data/cache/datasets'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

def local_files(name):
    folder = ROOT / 'data/raw' / name
    files = sorted(folder.rglob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No downloaded Parquet files in {folder}')
    result = {}
    for file in files:
        split = file.stem.split('-')[0]
        if name == 'CharXiv' and split == 'val':
            split = 'validation'
        result.setdefault(split, []).append(str(file))
    return result

def load_local(name):
    from datasets import load_dataset
    return load_dataset('parquet', data_files=local_files(name))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['TableVQA-Bench', 'CharXiv', 'ChartQAPro', 'Visual-TableQA', 'ChartQA', 'MMTU'])
    args = parser.parse_args()
    print(load_local(args.name))
