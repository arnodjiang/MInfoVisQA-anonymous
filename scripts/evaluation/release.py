pass
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def check_dataset(dataset):
    config = json.loads((ROOT / 'configs/benchmark_release.json').read_text())
    retired = {(ROOT / p).resolve() for p in config['retired_datasets']}
    if Path(dataset).resolve() in retired:
        raise ValueError('Dataset retired; current benchmark: ' + config['active_dataset'])
